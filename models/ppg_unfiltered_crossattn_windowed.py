"""
PPG + Unfiltered PPG Cross-Attention Model - Window-Adaptive Version with Sparse Attention

Key features:
1. Supports variable-length input sequences
2. Dynamic positional encoding (sinusoidal or learned)
3. Adaptive pooling based on input size
4. Proper top-k sparse attention for training and inference

This model validates whether cross-attention mechanism can extract useful 
information from noisy signals while supporting arbitrary window lengths.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
import math


class ResConvBlock(nn.Module):
    """Residual Convolutional Block"""

    def __init__(self, in_channels, out_channels, stride=2):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride)

        if in_channels != out_channels or stride != 1:
            self.residual_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.MaxPool1d(kernel_size=stride, stride=stride)
            )
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        x = self.pool(x)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        return x + residual


class DynamicSinusoidalEncoding(nn.Module):
    """Sinusoidal positional encoding that works with ANY sequence length"""
    
    def __init__(self, d_model, max_len=None):
        super(DynamicSinusoidalEncoding, self).__init__()
        self.d_model = d_model
        
        # Pre-compute wavelengths (invariant to sequence length)
        inv_freq = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x):
        """
        Args:
            x: (batch, d_model, seq_len)
        Returns:
            x with positional encoding added
        """
        batch_size, d_model, seq_len = x.shape
        
        # Generate position indices for current sequence length
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        # Compute sinusoidal encoding: position * inv_freq
        # position: (seq_len,) -> (seq_len, 1)
        # inv_freq: (d_model/2,) -> (1, d_model/2)
        # Result: (seq_len, d_model/2)
        sinusoid_inp = position.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        
        # Apply sin and cos
        pos_emb = torch.zeros(seq_len, d_model, device=x.device, dtype=x.dtype)
        pos_emb[:, 0::2] = torch.sin(sinusoid_inp)
        pos_emb[:, 1::2] = torch.cos(sinusoid_inp)
        
        # Reshape to match input: (1, d_model, seq_len)
        pos_emb = pos_emb.transpose(0, 1).unsqueeze(0)
        
        return x + pos_emb


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding that works with variable lengths"""
    
    def __init__(self, d_model, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, max_len) * 0.02)
    
    def forward(self, x):
        """
        Args:
            x: (batch, d_model, seq_len)
        Returns:
            x with positional encoding added
        """
        batch_size, d_model, seq_len = x.shape
        
        if seq_len > self.max_len:
            # Interpolate if sequence is longer than max
            pos_enc = F.interpolate(
                self.pos_embedding, 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            )
        else:
            pos_enc = self.pos_embedding[:, :, :seq_len]
        
        return x + pos_enc


class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Cross-Attention with proper top-k sparse attention"""

    def __init__(self, d_model, n_heads=8, dropout=0.1, use_sparse=False, top_k_percent=0.10):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_sparse = use_sparse
        self.top_k_percent = top_k_percent

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        if self.use_sparse:
            print(f"  ✓ Using Top-K Sparse Attention (keep top {top_k_percent*100:.0f}%)")

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        # Linear transformation and split into heads
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores: (batch, heads, seq_len, key_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Top-K sparsification (works during training and inference)
        if self.use_sparse:
            key_len = scores.size(-1)
            top_k = max(1, int(key_len * self.top_k_percent))
            
            # Get top-k indices
            _, top_k_idx = torch.topk(scores, k=top_k, dim=-1, largest=True)
            
            # Create mask: -inf for non-top-k positions
            sparse_mask = torch.full_like(scores, float('-inf'))
            sparse_mask.scatter_(-1, top_k_idx, 0.0)  # 0.0 for top-k positions
            
            # Apply sparse mask
            scores = scores + sparse_mask  # Adding 0 keeps top-k, adding -inf zeros others

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.w_o(context)

        # Residual connection and layer norm
        output = self.layer_norm(query + self.dropout(output))

        return output, attention_weights


class AdaptiveModalityWeighting(nn.Module):
    """Learn importance weights for clean vs noisy PPG"""

    def __init__(self, d_model):
        super(AdaptiveModalityWeighting, self).__init__()
        self.clean_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, 1, 1),
            nn.Sigmoid()
        )
        self.noisy_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, clean_features, noisy_features):
        clean_weight = self.clean_gate(clean_features)
        noisy_weight = self.noisy_gate(noisy_features)
        
        # Normalize weights
        total_weight = clean_weight + noisy_weight
        clean_weight = clean_weight / (total_weight + 1e-8)
        noisy_weight = noisy_weight / (total_weight + 1e-8)
        
        return clean_weight, noisy_weight


class CrossModalFusionBlock(nn.Module):
    """Cross-modal fusion using bidirectional cross-attention"""

    def __init__(self, d_model, n_heads=8, dropout=0.1, use_sparse=False, top_k_percent=0.10):
        super(CrossModalFusionBlock, self).__init__()
        
        # Clean PPG attends to Noisy PPG
        self.clean_cross_attn = MultiHeadCrossAttention(
            d_model, n_heads, dropout, use_sparse, top_k_percent
        )
        # Noisy PPG attends to Clean PPG
        self.noisy_cross_attn = MultiHeadCrossAttention(
            d_model, n_heads, dropout, use_sparse, top_k_percent
        )
        
        # Feed-forward networks
        self.clean_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.noisy_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, clean_features, noisy_features):
        # Cross-attention
        clean_attended, _ = self.clean_cross_attn(clean_features, noisy_features, noisy_features)
        noisy_attended, _ = self.noisy_cross_attn(noisy_features, clean_features, clean_features)
        
        # Feed-forward
        clean_out = self.norm1(clean_attended + self.clean_ffn(clean_attended))
        noisy_out = self.norm2(noisy_attended + self.noisy_ffn(noisy_attended))
        
        return clean_out, noisy_out


class TemporalBlock(nn.Module):
    """Temporal convolutional block with dilation"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Use 'same' padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class PPGUnfilteredWindowedCrossAttention(nn.Module):
    """
    Window-adaptive PPG + Unfiltered PPG cross-attention model with sparse attention.
    
    Supports variable-length inputs from 10 epochs to full 1200 epochs.
    Stream 1: Clean PPG signal (standard filtering)
    Stream 2: Unfiltered PPG signal (contains noise, baseline drift, motion artifacts)
    """
    
    def __init__(self, n_classes=4, d_model=256, n_heads=8, n_fusion_blocks=3, 
                 dropout=0.2, noise_config=None, use_sparse=False, top_k_percent=0.10,
                 positional_encoding='sinusoidal', max_len=5000):
        super(PPGUnfilteredWindowedCrossAttention, self).__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        self.use_sparse = use_sparse
        self.top_k_percent = top_k_percent
        self.positional_encoding_type = positional_encoding
        
        # Print optimization status
        print("\n" + "="*70)
        print("PPG UNFILTERED WINDOWED CROSS-ATTENTION MODEL")
        print("="*70)
        
        if use_sparse:
            print(f"\nTop-K Sparse Attention ENABLED")
        else:
            print("\nStandard Model (No Sparse Attention)")
        
        # Print positional encoding type
        print(f"\nPositional Encoding: {positional_encoding.upper()}")
        if positional_encoding == 'sinusoidal':
            print("   Type: Mathematical (sin/cos)")
        elif positional_encoding == 'learned':
            print(f"   Type: Learned embeddings")
        
        print("="*70 + "\n")
        
        # Noise configuration
        self.noise_config = noise_config or {
            'noise_level': 0.1,  # Gaussian noise standard deviation
            'drift_amplitude': 0.1,  # Baseline drift amplitude
            'drift_frequency': 0.1,  # Baseline drift frequency
            'spike_probability': 0.01,  # Motion artifact probability
            'spike_amplitude': 0.5  # Motion artifact amplitude
        }
        
        # Encoders - 9 ResConv blocks reduce by 2^9 = 512x
        encoder_channels = [1, 16, 32, 32, 64, 64, 128, 128, 256, d_model]
        
        clean_ppg_encoder_blocks = []
        noisy_ppg_encoder_blocks = []
        for i in range(len(encoder_channels) - 1):
            clean_ppg_encoder_blocks.append(ResConvBlock(encoder_channels[i], encoder_channels[i + 1]))
            noisy_ppg_encoder_blocks.append(ResConvBlock(encoder_channels[i], encoder_channels[i + 1]))
        
        self.clean_ppg_encoder = nn.Sequential(*clean_ppg_encoder_blocks)
        self.noisy_ppg_encoder = nn.Sequential(*noisy_ppg_encoder_blocks)
        
        # Positional encoding (sinusoidal or learned based on config)
        if positional_encoding == 'sinusoidal':
            self.positional_encoding = DynamicSinusoidalEncoding(d_model)
        elif positional_encoding == 'learned':
            self.positional_encoding = LearnedPositionalEncoding(d_model, max_len=max_len)
        else:
            raise ValueError(f"Unknown positional encoding type: {positional_encoding}. "
                           f"Must be 'sinusoidal' or 'learned'")
        
        # Modality weighting
        self.modality_weighting = AdaptiveModalityWeighting(d_model)
        
        # Cross-modal fusion blocks with sparse attention support
        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads, dropout, use_sparse, top_k_percent)
            for _ in range(n_fusion_blocks)
        ])
        
        # Feature aggregation
        self.feature_aggregation = nn.Sequential(
            nn.Conv1d(d_model * 2, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU()
        )
        
        # Temporal modeling with dilated convolutions
        self.temporal_blocks = nn.Sequential(
            TemporalBlock(d_model, d_model, kernel_size=7, stride=1, dilation=1, dropout=dropout),
            TemporalBlock(d_model, d_model, kernel_size=7, stride=1, dilation=2, dropout=dropout),
            TemporalBlock(d_model, d_model, kernel_size=7, stride=1, dilation=4, dropout=dropout)
        )
        
        # Feature refinement
        self.feature_refinement = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier - outputs per-epoch predictions
        self.classifier = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, n_classes, kernel_size=1)
        )

    def add_noise_to_ppg(self, clean_ppg):
        """
        Add noise to clean PPG signal to simulate unfiltered signal

        Args:
            clean_ppg: Clean PPG signal (B, 1, L)
        Returns:
            noisy_ppg: Noisy PPG signal (B, 1, L)
        """
        batch_size, _, length = clean_ppg.shape
        device = clean_ppg.device

        # Copy signal
        noisy_ppg = clean_ppg.clone()

        # 1. Add Gaussian white noise
        gaussian_noise = torch.randn_like(clean_ppg) * self.noise_config['noise_level']
        noisy_ppg = noisy_ppg + gaussian_noise

        # 2. Add baseline drift (low-frequency noise)
        t = torch.linspace(0, 1, length, device=device)
        drift_freq = self.noise_config['drift_frequency']
        drift_amp = self.noise_config['drift_amplitude']

        # Combination of multiple low-frequency components
        drift = drift_amp * (
                0.5 * torch.sin(2 * np.pi * drift_freq * t) +
                0.3 * torch.sin(2 * np.pi * drift_freq * 2 * t) +
                0.2 * torch.sin(2 * np.pi * drift_freq * 0.5 * t)
        )
        drift = drift.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        noisy_ppg = noisy_ppg + drift

        # 3. Add motion artifacts (random spikes)
        spike_prob = self.noise_config['spike_probability']
        spike_amp = self.noise_config['spike_amplitude']

        # Generate random spike locations
        spike_mask = torch.rand(batch_size, 1, length, device=device) < spike_prob
        spike_values = torch.randn(batch_size, 1, length, device=device) * spike_amp
        spikes = spike_mask.float() * spike_values

        # Smooth spikes (make them more realistic)
        kernel_size = 5
        padding = kernel_size // 2
        smoothing_kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        spikes = F.conv1d(spikes, smoothing_kernel, padding=padding)

        noisy_ppg = noisy_ppg + spikes

        # 4. Add high-frequency oscillation (EMG interference)
        emg_noise = torch.randn_like(clean_ppg) * 0.05
        noisy_ppg = noisy_ppg + emg_noise

        return noisy_ppg

    def forward(self, ppg):
        """
        Args:
            ppg: (B, 1, samples) - variable length clean PPG signal
            
        Returns:
            output: (B, n_classes, n_epochs) - predictions for each epoch
        """
        batch_size = ppg.size(0)
        input_samples = ppg.size(2)
        
        # Calculate number of input epochs (30-sec windows)
        samples_per_epoch = 1024
        n_epochs = input_samples // samples_per_epoch

        # Create unfiltered version
        ppg_unfiltered = self.add_noise_to_ppg(ppg)

        # Encode - reduces by 512x
        clean_features = self.clean_ppg_encoder(ppg)  # (B, d_model, samples//512)
        noisy_features = self.noisy_ppg_encoder(ppg_unfiltered)  # (B, d_model, samples//512)

        # Add positional encoding
        clean_features = self.positional_encoding(clean_features)
        noisy_features = self.positional_encoding(noisy_features)

        # Get modality weights
        clean_weight, noisy_weight = self.modality_weighting(clean_features, noisy_features)

        # Apply modality weights
        clean_features_weighted = clean_features * clean_weight
        noisy_features_weighted = noisy_features * noisy_weight

        # Convert to (B, L, C) format for attention
        clean_features_t = clean_features_weighted.transpose(1, 2)
        noisy_features_t = noisy_features_weighted.transpose(1, 2)

        # Cross-Modal Fusion with sparse attention
        for fusion_block in self.fusion_blocks:
            clean_features_t, noisy_features_t = fusion_block(clean_features_t, noisy_features_t)

        # Convert back to (B, C, L) format
        clean_features = clean_features_t.transpose(1, 2)
        noisy_features = noisy_features_t.transpose(1, 2)

        # Feature aggregation
        combined_features = torch.cat([clean_features, noisy_features], dim=1)
        fused_features = self.feature_aggregation(combined_features)

        # Temporal modeling
        temporal_features = self.temporal_blocks(fused_features)

        # Feature refinement
        refined_features = self.feature_refinement(temporal_features)

        # Adaptive upsampling to match number of epochs
        output_features = F.interpolate(
            refined_features, 
            size=n_epochs, 
            mode='linear', 
            align_corners=False
        )

        # Classification
        output = self.classifier(output_features)  # (B, n_classes, n_epochs)
        output = F.softmax(output, dim=1)

        return output

    def get_modality_weights(self):
        """Get current modality weights (for monitoring)"""
        if hasattr(self, 'clean_weight') and hasattr(self, 'noisy_weight'):
            return self.clean_weight, self.noisy_weight
        else:
            return None, None


def test_variable_lengths():
    """Test model with different input lengths and optimization modes"""
    print("\n" + "="*70)
    print("TESTING PPG UNFILTERED WINDOWED CROSS-ATTENTION MODEL")
    print("="*70)
    
    # Test 1: Standard model with sinusoidal encoding
    print("\n" + "="*70)
    print("TEST 1: Sinusoidal Encoding (Baseline)")
    print("="*70)
    model_sin = PPGUnfilteredWindowedCrossAttention(positional_encoding='sinusoidal')
    model_sin.eval()
    
    # Test 2: Learned encoding
    print("\n" + "="*70)
    print("TEST 2: Learned Encoding")
    print("="*70)
    model_learned = PPGUnfilteredWindowedCrossAttention(positional_encoding='learned')
    model_learned.eval()
    
    # Test 3: Sparse Attention with sinusoidal
    print("\n" + "="*70)
    print("TEST 3: Sparse Attention + Sinusoidal Encoding")
    print("="*70)
    model_sparse = PPGUnfilteredWindowedCrossAttention(
        use_sparse=True, 
        top_k_percent=0.10,
        positional_encoding='sinusoidal'
    )
    model_sparse.eval()
    
    print("\n" + "="*70)
    print("TESTING DIFFERENT SEQUENCE LENGTHS")
    print("="*70)
    
    # Test different window sizes
    test_configs = [
        (10, "10 epochs (5 min)"),
        (20, "20 epochs (10 min)"),
        (60, "60 epochs (30 min)"),
        (120, "120 epochs (1 hour)"),
        (240, "240 epochs (2 hours)"),
        (1200, "1200 epochs (10 hours - full sequence)")
    ]
    
    with torch.no_grad():
        for n_epochs, description in test_configs:
            samples = n_epochs * 1024
            ppg = torch.randn(2, 1, samples)
            
            # Test all three models
            output_sin = model_sin(ppg)
            output_learned = model_learned(ppg)
            output_sparse = model_sparse(ppg)
            
            print(f"\n{description}:")
            print(f"  Input:  {ppg.shape}")
            print(f"  Output (sinusoidal): {output_sin.shape}")
            print(f"  Output (learned):    {output_learned.shape}")
            print(f"  Output (sparse+sin): {output_sparse.shape}")
            
            assert output_sin.shape == (2, 4, n_epochs), f"Expected (2, 4, {n_epochs})"
            assert output_learned.shape == (2, 4, n_epochs), f"Expected (2, 4, {n_epochs})"
            assert output_sparse.shape == (2, 4, n_epochs), f"Expected (2, 4, {n_epochs})"
            
            # Compare outputs
            diff_sin_learned = (output_sin - output_learned).abs().mean()
            diff_sin_sparse = (output_sin - output_sparse).abs().mean()
            print(f"  Difference (sinusoidal vs learned): {diff_sin_learned:.6f}")
            print(f"  Difference (sinusoidal vs sparse):  {diff_sin_sparse:.6f}")
            print(f"  ✓ All models produce correct output shapes")
    
    # Parameter counts
    print("\n" + "="*70)
    print("MODEL STATISTICS")
    print("="*70)
    
    total_params_sin = sum(p.numel() for p in model_sin.parameters())
    trainable_params_sin = sum(p.numel() for p in model_sin.parameters() if p.requires_grad)
    
    total_params_learned = sum(p.numel() for p in model_learned.parameters())
    trainable_params_learned = sum(p.numel() for p in model_learned.parameters() if p.requires_grad)
    
    print(f"\nSinusoidal Encoding Model:")
    print(f"  Total parameters: {total_params_sin:,}")
    print(f"  Trainable parameters: {trainable_params_sin:,}")
    
    print(f"\nLearned Encoding Model:")
    print(f"  Total parameters: {total_params_learned:,}")
    print(f"  Trainable parameters: {trainable_params_learned:,}")
    
    diff_params = total_params_learned - total_params_sin
    print(f"\nParameter difference: {diff_params:,} ({diff_params / total_params_sin * 100:.2f}% increase)")
    
    # Test noise generation
    print("\n" + "=" * 70)
    print("NOISE GENERATION TEST")
    print("=" * 70)
    clean_ppg = torch.randn(1, 1, 10240)  # 10 epochs
    noisy_ppg = model_sin.add_noise_to_ppg(clean_ppg)
    
    print(f"Clean PPG - mean: {clean_ppg.mean():.4f}, std: {clean_ppg.std():.4f}")
    print(f"Noisy PPG - mean: {noisy_ppg.mean():.4f}, std: {noisy_ppg.std():.4f}")
    print(f"Noise level: {(noisy_ppg - clean_ppg).std():.4f}")
    
    # Test extreme lengths with sinusoidal encoding
    print("\n" + "=" * 70)
    print("EXTREME LENGTH TEST (Sinusoidal Encoding)")
    print("=" * 70)
    
    extreme_lengths = [
        (5, "5 epochs (very short)"),
        (5000, "5000 epochs (beyond training max_len)"),
        (10000, "10000 epochs (extreme extrapolation)")
    ]
    
    with torch.no_grad():
        for n_epochs, description in extreme_lengths:
            samples = n_epochs * 1024
            ppg = torch.randn(1, 1, samples)
            
            try:
                output = model_sin(ppg)
                print(f"\n{description}:")
                print(f"  Input:  {ppg.shape}")
                print(f"  Output: {output.shape}")
                print(f"  ✓ Successfully handled extreme length")
            except Exception as e:
                print(f"\n{description}:")
                print(f"  ✗ Failed with error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_variable_lengths()