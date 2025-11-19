"""
Improved Multi-modal Sleep Staging Model - Window-Adaptive Version

Key changes from original:
1. Supports variable-length input sequences
2. Dynamic positional encoding
3. No hardcoded output length (1200)
4. Adaptive pooling based on input size
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
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
    """Multi-Head Cross-Attention Mechanism"""

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        # Linear transformation and split into heads
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention calculation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

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
    """Learn importance weights for each modality"""

    def __init__(self, d_model):
        super(AdaptiveModalityWeighting, self).__init__()
        self.ppg_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, 1, 1),
            nn.Sigmoid()
        )
        self.ecg_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, ppg_features, ecg_features):
        ppg_weight = self.ppg_gate(ppg_features)
        ecg_weight = self.ecg_gate(ecg_features)
        
        # Normalize weights
        total_weight = ppg_weight + ecg_weight
        ppg_weight = ppg_weight / (total_weight + 1e-8)
        ecg_weight = ecg_weight / (total_weight + 1e-8)
        
        return ppg_weight, ecg_weight


class CrossModalFusionBlock(nn.Module):
    """Cross-modal fusion using bidirectional cross-attention"""

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(CrossModalFusionBlock, self).__init__()
        
        # PPG attends to ECG
        self.ppg_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        # ECG attends to PPG
        self.ecg_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        
        # Feed-forward networks
        self.ppg_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ecg_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, ppg_features, ecg_features):
        # Cross-attention
        ppg_attended, _ = self.ppg_cross_attn(ppg_features, ecg_features, ecg_features)
        ecg_attended, _ = self.ecg_cross_attn(ecg_features, ppg_features, ppg_features)
        
        # Feed-forward
        ppg_out = self.norm1(ppg_attended + self.ppg_ffn(ppg_attended))
        ecg_out = self.norm2(ecg_attended + self.ecg_ffn(ecg_attended))
        
        return ppg_out, ecg_out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
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


class WindowAdaptiveSleepNet(nn.Module):
    """
    Window-adaptive sleep staging model.
    
    Supports variable-length inputs from 10 epochs to full 1200 epochs.
    """
    
    def __init__(self, n_classes=4, d_model=256, n_heads=8, n_fusion_blocks=3, dropout=0.2):
        super(WindowAdaptiveSleepNet, self).__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        
        # Encoders - 9 ResConv blocks reduce by 2^9 = 512x
        # Input: 1 channel, various lengths
        # Output: d_model channels, length//512
        encoder_channels = [1, 16, 32, 32, 64, 64, 128, 128, 256, d_model]
        
        ppg_encoder_blocks = []
        ecg_encoder_blocks = []
        for i in range(len(encoder_channels) - 1):
            ppg_encoder_blocks.append(ResConvBlock(encoder_channels[i], encoder_channels[i + 1]))
            ecg_encoder_blocks.append(ResConvBlock(encoder_channels[i], encoder_channels[i + 1]))
        
        self.ppg_encoder = nn.Sequential(*ppg_encoder_blocks)
        self.ecg_encoder = nn.Sequential(*ecg_encoder_blocks)
        
        # Learned positional encoding (supports variable lengths)
        self.positional_encoding = LearnedPositionalEncoding(d_model, max_len=5000)
        
        # Modality weighting
        self.modality_weighting = AdaptiveModalityWeighting(d_model)
        
        # Cross-modal fusion blocks
        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads, dropout)
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

    def forward(self, ppg, ecg):
        """
        Args:
            ppg: (B, 1, samples) - variable length
            ecg: (B, 1, samples) - variable length
            
        Returns:
            output: (B, n_classes, n_epochs) - predictions for each epoch
        """
        batch_size = ppg.size(0)
        input_samples = ppg.size(2)
        
        # Calculate number of input epochs (30-sec windows)
        samples_per_epoch = 1024
        n_epochs = input_samples // samples_per_epoch

        # Encode - reduces by 512x
        ppg_features = self.ppg_encoder(ppg)  # (B, d_model, samples//512)
        ecg_features = self.ecg_encoder(ecg)  # (B, d_model, samples//512)

        # Add positional encoding
        ppg_features = self.positional_encoding(ppg_features)
        ecg_features = self.positional_encoding(ecg_features)

        # Get modality weights
        ppg_weight, ecg_weight = self.modality_weighting(ppg_features, ecg_features)

        # Apply modality weights
        ppg_features_weighted = ppg_features * ppg_weight
        ecg_features_weighted = ecg_features * ecg_weight

        # Convert to (B, L, C) format for attention
        ppg_features_t = ppg_features_weighted.transpose(1, 2)
        ecg_features_t = ecg_features_weighted.transpose(1, 2)

        # Cross-Modal Fusion
        for fusion_block in self.fusion_blocks:
            ppg_features_t, ecg_features_t = fusion_block(ppg_features_t, ecg_features_t)

        # Convert back to (B, C, L) format
        ppg_features = ppg_features_t.transpose(1, 2)
        ecg_features = ecg_features_t.transpose(1, 2)

        # Feature aggregation
        combined_features = torch.cat([ppg_features, ecg_features], dim=1)
        fused_features = self.feature_aggregation(combined_features)

        # Temporal modeling
        temporal_features = self.temporal_blocks(fused_features)

        # Feature refinement
        refined_features = self.feature_refinement(temporal_features)

        # Adaptive upsampling to match number of epochs
        # Current: samples//512 -> Target: n_epochs
        # Since 512 samples reduces to 1 feature, and 1024 samples = 1 epoch
        # We need to upsample by factor of 2
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


def test_variable_lengths():
    """Test model with different input lengths"""
    print("Testing Window-Adaptive Model")
    print("=" * 60)
    
    model = WindowAdaptiveSleepNet()
    model.eval()
    
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
            ecg = torch.randn(2, 1, samples)
            
            output = model(ppg, ecg)
            
            print(f"{description}:")
            print(f"  Input:  {ppg.shape}")
            print(f"  Output: {output.shape}")
            assert output.shape == (2, 4, n_epochs), f"Expected (2, 4, {n_epochs}), got {output.shape}"
            print(f"  ✓ Correct output shape\n")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("=" * 60)


if __name__ == "__main__":
    test_variable_lengths()
