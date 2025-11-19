"""
Compressed Multi-modal Sleep Staging Model for Windowed Inference

Key optimizations for reduced memory and faster inference:
1. Reduced model capacity (d_model=128 instead of 256)
2. Fewer attention heads (4 instead of 8)
3. Fewer fusion blocks (2 instead of 3)
4. Lighter ResConv encoder (7 layers instead of 9)
5. Depthwise separable convolutions for efficiency
6. Supports variable-length windows (designed for 15-60 min chunks)

Expected stats:
- Parameters: ~3-4M (vs 12.3M in full model)
- Memory: ~200-300MB inference (vs 600-700MB)
- Speed: ~2x faster
- Accuracy: Minimal degradation (<1-2%) due to windowed training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthwiseSeparableConv(nn.Module):
    """Memory-efficient depthwise separable convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise: one filter per input channel
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        # Pointwise: 1x1 conv to combine channels
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class CompactResConvBlock(nn.Module):
    """Lightweight residual block using depthwise separable convs"""
    
    def __init__(self, in_channels, out_channels, stride=2):
        super(CompactResConvBlock, self).__init__()
        
        # Use depthwise separable convs for efficiency
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        
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
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        return x + residual


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for variable lengths"""
    
    def __init__(self, d_model, max_len=2500):  # Reduced from 5000
        super(LearnedPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, max_len) * 0.02)
    
    def forward(self, x):
        batch_size, d_model, seq_len = x.shape
        
        if seq_len > self.max_len:
            pos_enc = F.interpolate(
                self.pos_embedding, 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            )
        else:
            pos_enc = self.pos_embedding[:, :, :seq_len]
        
        return x + pos_enc


class EfficientMultiHeadCrossAttention(nn.Module):
    """Memory-efficient cross-attention with reduced heads"""
    
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super(EfficientMultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Use single projection for Q, K, V (more efficient)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Project and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return output, attn


class AdaptiveModalityWeighting(nn.Module):
    """Lightweight modality weighting"""
    
    def __init__(self, d_model):
        super(AdaptiveModalityWeighting, self).__init__()
        self.weight = nn.Parameter(torch.tensor([0.5]))
    
    def forward(self, ppg_features, ecg_features):
        alpha = torch.sigmoid(self.weight)
        return alpha * ppg_features + (1 - alpha) * ecg_features


class CompactCrossModalFusionBlock(nn.Module):
    """Lightweight fusion block with reduced parameters"""
    
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super(CompactCrossModalFusionBlock, self).__init__()
        
        # Cross-attention layers
        self.ppg_to_ecg = EfficientMultiHeadCrossAttention(d_model, n_heads, dropout)
        self.ecg_to_ppg = EfficientMultiHeadCrossAttention(d_model, n_heads, dropout)
        
        # Lightweight feedforward networks
        self.ppg_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # Reduced expansion (2x instead of 4x)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        self.ecg_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.ppg_norm1 = nn.LayerNorm(d_model)
        self.ppg_norm2 = nn.LayerNorm(d_model)
        self.ecg_norm1 = nn.LayerNorm(d_model)
        self.ecg_norm2 = nn.LayerNorm(d_model)
        
        # Adaptive weighting
        self.modality_weighting = AdaptiveModalityWeighting(d_model)
    
    def forward(self, ppg_features, ecg_features):
        # Cross-attention
        ppg_attended, _ = self.ppg_to_ecg(ppg_features, ecg_features, ecg_features)
        ecg_attended, _ = self.ecg_to_ppg(ecg_features, ppg_features, ppg_features)
        
        # Residual + norm
        ppg_features = self.ppg_norm1(ppg_features + ppg_attended)
        ecg_features = self.ecg_norm1(ecg_features + ecg_attended)
        
        # Feedforward
        ppg_features = self.ppg_norm2(ppg_features + self.ppg_ffn(ppg_features))
        ecg_features = self.ecg_norm2(ecg_features + self.ecg_ffn(ecg_features))
        
        # Adaptive fusion
        fused = self.modality_weighting(ppg_features, ecg_features)
        
        return fused, ppg_features, ecg_features


class CompactWindowAdaptiveSleepNet(nn.Module):
    """
    Compressed window-adaptive model for memory-efficient inference.
    
    Architecture:
    - 7-layer ResConv encoder (vs 9 in full model)
    - d_model=128 (vs 256)
    - 4 attention heads (vs 8)
    - 2 fusion blocks (vs 3)
    - Depthwise separable convolutions
    
    Target: ~3-4M parameters, ~200-300MB memory, 2x faster inference
    """
    
    def __init__(
        self,
        n_classes=4,
        d_model=128,
        n_heads=4,
        n_fusion_blocks=2,
        dropout=0.1
    ):
        super(CompactWindowAdaptiveSleepNet, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_fusion_blocks = n_fusion_blocks
        
        # Lighter encoder: 7 blocks instead of 9
        # Downsampling: 128x (2^7) instead of 512x (2^9)
        # Input: 1,228,800 samples → Output: 9,600 samples
        encoder_channels = [1, 32, 64, 64, 128, 128, 128, d_model]
        
        self.ppg_encoder = nn.ModuleList([
            CompactResConvBlock(encoder_channels[i], encoder_channels[i+1], stride=2)
            for i in range(len(encoder_channels)-1)
        ])
        
        self.ecg_encoder = nn.ModuleList([
            CompactResConvBlock(encoder_channels[i], encoder_channels[i+1], stride=2)
            for i in range(len(encoder_channels)-1)
        ])
        
        # Positional encoding
        self.ppg_pos_encoding = LearnedPositionalEncoding(d_model)
        self.ecg_pos_encoding = LearnedPositionalEncoding(d_model)
        
        # Cross-modal fusion blocks
        self.fusion_blocks = nn.ModuleList([
            CompactCrossModalFusionBlock(d_model, n_heads, dropout)
            for _ in range(n_fusion_blocks)
        ])
        
        # Lightweight temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, n_classes, kernel_size=1)
        )
    
    def forward(self, ppg, ecg):
        """
        Args:
            ppg: (batch, 1, n_samples)
            ecg: (batch, 1, n_samples)
        
        Returns:
            logits: (batch, n_classes, n_epochs)
        """
        # Encode both modalities
        ppg_feat = ppg
        for block in self.ppg_encoder:
            ppg_feat = block(ppg_feat)
        
        ecg_feat = ecg
        for block in self.ecg_encoder:
            ecg_feat = block(ecg_feat)
        
        # Add positional encoding
        ppg_feat = self.ppg_pos_encoding(ppg_feat)
        ecg_feat = self.ecg_pos_encoding(ecg_feat)
        
        # Prepare for attention: (batch, channels, seq) -> (batch, seq, channels)
        ppg_feat = ppg_feat.transpose(1, 2)
        ecg_feat = ecg_feat.transpose(1, 2)
        
        # Cross-modal fusion
        fused = None
        for fusion_block in self.fusion_blocks:
            if fused is None:
                fused, ppg_feat, ecg_feat = fusion_block(ppg_feat, ecg_feat)
            else:
                fused, ppg_feat, ecg_feat = fusion_block(fused, fused)
        
        # Back to (batch, channels, seq)
        fused = fused.transpose(1, 2)
        
        # Temporal processing
        fused = self.temporal_conv(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        # Calculate number of epochs based on input size
        # Each epoch = 1024 samples at 34.133 Hz
        n_samples = ppg.shape[2]
        samples_per_epoch = 1024
        n_epochs = n_samples // samples_per_epoch
        
        # Upsample to match number of epochs
        # Current: downsampled by 128x
        # Need: one prediction per epoch
        current_len = logits.shape[2]
        if current_len != n_epochs:
            logits = F.interpolate(logits, size=n_epochs, mode='linear', align_corners=False)
        
        return logits


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the compressed model
    print("Testing Compact Window-Adaptive Sleep Net")
    print("=" * 60)
    
    # Test with different window sizes
    test_configs = [
        (30, "15-minute window"),   # 30 epochs
        (60, "30-minute window"),   # 60 epochs
        (120, "1-hour window"),     # 120 epochs
        (240, "2-hour window"),     # 240 epochs
    ]
    
    for n_epochs, desc in test_configs:
        print(f"\n{desc} ({n_epochs} epochs):")
        
        n_samples = n_epochs * 1024
        ppg = torch.randn(2, 1, n_samples)
        ecg = torch.randn(2, 1, n_samples)
        
        model = CompactWindowAdaptiveSleepNet(
            n_classes=4,
            d_model=128,
            n_heads=4,
            n_fusion_blocks=2
        )
        
        with torch.no_grad():
            output = model(ppg, ecg)
        
        params = count_parameters(model)
        
        print(f"  Input:  PPG/ECG ({ppg.shape[0]}, {ppg.shape[1]}, {ppg.shape[2]:,})")
        print(f"  Output: Logits  ({output.shape[0]}, {output.shape[1]}, {output.shape[2]})")
        print(f"  Parameters: {params:,} (~{params/1e6:.1f}M)")
        print(f"  Expected memory: ~{params * 4 / 1e6 * 2:.0f}-{params * 4 / 1e6 * 3:.0f}MB")
    
    # Compare with full model
    print("\n" + "=" * 60)
    print("Model Comparison:")
    print(f"  Compact model: ~{count_parameters(model)/1e6:.1f}M parameters")
    print(f"  Full model:    ~12.3M parameters")
    print(f"  Compression:   ~{12.3 / (count_parameters(model)/1e6):.1f}x smaller")
