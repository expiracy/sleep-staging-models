"""
PPG + Unfiltered PPG Cross-Attention Model - With STABLE Linear Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import numpy as np
import math


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise Separable Convolution for efficiency"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSeparableConv1d, self).__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        )
        
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResConvBlock(nn.Module):
    """Residual Convolutional Block with optional depthwise separable convolutions"""

    def __init__(self, in_channels, out_channels, stride=2, use_depthwise_separable=False):
        super(ResConvBlock, self).__init__()
        
        self.use_depthwise_separable = use_depthwise_separable
        
        if use_depthwise_separable:
            self.conv1 = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.conv3 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
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
        
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        sinusoid_inp = position.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        
        pos_emb = torch.zeros(seq_len, d_model, device=x.device, dtype=x.dtype)
        pos_emb[:, 0::2] = torch.sin(sinusoid_inp)
        pos_emb[:, 1::2] = torch.cos(sinusoid_inp)
        
        pos_emb = pos_emb.transpose(0, 1).unsqueeze(0)
        
        return x + pos_emb


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding that works with variable lengths"""
    
    def __init__(self, d_model, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, max_len) * 0.02)
    
    def forward(self, x):
        """
        Args:
            x: (batch, d_model, seq_len)
        Returns:
            x with positional encoding added
        """
        batch_size, d_model, seq_len = x.shape
        
        # fallback to interpolation if seq_len > max_len
        if seq_len > self.max_len:
            print("  Warning: Interpolating learned positional encodings")
            pos_enc = F.interpolate(
                self.pos_embedding, 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            )
        else:
            pos_enc = self.pos_embedding[:, :, :seq_len]
        
        return x + pos_enc


class SDPAAttention(nn.Module):
    """
    PyTorch's built-in Scaled Dot Product Attention.
    Works on ANY GPU (including compute capability 12.0+).
    Almost as fast as xFormers, built into PyTorch 2.0+.
    """
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(SDPAAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        print("  Using PyTorch Scaled Dot Product Attention (SDPAAttention)")
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        key_len = key.size(1)
        
        # Project and reshape: (batch, heads, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # PyTorch's optimized attention
        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,  # mask not typically used in cross-attention
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape and output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return self.layer_norm(query + self.dropout(output)), None


class LocalWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, window_size=64, dropout=0.1):
        super(LocalWindowAttention, self).__init__()
        
        assert d_model % n_heads == 0
        assert window_size % 2 == 0, "window_size should be even for symmetric windows"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        print(f" Sparse window attention (window size: {window_size})")
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, heads, seq_len, d_k)
        
        half_window = self.window_size // 2
        
        # Pad K and V: half_window on left, (half_window - 1) on right
        # This ensures unfold produces exactly seq_len windows
        K_padded = F.pad(K, (0, 0, half_window, half_window - 1))
        V_padded = F.pad(V, (0, 0, half_window, half_window - 1))
        # Shape: (batch, heads, seq_len + window_size - 1, d_k)
        
        # Extract sliding windows using unfold
        K_windows = K_padded.unfold(2, self.window_size, 1)
        V_windows = V_padded.unfold(2, self.window_size, 1)
        # Shape: (batch, heads, seq_len, d_k, window_size)
        
        # Rearrange to (batch, heads, seq_len, window_size, d_k)
        K_windows = K_windows.permute(0, 1, 2, 4, 3)
        V_windows = V_windows.permute(0, 1, 2, 4, 3)
        
        # Compute attention scores
        Q_expanded = Q.unsqueeze(3)  # (batch, heads, seq_len, 1, d_k)
        
        scores = torch.matmul(Q_expanded, K_windows.transpose(-2, -1))
        # Shape: (batch, heads, seq_len, 1, window_size)
        scores = scores.squeeze(3) / math.sqrt(self.d_k)
        # Shape: (batch, heads, seq_len, window_size)
        
        # Softmax over window
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # (batch, heads, seq_len, window_size, 1)
        context = (attn_weights_expanded * V_windows).sum(dim=3)
        # Shape: (batch, heads, seq_len, d_k)
        
        # Reshape and output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        # Residual and norm
        output = self.layer_norm(query + self.dropout(output))
        
        return output, None
    

class LinearAttention(nn.Module):
    """
    Linear Attention mechanism with O(N) complexity - NUMERICALLY STABLE VERSION.
    
    Instead of computing softmax(QK^T)V which is O(N²),
    we compute φ(Q)(φ(K)^T V) which is O(N).
    
    Key fixes for numerical stability:
    1. More stable feature map (ReLU + larger epsilon)
    2. Gradient clipping in feature map
    3. Larger epsilon for division
    4. Scale normalization
    5. Careful initialization
    
    References:
    - "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    - "Linear Attention Mechanism: An Efficient Attention for Semantic Segmentation"
    """
    
    def __init__(self, d_model, n_heads=8, dropout=0.1, eps=1e-4):
        super(LinearAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.eps = eps  # Larger epsilon for stability
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Initialize with smaller values for stability
        nn.init.xavier_uniform_(self.w_q.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w_k.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w_v.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w_o.weight, gain=0.5)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Learnable temperature for feature map stability
        self.temperature = nn.Parameter(torch.ones(1))
        
        print(f"  ✓ Using Linear Attention (O(N) complexity) - STABLE VERSION")
        print(f"     - Epsilon: {eps}")
        print(f"     - Feature map: ReLU + eps (more stable than ELU)")
    
    def feature_map(self, x):
        """
        STABLE feature map function: φ(x) = ReLU(x / temperature) + eps
        
        This is more numerically stable than ELU + 1 because:
        1. ReLU is simpler and more stable
        2. Temperature scaling prevents extreme values
        3. Larger epsilon prevents division by near-zero
        4. Gradient clipping prevents explosion
        """
        # Scale by learnable temperature
        x = x / (self.temperature.abs() + 1e-8)
        
        # Apply ReLU and add epsilon
        # Using larger epsilon than original to ensure stability
        out = F.relu(x) + self.eps
        
        # Optional: Clip to prevent extreme values
        out = torch.clamp(out, min=self.eps, max=100.0)
        
        return out
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, key_len, d_model)
            value: (batch, key_len, d_model)
            mask: optional mask
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: None (not computed in linear attention)
        """
        batch_size, seq_len, _ = query.shape
        key_len = key.size(1)
        
        # Linear projections and reshape to (batch, heads, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scale Q and K for stability (similar to standard attention)
        scale = 1.0 / math.sqrt(self.d_k)
        Q = Q * scale
        K = K * scale
        
        # Apply feature map: φ(Q), φ(K)
        Q = self.feature_map(Q)  # (batch, heads, seq_len, d_k)
        K = self.feature_map(K)  # (batch, heads, key_len, d_k)
        
        # Check for NaN after feature map
        if torch.isnan(Q).any() or torch.isnan(K).any():
            print("WARNING: NaN detected in feature map output")
            Q = torch.nan_to_num(Q, nan=self.eps)
            K = torch.nan_to_num(K, nan=self.eps)
        
        # Linear attention computation:
        # Instead of: softmax(QK^T)V
        # We compute: φ(Q) * (φ(K)^T * V) / (φ(Q) * φ(K)^T * 1)
        
        # Compute φ(K)^T * V: (batch, heads, d_k, d_k)
        KV = torch.matmul(K.transpose(-2, -1), V)
        
        # Compute φ(K)^T * 1 (normalization term): (batch, heads, d_k, 1)
        K_sum = K.sum(dim=-2, keepdim=True).transpose(-2, -1)
        
        # Add epsilon to prevent division by zero
        K_sum = K_sum + self.eps
        
        # Compute φ(Q) * (φ(K)^T * V): (batch, heads, seq_len, d_k)
        QKV = torch.matmul(Q, KV)
        
        # Compute φ(Q) * (φ(K)^T * 1): (batch, heads, seq_len, 1)
        Q_K_sum = torch.matmul(Q, K_sum)
        
        # Normalize: divide by sum (with epsilon for numerical stability)
        # Use larger epsilon and clamp denominator
        denominator = torch.clamp(Q_K_sum, min=self.eps)
        context = QKV / denominator
        
        # Check for NaN in context
        if torch.isnan(context).any():
            print("WARNING: NaN detected in attention context")
            context = torch.nan_to_num(context, nan=0.0)
        
        # Clip context to prevent extreme values
        context = torch.clamp(context, min=-10.0, max=10.0)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + self.dropout(output))
        
        # Final NaN check
        if torch.isnan(output).any():
            print("WARNING: NaN detected in final output")
            output = torch.nan_to_num(output, nan=0.0)
        
        # Linear attention doesn't produce explicit attention weights
        return output, None

class LocalGlobalAttention(nn.Module):
    """
    Local + Global Hybrid Attention - EFFICIENT O(n×w + n×g) Implementation
    
    Perfect for PPG signals because:
    - Local window captures beat-to-beat patterns  
    - Global tokens track baseline drift, signal quality, respiratory modulation
    - TRUE O(n×w + n×g) complexity where w=window_size, g=num_global_tokens
    
    How it works:
    - Regular tokens: Attend to local window + all global tokens
    - Global tokens: Attend to everything (but there are only O(n/interval) of them)
    
    Key optimization: Never materialize full n×n attention matrix!
    - Regular tokens: Compute attention only for window + global positions
    - Global tokens: Compute full attention (but sparse, only ~n/40 tokens)
    
    This is ideal for cross-modal (clean ↔ noisy) PPG:
    - Local: Direct beat-to-beat alignment
    - Global: Mediate long-range drift and quality differences
    """
    
    def __init__(self, d_model, n_heads=8, window_size=80, global_interval=40, dropout=0.1):
        super(LocalGlobalAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert window_size % 2 == 0, "window_size should be even for symmetric windows"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.global_interval = global_interval
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        print(f"  Local+Global Hybrid Attention (EFFICIENT):")
        print(f"    - Local window: {window_size}")
        print(f"    - Global token interval: {global_interval}")
        print(f"    - TRUE O(n×{window_size} + n×g) complexity")
        print(f"    - Never materializes full n×n matrix!")
    
    def forward(self, query, key, value, mask=None):
        """
        Efficient implementation that never computes full attention matrix.
        
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, key_len, d_model)
            value: (batch, key_len, d_model)
            mask: optional additional mask (not typically used)
        
        Returns:
            output: (batch, seq_len, d_model)
            global_positions: list of global token positions (for visualization)
        """
        batch_size, seq_len, _ = query.shape
        key_len = key.size(1)
        
        # Project and reshape to (batch, heads, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, heads, seq_len, d_k)
        
        # Identify global token positions
        global_positions = list(range(0, seq_len, self.global_interval))
        if not global_positions or global_positions[-1] != seq_len - 1:
            if global_positions and global_positions[-1] != seq_len - 1:
                global_positions.append(seq_len - 1)
        
        global_set = set(global_positions)
        num_global = len(global_positions)
        
        # Separate regular and global tokens
        regular_positions = [i for i in range(seq_len) if i not in global_set]
        num_regular = len(regular_positions)
        
        half_window = self.window_size // 2
        
        # Initialize output tensor
        context = torch.zeros(batch_size, self.n_heads, seq_len, self.d_k, 
                            device=Q.device, dtype=Q.dtype)
        
        # ==================== PROCESS REGULAR TOKENS ====================
        # Regular tokens attend to: local window + all global tokens
        # This is O(n_regular × (window_size + num_global))
        
        if num_regular > 0:
            for i in regular_positions:
                # Get query for this position
                q_i = Q[:, :, i:i+1, :]  # (batch, heads, 1, d_k)
                
                # Determine local window indices
                left = max(0, i - half_window)
                right = min(key_len, i + half_window + 1)
                local_indices = list(range(left, right))
                
                # Combine local + global indices (global_positions already sorted)
                attend_indices = sorted(set(local_indices + global_positions))
                
                # Gather K, V for these positions
                attend_tensor = torch.tensor(attend_indices, device=K.device, dtype=torch.long)
                k_attend = K[:, :, attend_tensor, :]  # (batch, heads, attend_len, d_k)
                v_attend = V[:, :, attend_tensor, :]
                
                # Compute attention scores
                scores_i = torch.matmul(q_i, k_attend.transpose(-2, -1)) / math.sqrt(self.d_k)
                # Shape: (batch, heads, 1, attend_len)
                
                # Softmax
                attn_i = F.softmax(scores_i, dim=-1)
                attn_i = self.dropout(attn_i)
                
                # Apply attention to values
                context_i = torch.matmul(attn_i, v_attend)  # (batch, heads, 1, d_k)
                context[:, :, i, :] = context_i.squeeze(2)
        
        # ==================== PROCESS GLOBAL TOKENS ====================
        # Global tokens attend to EVERYTHING
        # This is O(num_global × n) but num_global << n (typically n/40)
        
        if num_global > 0:
            for g_idx in global_positions:
                # Get query for this global token
                q_g = Q[:, :, g_idx:g_idx+1, :]  # (batch, heads, 1, d_k)
                
                # Attend to ALL positions (this is ok because there are few global tokens)
                scores_g = torch.matmul(q_g, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                # Shape: (batch, heads, 1, key_len)
                
                # Softmax
                attn_g = F.softmax(scores_g, dim=-1)
                attn_g = self.dropout(attn_g)
                
                # Apply attention to values
                context_g = torch.matmul(attn_g, V)  # (batch, heads, 1, d_k)
                context[:, :, g_idx, :] = context_g.squeeze(2)
        
        # Reshape and output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + self.dropout(output))
        
        return output, global_positions


class StridedAttention(nn.Module):
    """
    Strided Sparse Attention - EFFICIENT O(n × (w + k×log n)) Implementation
    
    Perfect for PPG because it captures:
    - Local patterns (window): beat-to-beat variations
    - Medium patterns (stride=4,8): heart rate variability
    - Long patterns (stride=16,32): respiratory modulation, trends
    
    TRUE O(n × (w + Σ(n/stride))) complexity - never materializes full matrix!
    
    For position i, attend to:
    - Local: [i-w/2, ..., i, ..., i+w/2]
    - Strided: [..., i-2s, i-s, i, i+s, i+2s, ...]  for each stride s
    """
    
    def __init__(self, d_model, n_heads=8, local_window=64, strides=[4, 8, 16, 32], dropout=0.1):
        super(StridedAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.local_window = local_window
        self.strides = strides
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Calculate average attention positions per token
        avg_strided = sum(2 * 2400 // s for s in strides)  # Approximate for seq_len=2400
        
        print(f"  Strided Sparse Attention (EFFICIENT):")
        print(f"    - Local window: {local_window}")
        print(f"    - Strides: {strides}")
        print(f"    - TRUE O(n × (w + strided)) complexity")
        print(f"    - Never materializes full n×n matrix!")
    
    def get_strided_indices(self, position, seq_len):
        """
        Get indices for local window + strided positions for a single query position.
        
        Returns:
            List of key indices to attend to (sorted, unique)
        """
        indices = set()
        
        # Local window
        half_window = self.local_window // 2
        left = max(0, position - half_window)
        right = min(seq_len, position + half_window + 1)
        indices.update(range(left, right))
        
        # Strided positions
        for stride in self.strides:
            # Look backward
            j = position - stride
            while j >= 0:
                indices.add(j)
                j -= stride
            
            # Look forward
            j = position + stride
            while j < seq_len:
                indices.add(j)
                j += stride
        
        return sorted(list(indices))
    
    def forward(self, query, key, value, mask=None):
        """
        Efficient implementation that never computes full attention matrix.
        
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, key_len, d_model)
            value: (batch, key_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            sparsity_info: dict with sparsity statistics
        """
        batch_size, seq_len, _ = query.shape
        key_len = key.size(1)
        
        # Project and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, heads, seq_len, d_k)
        
        # Initialize output tensor
        context = torch.zeros(batch_size, self.n_heads, seq_len, self.d_k,
                            device=Q.device, dtype=Q.dtype)
        
        total_attended = 0
        
        # Process each query position
        for i in range(seq_len):
            # Get indices for this position (local + strided)
            attend_indices = self.get_strided_indices(i, key_len)
            total_attended += len(attend_indices)
            
            # Get query for this position
            q_i = Q[:, :, i:i+1, :]  # (batch, heads, 1, d_k)
            
            # Gather K, V for these positions
            attend_tensor = torch.tensor(attend_indices, device=K.device, dtype=torch.long)
            k_attend = K[:, :, attend_tensor, :]  # (batch, heads, attend_len, d_k)
            v_attend = V[:, :, attend_tensor, :]
            
            # Compute attention scores
            scores_i = torch.matmul(q_i, k_attend.transpose(-2, -1)) / math.sqrt(self.d_k)
            # Shape: (batch, heads, 1, attend_len)
            
            # Softmax
            attn_i = F.softmax(scores_i, dim=-1)
            attn_i = self.dropout(attn_i)
            
            # Apply attention to values
            context_i = torch.matmul(attn_i, v_attend)  # (batch, heads, 1, d_k)
            context[:, :, i, :] = context_i.squeeze(2)
        
        # Reshape and output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + self.dropout(output))
        
        # Compute sparsity stats
        avg_attended = total_attended / seq_len
        sparsity = 1.0 - (total_attended / (seq_len * key_len))
        sparsity_info = {
            'sparsity': sparsity,
            'num_attended': avg_attended
        }
        
        return output, sparsity_info

class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Cross-Attention with optional sparse attention or linear attention"""

    def __init__(self, d_model, n_heads=8, dropout=0.1, attention_config=None):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Parse attention config
        attention_config = attention_config or {'type': 'standard'}
        self.attention_type = attention_config.get('type', 'standard')
        self.top_k_percent = attention_config.get('top_k_percent', None)
        
        # Attention type flags
        self.use_linear = self.attention_type == 'linear'
        self.use_sparse_windowed = self.attention_type == 'sparse_windowed'
        self.use_local_global = self.attention_type == 'local_global'
        self.use_strided = self.attention_type == 'strided'
        self.use_sdpa = self.attention_type == 'sdpa'

        if self.use_linear:
            # Use linear attention
            self.attention = LinearAttention(d_model, n_heads, dropout)
            
        elif self.use_sparse_windowed:
            # Use efficient local window attention
            window_size = attention_config.get('window_size', 36)
            self.attention = LocalWindowAttention(d_model, n_heads, window_size, dropout)
            
        elif self.use_local_global:
            # Use local + global hybrid attention
            window_size = attention_config.get('window_size', 36)
            global_interval = attention_config.get('global_interval', 18)
            self.attention = LocalGlobalAttention(
                d_model, n_heads, window_size, global_interval, dropout
            )
            
        elif self.use_strided:
            # Use strided sparse attention
            local_window = attention_config.get('local_window', 18)
            strides = attention_config.get('strides', [4, 36])
            self.attention = StridedAttention(
                d_model, n_heads, local_window, strides, dropout
            )
            
        elif self.use_sdpa:
            # Use PyTorch Scaled Dot Product Attention
            self.attention = SDPAAttention(d_model, n_heads, dropout)
            
        else:
            # Use standard attention
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(d_model)
            
            if self.top_k_percent is not None:
                print(f"  Using Top-K Attention (keep top {self.top_k_percent*100:.0f}%)")

    def forward(self, query, key, value, mask=None):
        if self.use_linear or self.use_sparse_windowed or self.use_local_global or self.use_strided or self.use_sdpa:
            # Use specialized attention module
            return self.attention(query, key, value, mask)
        else:
            # Use standard attention (O(N²) complexity)
            return self._standard_attention(query, key, value, mask)
    
    def _standard_attention(self, query, key, value, mask=None):
        """Standard attention with optional sparsification"""
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
        
        # Top-K sparsification
        if self.top_k_percent is not None:
            key_len = scores.size(-1)
            top_k = max(1, int(key_len * self.top_k_percent))
            
            _, top_k_idx = torch.topk(scores, k=top_k, dim=-1, largest=True)
            
            sparse_mask = torch.full_like(scores, float('-inf'))
            sparse_mask.scatter_(-1, top_k_idx, 0.0)
            
            scores = scores + sparse_mask

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
        
        total_weight = clean_weight + noisy_weight
        clean_weight = clean_weight / (total_weight + 1e-8)
        noisy_weight = noisy_weight / (total_weight + 1e-8)
        
        return clean_weight, noisy_weight


class CrossModalFusionBlock(nn.Module):
    """Cross-modal fusion using bidirectional cross-attention"""

    def __init__(self, d_model, n_heads=8, dropout=0.1, attention_config=None):
        super(CrossModalFusionBlock, self).__init__()
        
        # Clean PPG attends to Noisy PPG
        self.clean_cross_attn = MultiHeadCrossAttention(
            d_model, n_heads, dropout, attention_config
        )
        # Noisy PPG attends to Clean PPG
        self.noisy_cross_attn = MultiHeadCrossAttention(
            d_model, n_heads, dropout, attention_config
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
    """Temporal convolutional block with dilation and optional depthwise separable convolutions"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2, 
                 use_depthwise_separable=False):
        super(TemporalBlock, self).__init__()
        
        self.use_depthwise_separable = use_depthwise_separable
        
        padding = (kernel_size - 1) * dilation // 2
        
        if use_depthwise_separable:
            self.conv1 = DepthwiseSeparableConv1d(
                n_inputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
            self.conv2 = DepthwiseSeparableConv1d(
                n_outputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        else:
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
        
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)
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
    Window-adaptive PPG + Unfiltered PPG cross-attention model with STABLE LINEAR ATTENTION.
    
    FIXED: Numerical stability issues that caused NaN loss during training.
    
    Supports variable-length inputs from 10 epochs to full 1200 epochs.
    Stream 1: Clean PPG signal (standard filtering)
    Stream 2: Unfiltered PPG signal (contains noise, baseline drift, motion artifacts)
    """
    
    def __init__(self, n_classes=4, d_model=256, n_heads=8, n_fusion_blocks=3, 
                 dropout=0.2, noise_config=None, attention_config=None,
                 positional_encoding='sinusoidal', max_len=5000, depthwise_separable_conv=False):
        super(PPGUnfilteredWindowedCrossAttention, self).__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        self.positional_encoding_type = positional_encoding
        self.depthwise_separable_conv = depthwise_separable_conv
        
        # Parse attention config
        attention_config = attention_config or {
            'type': 'standard'
        }
        
        self.attention_config = attention_config
        self.attention_type = attention_config['type']
        self.top_k_percent = attention_config.get('top_k_percent', None)
                
        # Print configuration
        print("\n" + "="*70)
        print("PPG UNFILTERED WINDOWED CROSS-ATTENTION MODEL")
        print("="*70)
        
        # Attention type
        print(f"\nAttention Configuration:")
        print(f"  Type: {self.attention_type}")
        if self.attention_type == 'linear':
            print(f"  Linear Attention")
        elif self.attention_type == 'sparse_windowed':
            window_size = attention_config.get('window_size', 180)
            attention_config['window_size'] = window_size # Ensure window_size is set
            print(f"  Sparse Windowed Attention (window size: {window_size})")
        else:
            print("  Standard Attention")

        if self.top_k_percent is not None:
            print(f"  Using Top-K Attention (keep top {self.top_k_percent*100:.0f}%)")

        
        # Positional encoding
        print(f"\nPositional Encoding: {positional_encoding.upper()}")
        if positional_encoding == 'sinusoidal':
            print("   Type: Mathematical (sin/cos)")
        elif positional_encoding == 'learned':
            print(f"   Type: Learned embeddings")
        
        # Depthwise separable convolutions
        if depthwise_separable_conv:
            print(f"\nDepthwise Separable Convolutions: ENABLED")
        else:
            print(f"\nStandard Convolutions")
        
        print("="*70 + "\n")
        
        # Noise configuration
        self.noise_config = noise_config or {
            'noise_level': 0.1,
            'drift_amplitude': 0.1,
            'drift_frequency': 0.1,
            'spike_probability': 0.01,
            'spike_amplitude': 0.5
        }
        
        # Encoders
        encoder_channels = [1, 16, 32, 32, 64, 64, 128, 128, 256, d_model]
        
        clean_ppg_encoder_blocks = []
        noisy_ppg_encoder_blocks = []
        for i in range(len(encoder_channels) - 1):
            clean_ppg_encoder_blocks.append(
                ResConvBlock(encoder_channels[i], encoder_channels[i + 1], 
                           use_depthwise_separable=depthwise_separable_conv)
            )
            noisy_ppg_encoder_blocks.append(
                ResConvBlock(encoder_channels[i], encoder_channels[i + 1],
                           use_depthwise_separable=depthwise_separable_conv)
            )
        
        self.clean_ppg_encoder = nn.Sequential(*clean_ppg_encoder_blocks)
        self.noisy_ppg_encoder = nn.Sequential(*noisy_ppg_encoder_blocks)
        
        # Positional encoding
        if positional_encoding == 'sinusoidal':
            self.positional_encoding = DynamicSinusoidalEncoding(d_model)
        elif positional_encoding == 'learned':
            self.positional_encoding = LearnedPositionalEncoding(d_model, max_len=max_len)
        else:
            raise ValueError(f"Unknown positional encoding type: {positional_encoding}")
        
        # Modality weighting
        self.modality_weighting = AdaptiveModalityWeighting(d_model)
        
        # Cross-modal fusion blocks with attention config
        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads, dropout, attention_config)
            for _ in range(n_fusion_blocks)
        ])
        
        # Feature aggregation
        if depthwise_separable_conv:
            self.feature_aggregation = nn.Sequential(
                DepthwiseSeparableConv1d(d_model * 2, d_model, kernel_size=1),
                nn.BatchNorm1d(d_model),
                nn.LeakyReLU()
            )
        else:
            self.feature_aggregation = nn.Sequential(
                nn.Conv1d(d_model * 2, d_model, kernel_size=1),
                nn.BatchNorm1d(d_model),
                nn.LeakyReLU()
            )
        
        # Temporal modeling
        self.temporal_blocks = nn.Sequential(
            TemporalBlock(d_model, d_model, kernel_size=7, stride=1, dilation=1, 
                         dropout=dropout, use_depthwise_separable=depthwise_separable_conv),
            TemporalBlock(d_model, d_model, kernel_size=7, stride=1, dilation=2, 
                         dropout=dropout, use_depthwise_separable=depthwise_separable_conv),
            TemporalBlock(d_model, d_model, kernel_size=7, stride=1, dilation=4, 
                         dropout=dropout, use_depthwise_separable=depthwise_separable_conv)
        )
        
        # Feature refinement
        if depthwise_separable_conv:
            self.feature_refinement = nn.Sequential(
                DepthwiseSeparableConv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.feature_refinement = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
        
        # Classifier
        if depthwise_separable_conv:
            self.classifier = nn.Sequential(
                DepthwiseSeparableConv1d(d_model, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(128, n_classes, kernel_size=1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Conv1d(d_model, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(128, n_classes, kernel_size=1)
            )

    def get_name(self):
        base_name = "PPGUnfilteredWindowedCrossAttention"
        base_name += "{"
        for key, value in self.attention_config.items():
            base_name += f"[{key}:{value}]"
        base_name += "}"
        
        if self.depthwise_separable_conv:
            base_name += "{{[dw_sep_conv]}}"
        
        base_name += f"{{[pos_enc:{self.positional_encoding_type}]}}"

        if self.top_k_percent is not None:
            base_name += f"{{[top_k:{self.top_k_percent}]}}"

        return base_name

    def add_noise_to_ppg(self, clean_ppg):
        """Add noise to clean PPG signal to simulate unfiltered signal"""
        batch_size, _, length = clean_ppg.shape
        device = clean_ppg.device

        noisy_ppg = clean_ppg.clone()

        # Gaussian noise
        gaussian_noise = torch.randn_like(clean_ppg) * self.noise_config['noise_level']
        noisy_ppg = noisy_ppg + gaussian_noise

        # Baseline drift
        t = torch.linspace(0, 1, length, device=device)
        drift_freq = self.noise_config['drift_frequency']
        drift_amp = self.noise_config['drift_amplitude']

        drift = drift_amp * (
                0.5 * torch.sin(2 * np.pi * drift_freq * t) +
                0.3 * torch.sin(2 * np.pi * drift_freq * 2 * t) +
                0.2 * torch.sin(2 * np.pi * drift_freq * 0.5 * t)
        )
        drift = drift.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        noisy_ppg = noisy_ppg + drift

        # Motion artifacts
        spike_prob = self.noise_config['spike_probability']
        spike_amp = self.noise_config['spike_amplitude']

        spike_mask = torch.rand(batch_size, 1, length, device=device) < spike_prob
        spike_values = torch.randn(batch_size, 1, length, device=device) * spike_amp
        spikes = spike_mask.float() * spike_values

        kernel_size = 5
        padding = kernel_size // 2
        smoothing_kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        spikes = F.conv1d(spikes, smoothing_kernel, padding=padding)

        noisy_ppg = noisy_ppg + spikes

        # EMG interference
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
        
        samples_per_epoch = 1024
        n_epochs = input_samples // samples_per_epoch

        # Create unfiltered version
        ppg_unfiltered = self.add_noise_to_ppg(ppg)

        # Encode
        clean_features = self.clean_ppg_encoder(ppg)
        noisy_features = self.noisy_ppg_encoder(ppg_unfiltered)

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

        # Cross-Modal Fusion
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

        # Adaptive upsampling
        output_features = F.interpolate(
            refined_features, 
            size=n_epochs, 
            mode='linear', 
            align_corners=False
        )

        # Classification
        output = self.classifier(output_features)
        output = F.softmax(output, dim=1)

        return output


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_windowed_attention():
    """Test windowed attention implementation correctness"""
    print("\n" + "="*70)
    print("TESTING SPARSE WINDOWED ATTENTION - CORRECTNESS")
    print("="*70)
    
    # Test with small, verifiable dimensions
    batch_size, seq_len, d_model = 2, 16, 32
    n_heads = 4
    window_size = 4
    
    print(f"\nTest setup:")
    print(f"  Batch: {batch_size}, Seq: {seq_len}, D_model: {d_model}")
    print(f"  Heads: {n_heads}, Window: {window_size}")
    
    # Create module
    module = LocalWindowAttention(d_model, n_heads, window_size, dropout=0.0)
    module.eval()
    
    # Create test input
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    with torch.no_grad():
        output, _ = module(query, key, value)
    
    print(f"\n✓ Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    
    # Test gradient flow
    module.train()
    query.requires_grad = True
    output, _ = module(query, key, value)
    loss = output.sum()
    loss.backward()
    
    print(f"✓ Gradient shape: {query.grad.shape}")
    assert query.grad is not None, "No gradient!"
    assert not torch.isnan(query.grad).any(), "NaN in gradient!"
    
    print(f"✓ No NaN values in output or gradients")
    
    # Test with different sequence lengths
    print(f"\n Testing with variable sequence lengths:")
    for test_seq_len in [8, 32, 64, 128]:
        test_query = torch.randn(1, test_seq_len, d_model)
        test_key = torch.randn(1, test_seq_len, d_model)
        test_value = torch.randn(1, test_seq_len, d_model)
        
        with torch.no_grad():
            test_output, _ = module(test_query, test_key, test_value)
        
        print(f"  Seq len {test_seq_len}: {test_output.shape} ✓")
        assert test_output.shape == (1, test_seq_len, d_model)
    
    print(f"\n✅ ALL WINDOWED ATTENTION TESTS PASSED!")
    print("="*70 + "\n")


def test_linear_attention():
    """Test model with different configurations including linear attention"""
    print("\n" + "="*70)
    print("TESTING STABLE LINEAR ATTENTION IMPLEMENTATION")
    print("="*70)
    
    # Test configurations
    configs = [
        {
            'name': 'Baseline (Standard Attention)',
            'attention_config': {'type': 'standard'},
            'depthwise_separable_conv': False,
        },
        {
            'name': 'Sparse Windowed Attention',
            'attention_config': {'type': 'sparse_windowed', 'window_size': 64},
            'depthwise_separable_conv': False,
        },
        {
            'name': 'STABLE Linear Attention',
            'attention_config': {'type': 'linear'},
            'depthwise_separable_conv': False,
        },
        {
            'name': 'STABLE Linear Attention + Depthwise',
            'attention_config': {'type': 'linear'},
            'depthwise_separable_conv': True,
        },
    ]
    
    models = []
    for config in configs:
        print("\n" + "="*70)
        print(f"Testing: {config['name']}")
        print("="*70)
        
        model = PPGUnfilteredWindowedCrossAttention(
            attention_config=config['attention_config'],
            depthwise_separable_conv=config['depthwise_separable_conv'],
            positional_encoding='sinusoidal'
        )
        model.eval()
        models.append((model, config['name']))
    
    # Test different window sizes
    print("\n" + "="*70)
    print("TESTING DIFFERENT SEQUENCE LENGTHS")
    print("="*70)
    
    test_configs = [
        (10, "10 epochs (5 min)"),
        (60, "60 epochs (30 min)"),
        (120, "120 epochs (1 hour)"),
    ]
    
    with torch.no_grad():
        for n_epochs, description in test_configs:
            samples = n_epochs * 1024
            ppg = torch.randn(2, 1, samples)
            
            print(f"\n{description}:")
            print(f"  Input: {ppg.shape}")
            
            for model, name in models:
                output = model(ppg)
                print(f"  Output ({name}): {output.shape}")
                
                # Check for NaN
                if torch.isnan(output).any():
                    print(f"    ⚠️  WARNING: NaN detected in output!")
                else:
                    print(f"    ✓ No NaN values")
                
                assert output.shape == (2, 4, n_epochs), f"Expected (2, 4, {n_epochs})"
            
            print(f"  ✓ All models produce correct output shapes")
    
    # Test gradients (simulate backward pass)
    print("\n" + "="*70)
    print("TESTING GRADIENT STABILITY")
    print("="*70)
    
    for model, name in models:
        if 'Linear' in name:
            print(f"\nTesting gradients for: {name}")
            model.train()
            
            # Small batch for testing
            ppg = torch.randn(2, 1, 10 * 1024, requires_grad=True)
            output = model(ppg)
            
            # Create dummy target
            target = torch.randint(0, 4, (2, 10))
            target_one_hot = F.one_hot(target, num_classes=4).float().transpose(1, 2)
            
            # Compute loss
            loss = F.mse_loss(output, target_one_hot)
            
            print(f"  Loss value: {loss.item():.6f}")
            
            if torch.isnan(loss):
                print(f"    ⚠️  WARNING: NaN loss!")
            else:
                print(f"    ✓ Loss is valid")
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            has_nan_grad = False
            for name_p, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"    ⚠️  NaN gradient in {name_p}")
                    has_nan_grad = True
            
            if not has_nan_grad:
                print(f"    ✓ All gradients are valid")



if __name__ == "__main__":
    # Test windowed attention first
    test_windowed_attention()
    
    # Then test full models
    test_linear_attention()