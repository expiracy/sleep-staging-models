"""
改进的多模态睡眠分期模型 - 使用Cross-Attention (内存优化版)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class ResConvBlock(nn.Module):
    """残差卷积块"""

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


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力机制"""

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

        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        context = torch.matmul(attention_weights, V)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 输出投影
        output = self.w_o(context)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + query)

        return output, attention_weights


class CrossModalFusionBlock(nn.Module):
    """交叉模态融合块"""

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(CrossModalFusionBlock, self).__init__()

        # PPG作为query，ECG作为key/value
        self.ppg_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)

        # ECG作为query，PPG作为key/value
        self.ecg_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ppg_features, ecg_features):
        # PPG关注ECG
        ppg_attended, _ = self.ppg_cross_attn(ppg_features, ecg_features, ecg_features)

        # ECG关注PPG
        ecg_attended, _ = self.ecg_cross_attn(ecg_features, ppg_features, ppg_features)

        # 组合特征
        ppg_out = self.layer_norm(ppg_attended + self.dropout(self.ffn(ppg_attended)))
        ecg_out = self.layer_norm(ecg_attended + self.dropout(self.ffn(ecg_attended)))

        return ppg_out, ecg_out


class TemporalConvBlock(nn.Module):
    """时序卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TemporalConvBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2, dilation=dilation
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2, dilation=dilation
        ))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = None

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))

        if self.residual is not None:
            x = self.residual(x)

        return self.relu(out + x)


class AdaptiveModalityWeighting(nn.Module):
    """自适应模态权重模块"""

    def __init__(self, d_model):
        super(AdaptiveModalityWeighting, self).__init__()
        self.ppg_weight_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.ecg_weight_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, ppg_features, ecg_features):
        ppg_weight = self.ppg_weight_net(ppg_features)
        ecg_weight = self.ecg_weight_net(ecg_features)

        # 归一化权重
        total_weight = ppg_weight + ecg_weight + 1e-8
        ppg_weight = ppg_weight / total_weight
        ecg_weight = ecg_weight / total_weight

        return ppg_weight, ecg_weight


class ImprovedMultiModalSleepNet(nn.Module):
    """改进的多模态睡眠分期网络 - 内存优化版"""

    def __init__(self, n_classes=4, d_model=256, n_heads=8, n_fusion_blocks=3):
        super(ImprovedMultiModalSleepNet, self).__init__()

        # PPG编码器 - 增加更多下采样层
        self.ppg_encoder = nn.Sequential(
            ResConvBlock(1, 16, stride=2),  # 1228800 -> 614400
            ResConvBlock(16, 32, stride=2),  # 614400 -> 307200
            ResConvBlock(32, 64, stride=2),  # 307200 -> 153600
            ResConvBlock(64, 128, stride=2),  # 153600 -> 76800
            ResConvBlock(128, 256, stride=2),  # 76800 -> 38400
            ResConvBlock(256, 256, stride=2),  # 38400 -> 19200
            ResConvBlock(256, 256, stride=2),  # 19200 -> 9600
            ResConvBlock(256, 256, stride=2),  # 9600 -> 4800
            ResConvBlock(256, d_model, stride=2)  # 4800 -> 2400
        )

        # ECG编码器 - 同样的结构
        self.ecg_encoder = nn.Sequential(
            ResConvBlock(1, 16, stride=2),
            ResConvBlock(16, 32, stride=2),
            ResConvBlock(32, 64, stride=2),
            ResConvBlock(64, 128, stride=2),
            ResConvBlock(128, 256, stride=2),
            ResConvBlock(256, 256, stride=2),
            ResConvBlock(256, 256, stride=2),
            ResConvBlock(256, 256, stride=2),
            ResConvBlock(256, d_model, stride=2)
        )

        # 位置编码
        self.positional_encoding = self._create_positional_encoding(d_model, 3000)

        # 自适应模态权重
        self.modality_weighting = AdaptiveModalityWeighting(d_model)

        # Cross-Modal Fusion层
        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads)
            for _ in range(n_fusion_blocks)
        ])

        # 特征聚合
        self.feature_aggregation = nn.Sequential(
            nn.Conv1d(d_model * 2, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 时序建模
        self.temporal_blocks = nn.Sequential(
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=1),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=2),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=4),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=8)
        )

        # 特征细化
        self.feature_refinement = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Conv1d(d_model, 128, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, n_classes, 1)
        )

    def _create_positional_encoding(self, d_model, max_len):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0).transpose(1, 2), requires_grad=False)

    def forward(self, ppg, ecg):
        batch_size = ppg.size(0)

        # 编码
        ppg_features = self.ppg_encoder(ppg)  # (B, d_model, 2400)
        ecg_features = self.ecg_encoder(ecg)  # (B, d_model, 2400)

        # 添加位置编码
        seq_len = ppg_features.size(2)
        ppg_features = ppg_features + self.positional_encoding[:, :, :seq_len]
        ecg_features = ecg_features + self.positional_encoding[:, :, :seq_len]

        # 获取模态权重
        ppg_weight, ecg_weight = self.modality_weighting(ppg_features, ecg_features)

        # 应用模态权重
        ppg_features_weighted = ppg_features * ppg_weight.unsqueeze(-1)
        ecg_features_weighted = ecg_features * ecg_weight.unsqueeze(-1)

        # 转换为(B, L, C)格式用于attention
        ppg_features_t = ppg_features_weighted.transpose(1, 2)
        ecg_features_t = ecg_features_weighted.transpose(1, 2)

        # Cross-Modal Fusion
        for fusion_block in self.fusion_blocks:
            ppg_features_t, ecg_features_t = fusion_block(ppg_features_t, ecg_features_t)

        # 转回(B, C, L)格式
        ppg_features = ppg_features_t.transpose(1, 2)
        ecg_features = ecg_features_t.transpose(1, 2)

        # 特征聚合
        combined_features = torch.cat([ppg_features, ecg_features], dim=1)
        fused_features = self.feature_aggregation(combined_features)

        # 时序建模
        temporal_features = self.temporal_blocks(fused_features)

        # 特征细化
        refined_features = self.feature_refinement(temporal_features)

        # 插值到1200个窗口
        # 从2400到1200，正好是2:1的关系
        output_features = F.avg_pool1d(refined_features, kernel_size=2, stride=2)

        # 确保输出长度正好是1200
        if output_features.size(2) != 1200:
            output_features = F.interpolate(output_features, size=1200, mode='linear', align_corners=False)

        # 分类
        output = self.classifier(output_features)  # (B, 4, 1200)
        output = F.softmax(output, dim=1)

        return output

    def get_modality_weights(self):
        """获取当前的模态权重（用于监控）"""
        return self.ppg_weight, self.ecg_weight


def test_model():
    """测试模型"""
    model = ImprovedMultiModalSleepNet()

    # 测试输入
    ppg = torch.randn(2, 1, 1228800)  # 10小时数据
    ecg = torch.randn(2, 1, 1228800)

    # 测试前向传播
    output = model(ppg, ecg)
    print(f"Output shape: {output.shape}")  # 应该是 (2, 4, 1200)

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 测试内存使用
    if torch.cuda.is_available():
        model = model.cuda()
        ppg = ppg.cuda()
        ecg = ecg.cuda()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            output = model(ppg, ecg)

        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # 转换为GB
        print(f"Peak GPU memory usage: {max_memory:.2f} GB")


if __name__ == "__main__":
    test_model()