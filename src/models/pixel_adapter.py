"""Pixel-level TTA Adapter for learning input-space perturbations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DepthwiseSeparableConv(nn.Module):
    """现代的深度可分离卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return x


class AdapterResBlock(nn.Module):
    """
    轻量级残差块，使用深度可分离卷积。
    结构: x -> DSConv -> GELU -> DSConv -> + x
    """

    def __init__(self, channels: int, expansion: float = 2.0):
        super().__init__()
        hidden_channels = int(channels * expansion)

        self.conv1 = DepthwiseSeparableConv(channels, hidden_channels)
        self.act = nn.GELU()
        self.conv2 = DepthwiseSeparableConv(hidden_channels, channels)

        # 残差块内部的缩放因子，初始化为较小值
        self.block_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = self.act(residual)
        residual = self.conv2(residual)
        return x + self.block_scale * residual


class PixelTTAAdapter(nn.Module):
    """
    像素级TTA适配器。

    在分类器前学习输入空间的细微扰动，用于TTA时的域适应。

    关键设计:
    1. 残差学习: output = input + scale * learned_perturbation
    2. 零初始化: 初始时scale=0，输出完全等于输入
    3. 限制范围: scale被限制在[-max_scale, +max_scale]内
    4. 轻量现代: 使用深度可分离卷积和残差连接

    Args:
        in_channels: 输入通道数，默认3 (RGB)
        hidden_channels: 隐藏层通道数
        num_blocks: 残差块数量
        max_scale: 最大残差缩放系数，控制最大影响程度
        use_spatial_attention: 是否使用空间注意力
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        max_scale: float = 0.15,
        use_spatial_attention: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.max_scale = max_scale
        self.use_spatial_attention = use_spatial_attention

        # 全局残差缩放因子
        # 注意：不能同时让scale=0和conv_out=0，否则scale的梯度也为0
        # 方案：scale初始化为小值，让conv_out零初始化来保证初始输出=输入
        # 这样perturbation有值，scale的梯度就能正常传播
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)  # 初始scale ≈ 0.015

        # 输入投影：扩展到hidden_channels
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.GELU(),
        )

        # 残差块堆叠
        self.blocks = nn.ModuleList([AdapterResBlock(hidden_channels, expansion=2.0) for _ in range(num_blocks)])

        # 空间注意力（可选）
        if use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels // 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels // 4, 1, kernel_size=1),
                nn.Sigmoid(),
            )

        # 输出投影：回到原始通道数
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

        # 关键：将输出卷积初始化为零，确保初始时残差为0
        # 此时 scale ≈ 0.015 (tanh(0.1)*0.15)，但 perturbation = 0
        # 所以初始 output = input + 0.015 * 0 = input
        self._zero_init_output()

    def _zero_init_output(self):
        """将输出卷积层初始化为零，确保初始时残差为0。"""
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

    def get_effective_scale(self) -> torch.Tensor:
        """获取当前有效的缩放系数。"""
        # 使用tanh限制在[-1, 1]，然后乘以max_scale
        return torch.tanh(self.residual_scale) * self.max_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            适配后的图像 (B, 3, H, W)，shape不变
        """
        # 保存原始输入用于残差连接
        identity = x

        # 特征提取
        features = self.conv_in(x)

        # 残差块处理
        for block in self.blocks:
            features = block(features)

        # 空间注意力（如果启用）
        if self.use_spatial_attention:
            attention = self.spatial_attention(features)
            features = features * attention

        # 投影回原始通道
        perturbation = self.conv_out(features)

        # 计算有效缩放系数
        scale = self.get_effective_scale()

        # 残差连接：初始时scale=0，输出=输入
        output = identity + scale * perturbation

        return output

    def get_perturbation_stats(self, x: torch.Tensor) -> dict:
        """
        获取扰动统计信息，用于监控和调试。

        Returns:
            dict: 包含scale、perturbation的均值/标准差等
        """
        with torch.no_grad():
            identity = x
            features = self.conv_in(x)
            for block in self.blocks:
                features = block(features)
            if self.use_spatial_attention:
                attention = self.spatial_attention(features)
                features = features * attention
            perturbation = self.conv_out(features)
            scale = self.get_effective_scale()

            actual_change = scale * perturbation

            return {
                "scale": scale.item(),
                "perturbation_mean": perturbation.mean().item(),
                "perturbation_std": perturbation.std().item(),
                "actual_change_mean": actual_change.mean().item(),
                "actual_change_std": actual_change.std().item(),
                "actual_change_max": actual_change.abs().max().item(),
            }


class PixelTTAAdapterLight(nn.Module):
    """
    更轻量的像素级TTA适配器。

    只使用简单的卷积层，参数更少，适合快速适应。
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 16,
        max_scale: float = 0.1,
    ):
        super().__init__()

        self.max_scale = max_scale

        # 全局缩放因子，初始化为小值（不能为0，否则梯度消失）
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

        # 简单的3层卷积
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

        # 零初始化最后一层
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def get_effective_scale(self) -> torch.Tensor:
        return torch.tanh(self.residual_scale) * self.max_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        perturbation = self.net(x)
        scale = self.get_effective_scale()
        return x + scale * perturbation


# ============================================================================
# Enhanced Adapter Components
# ============================================================================


def get_num_groups(channels: int, preferred: int = 8) -> int:
    """找到合适的 GroupNorm num_groups。"""
    for g in [preferred, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class LayerNorm2d(nn.Module):
    """适用于 (B, C, H, W) 的 LayerNorm。"""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    """
    Simple Gate 机制 (来自 NAFNet)。

    将通道分成两半，一半作为门控信号，比 GELU 更高效。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    """
    简化版通道注意力 (来自 NAFNet)。

    使用全局平均池化 + 1x1 卷积，比 SE 更轻量。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class NAFBlock(nn.Module):
    """
    NAFNet 风格的基础块。

    设计理念：简单即高效
    - LayerNorm 替代 BatchNorm
    - SimpleGate 替代 GELU（通道减半的门控）
    - 简化的通道注意力
    - 大卷积核捕获更大感受野

    参考: Simple Baselines for Image Restoration (ECCV 2022)
    """

    def __init__(self, channels: int, expansion: float = 2.0, kernel_size: int = 3):
        super().__init__()
        hidden = int(channels * expansion)
        padding = kernel_size // 2

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden * 2, 1)  # *2 for SimpleGate
        self.conv2 = nn.Conv2d(hidden, hidden * 2, kernel_size, padding=padding, groups=hidden)
        self.conv3 = nn.Conv2d(hidden, channels, 1)

        self.sg = SimpleGate()
        self.sca = SimpleChannelAttention(hidden)

        # Layer scale
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.sg(x)  # hidden * 2 -> hidden

        x = self.conv2(x)
        x = self.sg(x)  # hidden * 2 -> hidden

        x = self.sca(x)
        x = self.conv3(x)

        return identity + x * self.beta


class DownSample(nn.Module):
    """下采样：PixelUnshuffle。"""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, 1, 1), nn.PixelUnshuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpSample(nn.Module):
    """上采样：PixelShuffle。"""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, 3, 1, 1), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class PixelTTAAdapterEnhance(nn.Module):
    """
    增强版像素级TTA适配器 (NAFNet + U-Net 风格)。

    设计理念：
    1. NAFNet 风格的基础块 - 简单高效，SOTA 级图像恢复
    2. 轻量 U-Net 结构 - 编码器-解码器 + 跳跃连接
    3. 多尺度特征处理 - 通过下/上采样捕获不同尺度信息
    4. 残差学习 - output = input + scale * perturbation

    架构:
        Input -> Stem -> [Encoder] -> Bottleneck -> [Decoder] -> Output
                           ↓                           ↑
                        Skip Connection ─────────────────

    参数量约为标准版的 3-5 倍，设计更优雅主流。

    Args:
        in_channels: 输入通道数，默认3 (RGB)
        base_channels: 基础通道数 (默认32)
        num_blocks: 每个阶段的 NAFBlock 数量 (默认[2, 2, 2])
        max_scale: 最大残差缩放系数
        use_multi_scale: 是否使用多尺度 U-Net 结构
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_blocks: Optional[list] = None,
        max_scale: float = 0.15,
        use_multi_scale: bool = True,
    ):
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 2, 2] if use_multi_scale else [4]

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.max_scale = max_scale
        self.use_multi_scale = use_multi_scale

        # 全局残差缩放因子
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Stem: 输入投影
        self.stem = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        if use_multi_scale:
            # ============ U-Net 结构 ============
            # Encoder
            self.encoder1 = nn.Sequential(*[NAFBlock(base_channels) for _ in range(num_blocks[0])])
            self.down1 = DownSample(base_channels)  # -> base_channels * 2

            self.encoder2 = nn.Sequential(*[NAFBlock(base_channels * 2) for _ in range(num_blocks[1])])
            self.down2 = DownSample(base_channels * 2)  # -> base_channels * 4

            # Bottleneck
            self.bottleneck = nn.Sequential(*[NAFBlock(base_channels * 4) for _ in range(num_blocks[2])])

            # Decoder
            self.up2 = UpSample(base_channels * 4)  # -> base_channels * 2
            self.fusion2 = nn.Conv2d(base_channels * 4, base_channels * 2, 1)
            self.decoder2 = nn.Sequential(*[NAFBlock(base_channels * 2) for _ in range(num_blocks[1])])

            self.up1 = UpSample(base_channels * 2)  # -> base_channels
            self.fusion1 = nn.Conv2d(base_channels * 2, base_channels, 1)
            self.decoder1 = nn.Sequential(*[NAFBlock(base_channels) for _ in range(num_blocks[0])])
        else:
            # ============ 简单堆叠结构 ============
            self.blocks = nn.Sequential(*[NAFBlock(base_channels) for _ in range(num_blocks[0])])

        # Output: 输出投影
        self.output = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

        # 零初始化输出层
        self._zero_init_output()

    def _zero_init_output(self):
        """将输出卷积层初始化为零。"""
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def get_effective_scale(self) -> torch.Tensor:
        """获取当前有效的缩放系数。"""
        return torch.tanh(self.residual_scale) * self.max_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            适配后的图像 (B, C, H, W)
        """
        identity = x

        # Stem
        feat = self.stem(x)

        if self.use_multi_scale:
            # Encoder path
            enc1 = self.encoder1(feat)
            feat = self.down1(enc1)

            enc2 = self.encoder2(feat)
            feat = self.down2(enc2)

            # Bottleneck
            feat = self.bottleneck(feat)

            # Decoder path with skip connections
            feat = self.up2(feat)
            feat = self.fusion2(torch.cat([feat, enc2], dim=1))
            feat = self.decoder2(feat)

            feat = self.up1(feat)
            feat = self.fusion1(torch.cat([feat, enc1], dim=1))
            feat = self.decoder1(feat)
        else:
            feat = self.blocks(feat)

        # Output projection
        perturbation = self.output(feat)

        # Residual connection
        scale = self.get_effective_scale()
        return identity + scale * perturbation

    def get_perturbation_stats(self, x: torch.Tensor) -> dict:
        """获取扰动统计信息。"""
        with torch.no_grad():
            identity = x
            feat = self.stem(x)

            if self.use_multi_scale:
                enc1 = self.encoder1(feat)
                feat = self.down1(enc1)
                enc2 = self.encoder2(feat)
                feat = self.down2(enc2)
                feat = self.bottleneck(feat)
                feat = self.up2(feat)
                feat = self.fusion2(torch.cat([feat, enc2], dim=1))
                feat = self.decoder2(feat)
                feat = self.up1(feat)
                feat = self.fusion1(torch.cat([feat, enc1], dim=1))
                feat = self.decoder1(feat)
            else:
                feat = self.blocks(feat)

            perturbation = self.output(feat)
            scale = self.get_effective_scale()
            actual_change = scale * perturbation

            return {
                "scale": scale.item(),
                "perturbation_mean": perturbation.mean().item(),
                "perturbation_std": perturbation.std().item(),
                "actual_change_mean": actual_change.mean().item(),
                "actual_change_std": actual_change.std().item(),
                "actual_change_max": actual_change.abs().max().item(),
            }

    def get_num_params(self) -> int:
        """获取模型参数数量。"""
        return sum(p.numel() for p in self.parameters())


def create_pixel_adapter(
    adapter_type: str = "standard",
    in_channels: int = 3,
    hidden_channels: int = 32,
    num_blocks: int = 2,
    max_scale: float = 0.15,
    use_spatial_attention: bool = True,
    use_multi_scale: bool = True,
) -> nn.Module:
    """
    工厂函数创建像素适配器。

    Args:
        adapter_type: "standard", "light", 或 "enhance"
        in_channels: 输入通道数
        hidden_channels: 隐藏层通道数 (enhance 下作为 base_channels)
        num_blocks: 残差块数量
        max_scale: 最大残差缩放系数
        use_spatial_attention: 是否使用空间注意力 (standard专用)
        use_multi_scale: 是否使用多尺度 U-Net 结构 (enhance专用)

    Returns:
        PixelTTAAdapter, PixelTTAAdapterLight 或 PixelTTAAdapterEnhance 实例
    """
    if adapter_type == "standard":
        return PixelTTAAdapter(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            max_scale=max_scale,
            use_spatial_attention=use_spatial_attention,
        )
    elif adapter_type == "light":
        return PixelTTAAdapterLight(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            max_scale=max_scale,
        )
    elif adapter_type == "enhance":
        # enhance 版本使用 base_channels
        base_ch = hidden_channels if hidden_channels >= 32 else 32
        return PixelTTAAdapterEnhance(
            in_channels=in_channels,
            base_channels=base_ch,
            num_blocks=[2, 2, 2] if use_multi_scale else [4],
            max_scale=max_scale,
            use_multi_scale=use_multi_scale,
        )
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}. Choose from 'standard', 'light', 'enhance'")
