import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Union
from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from unetr_pp.network_architecture.acdc.model_components import UnetrPPEncoder, UnetrUpBlock

# ==========================================
# 修正版：加入归一化与轻量投影的 3D 拉普拉斯算子
# ==========================================
class LaplacianEdgeExtractor3D(nn.Module):
    def __init__(self, in_channels=1, proj_channels=8):
        """
        in_channels: 原始图像通道数 (通常为1)
        proj_channels: 映射后的边缘特征通道数 (建议使用小通道数如 8 或 16，避免过度干扰主干)
        """
        super().__init__()

        # 1. 定义固定的 3D 拉普拉斯卷积核
        laplace_kernel = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(laplace_kernel.repeat(in_channels, 1, 1, 1, 1), requires_grad=False)

        # 2. 【核心优化】归一化与轻量级映射
        # InstanceNorm3d 用来抹平不同样本之间剧烈的绝对值差异 (解决 C 问题)
        # Conv3d 1x1x1 用来将单通道硬边缘转换为多通道软特征，让网络学会“挑选”边缘
        self.edge_proj = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.Conv3d(in_channels, proj_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(proj_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # 提取高频边缘 (未归一化，可能极大)
        raw_edge = F.conv3d(x, self.weight, padding=1, groups=x.shape[1])
        raw_edge = torch.abs(raw_edge)
        # 归一化并映射到 proj_channels 维度
        return self.edge_proj(raw_edge)

class UNETR_PP(SegmentationNetwork):
    # ... 注释保持不变 ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        depths=None,
        dims=None,
        conv_op=nn.Conv3d,
        do_ds=True,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")
        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.feat_size = (2, 5, 5,)
        self.hidden_size = hidden_size

        # 【修正】1. 实例化优化的边缘提取器，设定映射到 8 个通道
        self.edge_proj_channels = 8
        self.edge_extractor = LaplacianEdgeExtractor3D(in_channels=in_channels, proj_channels=self.edge_proj_channels)

        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)
        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=4 * 10 * 10,
        )

        # 【修正】2. 告诉 decoder 接收的是映射后的 edge_proj_channels (8)
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 20 * 20,
            edge_channels=self.edge_proj_channels,  # 改为 8
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 40 * 40,
            edge_channels=self.edge_proj_channels,  # 改为 8
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(1, 4, 4),
            norm_name=norm_name,
            out_size=16 * 160 * 160,
            conv_decoder=True,
        )

        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder1(x_in)

        # 提取并映射边缘特征 (输出已是稳定的 8 通道软特征)
        full_edge_map = self.edge_extractor(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)

        # 下采样并喂给 decoder4
        edge_4 = F.adaptive_avg_pool3d(full_edge_map, output_size=enc2.shape[2:])
        dec2 = self.decoder4(dec3, enc2, edge_feat=edge_4)

        # 下采样并喂给 decoder3
        edge_3 = F.adaptive_avg_pool3d(full_edge_map, output_size=enc1.shape[2:])
        dec1 = self.decoder3(dec2, enc1, edge_feat=edge_3)

        out = self.decoder2(dec1, convBlock)

        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits