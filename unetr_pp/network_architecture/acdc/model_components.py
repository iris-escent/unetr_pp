import torch
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from unetr_pp.network_architecture.layers import LayerNorm

# 下采样:sconnv+mamba交叉融合模块
from unetr_pp.network_architecture.acdc.transformerblock import TransformerBlock
# 上采样 单独mobile_mamba 供解码器使用
from unetr_pp.network_architecture.acdc.mobile_mamba import MambaTransformerBlock
from unetr_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock



einops, _ = optional_import("einops")

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[16 * 40 * 40, 8 * 20 * 20, 4 * 10 * 10, 2 * 5 * 5],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
        super().__init__()

        # 下采样层
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                '''下采样经过TransformerBlock'''
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        # 上采样层
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                '''上采样经过TransformerBlock处'''
                stage_blocks.append(MambaTransformerBlock(input_size=out_size, hidden_size= out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out


class PaperAttentionModule(nn.Module):
    """
    基于 AS-UNet 论文图4 提出的新型注意力模块。
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        # 1. 空间上进行压缩 (Spatial Squeeze & Excitation) -> 生成 U_{sCE}
        # 采用 1x1x1 卷积将通道数压缩至 1，加上 Sigmoid 归一化为权重
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 2. 通道上进行压缩 (Channel Squeeze & Excitation) -> 生成 \hat{U}_{cSE}
        # 全局池化 -> 1x1卷积(等同全连接)+ReLU -> 1x1卷积(等同全连接)+Sigmoid
        mid_channels = max(1, channels // reduction)
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, u):
        # 输入 u: [B, C, D, H, W]
        # 空间压缩特征 U_sCE: [B, 1, D, H, W]
        u_sce = self.spatial_conv(u)

        # 通道压缩向量 U_cSE: [B, C, 1, 1, 1]
        u_cse = self.channel_fc(u)

        # 两者相乘得到新的权重 W: [B, C, D, H, W] (利用PyTorch广播机制自动扩展维度)
        # 对应论文式(3): W = U_sCE x U_cSE
        w = u_sce * u_cse

        # 权重与原输入特征逐像素相乘
        return u * w

class BoundaryAttentionBlock(nn.Module):
    """
    边缘注意模块 (BAB) - 完全按照 AS-UNet 论文结构复现
    此代码省去了补充层(f_{i-1})的拼接(对应L7层的最简模式)，更贴合您目前的单层输入调用逻辑。
    """

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        # 1. 输入层 1x1 卷积
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        # 2. 拼接掩膜边缘图 (ring) 后的 3x3 卷积
        # 掩膜边缘图是单通道的，所以输入维度是 out_channels + 1
        self.conv3x3 = nn.Sequential(
            nn.Conv3d(out_channels + 1, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),  # 加入BN稳定3D网络训练
            nn.ReLU(inplace=True)
        )

        # 3. 论文提出的新型注意力模块
        self.attention = PaperAttentionModule(out_channels)

    def forward(self, x, mask_boundary):
        """
        x: 输入特征图 R_i [B, C, D, H, W]
        mask_boundary: 掩膜边缘图 (即您的 ring) [B, 1, D, H, W]
        """
        # 为了兼容验证/推理阶段 do_ds=False 导致 mask_boundary 为空的情况
        if mask_boundary is None:
            mask_boundary = torch.zeros((x.size(0), 1, x.size(2), x.size(3), x.size(4)), device=x.device,
                                        dtype=x.dtype)

        # 步骤 1：输入特征经过 1x1 卷积
        feat = self.conv1(x)

        # 步骤 2：将特征图与掩膜边缘图按通道拼接 (Concat)
        concat_feat = torch.cat([feat, mask_boundary], dim=1)

        # 步骤 3：进行 3x3 卷积提取全局信息
        feat_3x3 = self.conv3x3(concat_feat)

        # 步骤 4：经过论文提出的乘法注意力模块
        out = self.attention(feat_3x3)

        return out