import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Union
from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
#加入BrightnessEnhancement模块
from unetr_pp.network_architecture.acdc.model_components import UnetrPPEncoder, UnetrUpBlock,BoundaryAttentionBlock


def extract_mask_edge_from_logits(logits, thr: float = 0.5):
    """
    基于 AS-UNet 论文提出的“掩膜边缘提取算法”。
    论文逻辑："遍历像素点值为0，且九宫格内其余像素不都为0时，标记为边缘"
    数学等价操作：边界 = MaxPool(Mask) - Mask
    """
    # 1. 先得到二值化的掩膜 (Mask)
    prob = F.softmax(logits, dim=1)
    if prob.shape[1] > 1:
        # 排除背景通道(0)，其余视为前景
        union_prob = prob[:, 1:, ...].sum(dim=1, keepdim=True)
    else:
        union_prob = prob

    mask = (union_prob > thr).float()  # [B, 1, D, H, W]

    # 2. 执行论文中的“九宫格”判定逻辑
    # 论文提到的“九宫格”对应 3x3 范围。为了适应 ACDC 的 3D 数据，
    # 我们采用 kernel=(1, 3, 3)，即只在 H 和 W 切片平面上寻找九宫格边缘，避免 D 维度伪影。
    kernel_size = (1, 3, 3)
    padding = (0, 1, 1)

    # max_pool3d 模拟膨胀：如果九宫格内存在前景(1)，则中心点变为 1
    dilated = F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=padding)

    # 3. 提取精确边缘
    # 当原 mask 为 0 (背景)，且 dilated 为 1 (九宫格内有前景) 时，相减结果为 1
    # 这精确复现了论文中的掩膜边缘提取算法
    mask_boundary = (dilated - mask).clamp(min=0.0)

    return mask_boundary

class UNETR_PP(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """
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
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimensions of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """

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
            out_size=4*10*10,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8*20*20,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16*40*40,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(1, 4, 4),
            norm_name=norm_name,
            out_size=16*160*160,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        # 1-在这里加入亮度增强模块
        self.boundary_attention = BoundaryAttentionBlock(in_channels=feature_size * 2)

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

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        # ===================================================================
        # 1. 基础网络直接输出 (对应论文的"输出1")
        # 主干网络正常前向传播，完全不依赖 BAB
        # ===================================================================
        out = self.decoder2(dec1, convBlock)
        logits_main = self.out1(out)  # 这就是测试时最终使用的输出1

        # ===================================================================
        # 2. 边缘注意辅助分支 (对应论文的 Add: 训练时增加 BAB 得到"输出2")
        # nnUNet 框架下，do_ds=True 代表处于训练或验证阶段 (需要多尺度深度监督)
        # ===================================================================
        if self.do_ds:
            # 提取粗预测并计算掩膜边缘图
            logits3_coarse = self.out3(dec2)
            mask_boundary = extract_mask_edge_from_logits(logits3_coarse.detach(), thr=0.5)

            # 调整边缘图大小以匹配 dec1
            target_size = dec1.shape[2:]
            mask_boundary_resized = None
            if mask_boundary is not None:
                mask_boundary_resized = F.interpolate(mask_boundary, size=target_size, mode='nearest')

            # 【核心改变】将 dec1 作为一个分支传入 BAB，计算得到 BAB 特征
            dec1_bab = self.boundary_attention(dec1, mask_boundary_resized)

            # 从 BAB 特征得到预测结果 (对应论文的"输出2")
            # 这里重用了 out2 作为输出层
            logits_bab = self.out2(dec1_bab)

            # 返回列表供联合损失函数计算：Loss = L(输出1) + L(输出2) + L(其它层)
            # 在前向反向传播中，Loss(输出2) 的梯度会回传，强制优化网络内部提取边缘的能力
            logits = [logits_main, logits_bab, logits3_coarse]

        # ===================================================================
        # 3. 测试/推理阶段 (对应论文的 Subtract: 舍弃 BAB)
        # ===================================================================
        else:
            # do_ds=False 时（即推断时），直接返回主干网络的输出1。
            # 代码根本不会执行到 BAB 的模块，彻底实现了舍弃 BAB 以减少测试参数量和计算量！
            logits = logits_main

        return logits