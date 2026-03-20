# unetr_pp/network_architecture/acdc/transformerblock.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from unetr_pp.network_architecture.dynunet_block import UnetResBlock
# MobileMambaBlockSeq should exist in mobile_mamba_block.py
from unetr_pp.network_architecture.mobile_mamba_block import MobileMambaBlockSeq

from unetr_pp.network_architecture.mamba_scconv_fusion import MambaScConvCrossAttnFusion


class TransformerBlock(nn.Module):
    """
    Integrated TransformerBlock for ACDC with:
      - MobileMambaBlockSeq (token transformer-like block)
      - local conv path (conv51, conv52)
      - heavy Mamba-ScConv cross-attention fusion (alpha spatial + alpha channel + cross-attn)
      - residual projection conv8
    Notes:
      - This version prioritizes fusion quality. For large H*W*D you may OOM.
      - If memory is tight: reduce attn_dim in fusion or only use fusion at bottleneck.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed: bool = False,
    ) -> None:
        super().__init__()

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0 and 1")

        # normalization & scale used for mamba path
        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)

        # Mobile Mamba block (token-level)
        self.mobile_mamba_block = MobileMambaBlockSeq(dim=hidden_size)

        # Conv local branch (3D residual blocks)
        # UnetResBlock signature: UnetResBlock(spatial_dims, in_channels, out_channels, kernel_size=3, stride=1, norm_name="batch")
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv52 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")

        # Fusion module: heavy cross-attention fusion (may be memory heavy)
        # attn_dim chosen as hidden_size//2 by default; adjust if OOM
        self.fusion = MambaScConvCrossAttnFusion(channels=hidden_size)


        # residual conv (project / shortcut)
        self.conv8 = nn.Sequential(
            nn.Dropout3d(0.1),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=1)
        )

        # optional positional embedding for tokens (N = input_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size)) if pos_embed else None

    def forward(self, x):
        """
        x: [B, C, H, W, D]  (C == hidden_size)
        returns: [B, C, H, W, D]
        """
        B, C, H, W, D = x.shape

        # --- 1. tokens for mamba path ---
        # reshape to [B, N, C] where N = H*W*D
        x_seq = x.reshape(B, C, H * W * D).permute(0, 2, 1)  # [B, N, C]

        # add positional embedding if provided
        if self.pos_embed is not None:
            # self.pos_embed shape: [1, N, C]
            x_seq = x_seq + self.pos_embed

        # mamba transformer block (token-level)
        x_mamba_seq = x_seq + self.gamma * self.mobile_mamba_block(self.norm(x_seq))

        # reshape back to 3D feature map: [B, C, H, W, D]
        x_mamba_3d = x_mamba_seq.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)

        # --- 2. conv (local) branch ---
        conv_feat = self.conv51(x_mamba_3d)
        conv_feat = self.conv52(conv_feat)  # [B, C, H, W, D]

        # --- 3. fusion: heavy Mamba-ScConv cross-attn fusion ---
        # fusion expects (x_mamba_3d, conv_feat) both [B,C,H,W,D]
        fused = self.fusion(x_mamba_3d, conv_feat)  # [B, C, H, W, D]

        # --- 4. residual projection and return ---
        out = fused + self.conv8(fused)

        return out