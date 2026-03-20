import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from unetr_pp.network_architecture.ScConv import ScConv

class MambaScConvCrossAttnFusion(nn.Module):
    """
    Fusion:
      1) spatial_out = heavy spatial projector from concat(mamba, conv) -> project to d -> project back to C
         spatial_fused = sigmoid(alpha_s) * x_mamba + (1-sigmoid(alpha_s)) * spatial_out

      2) channel_out = ScConv(x_conv) (per-slice) concat with x_mamba -> 1x1x1 conv reduce -> channel_out
         channel_fused = sigmoid(alpha_c) * x_mamba + (1-sigmoid(alpha_c)) * channel_out

      3) Cross-Attention:
         queries = spatial_emb (N positions, dim d)
         keys/values = channel tokens (L= C tokens, dim d) generated from channel_fused GAP -> linear -> tokens
         attn_out -> reshape -> project back to C
         final out = sigmoid(gamma) * spatial_fused + (1-sigmoid(gamma)) * attn_proj
    Notes:
      - channels: C
      - attn_dim: embedding dim for attention (d). must be divisible by heads.
      - num_heads: MHA heads for cross-attn
      - This module is compute & memory heavy for large N = H*W*D. Use at lower-resolution stages or reduce attn_dim.
    """
    def __init__(self, channels: int, attn_dim: int = None, num_heads: int = 4):
        super().__init__()
        self.C = channels
        self.num_heads = num_heads

        if attn_dim is None:
            attn_dim = max(32, channels // 2)
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"

        self.d = attn_dim

        # --- Spatial projections (from concat 2C -> d) ---
        self.s_q_conv = nn.Conv3d(2 * channels, self.d, kernel_size=1, bias=False)
        # We will use s_q_conv output as queries (per-position embeddings).
        # We'll also optionally get a spatial->C projection:
        self.s_out = nn.Sequential(
            nn.Conv3d(self.d, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )

        # --- ScConv for channel path (2D per-slice) ---
        self.scconv = ScConv(op_channel=channels)

        # --- channel fusion conv (reduce 2C -> C) ---
        self.channel_reduce = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )

        # --- produce channel tokens for cross-attn: map per-channel GAP scalar -> d-dim embedding ---
        # We'll take channel_fused GAP over spatial dims -> (B, C) and map each channel to a token embedding (C tokens)
        self.channel_token_fc = nn.Linear(1, self.d, bias=False)  # applied per-channel on scalar descriptor

        # --- MultiheadAttention for cross-attention (queries: N positions, keys/values: C tokens) ---
        # Use PyTorch MHA with batch_first=True
        self.cross_mha = nn.MultiheadAttention(embed_dim=self.d, num_heads=self.num_heads, batch_first=True)

        # project attention output (d -> C) via conv
        self.attn_to_C = nn.Conv3d(self.d, channels, kernel_size=1, bias=False)

        # learnable per-channel alphas (initialized 0 -> sigmoid 0.5)
        self.raw_alpha_s = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.raw_alpha_c = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.raw_gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

        # small eps
        self.eps = 1e-6

    def forward(self, x_mamba, x_conv):
        """
        输入：
        x_mamba: [B, C, H, W, D]
        x_conv : [B, C, H, W, D]
        returns: fused [B, C, H, W, D]
        """
        B, C, H, W, D = x_mamba.shape
        N = H * W * D

        # ---------- 1) Spatial path ----------
        # concat along channel -> [B, 2C, H, W, D]
        cat_spatial = torch.cat([x_mamba, x_conv], dim=1)
        # project to d per spatial position (queries)
        q_emb = self.s_q_conv(cat_spatial)  # [B, d, H, W, D]

        # project q_emb back to channels to get spatial_out
        spatial_out = self.s_out(q_emb)  # [B, C, H, W, D]

        # spatial alpha (per-channel)
        alpha_s = torch.sigmoid(self.raw_alpha_s)  # [1,C,1,1,1]
        spatial_fused = alpha_s * x_mamba + (1 - alpha_s) * spatial_out  # [B,C,H,W,D]

        # ---------- 2) Channel path ----------
        # apply ScConv per-slice on x_conv
        x2d = x_conv.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)  # [B*D, C, H, W]
        x_sc = self.scconv(x2d)  # [B*D, C, H, W]
        x_sc = x_sc.reshape(B, D, C, H, W).permute(0, 2, 3, 4, 1)  # [B, C, H, W, D]

        # concat and reduce to channel_out
        cat_channel = torch.cat([x_mamba, x_sc], dim=1)  # [B, 2C, H, W, D]
        channel_out = self.channel_reduce(cat_channel)   # [B, C, H, W, D]

        alpha_c = torch.sigmoid(self.raw_alpha_c)
        channel_fused = alpha_c * x_mamba + (1 - alpha_c) * channel_out  # [B,C,H,W,D]

        # ---------- 3) Cross-Attention ----------
        # Queries: q_emb reshape -> (B, N, d)
        q = q_emb.view(B, self.d, N).permute(0, 2, 1)  # [B, N, d]

        # Keys/Values: build C tokens per batch from channel_fused via GAP per-channel
        ch_gap = channel_fused.mean(dim=(2, 3, 4))  # [B, C]  (scalar descriptor per channel)
        ch_gap_unsq = ch_gap.unsqueeze(-1)  # [B, C, 1]
        # map scalar -> d-dim token for each channel
        ch_tokens = self.channel_token_fc(ch_gap_unsq)  # [B, C, d]
        # MHA expects (B, L, E) where L = sequence len = C
        # Use these as keys/values
        # cross_mha(query, key, value) with shapes (B, N, E), (B, C, E), (B, C, E)
        attn_out, _ = self.cross_mha(q, ch_tokens, ch_tokens)  # [B, N, d]

        # reshape attn_out -> [B, d, H, W, D]
        attn_out = attn_out.permute(0, 2, 1).contiguous().view(B, self.d, H, W, D)
        # project attn_out back to C channels
        attn_proj = self.attn_to_C(attn_out)  # [B, C, H, W, D]

        # final gamma
        gamma = torch.sigmoid(self.raw_gamma)  # [1,C,1,1,1]
        out = gamma * spatial_fused + (1 - gamma) * attn_proj

        return out
