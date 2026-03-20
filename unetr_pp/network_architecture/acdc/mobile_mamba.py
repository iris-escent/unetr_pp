import torch
import torch.nn as nn
import torch.nn.functional as F
from unetr_pp.network_architecture.dynunet_block import UnetResBlock



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)



class MobileMambaModuleSeq(nn.Module):

    def __init__(self, dim, global_ratio=0.8, local_ratio=0.2, hidden_expand=2):
        super().__init__()
        self.global_dim = int(global_ratio * dim)
        self.local_dim = int(local_ratio * dim)
        self.identity_dim = dim - self.global_dim - self.local_dim

        # global 分支
        self.global_ffn = nn.Sequential(
            nn.LayerNorm(self.global_dim),
            FeedForward(self.global_dim, self.global_dim * hidden_expand)
        )

        # local 分支
        self.local_ffn = nn.Sequential(
            nn.LayerNorm(self.local_dim),
            FeedForward(self.local_dim, self.local_dim * hidden_expand)
        )

        # projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):  # x: [B, N, C]
        x1, x2, x3 = torch.split(x, [self.global_dim, self.local_dim, self.identity_dim], dim=-1)
        x1 = self.global_ffn(x1)
        x2 = self.local_ffn(x2)
        out = torch.cat([x1, x2, x3], dim=-1)
        return self.output_proj(out)



class MobileMambaBlockSeq(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = MobileMambaModuleSeq(dim=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim=dim * 2, dropout=dropout)

    def forward(self, x):  # x: [B, N, C]
        x = x + self.mamba(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


'''这里有修改:TransformerBlock->MambaTransformerBlock'''

class MambaTransformerBlock(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed=False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate 应该在 0 到 1 之间")

        if hidden_size % 8 != 0:
            raise ValueError("hidden_size 建议为 8 的倍数")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)

        self.mobile_mamba_block = MobileMambaBlockSeq(dim=hidden_size)


        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv52 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(
            nn.Dropout3d(0.1, False),
            nn.Conv3d(hidden_size, hidden_size, 1)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size)) if pos_embed else None

    def forward(self, x):

        B, C, H, W, D = x.shape


        x_reshaped = x.reshape(B, C, H * W * D).permute(0, 2, 1)  # [B, N, C]


        if self.pos_embed is not None:
            x_reshaped = x_reshaped + self.pos_embed


        x_mamba = x_reshaped + self.gamma * self.mobile_mamba_block(self.norm(x_reshaped))


        attn_skip = x_mamba.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]


        attn = self.conv51(attn_skip)
        attn = self.conv52(attn)
        x = attn_skip + self.conv8(attn)

        return x
