"""DiT with Cross-Attention (年齢のみ)。AdaLN を廃止し、条件は Cross-Attention で注入。

実験 A: DiT の学習不安定性が AdaLN 起因か切り分ける。
Time のみ AdaLN で注入し、年齢は Cross-Attention で渡す。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dit import (
    GaussianFourierProjection,
    PatchEmbed,
    sinusoidal_timestep_embedding,
)


__all__ = ["DiT_CrossAttn"]


# -----------------------------------------------------------------------------
# AdaLN for Time only（条件は Cross-Attention で別途注入）
# -----------------------------------------------------------------------------


class AdaLNTimeOnly(nn.Module):
    """Time Embedding のみから Scale/Shift を回帰。ゼロ初期化。"""

    def __init__(self, time_embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(time_embed_dim, hidden_dim * 6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        params = self.linear(t_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


# -----------------------------------------------------------------------------
# DiT Block: LayerNorm + Self-Attn + Cross-Attn (Age) + MLP、Time は AdaLN
# -----------------------------------------------------------------------------


class DiTBlockCrossAttn(nn.Module):
    """Self-Attention → Cross-Attention (Age) → MLP。Time は AdaLN で Scale/Shift。"""

    def __init__(
        self,
        hidden_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.adaLN_time = AdaLNTimeOnly(time_embed_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, N, D), t_emb: (B, time_dim), encoder_hidden_states: (B, 1, D)。"""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_time(t_emb)
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm = self.norm2(x)
        cross_out, _ = self.cross_attn(
            x_norm, encoder_hidden_states, encoder_hidden_states
        )
        x = x + cross_out

        x_norm = self.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


# -----------------------------------------------------------------------------
# Final Layer（Time のみ AdaLN）
# -----------------------------------------------------------------------------


class FinalLayerTimeOnly(nn.Module):
    """最終層。Time Embedding のみで AdaLN。"""

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int,
        out_channels: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        self.adaLN = nn.Linear(time_embed_dim, hidden_dim * 2)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(t_emb).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


# -----------------------------------------------------------------------------
# Age Encoder: (B,) -> (B, 1, hidden_dim)
# -----------------------------------------------------------------------------


class AgeEncoderDiT(nn.Module):
    """年齢スカラーを Fourier -> MLP で (B, 1, hidden_dim) に。Cross-Attention の encoder_hidden_states 用。"""

    def __init__(self, hidden_dim: int, fourier_dim: int = 64, scale: float = 30.0) -> None:
        super().__init__()
        self.fourier = GaussianFourierProjection(fourier_dim, scale=scale)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """age: (B,) -> (B, 1, hidden_dim)"""
        x = self.fourier(age)
        x = self.mlp(x)
        return x.unsqueeze(1)


# -----------------------------------------------------------------------------
# DiT_CrossAttn Model
# -----------------------------------------------------------------------------


class DiT_CrossAttn(nn.Module):
    """DiT with Cross-Attention（年齢のみ）。Time は AdaLN、Age は Cross-Attention。"""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 128,
        age_fourier_dim: int = 64,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_dim = hidden_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.age_encoder = AgeEncoderDiT(hidden_dim, fourier_dim=age_fourier_dim)

        self.blocks = nn.ModuleList([
            DiTBlockCrossAttn(hidden_dim, num_heads, mlp_ratio, time_embed_dim)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayerTimeOnly(
            hidden_dim, patch_size, in_channels, time_embed_dim
        )
        self.final_decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = self.patch_size
        h = w = self.img_size // p
        x = x.reshape(B, h, w, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.out_channels, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed

        t_emb = sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        encoder_hidden_states = self.age_encoder(age)

        for block in self.blocks:
            x = block(x, t_emb, encoder_hidden_states)
        x = self.final_layer(x, t_emb)
        x = self.unpatchify(x)
        x = self.final_decoder(x)
        return x
