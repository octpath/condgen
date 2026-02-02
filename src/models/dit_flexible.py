"""DiT (Diffusion Transformer) with AdaLN-Zero for Age+Gender.

U-Net の知見（Zero-Init, Fourier Scale 5.0）を活かした再実装。
DiTBlock: c -> SiLU -> Linear(6 params) Zero-Init。modulate で AdaLN。
Final Layer: AdaLN 変調後に線形射影（Zero-Init）。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# CFG 用: 性別の Null Token は Embedding の index 2
GENDER_NULL_INDEX = 2

__all__ = ["modulate", "DiTBlock", "DiT_Flexible", "DiT_Pixel", "GENDER_NULL_INDEX"]


# -----------------------------------------------------------------------------
# Embeddings（Fourier Scale 5.0 をデフォルトに）
# -----------------------------------------------------------------------------


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier Projection for continuous scalar (e.g. age)."""

    def __init__(self, embed_dim: int, scale: float = 5.0) -> None:
        super().__init__()
        half_dim = embed_dim // 2
        self.register_buffer("W", torch.randn(half_dim) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) -> (B, embed_dim)"""
        x_proj = x[:, None].float() * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """(B,) -> (B, embedding_dim)"""
    assert timesteps.dim() == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half_dim, dtype=torch.float32, device=timesteps.device
    ) / half_dim
    emb = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), value=0.0)
    return emb


class PatchEmbed(nn.Module):
    """(B, C, H, W) -> (B, N, embed_dim)"""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# -----------------------------------------------------------------------------
# modulate: x * (1 + scale) + shift
# -----------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x: (B, N, C), shift/scale: (B, C) -> (B, N, C)"""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# -----------------------------------------------------------------------------
# DiTBlock: AdaLN-Zero（c -> SiLU -> Linear 6 params, Zero-Init）
# -----------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """Transformer Block with AdaLN-Zero. c (B, C) -> SiLU -> Linear -> 6 params (Zero-Init)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        # c -> SiLU -> Linear -> 6 params (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 6),
        )
        nn.init.zeros_(self.adaLN_proj[1].weight)
        nn.init.zeros_(self.adaLN_proj[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C), c: (B, cond_dim)"""
        params = self.adaLN_proj(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)

        # Attention: x_norm = modulate(LayerNorm(x), shift_msa, scale_msa); x = x + gate_msa * Attn(x_norm)
        x_norm = self.norm1(x)
        x_norm = modulate(x_norm, shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP: x_norm = modulate(LayerNorm(x), shift_mlp, scale_mlp); x = x + gate_mlp * MLP(x_norm)
        x_norm = self.norm2(x)
        x_norm = modulate(x_norm, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


# -----------------------------------------------------------------------------
# FinalLayer: AdaLN 変調後に線形射影（Zero-Init）
# -----------------------------------------------------------------------------


class FinalLayer(nn.Module):
    """norm(x) -> AdaLN(cond) scale,shift -> modulate -> linear. いずれも Zero-Init。"""

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int,
        out_channels: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Linear(cond_dim, hidden_dim * 2)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, N, hidden_dim), cond: (B, cond_dim) -> (B, N, patch_size^2 * out_channels)"""
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = self.norm(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x


# -----------------------------------------------------------------------------
# DiT_Flexible: Age+Gender, Fourier Scale 5.0, pos_embed, AdaLN-Zero
# -----------------------------------------------------------------------------


class DiT_Flexible(nn.Module):
    """DiT with AdaLN-Zero. Time + Age/Gender (Fourier scale 5.0) -> c。pos_embed 学習可能。"""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        age_embed_dim: int = 64,
        gender_embed_dim: int = 64,
        fourier_scale: float = 5.0,
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

        cond_dim = time_embed_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.age_fourier = GaussianFourierProjection(age_embed_dim, scale=fourier_scale)
        self.age_mlp = nn.Sequential(
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.SiLU(),
        )
        self.null_age_embed = nn.Parameter(torch.randn(1, age_embed_dim))
        self.gender_embed = nn.Embedding(3, gender_embed_dim)
        self.fusion_linear = nn.Linear(age_embed_dim + gender_embed_dim, time_embed_dim)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio, cond_dim) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels, cond_dim)

        self.final_decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_size^2 * C) -> (B, C, H, W)"""
        B = x.shape[0]
        p = self.patch_size
        h = w = self.img_size // p
        x = x.reshape(B, h, w, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.out_channels, h * p, w * p)
        return x

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        use_null_age: Optional[Union[bool, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """sample: (B, 3, H, W), timestep: (B,), age: (B,) or (B,1), gender: (B,) long 0/1/2 (2=Null)."""
        if age.dim() == 2:
            age = age.squeeze(1)
        if gender.dim() == 2:
            gender = gender.squeeze(1)
        B = age.shape[0]
        device = age.device
        if use_null_age is None:
            use_null_age = torch.zeros(B, dtype=torch.bool, device=device)
        elif isinstance(use_null_age, bool):
            use_null_age = torch.full((B,), use_null_age, dtype=torch.bool, device=device)

        x = self.patch_embed(sample)
        x = x + self.pos_embed

        t_emb = sinusoidal_timestep_embedding(timestep.view(-1), self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)

        a_fourier = self.age_fourier(age)
        a_null = self.age_mlp(self.null_age_embed.expand(B, -1))
        a_emb = torch.where(use_null_age.unsqueeze(1), a_null, self.age_mlp(a_fourier))
        g_emb = self.gender_embed(gender.long().clamp(0, GENDER_NULL_INDEX))
        v_concat = torch.cat([a_emb, g_emb], dim=-1)
        v_fused = self.fusion_linear(v_concat)
        c = v_fused + t_emb

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        x = self.final_decoder(x)
        return x


# -----------------------------------------------------------------------------
# DiT_Pixel: Pixel-space 特化（patch_size=2, depth=12, AdaLN-Zero, Null Label）
# -----------------------------------------------------------------------------


class DiT_Pixel(DiT_Flexible):
    """Pixel-space 学習特化 DiT。patch_size=2, depth=12, AdaLN-Zero, Null Label 対応。

    生画像の細部を捉えるため patch_size=2。安定学習のため AdaLN 変調はすべて Zero-Init。
    残差: x = x + gate * Attention(modulate(Norm(x)))。Final Layer も AdaLN (Zero-Init) 後に線形射影。
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 2,
        in_channels: int = 3,
        hidden_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        age_embed_dim: int = 64,
        gender_embed_dim: int = 64,
        fourier_scale: float = 5.0,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            time_embed_dim=time_embed_dim,
            age_embed_dim=age_embed_dim,
            gender_embed_dim=gender_embed_dim,
            fourier_scale=fourier_scale,
        )
