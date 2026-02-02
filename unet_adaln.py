"""
U-Net with AdaLN-Zero (Age + Gender conditioning).

train_flexible.py の cond_method='adaln' で利用しているモデルと同じ単一ファイル版。
依存: torch, 標準ライブラリのみ。

forward(sample, timestep, age, gender) でノイズ予測。
- sample: (B, 3, H, W)
- timestep: (B,) long
- age: (B,) or (B, 1), 正規化 [0, 1]
- gender: (B,) long, 0=Male, 1=Female
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["GaussianFourierProjection", "AdaGroupNorm", "UNetAdaLNComplex"]


# -----------------------------------------------------------------------------
# Gaussian Fourier Projection（連続スカラー用）
# -----------------------------------------------------------------------------


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier Projection for continuous scalar (e.g. age)."""

    def __init__(self, embed_dim: int, scale: float = 30.0) -> None:
        super().__init__()
        half_dim = embed_dim // 2
        self.register_buffer("W", torch.randn(half_dim) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) -> (B, embed_dim)"""
        x_proj = x[:, None].float() * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# -----------------------------------------------------------------------------
# Time Embedding
# -----------------------------------------------------------------------------


def _sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
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


def _norm_groups(channels: int) -> int:
    if channels >= 32:
        return 32
    if channels >= 16:
        return 16
    return min(8, channels)


# -----------------------------------------------------------------------------
# AdaGroupNorm（Zero-Init）
# -----------------------------------------------------------------------------


class AdaGroupNorm(nn.Module):
    """条件埋め込みで scale/shift を予測する GroupNorm。Zero-Init。"""

    def __init__(
        self,
        embedding_dim: int,
        num_channels: int,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        num_groups = min(num_groups, num_channels, _norm_groups(num_channels))
        self.group_norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(embedding_dim, num_channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.group_norm(x)
        scale_shift = self.proj(emb)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1.0 + scale) + shift


# -----------------------------------------------------------------------------
# ResBlockAdaGN
# -----------------------------------------------------------------------------


class ResBlockAdaGN(nn.Module):
    """Conv -> AdaGroupNorm(cond) -> SiLU -> Conv -> AdaGroupNorm(cond) -> Time scale_shift -> + skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        cond_dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm1 = AdaGroupNorm(cond_dim, out_channels, num_groups=32)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm2 = AdaGroupNorm(cond_dim, out_channels, num_groups=32)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * out_channels),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h, cond)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h, cond)
        scale_shift = self.time_mlp(t_emb)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return h + self.skip(x)


# -----------------------------------------------------------------------------
# UNetAdaLNComplex
# -----------------------------------------------------------------------------


class UNetAdaLNComplex(nn.Module):
    """U-Net with Age+Gender conditioning via AdaGroupNorm (Zero-Init).

    3レベル: (64, 128, 256)。forward(sample, timestep, age, gender)。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64, 128, 256),
        time_embed_dim: int = 256,
        age_embed_dim: int = 64,
        gender_embed_dim: int = 64,
        fourier_scale: float = 5.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.time_embed_dim = time_embed_dim

        time_input_dim = 32
        self.time_proj = nn.Linear(time_input_dim, time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.age_fourier = GaussianFourierProjection(age_embed_dim, scale=fourier_scale)
        self.age_mlp = nn.Sequential(
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.SiLU(),
        )
        self.gender_embed = nn.Embedding(2, gender_embed_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(age_embed_dim + gender_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        cond_dim = time_embed_dim
        ch_list = list(block_out_channels)

        self.conv_in = nn.Conv2d(in_channels, ch_list[0], 3, padding=1)
        self.down1 = nn.ModuleList([
            ResBlockAdaGN(ch_list[0], ch_list[0], time_embed_dim, cond_dim),
            ResBlockAdaGN(ch_list[0], ch_list[0], time_embed_dim, cond_dim),
        ])
        self.down2 = nn.ModuleList([
            ResBlockAdaGN(ch_list[0], ch_list[1], time_embed_dim, cond_dim),
            ResBlockAdaGN(ch_list[1], ch_list[1], time_embed_dim, cond_dim),
        ])
        self.down3 = nn.ModuleList([
            ResBlockAdaGN(ch_list[1], ch_list[2], time_embed_dim, cond_dim),
            ResBlockAdaGN(ch_list[2], ch_list[2], time_embed_dim, cond_dim),
        ])
        self.downsample = nn.Conv2d(ch_list[0], ch_list[0], 3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(ch_list[1], ch_list[1], 3, stride=2, padding=1)
        self.mid_block = ResBlockAdaGN(ch_list[2], ch_list[2], time_embed_dim, cond_dim)
        self.up1 = nn.ModuleList([
            ResBlockAdaGN(ch_list[2] + ch_list[1], ch_list[1], time_embed_dim, cond_dim),
            ResBlockAdaGN(ch_list[1], ch_list[1], time_embed_dim, cond_dim),
        ])
        self.up2 = nn.ModuleList([
            ResBlockAdaGN(ch_list[1] + ch_list[0], ch_list[0], time_embed_dim, cond_dim),
            ResBlockAdaGN(ch_list[0], ch_list[0], time_embed_dim, cond_dim),
        ])
        self.upsample = nn.ConvTranspose2d(ch_list[2], ch_list[2], 4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(ch_list[1], ch_list[1], 4, stride=2, padding=1)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(_norm_groups(ch_list[0]), ch_list[0]),
            nn.SiLU(),
            nn.Conv2d(ch_list[0], out_channels, 3, padding=1),
        )
        self.ch_list = ch_list
        self.cond_dim = cond_dim

    def _get_time_embed(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = _sinusoidal_timestep_embedding(timesteps.view(-1), 32)
        t_emb = self.time_proj(t_emb)
        t_emb = self.time_mlp(t_emb)
        return t_emb

    def _get_cond(self, age: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        if age.dim() == 2:
            age = age.squeeze(1)
        if gender.dim() == 2:
            gender = gender.squeeze(1)
        a_emb = self.age_fourier(age)
        a_emb = self.age_mlp(a_emb)
        g_emb = self.gender_embed(gender.long().clamp(0, 1))
        cond = torch.cat([a_emb, g_emb], dim=-1)
        cond = self.cond_mlp(cond)
        return cond

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
    ) -> torch.Tensor:
        """sample: (B, 3, H, W), timestep: (B,), age: (B,) or (B,1), gender: (B,) long."""
        t_emb = self._get_time_embed(timestep)
        cond = self._get_cond(age, gender)
        h = self.conv_in(sample)
        s1 = h
        for b in self.down1:
            h = b(h, t_emb, cond)
        s1 = h
        h = self.downsample(h)
        for b in self.down2:
            h = b(h, t_emb, cond)
        s2 = h
        h = self.downsample2(h)
        for b in self.down3:
            h = b(h, t_emb, cond)
        h = self.mid_block(h, t_emb, cond)
        h = self.upsample(h)
        h = torch.cat([h, s2], dim=1)
        for b in self.up1:
            h = b(h, t_emb, cond)
        h = self.upsample2(h)
        h = torch.cat([h, s1], dim=1)
        for b in self.up2:
            h = b(h, t_emb, cond)
        return self.conv_out(h)
