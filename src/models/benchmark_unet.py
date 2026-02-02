"""ベンチマーク用 U-Net: Fourier Feature + FiLM による連続値条件付け。

DiT の構造の難しさを排除し、「Fourier Feature による連続値制御」の効果だけを検証する。
Standard UNet なので学習初期からきれいな顔が出る想定。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["BenchmarkUNet", "GaussianFourierProjection"]


# -----------------------------------------------------------------------------
# Gaussian Fourier Projection（連続値条件付け用）
# -----------------------------------------------------------------------------


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier Projection for continuous scalar (e.g. age).
    学習しないランダム重みでスカラーを sin/cos 特徴に写す。
    """

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


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """DDPM 風の正弦波タイムステップ埋め込み。(B,) -> (B, embedding_dim)."""
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


# -----------------------------------------------------------------------------
# FiLM (Scale & Shift)
# -----------------------------------------------------------------------------


def _norm_groups(channels: int) -> int:
    if channels >= 32:
        return 32
    if channels >= 16:
        return 16
    return min(8, channels)


class FiLM(nn.Module):
    """条件埋め込みから Scale (γ) と Shift (β) を生成し、x·(1+γ)+β を適用。"""

    def __init__(self, cond_embed_dim: int, num_channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * num_channels),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.mlp(cond)
        gamma, beta = scale_shift.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class ResBlockFiLM(nn.Module):
    """Conv -> GroupNorm -> SiLU -> FiLM。cond_emb で Scale & Shift。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_embed_dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.norm = nn.GroupNorm(_norm_groups(out_channels), out_channels)
        self.film = FiLM(cond_embed_dim, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        h = F.silu(h)
        h = self.film(h, cond)
        return h + self.skip(x)


# -----------------------------------------------------------------------------
# Benchmark U-Net (Fourier Feature + cond_emb = Time_MLP + Age_MLP)
# -----------------------------------------------------------------------------


class BenchmarkUNet(nn.Module):
    """ベンチマーク用軽量 U-Net。Fourier Feature で年齢を埋め込み、Time と加算して FiLM で注入。

    - 条件付け: cond_emb = MLP(Time_Emb) + MLP(Fourier(Age))
    - 各 ResBlock で cond_emb により Scale & Shift (FiLM/AdaGN)
    - channel=[64, 128, 256]、64x64 想定。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channel_list: Tuple[int, ...] = (64, 128, 256),
        cond_dim: int = 128,
        time_embed_dim: int = 64,
        age_fourier_dim: int = 64,
        fourier_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch_list = list(channel_list)
        self.cond_dim = cond_dim

        # Time: sinusoidal -> MLP -> cond_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # Age: Fourier Feature -> MLP -> cond_dim
        self.age_fourier = GaussianFourierProjection(age_fourier_dim, scale=fourier_scale)
        self.age_mlp = nn.Sequential(
            nn.Linear(age_fourier_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, self.ch_list[0], 3, padding=1)

        # Down
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch_in = self.ch_list[0]
        for i, ch_out in enumerate(self.ch_list):
            self.down_blocks.append(ResBlockFiLM(ch_in, ch_out, cond_dim))
            self.down_samples.append(
                nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1)
                if i < len(self.ch_list) - 1
                else nn.Identity()
            )
            ch_in = ch_out

        # Middle
        self.mid_block = ResBlockFiLM(self.ch_list[-1], self.ch_list[-1], cond_dim)

        # Up
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        ch_in = self.ch_list[-1]
        for i, ch_out in enumerate(reversed(self.ch_list[:-1])):
            ch_skip = self.ch_list[-(i + 2)]
            self.up_samples.append(nn.ConvTranspose2d(ch_in, ch_in, 4, stride=2, padding=1))
            self.up_blocks.append(ResBlockFiLM(ch_in + ch_skip, ch_out, cond_dim))
            ch_in = ch_out

        self.conv_out = nn.Sequential(
            nn.GroupNorm(_norm_groups(self.ch_list[0]), self.ch_list[0]),
            nn.SiLU(),
            nn.Conv2d(self.ch_list[0], out_channels, 3, padding=1),
        )

        self.time_embed_dim = time_embed_dim

    def _get_cond_embedding(self, timesteps: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        """cond_emb = MLP(Time_Emb) + MLP(Fourier(Age))。"""
        t_emb = sinusoidal_timestep_embedding(timesteps.view(-1), self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        a_emb = self.age_fourier(age.view(-1))
        a_emb = self.age_mlp(a_emb)
        return t_emb + a_emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, C, H, W), timesteps: (B,), age: (B,) 正規化 [0,1]。"""
        cond = self._get_cond_embedding(timesteps, age)

        h = self.conv_in(x)
        skips = []
        for block, sample in zip(self.down_blocks, self.down_samples):
            h = block(h, cond)
            skips.append(h)
            h = sample(h)

        h = self.mid_block(h, cond)

        for up_sample, block, skip in zip(
            self.up_samples,
            self.up_blocks,
            reversed(skips[:-1]),
        ):
            h = up_sample(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        return self.conv_out(h)


# -----------------------------------------------------------------------------
# 動作確認
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    B, C, H, W = 4, 3, 64, 64
    model = BenchmarkUNet(in_channels=3, out_channels=3)
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    age = torch.rand(B)

    out = model(x, t, age)
    assert out.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {out.shape}"
    print(f"BenchmarkUNet: input {x.shape} -> output {out.shape} (OK)")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
