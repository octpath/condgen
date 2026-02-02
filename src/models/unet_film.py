"""FiLM 条件付き U-Net（Diffusion 用バックボーン）。

連続値（年齢）で条件付け可能な U-Net。Time Embedding + Age Embedding を FiLM で
各層に注入する。本番では 2D Conv を 1D Conv に置き換え、年齢を温度・フィード量に
拡張する想定。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Time / Age Embedding（条件埋め込み）
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
# FiLM Layer
# -----------------------------------------------------------------------------


class FiLM(nn.Module):
    """条件埋め込みから Scale (γ) と Shift (β) を生成し、特徴マップに x·(1+γ)+β を適用する。

    本番（時系列）では同じロジックで 1D 特徴に適用可能。
    """

    def __init__(
        self,
        cond_embed_dim: int,
        num_channels: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or cond_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * num_channels),
        )
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W), cond: (B, cond_dim). Returns (B, C, H, W)."""
        # (B, 2*C)
        scale_shift = self.mlp(cond)
        gamma, beta = scale_shift.chunk(2, dim=1)
        # (B, C, 1, 1) でブロードキャスト
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


# -----------------------------------------------------------------------------
# ResBlock with FiLM
# -----------------------------------------------------------------------------


def _norm_groups(channels: int) -> int:
    """GroupNorm の num_groups（channels を割り切る値）。"""
    if channels >= 32:
        return 32
    if channels >= 16:
        return 16
    return min(8, channels)


class ResBlockFiLM(nn.Module):
    """Conv2d -> GroupNorm -> SiLU -> FiLM のカスタムブロック。Time + Age を条件とする。

    将来的に Conv2d を Conv1d に差し替えれば時系列 U-Net に拡張可能。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_embed_dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = nn.GroupNorm(_norm_groups(out_channels), out_channels)
        self.film = FiLM(cond_embed_dim, out_channels)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, C_in, H, W), cond: (B, cond_dim). Returns (B, C_out, H, W)."""
        h = self.conv(x)
        h = self.norm(h)
        h = F.silu(h)
        h = self.film(h, cond)
        return h + self.skip(x)


# -----------------------------------------------------------------------------
# Conditioned U-Net (UNetFiLM)
# -----------------------------------------------------------------------------


class UNetFiLM(nn.Module):
    """連続値（年齢）で条件付け可能な軽量 U-Net。

    64x64 解像度想定。Down: 64 -> 128 -> 256, Middle: 256, Up: 256 -> 128 -> 64。
    各レベルで ResBlockFiLM を使用。入力: ノイズ画像、タイムステップ t、年齢（正規化スカラー）。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        time_embed_dim: int = 256,
        age_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        ch_list = [base_channels * m for m in channel_mults]  # [64, 128, 256]
        cond_embed_dim = time_embed_dim + age_embed_dim
        self.age_embed_dim = age_embed_dim

        # Time embedding: sinusoidal -> MLP
        self.time_proj_dim = base_channels  # sinusoidal の出力次元
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_proj_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # Age embedding: スカラー -> MLP
        self.age_embed = nn.Sequential(
            nn.Linear(1, age_embed_dim),
            nn.SiLU(),
            nn.Linear(age_embed_dim, age_embed_dim),
        )
        # Null embedding for Classifier-Free Guidance (CFG)
        # 学習時に条件をドロップ (age=None) した際に使用する unconditional token
        self.null_age_embed = nn.Parameter(torch.randn(age_embed_dim))

        # Input
        self.conv_in = nn.Conv2d(in_channels, ch_list[0], 3, padding=1)

        # Down
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch_in = ch_list[0]
        for i, ch_out in enumerate(ch_list):
            self.down_blocks.append(
                ResBlockFiLM(ch_in, ch_out, cond_embed_dim)
            )
            self.down_samples.append(
                nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1)
                if i < len(ch_list) - 1
                else nn.Identity()
            )
            ch_in = ch_out

        # Middle
        self.mid_block = ResBlockFiLM(ch_list[-1], ch_list[-1], cond_embed_dim)

        # Up
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        ch_in = ch_list[-1]
        for i, ch_out in enumerate(reversed(ch_list[:-1])):  # [128, 64]
            # concat skip で ch_in + ch_skip -> ResBlock -> ch_out
            ch_skip = ch_list[-(i + 2)]  # down の対応するチャンネル
            self.up_samples.append(nn.ConvTranspose2d(ch_in, ch_in, 4, stride=2, padding=1))
            self.up_blocks.append(
                ResBlockFiLM(ch_in + ch_skip, ch_out, cond_embed_dim)
            )
            ch_in = ch_out
        # 最後の up で ch_in = 64
        self.conv_out = nn.Sequential(
            nn.GroupNorm(_norm_groups(ch_list[0]), ch_list[0]),
            nn.SiLU(),
            nn.Conv2d(ch_list[0], out_channels, 3, padding=1),
        )

        self.ch_list = ch_list
        self.cond_embed_dim = cond_embed_dim

    def _get_cond_embedding(
        self,
        timesteps: torch.Tensor,
        age: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """(B,) timesteps と (B,) or (B,1) or None age から (B, cond_embed_dim)。
        
        age が None の場合、CFG 用の null_age_embed を使用する。
        """
        # Time: (B,) -> (B, time_proj_dim)
        t_emb = sinusoidal_timestep_embedding(
            timesteps.view(-1),
            self.time_proj_dim,
        )
        t_emb = self.time_embed(t_emb)  # (B, time_embed_dim)
        
        # Age: (B,) or (B,1) or None -> (B, age_embed_dim)
        if age is None:
            # Unconditional: null_age_embed を全バッチで使用
            batch_size = t_emb.shape[0]
            a_emb = self.null_age_embed.unsqueeze(0).expand(batch_size, -1)
        else:
            if age.dim() == 1:
                age = age.unsqueeze(1)
            a_emb = self.age_embed(age.float())
        
        return torch.cat([t_emb, a_emb], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        age: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, C, H, W), timesteps: (B,) or (B,1), age: (B,) or (B,1) or None.
        
        age が None の場合、CFG 用の unconditional 生成モードになる。
        """
        cond = self._get_cond_embedding(timesteps, age)  # (B, cond_embed_dim)

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
    model = UNetFiLM(in_channels=3, out_channels=3)
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    age = torch.tensor([0.2, 0.5, 0.8, 0.1])

    # Conditional
    out = model(x, t, age)
    assert out.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {out.shape}"
    print(f"UNetFiLM (conditional): input {x.shape} -> output {out.shape} (OK)")

    # Unconditional (CFG)
    out_uncond = model(x, t, age=None)
    assert out_uncond.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {out_uncond.shape}"
    print(f"UNetFiLM (unconditional): input {x.shape} -> output {out_uncond.shape} (OK)")

