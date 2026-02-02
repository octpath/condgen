"""
Cross-Attention 条件付き U-Net（年齢+性別を encoder_hidden_states で注入）。

train_flexible.py の cond_method='cross_attn' で利用しているモデルと同じ単一ファイル版。
依存: torch, diffusers。

- UNet: forward(sample, timestep, encoder_hidden_states)
- AgeGenderEncoder: forward(age, gender) -> (B, 1, cross_attention_dim)
  年齢は Fourier、性別は Embedding で埋め込み、結合して MLP で cross_attention_dim に。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


__all__ = ["GaussianFourierProjection", "AgeGenderEncoder", "create_unet_crossattn_64"]


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
# AgeGender Encoder: 年齢＋性別 → (B, 1, cross_attention_dim)
# -----------------------------------------------------------------------------


class AgeGenderEncoder(nn.Module):
    """年齢（Fourier）と性別（Embedding）を結合し、MLP で cross_attention_dim に変換。

    (B,) or (B,1) の age、および (B,) long の gender → (B, 1, cross_attention_dim)
    gender: 0=Male, 1=Female
    """

    def __init__(
        self,
        cross_attention_dim: int = 128,
        fourier_embed_dim: int = 64,
        fourier_scale: float = 10.0,
        gender_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.fourier = GaussianFourierProjection(fourier_embed_dim, scale=fourier_scale)
        self.gender_embed = nn.Embedding(2, gender_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_embed_dim + gender_embed_dim, cross_attention_dim * 2),
            nn.SiLU(),
            nn.Linear(cross_attention_dim * 2, cross_attention_dim),
        )

    def forward(self, age: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        """age: (B,) or (B, 1). 正規化 [0, 1]. gender: (B,) long. → (B, 1, cross_attention_dim)"""
        if age.dim() == 2:
            age = age.squeeze(1)
        if gender.dim() == 2:
            gender = gender.squeeze(1)
        a_emb = self.fourier(age)
        g_emb = self.gender_embed(gender.long().clamp(0, 1))
        x = torch.cat([a_emb, g_emb], dim=-1)
        x = self.mlp(x)
        return x.unsqueeze(1)


# -----------------------------------------------------------------------------
# UNet2DConditionModel 64x64 ファクトリ
# -----------------------------------------------------------------------------


def create_unet_crossattn_64(
    in_channels: int = 3,
    out_channels: int = 3,
    cross_attention_dim: int = 128,
    block_out_channels: Tuple[int, ...] = (64, 128, 128, 256),
    norm_num_groups: int = 8,
    sample_size: int = 64,
    layers_per_block: int = 2,
    transformer_layers_per_block: int = 1,
) -> UNet2DConditionModel:
    """64x64 用の Cross-Attention 条件付き U-Net を作成。

    入力は RGB 3ch。条件は encoder_hidden_states で渡す（AgeGenderEncoder(age, gender) の出力）。
    forward(sample, timestep, encoder_hidden_states) でノイズ予測。
    """
    n = len(block_out_channels)
    down_block_types = ("CrossAttnDownBlock2D",) * (n - 1) + ("DownBlock2D",)
    up_block_types = ("UpBlock2D",) + ("CrossAttnUpBlock2D",) * (n - 1)

    return UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        mid_block_type="UNetMidBlock2DCrossAttn",
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
        cross_attention_dim=cross_attention_dim,
        transformer_layers_per_block=transformer_layers_per_block,
        downsample_padding=0,
        attention_head_dim=8,
    )
