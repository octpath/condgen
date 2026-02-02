"""Cross-Attention 条件付き U-Net（年齢を encoder_hidden_states で注入）。

diffusers.UNet2DConditionModel をベースに、年齢スカラーを Fourier Feature + MLP で
(B, 1, cross_attention_dim) に変換し、Attention 層を通じて注入する。
画像空間（Concatenation）を使わないため色の干渉を防ぐ。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

from src.models.dit import GaussianFourierProjection


__all__ = ["AgeEncoder", "create_unet_crossattn_64"]


# -----------------------------------------------------------------------------
# Age Encoder: 年齢スカラー → (B, 1, cross_attention_dim)
# -----------------------------------------------------------------------------


class AgeEncoder(nn.Module):
    """年齢スカラーを Fourier Feature 化し、MLP で cross_attention_dim 次元に変換する。

    出力は UNet2DConditionModel の encoder_hidden_states として渡す。
    (B,) or (B,1) → (B, 1, cross_attention_dim)
    """

    def __init__(
        self,
        cross_attention_dim: int = 128,
        fourier_embed_dim: int = 64,
        fourier_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.fourier = GaussianFourierProjection(fourier_embed_dim, scale=fourier_scale)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_embed_dim, cross_attention_dim * 2),
            nn.SiLU(),
            nn.Linear(cross_attention_dim * 2, cross_attention_dim),
        )

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """age: (B,) or (B, 1). 正規化 [0, 1]. → (B, 1, cross_attention_dim)"""
        if age.dim() == 2:
            age = age.squeeze(1)
        x = self.fourier(age)
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
    """64x64 用の Cross-Attention 条件付き U-Net を作成する。

    入力は RGB 3ch のみ。年齢条件は encoder_hidden_states で渡す。
    Time Embedding は標準のまま。色は画像空間に載せないため干渉が少ない。

    Args:
        in_channels: 入力チャネル数（3 = RGB）。
        out_channels: 出力チャネル数（3 = RGB）。
        cross_attention_dim: Cross-Attention の条件次元（128 or 256 推奨）。
        block_out_channels: 各ブロックの出力チャネル。
        norm_num_groups: GroupNorm のグループ数。
        sample_size: 入力画像サイズ。
        layers_per_block: 各ブロックの ResNet レイヤー数。
        transformer_layers_per_block: 各 CrossAttn ブロックの Transformer レイヤー数。

    Returns:
        UNet2DConditionModel。forward(sample, timestep, encoder_hidden_states) でノイズ予測。
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
