"""条件付け手法を引数で切り替える統合型 U-Net。

cond_method: 'concat' | 'cross_attn' | 'adaln'
fourier_scale: GaussianFourierProjection のスケール（adaln / cross_attn で使用）。
共通で Age と Gender を受け取り、内部でエンコードして使用する。
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import UNet2DModel, UNet2DConditionModel

from src.models.dit import GaussianFourierProjection
from src.models.unet_adaln_complex import GENDER_NULL_INDEX, UNetAdaLNComplex


__all__ = ["FlexibleConditionalUNet", "COND_METHODS", "GENDER_NULL_INDEX"]

COND_METHODS = ("concat", "cross_attn", "adaln")


# -----------------------------------------------------------------------------
# Concat: 入力チャネル拡張 (3 + Age(1) + Gender(1) = 5)
# -----------------------------------------------------------------------------


def _create_concat_backbone(
    sample_size: int = 64,
    block_out_channels: Tuple[int, ...] = (64, 128, 128, 256),
    norm_num_groups: int = 8,
    layers_per_block: int = 2,
) -> UNet2DModel:
    n = len(block_out_channels)
    down_block_types = ("DownBlock2D",) * (n - 1) + ("DownBlock2D",)
    up_block_types = ("UpBlock2D",) + ("UpBlock2D",) * (n - 1)
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=5,
        out_channels=3,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
        downsample_padding=0,
    )


# -----------------------------------------------------------------------------
# CrossAttn: Age/Gender を埋め込み encoder_hidden_states に注入
# -----------------------------------------------------------------------------


class AgeGenderEncoder(nn.Module):
    """Age (Fourier or null_age_embed) + Gender (Embedding, index 2=Null) -> (B, 1, cross_attention_dim)."""

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
        self.null_age_embed = nn.Parameter(torch.randn(1, fourier_embed_dim))
        self.gender_embed = nn.Embedding(3, gender_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_embed_dim + gender_embed_dim, cross_attention_dim * 2),
            nn.SiLU(),
            nn.Linear(cross_attention_dim * 2, cross_attention_dim),
        )

    def forward(
        self,
        age: torch.Tensor,
        gender: torch.Tensor,
        use_null_age: Optional[Union[bool, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """age: (B,) or (B,1), gender: (B,) long 0/1/2 (2=Null). use_null_age: True の箇所は null_age_embed を使用。"""
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
        a_fourier = self.fourier(age)
        a_null = self.null_age_embed.expand(B, -1)
        a_emb = torch.where(use_null_age.unsqueeze(1), a_null, a_fourier)
        g_emb = self.gender_embed(gender.long().clamp(0, GENDER_NULL_INDEX))
        x = torch.cat([a_emb, g_emb], dim=-1)
        x = self.mlp(x)
        return x.unsqueeze(1)


def _create_cross_attn_backbone(
    sample_size: int = 64,
    cross_attention_dim: int = 128,
    block_out_channels: Tuple[int, ...] = (64, 128, 128, 256),
    norm_num_groups: int = 8,
    layers_per_block: int = 2,
    transformer_layers_per_block: int = 1,
) -> UNet2DConditionModel:
    n = len(block_out_channels)
    down_block_types = ("CrossAttnDownBlock2D",) * (n - 1) + ("DownBlock2D",)
    up_block_types = ("UpBlock2D",) + ("CrossAttnUpBlock2D",) * (n - 1)
    return UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=3,
        out_channels=3,
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


# -----------------------------------------------------------------------------
# FlexibleConditionalUNet: 統合インターフェース
# -----------------------------------------------------------------------------


class FlexibleConditionalUNet(nn.Module):
    """条件付け手法を cond_method で切り替える U-Net。Age と Gender を共通で受け取る。"""

    def __init__(
        self,
        cond_method: Literal["concat", "cross_attn", "adaln"] = "adaln",
        fourier_scale: float = 10.0,
        sample_size: int = 64,
        block_out_channels: Tuple[int, ...] = (64, 128, 256),
        cross_attention_dim: int = 128,
        norm_num_groups: int = 8,
        layers_per_block: int = 2,
        transformer_layers_per_block: int = 1,
    ) -> None:
        super().__init__()
        self.cond_method = cond_method
        self.fourier_scale = fourier_scale
        self.sample_size = sample_size

        if cond_method == "concat":
            self.backbone = _create_concat_backbone(
                sample_size=sample_size,
                block_out_channels=block_out_channels,
                norm_num_groups=norm_num_groups,
                layers_per_block=layers_per_block,
            )
            self.encoder = None
        elif cond_method == "cross_attn":
            self.backbone = _create_cross_attn_backbone(
                sample_size=sample_size,
                cross_attention_dim=cross_attention_dim,
                block_out_channels=block_out_channels,
                norm_num_groups=norm_num_groups,
                layers_per_block=layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block,
            )
            self.encoder = AgeGenderEncoder(
                cross_attention_dim=cross_attention_dim,
                fourier_embed_dim=64,
                fourier_scale=fourier_scale,
                gender_embed_dim=64,
            )
        elif cond_method == "adaln":
            self.backbone = UNetAdaLNComplex(
                in_channels=3,
                out_channels=3,
                block_out_channels=block_out_channels,
                time_embed_dim=256,
                age_embed_dim=64,
                gender_embed_dim=64,
                fourier_scale=fourier_scale,
            )
            self.encoder = None
        else:
            raise ValueError(f"cond_method must be one of {COND_METHODS}, got {cond_method!r}")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        use_null_age: Optional[Union[bool, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """sample: (B, 3, H, W), timestep: (B,) or int, age: (B,) or (B,1), gender: (B,) long 0/1/2 (2=Null)."""
        if self.cond_method == "concat":
            B, _, H, W = sample.shape
            if age.dim() == 2:
                age = age.squeeze(1)
            if gender.dim() == 2:
                gender = gender.squeeze(1)
            device = sample.device
            if use_null_age is None:
                use_null_age = torch.zeros(B, dtype=torch.bool, device=device)
            elif isinstance(use_null_age, bool):
                use_null_age = torch.full((B,), use_null_age, dtype=torch.bool, device=device)
            age_ch = torch.where(use_null_age, torch.full_like(age, 0.5), age).view(B, 1, 1, 1).expand(B, 1, H, W).to(sample.dtype)
            gender_ch = torch.where(
                gender == GENDER_NULL_INDEX,
                torch.full((B,), 0.5, device=device, dtype=torch.float32),
                gender.float(),
            ).view(B, 1, 1, 1).expand(B, 1, H, W).to(sample.dtype)
            x = torch.cat([sample, age_ch, gender_ch], dim=1)
            return self.backbone(x, timestep).sample
        elif self.cond_method == "cross_attn":
            encoder_hidden_states = self.encoder(age, gender, use_null_age=use_null_age)
            return self.backbone(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample
        else:
            return self.backbone(sample, timestep, age, gender, use_null_age=use_null_age)
