"""共通埋め込み: Age/Gender 条件付けで使う定数と Fourier 射影。

train_flexible.py から呼ばれる unet_flexible / dit_flexible および
unet_adaln_complex で共有する。参照を一箇所にまとめ、重複定義を避ける。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


__all__ = ["GENDER_NULL_INDEX", "GaussianFourierProjection"]


# CFG (Classifier-Free Guidance) 用: 性別の Null Token は Embedding の index 2
GENDER_NULL_INDEX = 2


class GaussianFourierProjection(nn.Module):
    """連続スカラー（年齢など）の Gaussian Fourier 射影。

    U-Net/DiT の知見で scale=5.0 をデフォルトに。本番では温度・フィード量等に拡張。
    """

    def __init__(self, embed_dim: int, scale: float = 5.0) -> None:
        super().__init__()
        half_dim = embed_dim // 2
        self.register_buffer("W", torch.randn(half_dim) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) -> (B, embed_dim)"""
        x_proj = x[:, None].float() * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
