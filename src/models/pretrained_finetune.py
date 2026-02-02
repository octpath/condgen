"""学習済み DDPM-CelebAHQ-256 をベースにした年齢条件付き UNet。

入力チャンネルに年齢情報を結合することで、連続値（年齢）による条件付き生成を実現。
ファインチューニング時は学習開始直後から元の CelebA 品質を維持しつつ、
4チャンネル目の重みが徐々に学習される設計。
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from diffusers import DDPMPipeline, UNet2DModel


__all__ = ["AgeConditionedUNet"]


class AgeConditionedUNet(nn.Module):
    """年齢条件付き UNet2D。google/ddpm-celebahq-256 をベースに、
    入力チャンネルを RGB(3) + Age(1) = 4 に拡張したラッパー。

    - conv_in: 3ch → 4ch に置換。元の3ch分の重みは継承、4ch目はゼロ初期化。
    - Forward: sample (B,3,H,W) + age (B,) → age を空間展開して (B,1,H,W) と結合 → (B,4,H,W)
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "google/ddpm-celebahq-256",
        age_max: float = 116.0,
    ) -> None:
        """AgeConditionedUNet を初期化する。

        Args:
            pretrained_model_name_or_path: ベースとなる UNet のモデル ID またはパス。
            age_max: 年齢の正規化最大値。age / age_max で [0,1] に正規化。
        """
        super().__init__()
        self.age_max = age_max

        # Pipeline から UNet を取得（google/ddpm-celebahq-256 は subfolder なし）
        pipe = DDPMPipeline.from_pretrained(pretrained_model_name_or_path)
        self.unet: UNet2DModel = pipe.unet

        # conv_in を 3ch → 4ch に置換
        old_conv = self.unet.conv_in
        out_ch = old_conv.out_channels
        new_conv = nn.Conv2d(4, out_ch, kernel_size=3, stride=1, padding=1)

        # 重み継承: 元の3ch分をコピー、4ch目はゼロ初期化
        with torch.no_grad():
            new_conv.weight[:, :3, :, :].copy_(old_conv.weight)
            new_conv.weight[:, 3:4, :, :].zero_()
            new_conv.bias.copy_(old_conv.bias)

        self.unet.conv_in = new_conv

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        age: torch.Tensor,
        return_dict: bool = True,
    ):
        """ノイズ予測を行う。

        Args:
            sample: ノイズ付き画像 (B, 3, H, W)。
            timestep: タイムステップ (B,) または int。
            age: 年齢（正規化済み [0,1]）(B,)。生年齢の場合は age/age_max で正規化すること。
            return_dict: UNet2DModel の出力形式。

        Returns:
            UNet2DOutput または Tensor。ノイズ予測 (B, 3, H, W)。
        """
        B, _, H, W = sample.shape

        # age を空間方向に展開: (B,) -> (B, 1, H, W)
        age_map = age.view(B, 1, 1, 1).expand(B, 1, H, W)

        # sample (B,3,H,W) + age_map (B,1,H,W) -> (B,4,H,W)
        x = torch.cat([sample, age_map], dim=1)

        return self.unet(x, timestep, return_dict=return_dict)
