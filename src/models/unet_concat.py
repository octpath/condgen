"""入力チャネル結合 (Input Concatenation) 用 U-Net。

diffusers.UNet2DModel を in_channels=4 (RGB + Age) で初期化。
Time Embedding には一切干渉せず、標準のメカニズムのみ使用する。
"""

from __future__ import annotations

from typing import Tuple

from diffusers import UNet2DModel


__all__ = ["create_unet_concat_64"]


def create_unet_concat_64(
    in_channels: int = 4,
    out_channels: int = 3,
    block_out_channels: Tuple[int, ...] = (64, 128, 128, 256),
    norm_num_groups: int = 8,
    sample_size: int = 64,
    layers_per_block: int = 2,
) -> UNet2DModel:
    """64x64 用の入力チャネル結合 U-Net を作成する。

    RGB(3) + Age(1) = 4 チャネルを入力とし、ノイズ予測 (3 ch) を出力する。
    Time Embedding は標準のまま使用（介入なし）。

    Args:
        in_channels: 入力チャネル数（4 = RGB + Age）。
        out_channels: 出力チャネル数（3 = RGB）。
        block_out_channels: 各ブロックの出力チャネル（深さに応じて調整）。
        norm_num_groups: GroupNorm のグループ数（学習安定化）。
        sample_size: 入力画像サイズ。
        layers_per_block: 各ブロックのレイヤー数。

    Returns:
        UNet2DModel。forward(sample, timestep) でノイズ予測。
    """
    n = len(block_out_channels)
    down_block_types = ("DownBlock2D",) * (n - 1) + ("DownBlock2D",)
    up_block_types = ("UpBlock2D",) + ("UpBlock2D",) * (n - 1)

    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
        downsample_padding=0,
    )
