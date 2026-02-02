"""train_flexible.py で学習したチェックポイントの読み込みヘルパー。

config.json があれば FlexibleConditionalUNet / DiT_Pixel を構築して state_dict をロード。
なければ train_unet_adaln 用の UNetAdaLNComplex としてロード（後方互換）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_latest_checkpoint(checkpoint_dir: Path, prefer_ema: bool = True) -> Optional[Path]:
    """model_epoch_XXXX.pt または model_epoch_XXXX_ema.pt のうち最新を返す。"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    candidates = list(checkpoint_dir.glob("model_epoch_*_ema.pt")) + list(
        checkpoint_dir.glob("model_epoch_*.pt")
    )

    def epoch_num(p: Path) -> int:
        stem = p.stem.replace("_ema", "")
        if "epoch_" in stem:
            try:
                return int(stem.split("epoch_")[1].split(".")[0])
            except ValueError:
                return -1
        return -1

    candidates = sorted(set(candidates), key=epoch_num, reverse=True)
    if not candidates:
        return None
    if prefer_ema:
        ema_first = [p for p in candidates if "_ema" in p.name]
        if ema_first:
            return ema_first[0]
    return candidates[0]


def load_config(output_dir: Path) -> Optional[dict]:
    """output_dir/config.json を読む。なければ None。"""
    config_path = Path(output_dir) / "config.json"
    if not config_path.is_file():
        return None
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def build_and_load_model(
    checkpoint_dir: Path,
    device: torch.device,
    prefer_ema: bool = True,
) -> Tuple[torch.nn.Module, str]:
    """最新チェックポイントをロードしてモデルを返す。

    config.json が checkpoint_dir の親にある場合:
        model=unet -> FlexibleConditionalUNet, model=dit -> DiT_Pixel を構築してロード。
    ない場合:
        UNetAdaLNComplex を構築してロード（train_unet_adaln 用）。

    Returns:
        (model, model_type): model_type は "flexible" または "adaln_standalone"。
    """
    ckpt_path = _find_latest_checkpoint(checkpoint_dir, prefer_ema=prefer_ema)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    output_dir = checkpoint_dir.parent
    config = load_config(output_dir)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)

    if config is not None and config.get("model") == "dit":
        from src.models.dit_flexible import DiT_Pixel
        model = DiT_Pixel(
            img_size=64,
            patch_size=int(config.get("patch_size", 2)),
            in_channels=3,
            hidden_dim=int(config.get("hidden_size", 384)),
            depth=int(config.get("depth", 12)),
            num_heads=int(config.get("num_heads", 6)),
            mlp_ratio=4.0,
            time_embed_dim=256,
            age_embed_dim=64,
            gender_embed_dim=64,
            fourier_scale=float(config.get("fourier_scale", 5.0)),
        )
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        return model, "flexible"

    if config is not None and config.get("model") == "unet":
        from src.models.unet_flexible import FlexibleConditionalUNet
        cond_method = config.get("cond_method", "adaln")
        fourier_scale = float(config.get("fourier_scale", 5.0))
        use_fourier_features = not config.get("no_fourier", False)
        model = FlexibleConditionalUNet(
            cond_method=cond_method,
            fourier_scale=fourier_scale,
            sample_size=64,
            block_out_channels=(64, 128, 256),
            cross_attention_dim=128,
            norm_num_groups=8,
            layers_per_block=2,
            transformer_layers_per_block=1,
            use_fourier_features=use_fourier_features,
        )
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        return model, "flexible"

    # 後方互換: train_unet_adaln のチェックポイント（UNetAdaLNComplex 単体）
    from src.models.unet_adaln_complex import UNetAdaLNComplex
    model = UNetAdaLNComplex(
        in_channels=3,
        out_channels=3,
        block_out_channels=(64, 128, 256),
        time_embed_dim=256,
        age_embed_dim=64,
        gender_embed_dim=64,
        fourier_scale=5.0,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, "adaln_standalone"


def find_latest_checkpoint(checkpoint_dir: Path, prefer_ema: bool = True) -> Optional[Path]:
    """公開: 最新チェックポイントパスを返す。"""
    return _find_latest_checkpoint(Path(checkpoint_dir), prefer_ema=prefer_ema)
