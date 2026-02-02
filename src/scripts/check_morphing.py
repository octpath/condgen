#!/usr/bin/env python3
"""即時確認用: 最新チェックポイントで DDIM + CFG のモーフィング動画を生成する。

train_flexible.py で学習したモデル（FlexibleConditionalUNet / DiT_Pixel）、
train_unet_adaln で学習した UNetAdaLNComplex、または train_crossattn で学習した
CrossAttn U-Net + AgeEncoder をロードし、create_morphing_gif で年齢モーフィングGIFを保存する。

実行例（プロジェクトルートから）:
  uv run python src/scripts/check_morphing.py
  uv run python src/scripts/check_morphing.py --checkpoint_dir outputs/dit_pixel/checkpoints --output_dir outputs/dit_pixel/samples
  uv run python src/scripts/check_morphing.py --checkpoint_dir outputs/crossattn/checkpoints --cross_attention_dim 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from loguru import logger

from src.dataset.loader import AGE_MAX
from src.models.unet_crossattn import AgeEncoder, create_unet_crossattn_64
from src.scripts.load_flexible_checkpoint import (
    build_and_load_model,
    find_latest_checkpoint,
    load_config,
)
from src.training.train_flexible import sample_flexible_from_noise_cfg
from src.training.train_unet_adaln import sample_with_age_gender_from_noise_cfg
from src.utils.visualize import create_morphing_gif


@torch.no_grad()
def _sample_crossattn_ddim(
    unet,
    age_encoder,
    scheduler: DDIMScheduler,
    x_t_initial: torch.Tensor,
    age: float,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    guidance_scale: float | None = None,
) -> torch.Tensor:
    """固定ノイズから年齢条件付きで1枚 denoise（DDIM・オプションで CFG）。CrossAttn 用。"""
    unet.eval()
    age_encoder.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = x_t_initial.clone()
    B = x_t.shape[0]
    age_batch = torch.full((B,), age, device=device, dtype=x_t.dtype)
    encoder_hidden_states = age_encoder(age_batch)

    use_cfg = guidance_scale is not None and guidance_scale > 1.0
    if use_cfg:
        cross_attention_dim = encoder_hidden_states.shape[-1]
        null_encoder = torch.zeros(
            B, 1, cross_attention_dim, device=device, dtype=encoder_hidden_states.dtype
        )

    for t in scheduler.timesteps:
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        if use_cfg:
            pred_cond = unet(
                x_t, t_batch, encoder_hidden_states=encoder_hidden_states
            ).sample
            pred_uncond = unet(
                x_t, t_batch, encoder_hidden_states=null_encoder
            ).sample
            noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            noise_pred = unet(
                x_t, t_batch, encoder_hidden_states=encoder_hidden_states
            ).sample
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check morphing with DDIM (train_flexible / train_unet_adaln / crossattn)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/dit_pixel/checkpoints",
        help="チェックポイントディレクトリ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="GIF 保存先（未指定時は checkpoint_dir の親 / samples）",
    )
    parser.add_argument(
        "--gender",
        type=int,
        default=0,
        choices=[0, 1],
        help="モーフィングで固定する性別 (0=male, 1=female)。flexible/adaln のみ有効",
    )
    parser.add_argument(
        "--cross_attention_dim",
        type=int,
        default=128,
        help="CrossAttn モデル用の次元（crossattn チェックポイント時のみ使用）",
    )
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=[4.0, 8.0, 12.0],
        help="ガイダンススケールのリスト（各スケールで1本GIF保存）",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="DDIM ステップ数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--age_min",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--age_max",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--age_step",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="EMA チェックポイントを優先しない",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    ckpt_path = find_latest_checkpoint(checkpoint_dir, prefer_ema=not args.no_ema)
    if ckpt_path is None:
        logger.error(f"チェックポイントが見つかりません: {checkpoint_dir}")
        sys.exit(1)
    logger.info(f"使用チェックポイント: {ckpt_path}")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir.parent / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = load_config(checkpoint_dir.parent)

    # CrossAttn: config がなく state に unet / age_encoder がある場合
    if config is None and isinstance(state, dict) and "unet" in state and "age_encoder" in state:
        logger.info("Detected CrossAttn checkpoint (unet + age_encoder)")
        unet = create_unet_crossattn_64(cross_attention_dim=args.cross_attention_dim)
        age_encoder = AgeEncoder(cross_attention_dim=args.cross_attention_dim)
        unet.load_state_dict(state["unet"], strict=True)
        age_encoder.load_state_dict(state["age_encoder"], strict=True)
        unet = unet.to(device).eval()
        age_encoder = age_encoder.to(device).eval()
        ddpm = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        ddim = DDIMScheduler.from_config(ddpm.config)

        def sample_fn(age_norm: float, noise: torch.Tensor, guidance_scale: float | None = None):
            return _sample_crossattn_ddim(
                unet,
                age_encoder,
                ddim,
                noise,
                age_norm,
                num_inference_steps=args.num_inference_steps,
                device=device,
                guidance_scale=guidance_scale,
            )

        gif_path = output_dir / "check_morphing.gif"
        create_morphing_gif(
            sample_fn,
            gif_path,
            age_min=args.age_min,
            age_max=args.age_max,
            age_step=args.age_step,
            duration_per_frame=200,
            input_range=(-1.0, 1.0),
            noise_shape=(1, 3, 64, 64),
            device=device,
            seed=args.seed,
            age_scale=AGE_MAX,
            guidance_scales=args.guidance_scales,
        )
        logger.info(f"保存先: {output_dir}")
        for gs in args.guidance_scales:
            logger.info(f"  - {gif_path.stem}_gs{gs}.gif")
        return

    # Flexible または AdaLN standalone
    try:
        model, model_type = build_and_load_model(
            checkpoint_dir, device, prefer_ema=not args.no_ema
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Loaded model type: {model_type}")
    dtype = next(model.parameters()).dtype
    ddpm = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    ddim = DDIMScheduler.from_config(ddpm.config)
    gender_int = args.gender

    def sample_fn(age_norm: float, noise: torch.Tensor, guidance_scale: float | None = None):
        gs = guidance_scale if guidance_scale is not None else 1.0
        if model_type == "flexible":
            return sample_flexible_from_noise_cfg(
                model,
                ddim,
                noise,
                age_norm,
                gender_int,
                guidance_scale=gs,
                num_inference_steps=args.num_inference_steps,
                device=device,
            )
        return sample_with_age_gender_from_noise_cfg(
            model,
            ddim,
            noise,
            age_norm,
            gender_int,
            guidance_scale=gs,
            num_inference_steps=args.num_inference_steps,
            device=device,
        )

    gif_path = output_dir / "check_morphing.gif"
    create_morphing_gif(
        sample_fn,
        gif_path,
        age_min=args.age_min,
        age_max=args.age_max,
        age_step=args.age_step,
        duration_per_frame=200,
        input_range=(-1.0, 1.0),
        noise_shape=(1, 3, 64, 64),
        device=device,
        dtype=dtype,
        seed=args.seed,
        age_scale=AGE_MAX,
        guidance_scales=args.guidance_scales,
    )
    logger.info(f"保存先: {output_dir}")
    for gs in args.guidance_scales:
        logger.info(f"  - {gif_path.stem}_gs{gs}.gif")


if __name__ == "__main__":
    main()
