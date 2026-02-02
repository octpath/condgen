#!/usr/bin/env python3
"""即時確認用: 最新チェックポイントで Guidance Scale ごとのグリッドを生成する。

train_flexible.py で学習したモデル（FlexibleConditionalUNet / DiT_Pixel）または
train_unet_adaln で学習した UNetAdaLNComplex をロードし、save_multi_scale_grids で
スケールごとのグリッドを DDIM + CFG で決定論的に生成・保存する。
グリッド: (2*num_noise) 行 × Age 列。上半分=Male、下半分=Female。--num_noise でノイズ種類数、--ages で年齢列を指定可能。

実行例（プロジェクトルートから）:
  uv run python src/scripts/check_grids.py
  uv run python src/scripts/check_grids.py --ages 20 40 60 80 --num_noise 3
  uv run python src/scripts/check_grids.py --checkpoint_dir outputs/dit_pixel/checkpoints --no_ema
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

from src.scripts.load_flexible_checkpoint import build_and_load_model
from src.training.train_flexible import sample_flexible_from_noise_cfg
from src.training.train_unet_adaln import sample_with_age_gender_from_noise_cfg
from src.utils.visualize import save_multi_scale_grids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-scale grids from latest checkpoint (train_flexible or train_unet_adaln)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/dit_pixel/checkpoints",
        help="Directory containing model_epoch_*.pt (parent should have config.json for train_flexible)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save grid_scale_*.png (default: same parent as checkpoint_dir / samples)",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 7.5],
        help="Guidance scales to compare",
    )
    parser.add_argument(
        "--ages",
        type=float,
        nargs="+",
        default=[20.0, 40.0, 60.0, 80.0, 100.0],
        help="Age 列（生値）。例: --ages 20 40 60 80 100",
    )
    parser.add_argument(
        "--num_noise",
        type=int,
        default=1,
        help="使用するノイズ画像の種類数。2 以上で (2*num_noise) 行×Age 列（上半分=Male、下半分=Female）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fixed noise (num_noise>1 のとき seed, seed+1, ... を使用)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="DDIM steps",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Prefer plain checkpoint over _ema",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(checkpoint_dir.parent / "samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def sample_fn_factory(guidance_scale: float):
        def sample_fn(age_norm: float, gender_int: int, noise: torch.Tensor) -> torch.Tensor:
            if model_type == "flexible":
                return sample_flexible_from_noise_cfg(
                    model,
                    ddim,
                    noise,
                    age_norm,
                    gender_int,
                    guidance_scale=guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    device=device,
                )
            # adaln_standalone
            return sample_with_age_gender_from_noise_cfg(
                model,
                ddim,
                noise,
                age_norm,
                gender_int,
                guidance_scale=guidance_scale,
                num_inference_steps=args.num_inference_steps,
                device=device,
            )

        return sample_fn

    ages_list = [float(a) for a in args.ages]
    logger.info(
        f"Saving multi-scale grids to {output_dir} (scales={args.scales}, ages={ages_list}, num_noise={args.num_noise})"
    )
    save_multi_scale_grids(
        sample_fn_factory,
        output_dir,
        scales=args.scales,
        ages=ages_list,
        age_scale=116.0,
        seed=args.seed,
        sample_size=64,
        num_inference_steps=args.num_inference_steps,
        num_noise=args.num_noise,
        device=device,
        dtype=dtype,
    )
    logger.info("Done. Check grid_scale_*.png for scale comparison.")


if __name__ == "__main__":
    main()
