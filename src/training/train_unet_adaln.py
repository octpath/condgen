"""実験 B: U-Net with AdaLN（AdaGroupNorm）複合条件（年齢 + 性別）の学習。

条件は AdaGroupNorm（Zero-Init）で注入。Time は scale_shift のみ。
以前の FiLM 版の重みは互換性がないため、必ずスクラッチから学習すること。
検証時は「男性/女性 × 年齢変化」のグリッド・モーフィング動画を生成。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.training_utils import EMAModel
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.loader import AGE_MAX, UTKFaceDataset
from src.models.unet_adaln_complex import GENDER_NULL_INDEX, UNetAdaLNComplex
from src.utils.common import seed_everything
from src.utils.visualize import create_morphing_gif, ddim_step_with_scale_and_cfg, save_image_grid

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Logging to console only.")


# -----------------------------------------------------------------------------
# Sampling（年齢 + 性別、DDIM）
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_with_age_gender(
    model: UNetAdaLNComplex,
    scheduler,
    age: float,
    gender: int,
    batch_size: int = 1,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    sample_size: int = 64,
) -> torch.Tensor:
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = torch.randn(batch_size, 3, sample_size, sample_size, device=device, dtype=next(model.parameters()).dtype)
    age_t = torch.full((batch_size,), age, device=device, dtype=x_t.dtype)
    gender_t = torch.full((batch_size,), gender, dtype=torch.long, device=device)
    for t in tqdm(scheduler.timesteps, desc=f"Sampling (age={age:.2f}, gender={gender})", leave=False):
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        noise_pred = model(x_t, t_batch, age_t, gender_t)
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t


@torch.no_grad()
def sample_with_age_gender_from_noise(
    model: UNetAdaLNComplex,
    scheduler,
    x_t_initial: torch.Tensor,
    age: float,
    gender: int,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """DDIM サンプリング。scale_model_input を適用し、CFG なしと scale=1.0 で一致する。"""
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = x_t_initial.clone()
    B = x_t.shape[0]
    age_t = torch.full((B,), age, device=device, dtype=x_t.dtype)
    gender_t = torch.full((B,), gender, dtype=torch.long, device=device)

    def model_forward_fn(latent_model_input: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
        return model(latent_model_input, t_batch, age_t, gender_t)

    for t in scheduler.timesteps:
        x_t = ddim_step_with_scale_and_cfg(
            scheduler, x_t, t, model_forward_fn,
            do_classifier_free_guidance=False,
            guidance_scale=1.0,
            device=device,
        )
    return x_t


@torch.no_grad()
def sample_with_age_gender_from_noise_cfg(
    model: UNetAdaLNComplex,
    scheduler,
    x_t_initial: torch.Tensor,
    age: float,
    gender: int,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """DDIM + CFG。Uncond は Null (gender=2, use_null_age=True) で真の条件なし。"""
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = x_t_initial.clone()
    B = x_t.shape[0]
    dtype = next(model.parameters()).dtype
    age_uncond = torch.full((B,), 0.5, device=device, dtype=dtype)
    gender_uncond = torch.full((B,), GENDER_NULL_INDEX, dtype=torch.long, device=device)
    use_null_age_uncond = torch.ones(B, dtype=torch.bool, device=device)
    age_cond = torch.full((B,), age, device=device, dtype=dtype)
    gender_cond = torch.full((B,), gender, dtype=torch.long, device=device)
    use_null_age_cond = torch.zeros(B, dtype=torch.bool, device=device)
    age_2b = torch.cat([age_uncond, age_cond], dim=0)
    gender_2b = torch.cat([gender_uncond, gender_cond], dim=0)
    use_null_age_2b = torch.cat([use_null_age_uncond, use_null_age_cond], dim=0)

    def model_forward_fn(latent_model_input: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
        return model(latent_model_input, t_batch, age_2b, gender_2b, use_null_age=use_null_age_2b)

    for t in scheduler.timesteps:
        x_t = ddim_step_with_scale_and_cfg(
            scheduler, x_t, t, model_forward_fn,
            do_classifier_free_guidance=True,
            guidance_scale=guidance_scale,
            device=device,
        )
    return x_t


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(
    model: UNetAdaLNComplex,
    ema_model: Optional[EMAModel],
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int = 0,
    label_jitter_std: float = 0.05,
    cfg_drop_rate: float = 0.1,
) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
    for images, ages, genders in progress:
        images = images.to(accelerator.device)
        ages = ages.squeeze(1)
        genders = genders.squeeze(1)
        B = images.shape[0]
        if label_jitter_std > 0:
            age_input = ages + torch.randn_like(ages, device=ages.device) * label_jitter_std
            age_input = age_input.clamp(0.0, 1.0)
        else:
            age_input = ages
        use_null_age = torch.rand(B, device=images.device) < cfg_drop_rate
        gender_input = torch.where(use_null_age, torch.full_like(genders, GENDER_NULL_INDEX), genders)
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,), device=images.device
        ).long()
        noisy_images = scheduler.add_noise(images, noise, timesteps)
        noise_pred = model(noisy_images, timesteps, age_input, gender_input, use_null_age=use_null_age)
        loss = F.mse_loss(noise_pred, noise)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if ema_model is not None:
            ema_model.step(model.parameters())
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)


# -----------------------------------------------------------------------------
# Validation: create_gender_age_grid
# -----------------------------------------------------------------------------


def create_gender_age_grid(
    model: UNetAdaLNComplex,
    ema_model: Optional[EMAModel],
    ddpm_scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int,
    output_dir: Path,
    sample_size: int = 64,
    num_inference_steps: int = 50,
    morphing_seed: int = 42,
    ages_for_grid: list[float] = [20, 40, 60, 80],
    seeds_for_grid: list[int] = (42, 43, 44, 45),
    age_min: int = 0,
    age_max: int = 100,
    age_step: int = 5,
) -> None:
    """性別ごとにファイルを分け、各ファイルは 4行(Seed) × 4列(Age) のグリッドを保存。"""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    ddim = DDIMScheduler.from_config(ddpm_scheduler.config)
    device = accelerator.device
    dtype = next(model.parameters()).dtype
    ages_norm = [a / AGE_MAX for a in ages_for_grid]
    nrow = len(ages_for_grid)

    for gender_int, name in [(0, "male"), (1, "female")]:
        rows = []
        for seed in seeds_for_grid:
            torch.manual_seed(seed)
            fixed_noise = torch.randn(1, 3, sample_size, sample_size, device=device, dtype=dtype)
            row_samples = []
            for age_norm in ages_norm:
                s = sample_with_age_gender_from_noise(
                    model, ddim, fixed_noise, age_norm, gender_int,
                    num_inference_steps=num_inference_steps, device=device,
                )
                row_samples.append(s)
            rows.append(torch.cat(row_samples, dim=0))
        grid_tensor = torch.cat(rows, dim=0)
        grid_path = output_dir / f"gender_age_grid_{name}_epoch_{epoch:04d}.png"
        save_image_grid(grid_tensor, grid_path, nrow=nrow, input_range=(-1.0, 1.0))
        logger.info(f"Saved grid ({name}, 4 seeds × 4 ages): {grid_path}")

    if WANDB_AVAILABLE and wandb.run is not None:
        for name in ("male", "female"):
            p = output_dir / f"gender_age_grid_{name}_epoch_{epoch:04d}.png"
            if p.exists():
                wandb.log({f"gender_age_grid_{name}": wandb.Image(str(p)), "epoch": epoch})

    for gender_int, name in [(0, "male"), (1, "female")]:
        def sample_fn(age_norm: float, noise: torch.Tensor) -> torch.Tensor:
            return sample_with_age_gender_from_noise(
                model, ddim, noise, age_norm, gender_int,
                num_inference_steps=num_inference_steps, device=device,
            )
        gif_path = output_dir / f"morphing_{name}_epoch_{epoch:04d}.gif"
        create_morphing_gif(
            sample_fn, gif_path,
            age_min=age_min, age_max=age_max, age_step=age_step,
            duration_per_frame=200, input_range=(-1.0, 1.0),
            noise_shape=(1, 3, sample_size, sample_size),
            device=device,
            seed=morphing_seed,
            age_scale=AGE_MAX,
        )
        logger.info(f"Saved morphing ({name} 0→100歳): {gif_path}")
    if ema_model is not None:
        ema_model.restore(model.parameters())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment B: U-Net with AdaLN/FiLM (age + gender)")
    parser.add_argument("--data_root", type=str, default="data/raw/UTKFace")
    parser.add_argument("--output_dir", type=str, default="outputs/unet_adaln")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label_jitter_std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="unet-adaln")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--cfg_drop_rate", type=float, default=0.1,
                        help="CFG: 条件を Null にドロップする確率 (gender=2, use_null_age=True)")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 実験条件を config.json に保存（どの条件で学習したかチェックポイントと一緒に残す）
    config_path = output_dir / "config.json"
    try:
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved config: {config_path}")
    except Exception as e:
        logger.warning(f"Could not save config.json: {e}")

    accelerator = Accelerator(mixed_precision="fp16")
    logger.info(f"Device: {accelerator.device}, Mixed Precision: fp16")
    if accelerator.is_main_process and not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    dataset = UTKFaceDataset(root=args.data_root, image_size=64, return_gender=True)
    logger.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = UNetAdaLNComplex(
        in_channels=3,
        out_channels=3,
        block_out_channels=(64, 128, 256),
        time_embed_dim=256,
        age_embed_dim=64,
        gender_embed_dim=64,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ema_model = None
    if not args.no_ema:
        ema_model = EMAModel(model.parameters(), decay=args.ema_decay, use_ema_warmup=True, inv_gamma=1.0, power=0.75)
        logger.info(f"EMA enabled decay={args.ema_decay}")

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    if ema_model is not None:
        ema_model.to(accelerator.device)

    logger.info(f"Training UNetAdaLNComplex for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}, weight_decay=1e-2, CosineAnnealingLR")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train(
            model, ema_model, dataloader, optimizer, ddpm_scheduler, accelerator,
            epoch=epoch, label_jitter_std=args.label_jitter_std, cfg_drop_rate=args.cfg_drop_rate,
        )
        lr_scheduler.step()
        logger.info(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
        if accelerator.is_main_process:
            if not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"train/loss": avg_loss, "epoch": epoch})
            if epoch % 5 == 0 or epoch == args.epochs:
                create_gender_age_grid(
                    accelerator.unwrap_model(model), ema_model, ddpm_scheduler, accelerator, epoch, samples_dir,
                    sample_size=64,
                )
            if epoch % 10 == 0 or epoch == args.epochs:
                ckpt_path = checkpoints_dir / f"model_epoch_{epoch:04d}.pt"
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")
                if ema_model is not None:
                    ema_ckpt_path = checkpoints_dir / f"model_epoch_{epoch:04d}_ema.pt"
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    torch.save(accelerator.unwrap_model(model).state_dict(), ema_ckpt_path)
                    ema_model.restore(model.parameters())
                    logger.info(f"Saved EMA checkpoint: {ema_ckpt_path}")

    if accelerator.is_main_process and not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
