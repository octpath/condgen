"""ベンチマーク U-Net (Fourier Feature + FiLM) の学習スクリプト。

Fourier Feature による連続値条件付けの効果を検証する。
64x64、Batch 64、50 Epoch。エポック終了ごとにモーフィングGIFを保存。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.loader import AGE_MAX, UTKFaceDataset
from src.models.benchmark_unet import BenchmarkUNet
from src.utils.common import seed_everything
from src.utils.visualize import create_morphing_gif, save_image_grid

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Logging to console only.")


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_with_age(
    model: BenchmarkUNet,
    scheduler: DDPMScheduler,
    age: float,
    batch_size: int = 1,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    sample_size: int = 64,
) -> torch.Tensor:
    """年齢条件付きで画像サンプリング。"""
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = torch.randn(batch_size, 3, sample_size, sample_size, device=device)
    age_tensor = torch.full((batch_size,), age, device=device, dtype=x_t.dtype)

    for t in tqdm(scheduler.timesteps, desc=f"Sampling (age={age:.2f})", leave=False):
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        noise_pred = model(x_t, t_batch, age_tensor)
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample

    return x_t


@torch.no_grad()
def sample_with_age_from_noise(
    model: BenchmarkUNet,
    scheduler: DDPMScheduler,
    x_t_initial: torch.Tensor,
    age: float,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """固定ノイズから年齢条件付きで1枚 denoise。モーフィングGIF用。"""
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = x_t_initial.clone()
    age_tensor = torch.full((x_t.shape[0],), age, device=device, dtype=x_t.dtype)

    for t in scheduler.timesteps:
        t_batch = torch.full((x_t.shape[0],), t, dtype=torch.long, device=device)
        noise_pred = model(x_t, t_batch, age_tensor)
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample

    return x_t


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(
    model: BenchmarkUNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int = 0,
) -> float:
    """1エポックの学習。"""
    model.train()
    total_loss = 0.0
    progress = tqdm(
        dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process
    )

    for images, ages in progress:
        ages = ages.squeeze(1)

        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device
        ).long()
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, ages)
        loss = F.mse_loss(noise_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


# -----------------------------------------------------------------------------
# Validation & Morphing GIF
# -----------------------------------------------------------------------------


@torch.no_grad()
def validate_and_visualize(
    model: BenchmarkUNet,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int,
    output_dir: Path,
    ages_to_sample: list[float] = [20, 40, 60, 80],
    samples_per_age: int = 4,
    sample_size: int = 64,
    morphing_seed: int = 42,
) -> None:
    """検証: 固定年齢で画像生成 + モーフィングGIF (0〜100歳)。"""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    ages_normalized = [a / AGE_MAX for a in ages_to_sample]

    all_samples = []
    for age_norm in ages_normalized:
        samples = sample_with_age(
            model, scheduler, age=age_norm,
            batch_size=samples_per_age, device=accelerator.device, sample_size=sample_size,
        )
        all_samples.append(samples)
    all_samples = torch.cat(all_samples, dim=0)

    grid_path = output_dir / f"epoch_{epoch:04d}.png"
    save_image_grid(all_samples, grid_path, nrow=samples_per_age, input_range=(-1.0, 1.0))
    logger.info(f"Saved grid: {grid_path}")

    # モーフィングGIF: 固定ノイズで年齢 0〜100 を 5 歳刻み（同じ顔が年齢だけ変化）
    def sample_fn(age_norm: float, noise: torch.Tensor) -> torch.Tensor:
        return sample_with_age_from_noise(
            model, scheduler, noise, age_norm,
            num_inference_steps=50, device=accelerator.device,
        )

    gif_path = output_dir / f"morphing_epoch_{epoch:04d}.gif"
    create_morphing_gif(
        sample_fn, gif_path,
        age_min=0, age_max=100, age_step=5,
        duration_per_frame=200, input_range=(-1.0, 1.0),
        noise_shape=(1, 3, sample_size, sample_size),
        device=accelerator.device,
        seed=morphing_seed,
        age_scale=AGE_MAX,
    )
    logger.info(f"Saved morphing GIF: {gif_path}")

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "samples": wandb.Image(str(grid_path)),
            "morphing_gif": wandb.Video(str(gif_path), format="gif"),
            "epoch": epoch,
        })


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Benchmark U-Net (Fourier Feature + FiLM)")
    parser.add_argument("--data_root", type=str, default="data/raw/UTKFace", help="UTKFace root")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark", help="Output dir")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="benchmark-unet-fourier")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16")
    logger.info(f"Device: {accelerator.device}")

    if accelerator.is_main_process and not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    dataset = UTKFaceDataset(root=args.data_root, image_size=64)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    model = BenchmarkUNet(in_channels=3, out_channels=3, channel_list=(64, 128, 256))
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    logger.info(
        f"Training Benchmark U-Net for {args.epochs} epochs, "
        f"batch_size={args.batch_size}, lr={args.lr}"
    )
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train(model, dataloader, optimizer, scheduler, accelerator, epoch=epoch)
        logger.info(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            if not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"train/loss": avg_loss, "epoch": epoch})

            validate_and_visualize(
                accelerator.unwrap_model(model),
                scheduler,
                accelerator,
                epoch,
                samples_dir,
                sample_size=64,
            )

            if epoch % 10 == 0 or epoch == args.epochs:
                ckpt_path = checkpoints_dir / f"model_epoch_{epoch:04d}.pt"
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

    if accelerator.is_main_process and not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
