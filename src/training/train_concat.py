"""入力チャネル結合 (Input Concatenation) U-Net の学習スクリプト。

Time Embedding に干渉せず、RGB + Age(4ch) で学習。VRM (Label Jittering) で条件空間を滑らかにする。
検証時に 20歳→80歳 のモーフィングGIF を保存。
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
from diffusers.training_utils import EMAModel
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.loader import AGE_MAX, UTKFaceDataset, expand_age_to_spatial
from src.models.unet_concat import create_unet_concat_64
from src.utils.common import seed_everything
from src.utils.visualize import create_morphing_gif, save_image_grid

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Logging to console only.")


# -----------------------------------------------------------------------------
# Sampling（4ch 入力: x_t + age_map）
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_with_age(
    model,
    scheduler: DDPMScheduler,
    age: float,
    batch_size: int = 1,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    sample_size: int = 64,
) -> torch.Tensor:
    """年齢条件付きで画像サンプリング。検証時はノイズなしの正確な age を使用。"""
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = torch.randn(batch_size, 3, sample_size, sample_size, device=device)
    age_map = expand_age_to_spatial(
        torch.full((batch_size,), age, device=device, dtype=x_t.dtype),
        sample_size, sample_size,
    )
    for t in tqdm(scheduler.timesteps, desc=f"Sampling (age={age:.2f})", leave=False):
        inputs = torch.cat([x_t, age_map], dim=1)
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        noise_pred = model(inputs, t_batch).sample
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t


@torch.no_grad()
def sample_with_age_from_noise(
    model,
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
    B = x_t.shape[0]
    H, W = x_t.shape[2], x_t.shape[3]
    age_map = expand_age_to_spatial(
        torch.full((B,), age, device=device, dtype=x_t.dtype), H, W,
    )
    for t in scheduler.timesteps:
        inputs = torch.cat([x_t, age_map], dim=1)
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        noise_pred = model(inputs, t_batch).sample
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(
    model,
    ema_model: Optional[EMAModel],
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int = 0,
    label_jitter_std: float = 0.05,
    sample_size: int = 64,
) -> float:
    """1エポックの学習。VRM: 学習時のみ age にノイズを加える。"""
    model.train()
    total_loss = 0.0
    progress = tqdm(
        dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process
    )
    for images, ages in progress:
        images = images.to(accelerator.device)
        ages = ages.squeeze(1)
        B, _, H, W = images.shape

        # VRM (Label Jittering): 学習時のみ
        if label_jitter_std > 0:
            age_input = ages + torch.randn_like(ages, device=ages.device) * label_jitter_std
            age_input = age_input.clamp(0.0, 1.0)
        else:
            age_input = ages

        age_map = expand_age_to_spatial(age_input, H, W)

        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,), device=images.device
        ).long()
        noisy_images = scheduler.add_noise(images, noise, timesteps)
        inputs = torch.cat([noisy_images, age_map], dim=1)

        noise_pred = model(inputs, timesteps).sample
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
# Validation & Morphing 20→80
# -----------------------------------------------------------------------------


@torch.no_grad()
def validate_and_visualize(
    model,
    ema_model: Optional[EMAModel],
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int,
    output_dir: Path,
    ages_to_sample: list[float] = [20, 40, 60, 80],
    samples_per_age: int = 4,
    sample_size: int = 64,
    morphing_seed: int = 42,
) -> None:
    """検証: ノイズなしの正確な age で生成。20歳→80歳のモーフィングGIFを保存。"""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())

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

    # モーフィングGIF: 20歳→80歳。固定ノイズで「同じ顔が年齢だけ変化」を可視化
    def sample_fn(age_norm: float, noise: torch.Tensor) -> torch.Tensor:
        return sample_with_age_from_noise(
            model, scheduler, noise, age_norm,
            num_inference_steps=50, device=accelerator.device,
        )

    gif_path = output_dir / f"morphing_epoch_{epoch:04d}.gif"
    create_morphing_gif(
        sample_fn, gif_path,
        age_min=20, age_max=80, age_step=5,
        duration_per_frame=200, input_range=(-1.0, 1.0),
        noise_shape=(1, 3, sample_size, sample_size),
        device=accelerator.device,
        seed=morphing_seed,
        age_scale=AGE_MAX,
    )
    logger.info(f"Saved morphing GIF (20→80歳): {gif_path}")

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "samples": wandb.Image(str(grid_path)),
            "morphing_gif": wandb.Video(str(gif_path), format="gif"),
            "epoch": epoch,
        })
    if ema_model is not None:
        ema_model.restore(model.parameters())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Input Concatenation U-Net (RGB+Age 4ch)")
    parser.add_argument("--data_root", type=str, default="data/raw/UTKFace")
    parser.add_argument("--output_dir", type=str, default="outputs/concat")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label_jitter_std", type=float, default=0.05, help="VRM: std of noise added to age (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="concat-unet")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--no_ema", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16")
    logger.info(f"Device: {accelerator.device}, Mixed Precision: fp16")

    if accelerator.is_main_process and not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    dataset = UTKFaceDataset(root=args.data_root, image_size=64)
    logger.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    model = create_unet_concat_64()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    ema_model = None
    if not args.no_ema:
        ema_model = EMAModel(
            model.parameters(), decay=args.ema_decay,
            use_ema_warmup=True, inv_gamma=1.0, power=0.75,
        )
        logger.info(f"EMA enabled decay={args.ema_decay}")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if ema_model is not None:
        ema_model.to(accelerator.device)

    logger.info(f"Training Input Concatenation U-Net for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    logger.info(f"VRM label_jitter_std={args.label_jitter_std}")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train(
            model, ema_model, dataloader, optimizer, scheduler, accelerator,
            epoch=epoch, label_jitter_std=args.label_jitter_std, sample_size=64,
        )
        logger.info(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            if not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"train/loss": avg_loss, "epoch": epoch})
            if epoch % 5 == 0 or epoch == args.epochs:
                validate_and_visualize(
                    accelerator.unwrap_model(model), ema_model, scheduler, accelerator,
                    epoch, samples_dir, sample_size=64,
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
