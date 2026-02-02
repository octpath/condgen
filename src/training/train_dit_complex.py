"""DiT 複合条件付け（年齢 + 性別）の学習スクリプト。

DiT_ComplexCond で連続値(年齢)とカテゴリ値(性別)を組み合わせた条件付けを検証する。
検証時は DDIM + 固定ノイズで「男性×0→100歳」「女性×0→100歳」のモーフィング動画を生成する。
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
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.training_utils import EMAModel
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.loader import AGE_MAX, UTKFaceDataset
from src.models.dit import DiT_ComplexCond
from src.utils.common import seed_everything
from src.utils.visualize import create_morphing_gif, save_image_grid

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Logging to console only.")


# -----------------------------------------------------------------------------
# Sampling（年齢 + 性別、DDIM 推奨）
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_with_age_gender(
    model: DiT_ComplexCond,
    scheduler,
    age: float,
    gender: int,
    batch_size: int = 1,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    sample_size: int = 64,
) -> torch.Tensor:
    """年齢・性別条件付きで画像サンプリング。"""
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = torch.randn(batch_size, 3, sample_size, sample_size, device=device, dtype=model.patch_embed.proj.weight.dtype)
    age_t = torch.full((batch_size,), age, device=device, dtype=x_t.dtype)
    gender_t = torch.full((batch_size,), gender, dtype=torch.long, device=device)
    for t in tqdm(scheduler.timesteps, desc=f"Sampling (age={age:.2f}, gender={gender})", leave=False):
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        noise_pred = model(x_t, t_batch, age_t, gender_t)
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t


@torch.no_grad()
def sample_with_age_gender_from_noise(
    model: DiT_ComplexCond,
    scheduler,
    x_t_initial: torch.Tensor,
    age: float,
    gender: int,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """固定ノイズから年齢・性別条件付きで1枚 denoise。モーフィングGIF用。"""
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = x_t_initial.clone()
    B = x_t.shape[0]
    age_t = torch.full((B,), age, device=device, dtype=x_t.dtype)
    gender_t = torch.full((B,), gender, dtype=torch.long, device=device)
    for t in scheduler.timesteps:
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        noise_pred = model(x_t, t_batch, age_t, gender_t)
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(
    model: DiT_ComplexCond,
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
    for images, ages, genders in progress:
        images = images.to(accelerator.device)
        ages = ages.squeeze(1)
        genders = genders.squeeze(1)
        B, _, H, W = images.shape

        if label_jitter_std > 0:
            age_input = ages + torch.randn_like(ages, device=ages.device) * label_jitter_std
            age_input = age_input.clamp(0.0, 1.0)
        else:
            age_input = ages

        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,), device=images.device
        ).long()
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, age_input, genders)
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
# Validation: create_gender_age_grid（男性×0→100歳、女性×0→100歳）
# -----------------------------------------------------------------------------


def create_gender_age_grid(
    model: DiT_ComplexCond,
    ema_model: Optional[EMAModel],
    ddpm_scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int,
    output_dir: Path,
    sample_size: int = 64,
    num_inference_steps: int = 50,
    morphing_seed: int = 42,
    ages_for_grid: list[float] = [20, 40, 60, 80],
    age_min: int = 0,
    age_max: int = 100,
    age_step: int = 5,
) -> None:
    """性別×年齢の検証: グリッド画像 + 男性/女性それぞれのモーフィング動画（DDIM・固定ノイズ）。"""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())

    ddpm = ddpm_scheduler
    ddim = DDIMScheduler.from_config(ddpm.config)
    device = accelerator.device

    # 1) グリッド: Male × [20,40,60,80], Female × [20,40,60,80]（2行×4列）
    ages_norm = [a / AGE_MAX for a in ages_for_grid]
    rows = []
    for gender_int, label in [(0, "Male"), (1, "Female")]:
        row_samples = []
        for age_norm in ages_norm:
            s = sample_with_age_gender(
                model, ddim, age=age_norm, gender=gender_int,
                batch_size=1, num_inference_steps=num_inference_steps,
                device=device, sample_size=sample_size,
            )
            row_samples.append(s)
        rows.append(torch.cat(row_samples, dim=0))
    grid_tensor = torch.cat(rows, dim=0)
    grid_path = output_dir / f"gender_age_grid_epoch_{epoch:04d}.png"
    save_image_grid(grid_tensor, grid_path, nrow=len(ages_for_grid), input_range=(-1.0, 1.0))
    logger.info(f"Saved grid: {grid_path}")

    # 2) モーフィング: 男性 × (0歳→100歳)、女性 × (0歳→100歳)（DDIM + 固定ノイズ、create_morphing_gif 内で seed 固定）
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

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "gender_age_grid": wandb.Image(str(grid_path)),
            "epoch": epoch,
        })
    if ema_model is not None:
        ema_model.restore(model.parameters())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DiT with complex conditioning (age + gender)")
    parser.add_argument("--data_root", type=str, default="data/raw/UTKFace")
    parser.add_argument("--output_dir", type=str, default="outputs/dit_complex")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--label_jitter_std", type=float, default=0.05, help="VRM: std of noise added to age (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="dit-complex")
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

    dataset = UTKFaceDataset(root=args.data_root, image_size=64, return_gender=True)
    logger.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    model = DiT_ComplexCond(
        img_size=64,
        patch_size=args.patch_size,
        in_channels=3,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=6,
        mlp_ratio=4.0,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    ema_model = None
    if not args.no_ema:
        ema_model = EMAModel(
            model.parameters(), decay=args.ema_decay,
            use_ema_warmup=True, inv_gamma=1.0, power=0.75,
        )
        logger.info(f"EMA enabled decay={args.ema_decay}")

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if ema_model is not None:
        ema_model.to(accelerator.device)

    logger.info(f"Training DiT_ComplexCond for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
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
                create_gender_age_grid(
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
