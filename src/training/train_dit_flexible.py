"""DiT_Flexible (AdaLN-Zero, Age+Gender) の学習スクリプト。

U-Net の知見を活かす: Fourier Scale 5.0、Zero-Init、推論時は DDIM + scale_model_input。
年齢・性別条件付きで学習し、検証時は「男性/女性 × 年齢」グリッドとモーフィングGIFを生成。
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
from src.models.dit_flexible import DiT_Flexible, GENDER_NULL_INDEX
from src.utils.common import seed_everything
from src.utils.visualize import create_morphing_gif, ddim_step_with_scale_and_cfg, save_image_grid

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Logging to console only.")


# -----------------------------------------------------------------------------
# Sampling（DDIM + scale_model_input）
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_dit_flexible_from_noise(
    model: DiT_Flexible,
    scheduler,
    x_t_initial: torch.Tensor,
    age: float,
    gender: int,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """DDIM サンプリング。scale_model_input を適用。forward(sample, timestep, age, gender)。"""
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = x_t_initial.clone()
    B = x_t.shape[0]
    dtype = next(model.parameters()).dtype
    age_t = torch.full((B,), age, device=device, dtype=dtype)
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


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(
    model: DiT_Flexible,
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
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
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
    model: DiT_Flexible,
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
    """性別ごとにファイルを分け、各ファイルは 4行(Seed) × 4列(Age) のグリッドを保存。scale_model_input 適用。"""
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
                s = sample_dit_flexible_from_noise(
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
            return sample_dit_flexible_from_noise(
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
    parser = argparse.ArgumentParser(description="Train DiT_Flexible (AdaLN-Zero, Age+Gender)")
    parser.add_argument("--data_root", type=str, default="data/raw/UTKFace")
    parser.add_argument("--output_dir", type=str, default="outputs/dit_flexible")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label_jitter_std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="dit-flexible")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument(
        "--fourier_scale",
        type=float,
        default=5.0,
        help="GaussianFourierProjection scale (U-Net 知見: 5.0)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extra_epochs", type=int, default=100)
    parser.add_argument("--cfg_drop_rate", type=float, default=0.1,
                        help="CFG: 条件を Null にドロップする確率 (gender=2, use_null_age=True)")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "DiT_Flexible",
        "fourier_scale": args.fourier_scale,
        "patch_size": args.patch_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "label_jitter_std": args.label_jitter_std,
        "cfg_drop_rate": args.cfg_drop_rate,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved config: {output_dir / 'config.json'}")

    start_epoch = 0
    total_epochs = args.epochs
    resume_state: Optional[dict] = None
    if args.resume:
        ckpt_files = list(checkpoints_dir.glob("model_epoch_*.pt"))
        ckpt_files = [f for f in ckpt_files if "_ema.pt" not in f.name]
        if not ckpt_files:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoints_dir}. Remove --resume or run training first."
            )
        def epoch_from_path(p: Path) -> int:
            return int(p.stem.replace("model_epoch_", "").replace(".pt", ""))
        latest = max(ckpt_files, key=epoch_from_path)
        start_epoch = epoch_from_path(latest)
        total_epochs = args.extra_epochs
        resume_state = torch.load(latest, map_location="cpu")
        logger.info(f"Resume: {latest}, training {total_epochs} more epochs (from {start_epoch + 1})")

    accelerator = Accelerator(mixed_precision="fp16")
    logger.info(f"Device: {accelerator.device}, Mixed Precision: fp16")

    if accelerator.is_main_process and not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    dataset = UTKFaceDataset(root=args.data_root, image_size=64, return_gender=True)
    logger.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = DiT_Flexible(
        img_size=64,
        patch_size=args.patch_size,
        in_channels=3,
        hidden_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        time_embed_dim=256,
        age_embed_dim=64,
        gender_embed_dim=64,
        fourier_scale=args.fourier_scale,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    logger.info(f"Fourier scale: {args.fourier_scale}, patch_size: {args.patch_size}")

    ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    if resume_state is not None:
        model.load_state_dict(resume_state, strict=True)
        logger.info("Loaded checkpoint for resume.")

    ema_model = None
    if not args.no_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
            use_ema_warmup=True,
            inv_gamma=1.0,
            power=0.75,
        )
        logger.info(f"EMA enabled with decay={args.ema_decay}")

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    if ema_model is not None:
        ema_model.to(accelerator.device)

    logger.info(
        f"Training DiT_Flexible for {total_epochs} epochs (start_epoch={start_epoch}) "
        f"with batch_size={args.batch_size}, lr={args.lr}, fourier_scale={args.fourier_scale}"
    )

    for epoch in range(start_epoch + 1, start_epoch + total_epochs + 1):
        avg_loss = train(
            model, ema_model, dataloader, optimizer, ddpm_scheduler, accelerator,
            epoch=epoch, label_jitter_std=args.label_jitter_std, cfg_drop_rate=args.cfg_drop_rate,
        )
        if accelerator.is_main_process:
            lr_scheduler.step()
            if not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "train/loss": avg_loss,
                    "epoch": epoch,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                })
            if epoch % 5 == 0 or epoch == start_epoch + total_epochs:
                create_gender_age_grid(
                    accelerator.unwrap_model(model),
                    ema_model,
                    ddpm_scheduler,
                    accelerator,
                    epoch,
                    samples_dir,
                    sample_size=64,
                    num_inference_steps=50,
                )
            if epoch % 10 == 0 or epoch == start_epoch + total_epochs:
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
