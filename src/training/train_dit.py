"""DiT (Diffusion Transformer) の学習スクリプト。

AdaLN-Zero による強力な年齢条件付けを検証する。
64x64 の軽量設計で、連続条件（年齢）の効きを確認する。
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

from src.dataset.loader import AGE_MAX, UTKFaceDataset
from src.models.dit import DiT_Tiny
from src.utils.common import seed_everything
from src.utils.visualize import create_morphing_gif, save_image_grid

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Logging to console only.")


# -----------------------------------------------------------------------------
# Sampling（年齢条件付き）
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_with_age(
    model: DiT_Tiny,
    scheduler: DDPMScheduler,
    age: float,
    batch_size: int = 1,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    sample_size: int = 64,
) -> torch.Tensor:
    """年齢条件付きで画像サンプリング。
    
    Args:
        model: DiT_Tiny。
        scheduler: DDPMScheduler。
        age: 年齢（正規化済み 0〜1）。
        batch_size: 生成枚数。
        num_inference_steps: サンプリングステップ数。
        device: デバイス。
        sample_size: 画像サイズ（64）。
    
    Returns:
        生成画像 (B, C, H, W)、[-1, 1] 範囲。
    """
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
    model: DiT_Tiny,
    scheduler: DDPMScheduler,
    x_t_initial: torch.Tensor,
    age: float,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """固定ノイズから年齢条件付きで1枚 denoise する。モーフィングGIF用。

    Args:
        model: DiT_Tiny。
        scheduler: DDPMScheduler。
        x_t_initial: 初期ノイズ (1, C, H, W)。
        age: 年齢（正規化済み 0〜1）。
        num_inference_steps: サンプリングステップ数。
        device: デバイス。

    Returns:
        生成画像 (1, C, H, W)、[-1, 1] 範囲。
    """
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
# Training Loop
# -----------------------------------------------------------------------------


def train(
    model: DiT_Tiny,
    ema_model: Optional[EMAModel],
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int = 0,
    label_jitter_std: float = 0.05,
) -> float:
    """1エポックの学習。
    
    Args:
        model: DiT_Tiny。
        ema_model: EMAModel（オプション）。
        dataloader: 学習データローダー。
        optimizer: Optimizer。
        scheduler: DDPMScheduler。
        accelerator: Accelerator。
        epoch: 現在のエポック番号。
        label_jitter_std: Vicinal Label Augmentation のノイズ標準偏差（0で無効）。
    
    Returns:
        平均損失。
    """
    model.train()
    total_loss = 0.0
    progress = tqdm(
        dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process
    )

    for batch_idx, (images, ages) in enumerate(progress):
        # images: (B, 3, 64, 64) [-1, 1]、ages: (B, 1) 正規化済み
        ages = ages.squeeze(1)  # (B,)

        # Vicinal Label Augmentation (VRM): 学習時のみ年齢にノイズを加える
        if label_jitter_std > 0:
            age_noisy = ages + torch.randn_like(ages, device=ages.device) * label_jitter_std
            age_noisy = age_noisy.clamp(0.0, 1.0)
        else:
            age_noisy = ages

        # ランダムタイムステップでノイズを追加
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device
        ).long()
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        # ノイズ予測（学習時は age_noisy で近傍分布を学習、Validation 時はノイズなし）
        noise_pred = model(noisy_images, timesteps, age_noisy)

        # MSE Loss
        loss = F.mse_loss(noise_pred, noise)

        accelerator.backward(loss)
        
        # Gradient clipping（DiT は不安定になりやすいため）
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()

        if ema_model is not None:
            ema_model.step(model.parameters())

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        
        # Loss の推移を定期的にログ（不安定性の監視）
        if batch_idx % 50 == 0:
            logger.debug(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    return avg_loss


# -----------------------------------------------------------------------------
# Validation & Visualization
# -----------------------------------------------------------------------------


@torch.no_grad()
def validate_and_visualize(
    model: DiT_Tiny,
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
    """検証: 固定年齢で画像を生成し、保存・ログ。EMA モデルを使用。モーフィングGIFも生成。"""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())

    ages_normalized = [a / AGE_MAX for a in ages_to_sample]

    all_samples = []
    for age_norm in ages_normalized:
        samples = sample_with_age(
            model,
            scheduler,
            age=age_norm,
            batch_size=samples_per_age,
            device=accelerator.device,
            sample_size=sample_size,
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

    if ema_model is not None:
        ema_model.restore(model.parameters())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DiT with AdaLN-Zero")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw/UTKFace",
        help="UTKFace dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dit",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="dit-age-conditioning",
        help="Wandb project name",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA")
    parser.add_argument(
        "--patch_size",
        type=int,
        default=8,
        help="Patch size (4 or 8 for 64x64 images)",
    )
    parser.add_argument(
        "--label_jitter_std",
        type=float,
        default=0.05,
        help="Vicinal label augmentation: std of Gaussian noise added to age (0 to disable)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint and train extra_epochs more",
    )
    parser.add_argument(
        "--extra_epochs",
        type=int,
        default=100,
        help="When --resume: number of additional epochs to train (default 100)",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Resume: 最新チェックポイントを探す
    start_epoch = 0
    total_epochs = args.epochs
    resume_ckpt_path: Optional[Path] = None
    resume_patch_size: Optional[int] = None
    resume_state: Optional[dict] = None
    if args.resume:
        ckpt_files = list(checkpoints_dir.glob("model_epoch_*.pt"))
        ckpt_files = [f for f in ckpt_files if "_ema.pt" not in f.name]
        if not ckpt_files:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoints_dir}. Remove --resume or run training first."
            )
        # エポック番号で最大を取得
        def epoch_from_path(p: Path) -> int:
            stem = p.stem
            return int(stem.replace("model_epoch_", "").replace(".pt", ""))

        latest = max(ckpt_files, key=epoch_from_path)
        start_epoch = epoch_from_path(latest)
        resume_ckpt_path = latest
        total_epochs = args.extra_epochs
        # チェックポイントを1回だけロードし、patch_size を推定
        resume_state = torch.load(latest, map_location="cpu")
        num_patches = resume_state["pos_embed"].shape[1]
        img_size = 64
        resume_patch_size = img_size // int(num_patches**0.5)
        logger.info(
            f"Resume: {resume_ckpt_path}, patch_size inferred={resume_patch_size} "
            f"(num_patches={num_patches}), training {total_epochs} more epochs "
            f"(from {start_epoch + 1} to {start_epoch + total_epochs})"
        )

    accelerator = Accelerator(mixed_precision="fp16")
    logger.info(f"Device: {accelerator.device}, Mixed Precision: fp16")

    if accelerator.is_main_process and not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Dataset & Dataloader（64x64、batch_size は引数で制御）
    dataset = UTKFaceDataset(root=args.data_root, image_size=64)
    logger.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Model, Scheduler, Optimizer（Resume 時はチェックポイントの patch_size を使用）
    patch_size = resume_patch_size if resume_patch_size is not None else args.patch_size
    model = DiT_Tiny(
        img_size=64,
        patch_size=patch_size,
        in_channels=3,
        hidden_dim=384,
        depth=6,
        num_heads=6,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    logger.info(f"Patch size: {patch_size} (64x64 → {64//patch_size}x{64//patch_size} patches)")
    
    ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )

    if resume_state is not None:
        model.load_state_dict(resume_state, strict=True)
        logger.info(f"Loaded model from {resume_ckpt_path}")

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
        f"Training DiT for {total_epochs} epochs (start_epoch={start_epoch}) "
        f"with batch_size={args.batch_size}, lr={args.lr}, T_max={total_epochs} (CosineAnnealingLR)"
    )

    for epoch in range(start_epoch + 1, start_epoch + total_epochs + 1):
        avg_loss = train(
            model,
            ema_model,
            dataloader,
            optimizer,
            ddpm_scheduler,
            accelerator,
            epoch=epoch,
            label_jitter_std=args.label_jitter_std,
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
                validate_and_visualize(
                    accelerator.unwrap_model(model),
                    ema_model,
                    ddpm_scheduler,
                    accelerator,
                    epoch,
                    samples_dir,
                    sample_size=64,
                )

            if epoch % 10 == 0 or epoch == start_epoch + total_epochs:
                ckpt_path = checkpoints_dir / f"model_epoch_{epoch:04d}.pt"
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

                if ema_model is not None:
                    ema_ckpt_path = checkpoints_dir / f"model_epoch_{epoch:04d}_ema.pt"
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        ema_ckpt_path,
                    )
                    ema_model.restore(model.parameters())
                    logger.info(f"Saved EMA checkpoint: {ema_ckpt_path}")

    if (
        accelerator.is_main_process
        and not args.no_wandb
        and WANDB_AVAILABLE
        and wandb.run is not None
    ):
        wandb.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
