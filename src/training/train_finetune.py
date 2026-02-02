"""DDPM-CelebAHQ-256 のファインチューニングスクリプト。

年齢条件を入力チャンネルに結合した AgeConditionedUNet を用い、
UTKFace でファインチューニングする。最初のエポックから CelebA 品質の顔が生成され、
数エポックで年齢条件に応じた変化（シワ、髪の色等）が現れ始める。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# プロジェクトルートを sys.path に追加（スクリプト直接実行時用）
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
from src.models.pretrained_finetune import AgeConditionedUNet
from src.utils.common import seed_everything
from src.utils.visualize import save_image_grid

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Logging to console only.")


# -----------------------------------------------------------------------------
# Sampling（年齢条件付き、CFG なし）
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_with_age(
    model: AgeConditionedUNet,
    scheduler: DDPMScheduler,
    age: float,
    batch_size: int = 1,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    sample_size: int = 256,
) -> torch.Tensor:
    """年齢条件付きで画像サンプリング。ノイズ (B,3,H,W) から開始し、
    モデル内部で age と結合されて (B,4,H,W) として処理される。

    Args:
        model: AgeConditionedUNet。
        scheduler: DDPMScheduler。
        age: 年齢（正規化済み 0〜1）。
        batch_size: 生成枚数。
        num_inference_steps: サンプリングステップ数。
        device: デバイス。
        sample_size: 画像サイズ（256）。

    Returns:
        生成画像 (B, C, H, W)、[-1, 1] 範囲。
    """
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)

    # ランダムノイズ (B, 3, H, W) から開始
    x_t = torch.randn(batch_size, 3, sample_size, sample_size, device=device)
    age_tensor = torch.full((batch_size,), age, device=device, dtype=x_t.dtype)

    for t in tqdm(scheduler.timesteps, desc=f"Sampling (age={age:.2f})", leave=False):
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        out = model(x_t, t_batch, age_tensor, return_dict=True)
        noise_pred = out.sample
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample

    return x_t


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------


def train(
    model: AgeConditionedUNet,
    ema_model: Optional[EMAModel],
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int = 0,
) -> float:
    """1エポックの学習。

    Args:
        model: AgeConditionedUNet。
        ema_model: EMAModel（オプション）。
        dataloader: 学習データローダー。
        optimizer: Optimizer。
        scheduler: DDPMScheduler。
        accelerator: Accelerator。
        epoch: 現在のエポック番号。

    Returns:
        平均損失。
    """
    model.train()
    total_loss = 0.0
    progress = tqdm(
        dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process
    )

    for batch_idx, (images, ages) in enumerate(progress):
        # images: (B, 3, 256, 256) [-1, 1]、ages: (B, 1) 正規化済み
        ages = ages.squeeze(1)  # (B,)

        # ランダムタイムステップでノイズを追加
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device
        ).long()
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        # ノイズ予測（年齢条件は常に使用）
        out = model(noisy_images, timesteps, ages, return_dict=True)
        noise_pred = out.sample

        # MSE Loss
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
# Validation & Visualization
# -----------------------------------------------------------------------------


@torch.no_grad()
def validate_and_visualize(
    model: AgeConditionedUNet,
    ema_model: Optional[EMAModel],
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int,
    output_dir: Path,
    ages_to_sample: list[float] = [20, 40, 60, 80],
    samples_per_age: int = 4,
    sample_size: int = 256,
) -> None:
    """検証: 固定年齢で画像を生成し、保存・ログ。EMA モデルを使用。"""
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

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"samples": wandb.Image(str(grid_path)), "epoch": epoch})

    if ema_model is not None:
        ema_model.restore(model.parameters())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DDPM-CelebAHQ-256 with age conditioning"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw/UTKFace",
        help="UTKFace dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/finetune",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default 16 for 256x256 GPU memory)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (low for finetuning)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="age-conditioned-finetune",
        help="Wandb project name",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate",
    )
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="google/ddpm-celebahq-256",
        help="Pretrained model ID",
    )
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
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Dataset & Dataloader（256x256、batch_size は引数で制御）
    dataset = UTKFaceDataset(root=args.data_root, image_size=256)
    logger.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Model, Scheduler, Optimizer
    model = AgeConditionedUNet(pretrained_model_name_or_path=args.pretrained_model)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear"
    )  # DDPM 標準
    optimizer = AdamW(model.parameters(), lr=args.lr)

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

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if ema_model is not None:
        ema_model.to(accelerator.device)

    logger.info(
        f"Fine-tuning for {args.epochs} epochs with batch_size={args.batch_size}, lr={args.lr}"
    )

    for epoch in range(1, args.epochs + 1):
        avg_loss = train(
            model,
            ema_model,
            dataloader,
            optimizer,
            scheduler,
            accelerator,
            epoch=epoch,
        )
        logger.info(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            if not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"train/loss": avg_loss, "epoch": epoch})

            if epoch % 2 == 0 or epoch == args.epochs:
                validate_and_visualize(
                    accelerator.unwrap_model(model),
                    ema_model,
                    scheduler,
                    accelerator,
                    epoch,
                    samples_dir,
                    sample_size=256,
                )

            if epoch % 5 == 0 or epoch == args.epochs:
                ckpt_path = checkpoints_dir / f"model_epoch_{epoch:04d}.pt"
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    ckpt_path,
                )
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
    logger.info("Fine-tuning complete!")


if __name__ == "__main__":
    main()
