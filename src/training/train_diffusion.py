"""Diffusion Model (UNetFiLM) の学習スクリプト。

Classifier-Free Guidance (CFG) 対応。年齢条件をランダムにドロップし、
推論時に条件付き・条件なし予測を組み合わせることで、指定年齢への制御性を向上。
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

from src.dataset.loader import UTKFaceDataset
from src.models.unet_film import UNetFiLM
from src.utils.common import seed_everything
from src.utils.visualize import save_image_grid

# Wandb import (optional - wandb が無ければスキップ)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Logging to console only.")


# -----------------------------------------------------------------------------
# CFG Sampling (推論時)
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_with_cfg(
    model: UNetFiLM,
    scheduler: DDPMScheduler,
    age: float,
    guidance_scale: float = 1.5,
    batch_size: int = 1,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """CFG を用いた画像サンプリング。
    
    Args:
        model: 学習済み UNetFiLM。
        scheduler: DDPMScheduler。
        age: 年齢（正規化済み、0〜1）。
        guidance_scale: CFG のガイダンススケール（4.0 推奨）。
        batch_size: 生成枚数。
        num_inference_steps: サンプリングステップ数。
        device: デバイス。
    
    Returns:
        生成画像 (B, C, H, W)、[-1, 1] 範囲。
    """
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    # ランダムノイズから開始
    x_t = torch.randn(batch_size, 3, 64, 64, device=device)
    age_tensor = torch.full((batch_size,), age, device=device)
    
    for t in tqdm(scheduler.timesteps, desc=f"Sampling (age={age:.2f})", leave=False):
        # Time embedding
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        
        # Conditional & Unconditional prediction
        noise_pred_cond = model(x_t, t_batch, age_tensor)
        noise_pred_uncond = model(x_t, t_batch, age=None)
        
        # CFG: noise_pred = uncond + guidance_scale * (cond - uncond)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Denoise step
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    
    return x_t


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------


def train(
    model: UNetFiLM,
    ema_model: Optional[EMAModel],
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    cfg_dropout_prob: float = 0.1,
    epoch: int = 0,
) -> float:
    """1エポックの学習。
    
    Args:
        model: UNetFiLM。
        ema_model: EMAModel（ステップごとに更新）。
        dataloader: 学習データローダー。
        optimizer: Optimizer。
        scheduler: DDPMScheduler。
        accelerator: Accelerator。
        cfg_dropout_prob: CFG Dropout 確率（0.1 = 10%）。
        epoch: 現在のエポック番号。
    
    Returns:
        平均損失。
    """
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
    
    for batch_idx, (images, ages) in enumerate(progress):
        # images: (B, 3, 64, 64) 既に [-1, 1]（Dataset で Normalize 済み）、ages: (B, 1) 正規化済み
        ages = ages.squeeze(1)  # (B,)
        
        # CFG Dropout: 10% の確率で age を None (unconditional)
        if torch.rand(1).item() < cfg_dropout_prob:
            ages_input = None
        else:
            ages_input = ages
        
        # ランダムタイムステップでノイズを追加
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device
        ).long()
        noisy_images = scheduler.add_noise(images, noise, timesteps)
        
        # ノイズ予測
        noise_pred = model(noisy_images, timesteps, ages_input)
        
        # MSE Loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backprop
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # EMA 更新
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
    model: UNetFiLM,
    ema_model: Optional[EMAModel],
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    epoch: int,
    output_dir: Path,
    guidance_scale: float = 1.5,
    ages_to_sample: list[float] = [20, 40, 60, 80],
    samples_per_age: int = 4,
) -> None:
    """検証: 固定年齢で画像を生成し、保存・ログ。EMAモデルを使用。
    
    Args:
        model: UNetFiLM（生のモデル）。
        ema_model: EMAModel（推論時はこちらを使用）。
        scheduler: DDPMScheduler。
        accelerator: Accelerator。
        epoch: 現在のエポック番号。
        output_dir: 出力ディレクトリ。
        guidance_scale: CFG ガイダンススケール（1.5 推奨、比較用に 1.0 も生成）。
        ages_to_sample: サンプリングする年齢リスト（生値、正規化前）。
        samples_per_age: 年齢ごとのサンプル数。
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # EMAモデルの重みをコピー（推論時はEMAを使用）
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    
    # 年齢を正規化 (0-116 -> 0-1)
    from src.dataset.loader import AGE_MAX
    ages_normalized = [age / AGE_MAX for age in ages_to_sample]
    
    # guidance_scale=1.5 での生成
    all_samples_guided = []
    for age_norm in ages_normalized:
        samples = sample_with_cfg(
            model,
            scheduler,
            age=age_norm,
            guidance_scale=guidance_scale,
            batch_size=samples_per_age,
            device=accelerator.device,
        )
        all_samples_guided.append(samples)
    all_samples_guided = torch.cat(all_samples_guided, dim=0)
    
    # guidance_scale=1.0 での生成（Guidanceほぼなし、比較用）
    all_samples_no_guide = []
    for age_norm in ages_normalized:
        samples = sample_with_cfg(
            model,
            scheduler,
            age=age_norm,
            guidance_scale=1.0,
            batch_size=samples_per_age,
            device=accelerator.device,
        )
        all_samples_no_guide.append(samples)
    all_samples_no_guide = torch.cat(all_samples_no_guide, dim=0)
    
    # グリッド保存（input_range=(-1, 1) で正しく [0, 1] に変換してから保存）
    grid_path_guided = output_dir / f"epoch_{epoch:04d}_scale{guidance_scale:.1f}.png"
    save_image_grid(all_samples_guided, grid_path_guided, nrow=samples_per_age, input_range=(-1.0, 1.0))
    logger.info(f"Saved grid (scale={guidance_scale}): {grid_path_guided}")
    
    grid_path_no_guide = output_dir / f"epoch_{epoch:04d}_scale1.0.png"
    save_image_grid(all_samples_no_guide, grid_path_no_guide, nrow=samples_per_age, input_range=(-1.0, 1.0))
    logger.info(f"Saved grid (scale=1.0): {grid_path_no_guide}")
    
    # Wandb log
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            f"samples/scale_{guidance_scale}": wandb.Image(str(grid_path_guided)),
            "samples/scale_1.0": wandb.Image(str(grid_path_no_guide)),
            "epoch": epoch,
        })
    
    # EMAモデルの重みを元に戻す
    if ema_model is not None:
        ema_model.restore(model.parameters())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Diffusion Model with CFG")
    parser.add_argument("--data_root", type=str, default="data/raw/UTKFace", help="UTKFace dataset root")
    parser.add_argument("--output_dir", type=str, default="outputs/diffusion", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--cfg_dropout", type=float, default=0.1, help="CFG dropout probability")
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="CFG guidance scale for sampling (1.5 for stability)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="age-conditioned-diffusion", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA (not recommended)")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    logger.info(f"Device: {device}, Mixed Precision: fp16")
    
    # Wandb init (main process only)
    if accelerator.is_main_process and not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    
    # Dataset & Dataloader
    dataset = UTKFaceDataset(root=args.data_root, image_size=64)
    logger.info(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Model, Scheduler, Optimizer
    model = UNetFiLM(in_channels=3, out_channels=3)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # EMA Model (重み安定化のため推奨)
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
    
    # Accelerate prepare
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # EMA を GPU に移動（accelerator.prepare 後に実行）
    if ema_model is not None:
        ema_model.to(accelerator.device)
    
    logger.info(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        avg_loss = train(
            model,
            ema_model,
            dataloader,
            optimizer,
            scheduler,
            accelerator,
            cfg_dropout_prob=args.cfg_dropout,
            epoch=epoch,
        )
        logger.info(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
        
        if accelerator.is_main_process:
            # Log to wandb
            if not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"train/loss": avg_loss, "epoch": epoch})
            
            # Validation & Visualization
            if epoch % 5 == 0 or epoch == args.epochs:
                validate_and_visualize(
                    accelerator.unwrap_model(model),
                    ema_model,
                    scheduler,
                    accelerator,
                    epoch,
                    samples_dir,
                    guidance_scale=args.guidance_scale,
                )
            
            # Checkpoint
            if epoch % 10 == 0 or epoch == args.epochs:
                # 生のモデルを保存
                ckpt_path = checkpoints_dir / f"model_epoch_{epoch:04d}.pt"
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")
                
                # EMAモデルも保存（推論時はこちらを使用）
                if ema_model is not None:
                    ema_ckpt_path = checkpoints_dir / f"model_epoch_{epoch:04d}_ema.pt"
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    torch.save(accelerator.unwrap_model(model).state_dict(), ema_ckpt_path)
                    ema_model.restore(model.parameters())
                    logger.info(f"Saved EMA checkpoint: {ema_ckpt_path}")
    
    # Finish
    if accelerator.is_main_process and not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
