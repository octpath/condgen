"""可視化ユーティリティ: 画像グリッド・モーフィングGIFの保存。

PyTorch の (B, C, H, W) テンソルをグリッド画像として保存する。
年齢条件を連続変化させたモーフィングGIFで「同じ顔立ちが年齢だけ変化する」様子を視覚化する。

CFG (Classifier-Free Guidance) 用の uncond 入力:
  Null Label 学習をしたモデルでは、uncond は gender=2 (Null), use_null_age=True で
  「真の条件なし」とする。男性(0)や平均年齢(0.5)は使わない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
from torchvision.utils import make_grid, save_image


# -----------------------------------------------------------------------------
# DDIM ステップ（scale_model_input 必須・CFG 時は入力を2倍に複製）
# -----------------------------------------------------------------------------


def ddim_step_with_scale_and_cfg(
    scheduler,
    x_t: torch.Tensor,
    t: torch.Tensor,
    model_forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    do_classifier_free_guidance: bool,
    guidance_scale: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """1ステップの DDIM 更新。スケジューラに応じた scale_model_input と CFG 時の入力複製を行う。

    CFG なしと CFG scale=1.0 で同じ結果になるよう、CFG 時は入力を [uncond, cond] の2倍にし
    1回の forward で両方の予測を取り、pred_uncond + guidance_scale * (pred_cond - pred_uncond) で合成する。
    Null Label 学習済みモデルでは uncond は gender=2 (Null), use_null_age=True で渡すこと。

    Args:
        scheduler: DDIM 等のスケジューラ（scale_model_input メソッドを持つ）。
        x_t: 現在の latent (B, C, H, W)。
        t: 現在の timestep (スカラーまたは 1要素テンソル)。
        model_forward_fn: (latent_model_input, t_batch) -> noise_pred。
                         do_classifier_free_guidance 時は latent_model_input は (2B,...)、
                         noise_pred は (2B,...) で前半が uncond、後半が cond。
        do_classifier_free_guidance: True なら入力を2倍にし CFG で合成。
        guidance_scale: CFG のスケール（do_classifier_free_guidance 時のみ使用）。
        device: テンソル用デバイス。

    Returns:
        scheduler.step 後の prev_sample (B, C, H, W)。
    """
    latent_model_input = scheduler.scale_model_input(x_t, t)
    if do_classifier_free_guidance:
        latent_model_input = torch.cat([latent_model_input] * 2)
    B = x_t.shape[0]
    t_batch = torch.full(
        (latent_model_input.shape[0],),
        t.item() if hasattr(t, "item") else t,
        dtype=torch.long,
        device=x_t.device,
    )
    noise_pred = model_forward_fn(latent_model_input, t_batch)
    if do_classifier_free_guidance:
        pred_uncond, pred_cond = noise_pred.chunk(2, dim=0)
        noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    return scheduler.step(noise_pred, t, x_t).prev_sample

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def save_image_grid(
    tensor: torch.Tensor,
    path: Union[str, Path],
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[tuple[float, float]] = None,
    pad_value: float = 1.0,
    input_range: Optional[tuple[float, float]] = None,
) -> None:
    """バッチ画像テンソルをグリッド状に並べて1枚の画像として保存する。

    Args:
        tensor: 画像バッチ。(B, C, H, W)、float。
        path: 保存先ファイルパス。
        nrow: グリッドの1行あたりの画像数。
        normalize: True の場合、value_range で正規化してから保存する。
        value_range: 正規化時の (min, max)。None なら tensor の min/max を使用。
        pad_value: パディングの値（0〜1 で指定）。
        input_range: 入力テンソルの範囲 (min, max)。(-1, 1) のとき [0, 1] へ逆変換してから保存。
                     Diffusion 出力等、[-1, 1] 画像は input_range=(-1, 1) を指定すること。

    Raises:
        ValueError: tensor が 4D (B, C, H, W) でない場合。
    """
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got {tensor.dim()}D")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # [-1, 1] -> [0, 1] の逆変換（色の正規化を保証）
    if input_range == (-1.0, 1.0):
        tensor = (tensor / 2.0 + 0.5).clamp(0.0, 1.0)
        normalize = False  # 既に [0, 1] なので normalize 不要

    grid = make_grid(
        tensor,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        pad_value=pad_value,
    )
    save_image(grid, str(path))


def _frames_to_gif(
    frames_tensor: list[torch.Tensor],
    path: Path,
    duration_per_frame: int,
    input_range: Optional[tuple[float, float]],
) -> None:
    """フレームテンソルを [0,1] 変換してGIF保存。"""
    if input_range == (-1.0, 1.0):
        frames_tensor = [(t / 2.0 + 0.5).clamp(0.0, 1.0) for t in frames_tensor]
    pil_frames = []
    for t in frames_tensor:
        t = t.cpu()
        if t.dim() == 3:
            arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
        else:
            arr = (t.permute(0, 1, 2).numpy() * 255).astype("uint8")
            arr = arr[0]
        pil_frames.append(Image.fromarray(arr))
    pil_frames[0].save(
        str(path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_per_frame,
        loop=0,
    )


def create_morphing_gif(
    sample_fn: Callable[..., torch.Tensor],
    path: Union[str, Path],
    age_min: int = 0,
    age_max: int = 100,
    age_step: int = 5,
    duration_per_frame: int = 200,
    input_range: Optional[tuple[float, float]] = (-1.0, 1.0),
    noise_shape: Tuple[int, ...] = (1, 3, 64, 64),
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    seed: int = 42,
    age_scale: float = 100.0,
    guidance_scales: Optional[Iterable[float]] = None,
) -> None:
    """年齢を連続変化させた生成画像をGIFアニメーションとして保存する（高機能版）。

    1. **DDIM**: 推論時は呼び出し側で DDIMScheduler を用い、sample_fn 内で決定論的サンプリングを行うと
       品質が向上する（本関数は sample_fn に依存するため、DDIM 切り替えは sample_fn を渡す側で行う）。

    2. **初期ノイズの完全固定**: ループの外で latents を1つだけ生成し、全年齢（0〜100）で同じノイズを
       使い回す。これにより「同じ顔立ちが年齢だけ変化する」動画になる。

    3. **マルチスケール**: guidance_scales にリスト（例: [4.0, 8.0, 12.0]）を渡すと、各スケールで
       動画を生成・保存する。ファイル名は path の stem に _gs{scale} を付与（例: morphing_gs4.0.gif）。

    Args:
        sample_fn: (age_norm, noise, **kwargs) を受け取り生成画像 (1,C,H,W) を返す関数。
                   kwargs に guidance_scale が渡る場合がある（guidance_scales 指定時）。
        path: 保存先GIFファイルパス（guidance_scales 指定時は stem に _gs{scale} が付く）。
        age_min: 年齢の最小値（生値、例: 20）。
        age_max: 年齢の最大値（生値、例: 80）。
        age_step: 年齢の刻み（例: 5 → 20, 25, ..., 80）。
        duration_per_frame: 1フレームあたりの表示時間 [ms]。
        input_range: 入力テンソルの範囲。(-1, 1) のとき [0, 1] に変換してから保存。
        noise_shape: 固定ノイズの shape。デフォルト (1, 3, 64, 64)。
        device: ノイズを生成するデバイス。None のとき CPU。
        dtype: ノイズの dtype。None のとき torch.float32。
        seed: 固定ノイズ生成用の乱数シード。
        age_scale: 年齢正規化の分母。AGE_MAX=116 なら 116 を指定。
        guidance_scales: ガイダンススケールのリスト。指定時は各スケールで1本ずつGIFを保存する。
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL is required for create_morphing_gif. Install Pillow.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    torch.manual_seed(seed)
    fixed_noise = torch.randn(*noise_shape, device=device, dtype=dtype)

    ages = list(range(age_min, age_max + 1, age_step))
    scales_to_use: list[Optional[float]] = [None] if guidance_scales is None else list(guidance_scales)

    for gs in scales_to_use:
        frames_tensor: list[torch.Tensor] = []
        for age_val in ages:
            age_norm = age_val / age_scale
            if gs is not None:
                img = sample_fn(age_norm, fixed_noise, guidance_scale=gs)
            else:
                img = sample_fn(age_norm, fixed_noise)
            if img.dim() == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            frames_tensor.append(img)

        if gs is not None:
            out_path = path.parent / f"{path.stem}_gs{gs}.gif"
        else:
            out_path = path
        _frames_to_gif(frames_tensor, out_path, duration_per_frame, input_range)


# -----------------------------------------------------------------------------
# 性別×年齢グリッド（guidance_scale 対応）とスケール比較用マルチグリッド
# -----------------------------------------------------------------------------


def create_gender_age_grid_from_fn(
    sample_fn: Callable[[float, int, torch.Tensor], torch.Tensor],
    path: Union[str, Path],
    ages_norm: List[float],
    genders: List[int] = (0, 1),
    fixed_noise: Optional[torch.Tensor] = None,
    seed: int = 42,
    sample_size: int = 64,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    nrow: Optional[int] = None,
) -> None:
    """性別(縦)×年齢(横)のグリッドを生成・保存する。DDIM 用に固定ノイズで決定論的。

    sample_fn(age_norm, gender_int, noise) -> (1, C, H, W)。
    同一 fixed_noise で全セルを生成するため、条件の違いのみが画質に反映される。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    if fixed_noise is None:
        torch.manual_seed(seed)
        fixed_noise = torch.randn(1, 3, sample_size, sample_size, device=device, dtype=dtype)
    nrow = nrow or len(ages_norm)
    rows = []
    for gender_int in genders:
        row_samples = []
        for age_norm in ages_norm:
            out = sample_fn(age_norm, gender_int, fixed_noise)
            if out.dim() == 4 and out.shape[0] == 1:
                out = out  # (1, C, H, W)
            elif out.dim() == 3:
                out = out.unsqueeze(0)
            row_samples.append(out)
        rows.append(torch.cat(row_samples, dim=0))
    grid_tensor = torch.cat(rows, dim=0)
    save_image_grid(grid_tensor, path, nrow=nrow, input_range=(-1.0, 1.0))


def create_multi_noise_gender_age_grid_from_fn(
    sample_fn: Callable[[float, int, torch.Tensor], torch.Tensor],
    path: Union[str, Path],
    ages_norm: List[float],
    num_noise: int = 1,
    seed: int = 42,
    sample_size: int = 64,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    nrow: Optional[int] = None,
) -> None:
    """複数ノイズ × 性別 × 年齢のグリッドを生成・保存する。

    レイアウト: (2 * num_noise) 行 × len(ages_norm) 列。
    上半分 num_noise 行: Male (gender=0)、各行で異なるノイズ（seed, seed+1, ...）。
    下半分 num_noise 行: Female (gender=1)、同様に各行で異なるノイズ。
    sample_fn(age_norm, gender_int, noise) -> (1, C, H, W)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    nrow = nrow or len(ages_norm)
    rows = []
    for gender_int in (0, 1):
        for i in range(num_noise):
            torch.manual_seed(seed + i)
            noise = torch.randn(1, 3, sample_size, sample_size, device=device, dtype=dtype)
            row_samples = []
            for age_norm in ages_norm:
                out = sample_fn(age_norm, gender_int, noise)
                if out.dim() == 4 and out.shape[0] == 1:
                    out = out
                elif out.dim() == 3:
                    out = out.unsqueeze(0)
                row_samples.append(out)
            rows.append(torch.cat(row_samples, dim=0))
    grid_tensor = torch.cat(rows, dim=0)
    save_image_grid(grid_tensor, path, nrow=nrow, input_range=(-1.0, 1.0))


def save_multi_scale_grids(
    sample_fn_factory: Callable[[float], Callable[[float, int, torch.Tensor], torch.Tensor]],
    output_dir: Union[str, Path],
    scales: List[float] = (1.0, 2.0, 4.0, 7.5),
    ages: List[float] = (20, 40, 60, 80, 100),
    age_scale: float = 116.0,
    seed: int = 42,
    sample_size: int = 64,
    num_inference_steps: int = 50,
    num_noise: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """Guidance Scale ごとのグリッドを生成・保存する（スケール比較用）。

    sample_fn_factory(guidance_scale) が sample_fn(age_norm, gender_int, noise) を返す。
    num_noise=1 のときは 2 行(Male/Female)×Age 列。num_noise>1 のときは (2*num_noise) 行×Age 列（上半分=Male、下半分=Female、各行で異なるノイズ）。
    保存先: output_dir/grid_scale_1.5.png ... grid_scale_12.0.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    ages_norm = [a / age_scale for a in ages]
    for gs in scales:
        sample_fn = sample_fn_factory(gs)
        path = output_dir / f"grid_scale_{gs}.png"
        if num_noise <= 1:
            torch.manual_seed(seed)
            fixed_noise = torch.randn(1, 3, sample_size, sample_size, device=device, dtype=dtype)
            create_gender_age_grid_from_fn(
                sample_fn,
                path,
                ages_norm=ages_norm,
                genders=[0, 1],
                fixed_noise=fixed_noise,
                seed=seed,
                sample_size=sample_size,
                device=device,
                dtype=dtype,
                nrow=len(ages),
            )
        else:
            create_multi_noise_gender_age_grid_from_fn(
                sample_fn,
                path,
                ages_norm=ages_norm,
                num_noise=num_noise,
                seed=seed,
                sample_size=sample_size,
                device=device,
                dtype=dtype,
                nrow=len(ages),
            )
