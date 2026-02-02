"""Hugging Face から UTKFace データセットをダウンロードし、画像をフラットに保存する。

nu-delta/utkface は Parquet 形式のため、snapshot_download で Parquet のみ取得し、
datasets で読み込んでから画像を data/raw/UTKFace に展開する。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

# プロジェクトルートを基準に data/raw を決定
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
UTKFACE_SAVE_DIR = DATA_RAW / "UTKFace"

REPO_ID = "nu-delta/utkface"
REPO_TYPE = "dataset"
# README.md / .gitattributes を除外し、データファイル（Parquet）のみ取得
ALLOW_PATTERNS = ["data/*.parquet"]


def download_utkface(
    save_dir: Union[str, Path] = UTKFACE_SAVE_DIR,
    repo_id: str = REPO_ID,
    repo_type: str = REPO_TYPE,
    allow_patterns: list[str] | None = None,
) -> Path:
    """UTKFace を Hugging Face から取得し、画像を save_dir にフラットに保存する。

    1. snapshot_download で Parquet のみダウンロード（README / .gitattributes は取得しない）
    2. datasets で Parquet を読み込み、画像を save_dir/*.jpg として保存する。

    Args:
        save_dir: 画像の保存先（フラットな1フォルダ）。Parquet は save_dir の親に展開される。
        repo_id: Hugging Face のリポジトリ ID。
        repo_type: リポジトリタイプ（"dataset"）。
        allow_patterns: ダウンロードするファイルパターン。None なら data/*.parquet のみ。

    Returns:
        画像が保存されたディレクトリ（save_dir）。
    """
    from huggingface_hub import snapshot_download

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Parquet のみダウンロード（画像のみ取得するため、メタデータファイルは除外）
    patterns = allow_patterns or ALLOW_PATTERNS
    # 保存先: save_dir の親に repo が展開されるので、local_dir = save_dir だと
    # save_dir/data/*.parquet になる。画像は save_dir に書きたいので、
    # local_dir = save_dir でダウンロードし、Parquet は save_dir/data/ に入る。
    cache_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=save_dir,
        allow_patterns=patterns,
    )
    cache_dir = Path(cache_dir)

    # Parquet を読み込み、画像を save_dir に展開
    parquet_dir = cache_dir / "data"
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Parquet が見つかりません: {parquet_dir}")

    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files={"train": [str(p) for p in parquet_files]},
        split="train",
    )

    # 画像列と file_name を取得して保存（フラットに *.jpg）
    for i, row in enumerate(ds):
        img = row.get("image")
        file_name = row.get("file_name")
        if file_name is None:
            file_name = f"image_{i:06d}.jpg"
        elif not str(file_name).lower().endswith((".jpg", ".jpeg", ".png")):
            file_name = f"{file_name}.jpg"
        # パスが含まれていれば basename のみ使用（フラット前提）
        file_name = Path(file_name).name
        out_path = save_dir / file_name

        if img is not None:
            import io
            from PIL import Image as PILImage
            if hasattr(img, "save"):
                img.save(out_path)
            elif isinstance(img, dict) and "bytes" in img:
                PILImage.open(io.BytesIO(img["bytes"])).save(out_path)
            elif isinstance(img, bytes):
                PILImage.open(io.BytesIO(img)).save(out_path)

    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hugging Face から UTKFace をダウンロードし、data/raw/UTKFace に画像を展開"
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=UTKFACE_SAVE_DIR,
        help="画像の保存先 (default: data/raw/UTKFace)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help="Hugging Face の dataset リポジトリ ID",
    )
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_dir = download_utkface(save_dir=args.save_dir, repo_id=args.repo_id)
    print(f"画像を保存しました: {out_dir}")


if __name__ == "__main__":
    main()
