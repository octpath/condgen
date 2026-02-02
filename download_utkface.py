"""UTKFace データセットを data/raw にダウンロード・展開するスクリプト。

Aligned & Cropped Faces (約107MB) を Google Drive から取得する。
フォルダ内ファイル数制限のため、手動で ZIP を置いた場合は展開のみ行う。

公式: https://susanqq.github.io/UTKFace/
Aligned & Cropped: https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# プロジェクトルートを基準に data/raw を決定
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
UTKFACE_FOLDER_NAME = "utkface"
# Google Drive フォルダ ID (Aligned & Cropped Faces)
GDOWN_FOLDER_ID = "0BxYys69jI14kU0I1YUQyY1ZDRUE"


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """ZIP を out_dir に展開する。"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Extracted {zip_path.name} -> {out_dir}")


def download_with_gdown(out_dir: Path) -> bool:
    """gdown で Google Drive フォルダをダウンロードする。成功時 True。"""
    try:
        import gdown
    except ImportError:
        print("gdown がインストールされていません: uv add gdown で追加してください。")
        return False

    url = f"https://drive.google.com/drive/folders/{GDOWN_FOLDER_ID}"
    # フォルダ全体はファイル数制限で失敗する可能性あり。その場合は手動ダウンロード案内を出す。
    try:
        gdown.download_folder(url, output=str(out_dir), quiet=False)
        return True
    except Exception as e:
        print(f"gdown でのダウンロードに失敗しました: {e}")
        print("手動で以下から ZIP をダウンロードし、data/raw/ に utkface.zip として保存してから")
        print("再度このスクリプトを実行（--extract-only）してください。")
        print(f"  {url}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="UTKFace を data/raw にダウンロード・展開")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_RAW / UTKFACE_FOLDER_NAME,
        help="展開先ディレクトリ (default: data/raw/utkface)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="data/raw 内の utkface*.zip を探して展開のみ行う（ダウンロードは行わない）",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="gdown でダウンロードのみ試行し、展開は行わない",
    )
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_dir = args.output_dir.resolve()

    if args.extract_only:
        zips = list(DATA_RAW.glob("utkface*.zip")) + list(DATA_RAW.glob("*.zip"))
        if not zips:
            print(f"data/raw に utkface*.zip が見つかりません。手動で ZIP を配置してください。")
            return
        for zip_path in zips:
            extract_zip(zip_path, out_dir)
        print(f"展開先: {out_dir}")
        return

    if args.download_only:
        success = download_with_gdown(out_dir)
        if success:
            print(f"ダウンロード先: {out_dir}")
        return

    # 既に展開済みかチェック
    if out_dir.exists() and any(out_dir.iterdir()):
        # サブディレクトリまたは jpg が含まれるか
        jpgs = list(out_dir.rglob("*.jpg"))
        if jpgs:
            print(f"既に展開済みです: {out_dir} ({len(jpgs)} images)")
            return

    # data/raw に zip があれば展開を優先
    zips = list(DATA_RAW.glob("utkface*.zip")) + list(DATA_RAW.glob("*.zip"))
    if zips:
        for zip_path in zips:
            extract_zip(zip_path, out_dir)
        print(f"展開先: {out_dir}")
        return

    # なければ gdown でダウンロード試行
    success = download_with_gdown(out_dir)
    if success:
        print(f"ダウンロード先: {out_dir}")


if __name__ == "__main__":
    main()
