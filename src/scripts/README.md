# src/scripts/ — 単独実行用 CLI

このフォルダには、**学習ループに含まれず単独で実行する CLI スクリプト**を置きます。

- **download_data.py** — UTKFace を Hugging Face から取得し `data/raw/UTKFace` に展開
- **check_grids.py** — 学習済みチェックポイントから複数 Guidance Scale のグリッドを生成
- **check_morphing.py** — 学習済みチェックポイントから年齢モーフィング GIF を生成
- **load_flexible_checkpoint.py** — 上記スクリプト用のチェックポイント読み込みヘルパー（直接実行しない）

共通ライブラリ（乱数固定・可視化など）は `src/utils/` にあります。
