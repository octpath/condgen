"""UTKFace データセットの Dataset / DataLoader 定義。

画像は指定サイズ（デフォルト 256x256）にリサイズし、正規化する。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms

# ファイル名形式: [age]_[gender]_[race]_[date].jpg
# gender: 0=Male, 1=Female
FILENAME_PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+)_\d+\.jpg$", re.IGNORECASE)
# 年齢の最大値（正規化用）。データセットは 0–116。
AGE_MAX = 116.0

__all__ = [
    "UTKFaceDataset",
    "get_utkface_splits",
    "parse_utkface_filename",
    "parse_utkface_age_gender",
    "expand_age_to_spatial",
    "AGE_MAX",
]


def parse_utkface_filename(path: Union[str, Path]) -> Optional[int]:
    """UTKFace のファイル名から年齢を抽出する。パースに失敗したら None。"""
    name = Path(path).name
    m = FILENAME_PATTERN.match(name)
    if m is None:
        return None
    return int(m.group(1))


def parse_utkface_age_gender(path: Union[str, Path]) -> Optional[Tuple[int, int]]:
    """UTKFace のファイル名から年齢と性別を抽出する。パースに失敗したら None。

    Returns:
        (age, gender)。gender は 0=Male, 1=Female。
    """
    name = Path(path).name
    m = FILENAME_PATTERN.match(name)
    if m is None:
        return None
    age = int(m.group(1))
    gender = int(m.group(2))  # 0 or 1
    return (age, gender)


class UTKFaceDataset(Dataset):
    """UTKFace の画像・年齢・性別を返す Dataset。

    画像は指定サイズにリサイズし、テンソル化して [-1, 1] に正規化する。
    年齢は [0, AGE_MAX] を [0, 1] に線形正規化する（連続条件のプロキシとして利用）。
    性別は 0=Male, 1=Female（nn.Embedding 用に long）。
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_size: int = 256,
        transform: Optional[Callable] = None,
        age_max: float = AGE_MAX,
        filter_invalid: bool = True,
        return_gender: bool = False,
    ) -> None:
        """UTKFaceDataset を初期化する。

        Args:
            root: UTKFace 画像が入ったディレクトリ（data/raw/UTKFace 等）。
                  フラット構造を前提とし、直下の *.jpg のみ列挙する。
            image_size: リサイズ後の一辺の長さ。デフォルト 256（学習済みモデル対応）。
            transform: 追加の変換。None のときはリサイズ＋Tensor＋Normalize([-1,1]) を行う。
            age_max: 年齢正規化の最大値。age_normalized = age / age_max。
            filter_invalid: True のとき、ファイル名から年齢を取得できない画像を除外する。
            return_gender: True のとき __getitem__ が (image, age, gender) を返す。False のとき (image, age) のみ（後方互換）。
        """
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform
        self.age_max = age_max
        self.filter_invalid = filter_invalid
        self.return_gender = return_gender

        self.image_paths: list[Path] = []
        self.ages: list[int] = []
        self.genders: list[int] = []

        for path in sorted(self.root.glob("*.jpg")):
            parsed = parse_utkface_age_gender(path)
            if parsed is None and filter_invalid:
                continue
            if parsed is None:
                age, gender = 0, 0
            else:
                age, gender = parsed
            self.image_paths.append(path)
            self.ages.append(age)
            self.genders.append(gender)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """画像 (C, H, W)、正規化年齢 (1,)、および return_gender 時は性別 (1,) long を返す。"""
        path = self.image_paths[index]
        age = self.ages[index]
        gender = self.genders[index]

        image = Image.open(path).convert("RGB")
        t = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        img_tensor = t(image)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        age_normalized = float(age) / self.age_max
        age_tensor = torch.tensor([age_normalized], dtype=torch.float32)
        gender_tensor = torch.tensor([gender], dtype=torch.long)

        if self.return_gender:
            return img_tensor, age_tensor, gender_tensor
        return img_tensor, age_tensor


def expand_age_to_spatial(
    age: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """年齢スカラーを空間マップに拡張する。入力チャネル結合用。

    Args:
        age: (B,) または (B, 1)。正規化済み [0, 1]。
        height: 画像の高さ。
        width: 画像の幅。

    Returns:
        (B, 1, height, width)。同一値が H×W に広がったマップ。
    """
    if age.dim() == 1:
        age = age.unsqueeze(1)
    age = age.unsqueeze(-1).unsqueeze(-1)
    return age.expand(-1, -1, height, width)


def get_utkface_splits(
    root: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    **dataset_kwargs: object,
) -> Tuple[Subset, Subset, Subset]:
    """UTKFace を train / val / test に動的に分割する。

    フォルダ分けではなくリスト操作で分割するため、「特定の年齢層を除外して学習」等の
    実験がしやすい。random_split により再現性のため seed で固定する。

    Args:
        root: UTKFace 画像が入ったディレクトリ（フラット構造）。
        train_ratio: 訓練セットの割合（0〜1）。
        val_ratio: 検証セットの割合（0〜1）。
        test_ratio: テストセットの割合（0〜1）。train_ratio + val_ratio + test_ratio == 1 であること。
        seed: random_split 用の乱数シード。
        **dataset_kwargs: UTKFaceDataset に渡す追加引数（image_size, transform 等）。

    Returns:
        (train_subset, val_subset, test_subset)。各要素は torch.utils.data.Subset。
    """
    full = UTKFaceDataset(root=root, **dataset_kwargs)
    n = len(full)
    if n == 0:
        raise ValueError(f"Dataset is empty (root={root}). 画像が1枚も見つかりません。")
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test < 0:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio が 1 を超えています: "
            f"{train_ratio} + {val_ratio} + {test_ratio}"
        )
    generator = torch.Generator().manual_seed(seed)
    return random_split(full, [n_train, n_val, n_test], generator=generator)
