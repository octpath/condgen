"""汎用ユーティリティ: 再現性のためのシード固定とデバイス判定。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """再現性のため、Python / NumPy / PyTorch / CUDA の乱数シードを固定する。

    Args:
        seed: 乱数シード。デフォルト 42。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = True


def get_device(prefer_cuda: bool = True) -> torch.device:
    """利用可能なデバイス（cuda / cpu）を返す。

    Args:
        prefer_cuda: True のとき CUDA が利用可能なら "cuda" を返す。

    Returns:
        使用する torch.device。
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
