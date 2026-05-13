from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Best-effort reproducibility (matches the original eval script behavior)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(int(seed))


def sanitize_accelerate_env(keys: Iterable[str] = ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "WORLD_SIZE")) -> None:
    """Remove empty-string env vars that can confuse Accelerate/torch.distributed."""
    for k in keys:
        if os.environ.get(k) == "":
            del os.environ[k]
