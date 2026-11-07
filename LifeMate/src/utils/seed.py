"""Global seed locking utility to guarantee deterministic execution across runs."""

import os
import random
from typing import Optional

import numpy as np


def set_deterministic_seeds(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Fix all random seeds for Python, NumPy, and PyTorch (if installed).

    Args:
        seed: Integer random seed.
        deterministic_torch: Whether to enforce deterministic cuDNN/CUDA algorithms.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass


def get_worker_seed(base_seed: int, worker_id: Optional[int] = None) -> int:
    """Compute a deterministic seed for a given worker index."""
    if worker_id is None:
        return base_seed
    return base_seed + int(worker_id) * 1009