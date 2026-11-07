import os
import random
import numpy as np
import torch

def lock_deterministic_environment(seed: int = 42):
    """Cố định toàn bộ hạt giống ngẫu nhiên (Python, NumPy, PyTorch, CUDA) theo quy định thi."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
