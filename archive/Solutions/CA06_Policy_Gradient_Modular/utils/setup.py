import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import deque, namedtuple
import random
import multiprocessing as mp
import threading
import time
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

print("Environment setup complete!")
print(f"PyTorch version: {torch.__version__}")
print(f"Gymnasium version: {gym.__version__}")
print(f"NumPy version: {np.__version__}")


import importlib
import torch.distributions as _torch_distributions


try:
    _orig_module = importlib.reload(importlib.import_module("torch.distributions"))
    _OrigCategorical = getattr(_orig_module, "Categorical")
except Exception:

    _OrigCategorical = getattr(_torch_distributions, "Categorical", None)


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(logits):
        return logits

    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

    try:
        max_val = logits.max(dim=-1, keepdim=True)[0]
        logits = logits - max_val
    except Exception:
        pass
    return logits


def Categorical(*args, **kwargs):

    if "logits" in kwargs:
        kwargs["logits"] = _sanitize_logits(kwargs["logits"])
        return _OrigCategorical(**kwargs)
    if "probs" in kwargs:
        probs = kwargs["probs"]
        if torch.is_tensor(probs):
            probs = torch.nan_to_num(
                probs, nan=1.0 / probs.size(-1), posinf=1.0, neginf=0.0
            )
            probs = probs.clamp(min=0.0)
            s = probs.sum(dim=-1, keepdim=True)
            s[s == 0] = 1.0
            probs = probs / s
            kwargs["probs"] = probs
        return _OrigCategorical(**kwargs)
    if len(args) == 1 and torch.is_tensor(args[0]):
        logits = _sanitize_logits(args[0])
        return _OrigCategorical(logits=logits)
    return _OrigCategorical(*args, **kwargs)


try:
    _torch_distributions.Categorical = Categorical
    torch.distributions.Categorical = Categorical
    print("[stability wrapper] Monkey-patched torch.distributions.Categorical")
except Exception as e:
    print("[stability wrapper] Warning: could not monkey-patch torch.distributions:", e)


print("\n[stability wrapper] Running quick sanity checks for Categorical wrapper...")
with torch.no_grad():
    t1 = torch.tensor([[1e9, -1e9]], dtype=torch.float32)
    c1 = Categorical(logits=t1)
    print("large logits -> probs:", c1.probs)

    t2 = torch.tensor([[float("nan"), float("nan")]], dtype=torch.float32)
    c2 = Categorical(logits=t2)
    print("nan logits -> probs:", c2.probs)

print("[stability wrapper] Sanity checks complete.\n")
