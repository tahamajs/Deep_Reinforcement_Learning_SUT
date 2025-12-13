import random
import numpy as np
import torch
import warnings


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across multiple libraries.

    Args:
        seed: The integer seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    warnings.filterwarnings("ignore")


def smooth_curve(scores: list, window_size: int = 20) -> np.ndarray:
    """
    Smoothes a list of scores using a rolling average.

    Args:
        scores: A list of numerical scores.
        window_size: The size of the rolling window for smoothing.

    Returns:
        A numpy array of smoothed scores.
    """
    return np.convolve(scores, np.ones(window_size) / window_size, mode="valid")


