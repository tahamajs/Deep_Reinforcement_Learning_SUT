import torch
import numpy as np
import random
import os
from datetime import datetime

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across torch, numpy, and random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed) # This can affect dict iteration order

def get_device() -> torch.device:
    """
    Returns the appropriate device (GPU if available, otherwise CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger:
    """
    A simple logger to track training metrics and save them to a file.
    """
    def __init__(self, log_dir: str = "./logs", experiment_name: str = "dreamer_fun"):
        self.log_dir = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = open(os.path.join(self.log_dir, "log.txt"), "w")
        self.metrics = {}

    def log(self, step: int, metrics: dict):
        """
        Logs a dictionary of metrics at a given step.
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
            self.log_file.write(f"[Step {step}] {key}: {value:.4f}\n")
        self.log_file.flush()

    def get_metric_history(self, key: str) -> list:
        """
        Returns the history of a specific metric.
        """
        return self.metrics.get(key, [])

    def close(self):
        """
        Closes the log file.
        """
        self.log_file.close()

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, log_dir: str, model_name: str):
    """
    Saves the model and optimizer state.
    """
    path = os.path.join(log_dir, f"{model_name}_step_{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str, device: torch.device):
    """
    Loads the model and optimizer state from a checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path} at step {checkpoint['step']}")
    return checkpoint['step']
