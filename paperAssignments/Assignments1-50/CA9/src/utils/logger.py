import csv
from typing import Dict, List
import os

import matplotlib.pyplot as plt


class CSVLogger:
    def __init__(self, path: str, fieldnames: List[str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def log(self, row: Dict):
        self.writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


def plot_series(x, ys: Dict[str, List[float]], out_path: str, title: str = ""):
    plt.figure(figsize=(8, 4))
    for name, y in ys.items():
        plt.plot(x, y, label=name)
    plt.legend()
    plt.title(title)
    plt.xlabel("step")
    plt.grid(True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()











