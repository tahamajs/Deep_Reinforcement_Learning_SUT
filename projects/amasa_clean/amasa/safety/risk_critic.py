"""Trajectory risk critic for runtime action gating."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskMLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class RiskCriticConfig:
    state_dim: int
    action_dim: int
    history_dim: int = 4
    lr: float = 1e-3
    device: str = "cpu"


class RiskCritic:
    def __init__(self, cfg: RiskCriticConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = RiskMLP(cfg.state_dim + cfg.action_dim + cfg.history_dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def _features(self, state: np.ndarray, action: np.ndarray, history: np.ndarray):
        feat = np.concatenate([state, action, history], axis=-1).astype(np.float32)
        return torch.as_tensor(feat, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def score(self, state: np.ndarray, action: np.ndarray, history: np.ndarray) -> float:
        x = self._features(state, action, history).unsqueeze(0)
        logit = self.model(x)
        return float(torch.sigmoid(logit).item())

    def update_batch(self, states: np.ndarray, actions: np.ndarray, history: np.ndarray, labels: np.ndarray):
        x = np.concatenate([states, actions, history], axis=-1).astype(np.float32)
        y = labels.astype(np.float32).reshape(-1, 1)
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        logit = self.model(x_t)
        loss = F.binary_cross_entropy_with_logits(logit, y_t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def save(self, path: str):
        payload = {"model": self.model.state_dict(), "cfg": self.cfg}
        try:
            torch.save(payload, path)
        except RuntimeError:
            torch.save(payload, path, _use_new_zipfile_serialization=False)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        return ckpt
