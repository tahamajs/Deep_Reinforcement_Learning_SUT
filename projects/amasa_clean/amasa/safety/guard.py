"""Safety guard combining PID lambda, decision-tree shield, and risk critic."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from projects.amasa_clean.amasa.safety.pid_lagrangian import PIDConfig, PIDLagrangian
from projects.amasa_clean.amasa.safety.shield import SafetyShield
from projects.amasa_clean.amasa.safety.risk_critic import RiskCritic, RiskCriticConfig


@dataclass
class GuardConfig:
    state_dim: int
    action_dim: int
    kp: float = 0.5
    ki: float = 0.05
    kd: float = 0.1
    lambda_max: float = 10.0
    cost_limit: float = 0.0
    risk_threshold: float = 0.65
    shield_enabled: bool = True
    risk_enabled: bool = True
    history_dim: int = 4
    device: str = "cpu"


class SafetyGuard:
    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg
        self.pid = PIDLagrangian(PIDConfig(kp=cfg.kp, ki=cfg.ki, kd=cfg.kd, lambda_max=cfg.lambda_max))
        self.shield = SafetyShield() if cfg.shield_enabled else None
        self.risk = RiskCritic(
            RiskCriticConfig(
                state_dim=cfg.state_dim,
                action_dim=cfg.action_dim,
                history_dim=cfg.history_dim,
                device=cfg.device,
            )
        ) if cfg.risk_enabled else None
        self.cost_history = deque([0.0] * cfg.history_dim, maxlen=cfg.history_dim)

    @property
    def lambda_value(self) -> float:
        return float(self.pid.lmbda)

    def _history_vec(self):
        return np.array(self.cost_history, dtype=np.float32)

    def process_action(self, state: np.ndarray, action: np.ndarray):
        risk_score = 0.0
        blocked = 0
        final_action = action.copy()

        if self.risk is not None:
            risk_score = self.risk.score(state, action, self._history_vec())
            if risk_score > self.cfg.risk_threshold:
                final_action = 0.3 * np.tanh(final_action)
                blocked = 1

        if self.shield is not None and getattr(self.shield, "trained", False):
            shield_action, explanation = self.shield.filter(state, final_action)
            if np.linalg.norm(shield_action - final_action) > 1e-6:
                blocked = 1
            final_action = shield_action

        return final_action, {"risk_score": risk_score, "shield_blocked": blocked}

    def update_cost(self, cost: float):
        self.cost_history.append(float(cost))
        violation = max(0.0, float(cost) - self.cfg.cost_limit)
        return self.pid.update(violation)

    def observe_for_risk(self, state: np.ndarray, action: np.ndarray, cost: float):
        if self.risk is None:
            return None
        labels = np.array([1.0 if cost > self.cfg.cost_limit else 0.0], dtype=np.float32)
        states = state[None, :].astype(np.float32)
        actions = action[None, :].astype(np.float32)
        hist = self._history_vec()[None, :]
        return self.risk.update_batch(states, actions, hist, labels)

    def maybe_fit_shield(self, states, actions, costs, terminals):
        if self.shield is None:
            return
        if not getattr(self.shield, "trained", False):
            self.shield.fit(states, actions, costs, terminals)

    def reset_episode(self):
        self.pid.reset()
        self.cost_history = deque([0.0] * self.cfg.history_dim, maxlen=self.cfg.history_dim)

    def safety_log(self, info: Dict[str, Any], risk_score: float, blocked: int):
        return {
            "lambda": float(self.pid.lmbda),
            "risk_score": float(risk_score),
            "shield_blocked": int(blocked),
            "cost": float(info.get("cost", 0.0)),
            "force": float(info.get("force", 0.0)),
            "corridor_violation": int(info.get("corridor_violation", 0)),
        }
