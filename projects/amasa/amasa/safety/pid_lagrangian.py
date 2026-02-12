"""PID controller for Lagrange multiplier used in safe RL fine-tuning."""
from dataclasses import dataclass


@dataclass
class PIDConfig:
    kp: float = 0.5
    ki: float = 0.05
    kd: float = 0.1
    lambda_min: float = 0.0
    lambda_max: float = 10.0


class PIDLagrangian:
    def __init__(self, cfg: PIDConfig = PIDConfig()):
        self.cfg = cfg
        self.integral = 0.0
        self.prev_error = 0.0
        self.lmbda = 1.0

    def update(self, cost_violation: float):
        # cost_violation >0 means constraint exceeded
        self.integral += cost_violation
        derivative = cost_violation - self.prev_error
        self.prev_error = cost_violation
        delta = (
            self.cfg.kp * cost_violation
            + self.cfg.ki * self.integral
            + self.cfg.kd * derivative
        )
        self.lmbda = min(self.cfg.lambda_max, max(self.cfg.lambda_min, self.lmbda + delta))
        return self.lmbda

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.lmbda = 1.0
