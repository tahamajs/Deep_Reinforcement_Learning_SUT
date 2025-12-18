"""Trajectory collection and utility functions."""
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import gym


@dataclass
class Trajectory:
    """Container for trajectory data."""
    states: List[np.ndarray] = None
    actions: List[Any] = None
    rewards: List[float] = None
    log_probs: List[torch.Tensor] = None
    values: List[torch.Tensor] = None
    dones: List[bool] = None

    def __post_init__(self):
        if self.states is None:
            self.states = []
        if self.actions is None:
            self.actions = []
        if self.rewards is None:
            self.rewards = []
        if self.log_probs is None:
            self.log_probs = []
        if self.values is None:
            self.values = []
        if self.dones is None:
            self.dones = []

    def add(self, state: np.ndarray, action: Any, reward: float,
            log_prob: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None, done: bool = False):
        """Add a step to the trajectory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.states)

    def to_tensors(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert trajectory to tensors."""
        return {
            'states': torch.FloatTensor(self.states).to(device),
            'actions': torch.stack(self.actions).to(device) if self.actions and torch.is_tensor(self.actions[0]) else self.actions,
            'rewards': torch.FloatTensor(self.rewards).to(device),
            'log_probs': torch.stack(self.log_probs).to(device) if self.log_probs else None,
            'values': torch.stack(self.values).to(device) if self.values else None,
            'dones': torch.BoolTensor(self.dones).to(device)
        }


def collect_trajectory(env, policy, max_steps: int = 200, render: bool = False) -> Trajectory:
    """Collect one trajectory from environment using policy."""
    trajectory = Trajectory()
    state = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if hasattr(policy, 'get_action'):
            action, log_prob, value = policy.get_action(state_tensor)
        else:
            # For simple policies
            logits = policy(state_tensor)
            try:
                import gym
                if isinstance(env.action_space, gym.spaces.Discrete):
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    value = None
                else:
                    # Continuous action space
                    mean = logits
                    std = torch.ones_like(mean) * 0.1
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum()
                    value = None
            except ImportError:
                # Default to discrete
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = None

        next_state, reward, done, _ = env.step(action.item() if torch.is_tensor(action) else action)

        trajectory.add(state, action, reward, log_prob, value, done)

        state = next_state
        steps += 1

        if render:
            env.render()

    return trajectory


def compute_returns(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted returns."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    return (returns - returns.mean()) / (returns.std() + 1e-8)


def compute_gae_returns(rewards: List[float], values: List[torch.Tensor],
                       dones: List[bool], gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) returns and advantages."""
    returns = []
    advantages = []
    gae = 0
    next_value = 0

    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[step]
            next_value = values[step]
        else:
            next_non_terminal = 1.0 - dones[step]
            next_value = values[step + 1]

        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        gae = delta + gamma * lam * next_non_terminal * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)

    returns = torch.stack(returns)
    advantages = torch.stack(advantages)
    return (returns - returns.mean()) / (returns.std() + 1e-8), (advantages - advantages.mean()) / (advantages.std() + 1e-8)