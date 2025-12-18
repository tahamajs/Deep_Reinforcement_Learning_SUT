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
        """Convert trajectory to tensors.

        Returns a dictionary with tensors for states, actions (if convertible), rewards,
        log_probs, values and dones.
        """
        # States
        states = torch.FloatTensor(self.states).to(device) if len(self.states) > 0 else torch.empty((0,))

        # Actions: try stacking if actions are tensors, otherwise convert to tensor
        actions = None
        if len(self.actions) > 0:
            if torch.is_tensor(self.actions[0]):
                actions = torch.stack(self.actions).to(device)
            else:
                try:
                    actions = torch.tensor(self.actions).to(device)
                except Exception:
                    actions = self.actions  # leave as-is if non-convertible

        rewards = torch.FloatTensor(self.rewards).to(device) if len(self.rewards) > 0 else torch.empty((0,))
        log_probs = torch.stack(self.log_probs).to(device) if self.log_probs else None
        values = torch.stack(self.values).to(device) if self.values else None
        dones = torch.BoolTensor(self.dones).to(device) if len(self.dones) > 0 else torch.BoolTensor([])

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'values': values,
            'dones': dones
        }


def collect_trajectory(env, policy, max_steps: int = 200, render: bool = False) -> Trajectory:
    """Collect one trajectory from environment using policy.

    This function is robust to different Gym API versions: ``env.reset()`` may
    return either ``obs`` or ``(obs, info)``; ``env.step()`` may return either
    ``(obs, reward, done, info)`` or ``(obs, reward, terminated, truncated, info)``.

    The policy is expected to either implement ``get_action(state_tensor) -> (action, log_prob, value)``
    or be a callable that returns logits / action parameters.
    """
    trajectory = Trajectory()
    state = env.reset()
    # Support new Gym that returns (obs, info)
    if isinstance(state, tuple):
        state = state[0]

    done = False
    steps = 0

    while not done and steps < max_steps:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if hasattr(policy, 'get_action'):
            action, log_prob, value = policy.get_action(state_tensor)
        else:
            # For simple policies that output logits or means
            logits = policy(state_tensor)
            # Try to infer action space from env if available
            try:
                import gym
                if hasattr(env, 'action_space') and isinstance(env.action_space, gym.spaces.Discrete):
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    value = None
                else:
                    mean = logits
                    std = torch.ones_like(mean) * 0.1
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum()
                    value = None
            except Exception:
                # Fallback: treat as discrete logits
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = None

        # Step the environment, supporting both old and new Gym step signatures
        step_result = env.step(action.item() if torch.is_tensor(action) else action)
        if len(step_result) == 4:
            next_state, reward, done, info = step_result
        elif len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            # Best-effort unpacking
            next_state, reward, done = step_result[0], step_result[1], bool(step_result[2])

        # Support new reset that returns (obs, info)
        if isinstance(next_state, tuple):
            next_state = next_state[0]

        trajectory.add(state, action, float(reward), log_prob, value, bool(done))

        state = next_state
        steps += 1

        if render and hasattr(env, 'render'):
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