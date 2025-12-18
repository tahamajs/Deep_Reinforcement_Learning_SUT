from typing import List
import torch


def compute_retrace_targets(rewards: torch.Tensor,
                            states: torch.Tensor,
                            actions: torch.Tensor,
                            next_states: torch.Tensor,
                            next_actions: torch.Tensor,
                            dones: torch.Tensor,
                            q_function: callable,
                            policy,
                            behavior_policy,
                            gamma: float = 0.99,
                            lambda_: float = 0.95) -> torch.Tensor:
    """
    Compute Retrace(lambda) targets for a single trajectory batch.
    Inputs are tensors with time first: shape [T, ...] or [B, T, ...] depending on how user calls.
    This implementation assumes sequences in first dim (T).
    """
    T = rewards.shape[0]
    q_next = q_function(next_states, next_actions)

    pi_probs = policy.action_prob(next_states, next_actions)
    mu_probs = behavior_policy.action_prob(next_states, next_actions)
    rho = pi_probs / (mu_probs + 1e-8)
    c = lambda_ * torch.min(torch.ones_like(rho), rho)

    targets = torch.zeros_like(rewards)
    retrace = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            retrace = rewards[t]
        else:
            retrace = rewards[t] + gamma * q_next[t] + gamma * c[t] * retrace
        targets[t] = retrace
    return targets




