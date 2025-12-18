from __future__ import annotations
from typing import List, Callable, Any
import torch


def branch_to_action_distribution(
    branches: List[Any], topk_frac: float = 0.5, tau: float = 1.0
):
    """
    Given a list of branches (each with .ret and .traj), return a probability
    distribution over the first actions of the top-k branches.

    Returns: (actions_tensor, probs_tensor)
      - actions_tensor: [k, action_dim]
      - probs_tensor: [k] probabilities summing to 1
    """
    if not branches:
        return None, None
    k = max(1, int(len(branches) * topk_frac))
    selected = branches[:k]
    first_actions = []
    rets = []
    for br in selected:
        if len(br.traj) == 0:
            continue
        a0 = br.traj[0][1]
        if isinstance(a0, torch.Tensor):
            first_actions.append(a0.view(-1).detach())
        else:
            first_actions.append(torch.tensor(a0, dtype=torch.float32))
        rets.append(br.ret)
    if len(first_actions) == 0:
        return None, None
    actions = torch.stack(first_actions)  # [k, action_dim]
    rets_t = torch.tensor(rets, dtype=torch.float32)
    probs = torch.softmax(rets_t / float(tau), dim=0)
    return actions, probs


def select_branch_action(branches: List[Any], topk_frac: float = 0.5, tau: float = 1.0):
    actions, probs = branch_to_action_distribution(
        branches, topk_frac=topk_frac, tau=tau
    )
    if actions is None:
        return None
    idx = torch.multinomial(probs, 1).item()
    return actions[idx]














