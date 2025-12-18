from typing import Iterable, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallDiscreteModel(nn.Module):
    """Small policy+value network for discrete toy MCTS.

    Accepts integer scalar states or batched tensor states of shape (B,).
    Produces:
      - policy(state) -> list of priors over actions
      - value(state) -> scalar value
    """

    def __init__(self, n_actions: int = 3, hidden: int = 64):
        super().__init__()
        # embed scalar state to vector
        self.embed = nn.Sequential(nn.Linear(1, hidden), nn.ReLU())
        self.policy_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor):
        # state shape (B,) or (B,1)
        if state.dim() == 1:
            state = state.unsqueeze(-1).float()
        else:
            state = state.float()
        h = self.embed(state)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value.squeeze(-1)

    # convenience wrappers used by PUCT
    def policy(self, state) -> List[float]:
        """Return priors for a single state (or sequence)."""
        self.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor([state], dtype=torch.float32)
            logits, _ = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            probs = probs[0].cpu().numpy().tolist()
            return probs

    def value(self, state) -> float:
        self.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor([state], dtype=torch.float32)
            _, v = self.forward(state)
            return float(v[0].item())

