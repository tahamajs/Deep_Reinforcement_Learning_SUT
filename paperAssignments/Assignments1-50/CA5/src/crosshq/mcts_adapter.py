from typing import List, Sequence
import torch
import torch.nn.functional as F
from src.crosshq.model import CrossQCritic, GaussianPolicy


class CrossHQMCTSAdapter:
    """Adapter to expose CrossHQ actor+critic to discrete-action PUCT.

    It maps a discrete action set (list of continuous action vectors) to priors using the
    GaussianPolicy log-probabilities and computes value estimates by averaging critic Q
    over sampled actions or evaluating Q for each discrete action.
    """

    def __init__(
        self,
        critic: CrossQCritic,
        actor: GaussianPolicy,
        action_set: Sequence[torch.Tensor],
        device=None,
    ):
        self.critic = critic
        self.actor = actor
        self.action_set = list(action_set)
        self.device = device or torch.device("cpu")

    def policy(self, state) -> List[float]:
        """Return prior probabilities over discrete action_set for a single state."""
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor([state], dtype=torch.float32, device=self.device)
        else:
            state_t = state.unsqueeze(0).float().to(self.device)
        # compute log probs for each candidate action under actor
        logps = []
        with torch.no_grad():
            for a in self.action_set:
                a_t = a.to(self.device).unsqueeze(0).float()
                dist = self.actor.dist(state_t)
                # compute log prob of this specific action under the actor distribution
                lp = dist.log_prob(a_t).sum(-1).item()
                logps.append(lp)
        # softmax over logps
        probs = F.softmax(torch.tensor(logps), dim=0).cpu().numpy().tolist()
        return probs

    def value(self, state) -> float:
        """Estimate value for state by averaging critic Q over actions in action_set."""
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor([state], dtype=torch.float32, device=self.device)
        else:
            state_t = state.unsqueeze(0).float().to(self.device)
        qs = []
        with torch.no_grad():
            for a in self.action_set:
                a_t = a.to(self.device).unsqueeze(0).float()
                x = torch.cat([state_t, a_t], dim=-1)
                q1 = self.critic.q1_forward(x).item()
                qs.append(q1)
        return float(sum(qs) / max(1, len(qs)))













