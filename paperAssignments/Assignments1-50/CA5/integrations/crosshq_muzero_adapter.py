from typing import Any, Dict, Iterable, List, Optional, Sequence
import torch
import torch.nn.functional as F
from src.crosshq.model import CrossQCritic, GaussianPolicy


class CrossHQMuZeroAdapter:
    """Adapter exposing CrossHQ actor+critic to a MuZero-like API.

    If `action_set` is provided, `initial_inference`/`recurrent_inference` will return
    discrete priors over those candidate actions. Otherwise they return the actor's
    parameterization (mu, std) under keys 'policy_mu' and 'policy_std'.
    """

    def __init__(
        self,
        critic: CrossQCritic,
        actor: GaussianPolicy,
        action_set: Optional[Sequence[torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ):
        self.critic = critic
        self.actor = actor
        self.action_set = list(action_set) if action_set is not None else None
        self.device = device or torch.device("cpu")

    def _priors_from_action_set(self, state: Any) -> List[float]:
        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor([state], dtype=torch.float32, device=self.device)
        else:
            state_t = state.unsqueeze(0).float().to(self.device)
        logps = []
        with torch.no_grad():
            dist = self.actor.dist(state_t)
            for a in self.action_set:
                a_t = a.to(self.device).unsqueeze(0).float()
                lp = dist.log_prob(a_t).sum(-1).item()
                logps.append(lp)
        probs = (
            F.softmax(torch.tensor(logps, dtype=torch.float32), dim=0)
            .cpu()
            .numpy()
            .tolist()
        )
        return probs

    def _value_from_action_set(self, state: Any) -> float:
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

    def initial_inference(self, observation: Any) -> Dict[str, Any]:
        if self.action_set is not None:
            priors = self._priors_from_action_set(observation)
            value = self._value_from_action_set(observation)
            hidden = None
            # attempt to get embedding from actor backbone if available
            try:
                with torch.no_grad():
                    if not isinstance(observation, torch.Tensor):
                        s = torch.tensor(
                            [observation], dtype=torch.float32, device=self.device
                        )
                    else:
                        s = observation.unsqueeze(0).float().to(self.device)
                    emb = self.actor.backbone(s)
                    hidden = emb.squeeze(0).cpu()
            except Exception:
                hidden = observation
            return {"policy": priors, "value": value, "hidden_state": hidden}
        else:
            # return parametric actor outputs
            if not isinstance(observation, torch.Tensor):
                s = torch.tensor([observation], dtype=torch.float32, device=self.device)
            else:
                s = observation.unsqueeze(0).float().to(self.device)
            mu, log_std = self.actor.forward(s)
            return {
                "policy_mu": mu.squeeze(0).cpu(),
                "policy_logstd": log_std.squeeze(0).cpu(),
                "value": None,
            }

    def recurrent_inference(self, hidden_state: Any, action: int) -> Dict[str, Any]:
        # For toy environments we interpret hidden_state as integer state
        try:
            next_state = int(hidden_state) + (action - 1)
        except Exception:
            next_state = hidden_state
        return self.initial_inference(next_state)













