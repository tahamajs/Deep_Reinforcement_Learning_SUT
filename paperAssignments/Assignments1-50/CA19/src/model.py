from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCriticEnsemble(nn.Module):
    """Actor-Critic with an ensemble of value heads for uncertainty estimation.

    policy: outputs logits over discrete actions
    value_ensemble: returns value estimates for each member (M, B)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        ensemble_size: int = 3,
    ):
        super().__init__()
        self.policy_net = MLP(obs_dim, hidden_dim, action_dim)
        # shared trunk for value ensemble
        self.value_trunk = MLP(obs_dim, hidden_dim, hidden_dim)
        # ensemble heads
        self.ensemble_size = ensemble_size
        self.value_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(ensemble_size)]
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, values)

        logits: (B, action_dim)
        values: (ensemble_size, B)
        """
        logits = self.policy_net(obs)
        trunk = self.value_trunk(obs)
        values = torch.stack(
            [head(trunk).squeeze(-1) for head in self.value_heads], dim=0
        )
        return logits, values

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            actions = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
        logp = F.log_softmax(logits, dim=-1)
        chosen_logp = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        # return actions and mean value across ensemble
        mean_value = values.mean(0)
        return actions, chosen_logp, mean_value






