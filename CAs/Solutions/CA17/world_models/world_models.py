import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent
import numpy as np
from typing import Dict, Tuple, Optional, List
import random

class RSSMCore(nn.Module):
    """Core RSSM architecture for world modeling"""

    def __init__(self, obs_dim: int, action_dim: int, state_dim: int = 32,
                 hidden_dim: int = 256, min_std: float = 0.1):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * state_dim)  # mean and std
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * state_dim)  # mean and std
        )

        self.rnn = nn.GRUCell(hidden_dim + state_dim + action_dim, hidden_dim)

        self.obs_decoder = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

        self.reward_decoder = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.cont_decoder = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def get_initial_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get initial hidden and stochastic states"""
        return {
            'hidden': torch.zeros(1, batch_size, self.hidden_dim),
            'stoch': torch.zeros(batch_size, self.state_dim)
        }

    def prior(self, hidden: torch.Tensor) -> Independent:
        """Compute prior distribution p(z_t | h_t)"""
        stats = self.prior_net(hidden)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        return Independent(Normal(mean, std), 1)

    def posterior(self, hidden: torch.Tensor, obs: torch.Tensor) -> Independent:
        """Compute posterior distribution q(z_t | h_t, o_t)"""
        stats = self.posterior_net(torch.cat([hidden, obs], dim=-1))
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        return Independent(Normal(mean, std), 1)

    def transition(self, prev_state: Dict[str, torch.Tensor],
                   action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute deterministic transition h_t = f(h_{t-1}, z_{t-1}, a_{t-1})"""
        prev_hidden = prev_state['hidden']
        prev_stoch = prev_state['stoch']

        rnn_input = torch.cat([prev_stoch, action], dim=-1)
        rnn_input = rnn_input.unsqueeze(0)  # Add sequence dimension

        hidden, _ = self.rnn(rnn_input, prev_hidden)

        return {
            'hidden': hidden,
            'stoch': prev_stoch  # Will be updated separately
        }

    def observe(self, hidden: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Update stochastic state using observation"""
        posterior_dist = self.posterior(hidden.squeeze(0), obs)
        return posterior_dist.rsample()

    def imagine(self, hidden: torch.Tensor) -> torch.Tensor:
        """Sample stochastic state from prior for imagination"""
        prior_dist = self.prior(hidden.squeeze(0))
        return prior_dist.rsample()

    def decode_obs(self, hidden: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        """Decode observation from state"""
        state_features = torch.cat([hidden.squeeze(0), stoch], dim=-1)
        return self.obs_decoder(state_features)

    def decode_reward(self, hidden: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        """Decode reward from state"""
        state_features = torch.cat([hidden.squeeze(0), stoch], dim=-1)
        return self.reward_decoder(state_features)

    def decode_cont(self, hidden: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        """Decode continuation probability from state"""
        state_features = torch.cat([hidden.squeeze(0), stoch], dim=-1)
        return self.cont_decoder(state_features)


class WorldModel(nn.Module):
    """Complete world model using RSSM"""

    def __init__(self, obs_dim: int, action_dim: int, **kwargs):
        super().__init__()
        self.rssm = RSSMCore(obs_dim, action_dim, **kwargs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor,
                initial_state: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through sequence of observations and actions"""
        batch_size, seq_len = obs_seq.shape[:2]

        if initial_state is None:
            state = self.rssm.get_initial_state(batch_size)
        else:
            state = initial_state

        hidden_seq = []
        stoch_seq = []
        prior_seq = []
        posterior_seq = []
        pred_obs_seq = []
        pred_reward_seq = []
        pred_cont_seq = []

        for t in range(seq_len):
            if t > 0:
                state = self.rssm.transition(state, action_seq[:, t-1])

            hidden = state['hidden']
            stoch = self.rssm.observe(hidden, obs_seq[:, t])

            prior_dist = self.rssm.prior(hidden.squeeze(0))
            posterior_dist = self.rssm.posterior(hidden.squeeze(0), obs_seq[:, t])
            pred_obs = self.rssm.decode_obs(hidden, stoch)
            pred_reward = self.rssm.decode_reward(hidden, stoch)
            pred_cont = self.rssm.decode_cont(hidden, stoch)

            hidden_seq.append(hidden.squeeze(0))
            stoch_seq.append(stoch)
            prior_seq.append(prior_dist)
            posterior_seq.append(posterior_dist)
            pred_obs_seq.append(pred_obs)
            pred_reward_seq.append(pred_reward)
            pred_cont_seq.append(pred_cont)

            state['stoch'] = stoch

        return {
            'hidden': torch.stack(hidden_seq, dim=1),
            'stoch': torch.stack(stoch_seq, dim=1),
            'prior': prior_seq,
            'posterior': posterior_seq,
            'pred_obs': torch.stack(pred_obs_seq, dim=1),
            'pred_reward': torch.stack(pred_reward_seq, dim=1),
            'pred_cont': torch.stack(pred_cont_seq, dim=1)
        }

    def imagine_rollout(self, initial_state: Dict[str, torch.Tensor],
                       actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Imagine rollout using learned dynamics"""
        batch_size, rollout_len = actions.shape[:2]

        hidden_seq = [initial_state['hidden'].squeeze(0)]
        stoch_seq = [initial_state['stoch']]
        pred_obs_seq = []
        pred_reward_seq = []
        pred_cont_seq = []

        state = initial_state.copy()

        for t in range(rollout_len):
            state = self.rssm.transition(state, actions[:, t])
            hidden = state['hidden']

            stoch = self.rssm.imagine(hidden)

            pred_obs = self.rssm.decode_obs(hidden, stoch)
            pred_reward = self.rssm.decode_reward(hidden, stoch)
            pred_cont = self.rssm.decode_cont(hidden, stoch)

            hidden_seq.append(hidden.squeeze(0))
            stoch_seq.append(stoch)
            pred_obs_seq.append(pred_obs)
            pred_reward_seq.append(pred_reward)
            pred_cont_seq.append(pred_cont)

            state['stoch'] = stoch

        return {
            'hidden': torch.stack(hidden_seq[1:], dim=1),  # Exclude initial
            'stoch': torch.stack(stoch_seq[1:], dim=1),
            'pred_obs': torch.stack(pred_obs_seq, dim=1),
            'pred_reward': torch.stack(pred_reward_seq, dim=1),
            'pred_cont': torch.stack(pred_cont_seq, dim=1)
        }


class MPCPlanner:
    """Model Predictive Control using learned world model"""

    def __init__(self, world_model: WorldModel, action_dim: int,
                 horizon: int = 15, num_samples: int = 1000,
                 top_k: int = 100, iterations: int = 10):
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.top_k = top_k
        self.iterations = iterations

    def plan(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Plan using Cross-Entropy Method (CEM)"""
        batch_size = state['hidden'].shape[1] if len(state['hidden'].shape) > 2 else state['hidden'].shape[0]

        mean = torch.zeros(batch_size, self.horizon, self.action_dim)
        std = torch.ones(batch_size, self.horizon, self.action_dim)

        for iteration in range(self.iterations):
            actions = torch.normal(mean.unsqueeze(1).expand(-1, self.num_samples, -1, -1),
                                 std.unsqueeze(1).expand(-1, self.num_samples, -1, -1))
            actions = torch.tanh(actions)  # Bound actions

            values = self._evaluate_sequences(state, actions)

            _, top_indices = torch.topk(values, self.top_k, dim=1)

            top_actions = actions.gather(1, top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon, self.action_dim))
            mean = top_actions.mean(dim=1)
            std = top_actions.std(dim=1) + 1e-4

        best_idx = torch.argmax(values, dim=1)
        best_actions = actions.gather(1, best_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.horizon, self.action_dim))

        return best_actions.squeeze(1)[:, 0]  # First action

    def _evaluate_sequences(self, state: Dict[str, torch.Tensor],
                          actions: torch.Tensor) -> torch.Tensor:
        """Evaluate action sequences using world model"""
        batch_size, num_samples = actions.shape[:2]

        expanded_state = {
            'hidden': state['hidden'].unsqueeze(1).expand(-1, num_samples, -1),
            'stoch': state['stoch'].unsqueeze(1).expand(-1, num_samples, -1)
        }

        flat_state = {
            'hidden': expanded_state['hidden'].reshape(-1, expanded_state['hidden'].shape[-1]).unsqueeze(0),
            'stoch': expanded_state['stoch'].reshape(-1, expanded_state['stoch'].shape[-1])
        }
        flat_actions = actions.reshape(-1, self.horizon, self.action_dim)

        with torch.no_grad():
            rollout = self.world_model.imagine_rollout(flat_state, flat_actions)

            rewards = rollout['pred_reward'].squeeze(-1)  # [batch*samples, horizon]
            continues = rollout['pred_cont'].squeeze(-1)

            discount = torch.cumprod(continues, dim=-1)
            discount = F.pad(discount[:, :-1], (1, 0), value=1.0)

            returns = (rewards * discount).sum(dim=-1)

        returns = returns.reshape(batch_size, num_samples)

        return returns


class ImaginationAugmentedAgent(nn.Module):
    """I2A-style agent combining model-free and model-based paths"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_rollouts: int = 5, rollout_length: int = 10):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_rollouts = num_rollouts
        self.rollout_length = rollout_length

        self.world_model = None

        self.model_free_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.rollout_encoder = nn.Sequential(
            nn.Linear(obs_dim + 1 + 1, hidden_dim // 2),  # obs + reward + continue
            nn.ReLU(),
            nn.LSTM(hidden_dim // 2, hidden_dim // 2, batch_first=True)
        )

        self.imagination_core = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )

        agg_input_dim = hidden_dim + num_rollouts * (hidden_dim // 4)
        self.aggregator = nn.Sequential(
            nn.Linear(agg_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def set_world_model(self, world_model: WorldModel):
        """Set the world model for imagination"""
        self.world_model = world_model

    def forward(self, obs: torch.Tensor, state: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward pass with imagination augmentation"""
        batch_size = obs.shape[0]

        mf_features = self.model_free_net(obs)

        if self.world_model is not None and state is not None:
            imagination_features = self._imagine_trajectories(state, batch_size)
        else:
            imagination_features = torch.zeros(batch_size, self.num_rollouts * (self.model_free_net[0].out_features // 4))

        combined_features = torch.cat([mf_features, imagination_features], dim=-1)
        agg_features = self.aggregator(combined_features)

        action_logits = self.policy_head(agg_features)
        values = self.value_head(agg_features)

        return action_logits, values, {}

    def _imagine_trajectories(self, state: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """Generate and encode imagined trajectories"""
        rollout_features = []

        for _ in range(self.num_rollouts):
            actions = torch.randn(batch_size, self.rollout_length, self.action_dim)
            actions = torch.tanh(actions)  # Bound actions

            with torch.no_grad():
                rollout = self.world_model.imagine_rollout(state, actions)

                obs_seq = rollout['pred_obs']
                reward_seq = rollout['pred_reward']
                cont_seq = rollout['pred_cont']

                rollout_seq = torch.cat([obs_seq, reward_seq, cont_seq], dim=-1)

                encoded, (hidden, _) = self.rollout_encoder(rollout_seq)
                rollout_feature = self.imagination_core(hidden[-1])  # Use final hidden state
                rollout_features.append(rollout_feature)

        return torch.cat(rollout_features, dim=-1)
