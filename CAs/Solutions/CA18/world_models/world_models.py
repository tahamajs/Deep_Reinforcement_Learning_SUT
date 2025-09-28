"""
World Models Module

This module implements world model-based reinforcement learning algorithms,
including recurrent state space models (RSSM), model predictive control (MPC),
and imagination-augmented agents (I2A).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class RSSMCore(nn.Module):
    """
    Recurrent State Space Model (RSSM) Core

    Learns to predict future states using deterministic and stochastic components.
    """

    def __init__(self, state_dim: int = 30, hidden_dim: int = 200,
                 action_dim: int = 2, embed_dim: int = 1024):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self.rnn = nn.GRUCell(embed_dim + state_dim + action_dim, hidden_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * state_dim)  # mean and log_std
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * state_dim)  # mean and log_std
        )

    def initial_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get initial state for a batch"""
        device = next(self.parameters()).device
        return {
            'hidden': torch.zeros(batch_size, self.hidden_dim, device=device),
            'stoch': torch.zeros(batch_size, self.state_dim, device=device)
        }

    def observe(self, embed: torch.Tensor, action: torch.Tensor,
                state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Update state using observation (posterior update)
        """
        hidden = self.rnn(
            torch.cat([state['stoch'], action], dim=1),
            state['hidden']
        )

        posterior_input = torch.cat([hidden, embed], dim=1)
        posterior_params = self.posterior_net(posterior_input)
        posterior_mean, posterior_logstd = posterior_params.chunk(2, dim=1)
        posterior_std = torch.exp(posterior_logstd)

        stoch = posterior_mean + posterior_std * torch.randn_like(posterior_std)

        prior_params = self.prior_net(hidden)
        prior_mean, prior_logstd = prior_params.chunk(2, dim=1)
        prior_std = torch.exp(prior_logstd)

        return {
            'hidden': hidden,
            'stoch': stoch,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std
        }

    def imagine(self, action: torch.Tensor,
                state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict next state using action (prior update)
        """
        hidden = self.rnn(
            torch.cat([state['stoch'], action], dim=1),
            state['hidden']
        )

        prior_params = self.prior_net(hidden)
        prior_mean, prior_logstd = prior_params.chunk(2, dim=1)
        prior_std = torch.exp(prior_logstd)

        stoch = prior_mean + prior_std * torch.randn_like(prior_std)

        return {
            'hidden': hidden,
            'stoch': stoch,
            'prior_mean': prior_mean,
            'prior_std': prior_std
        }


class WorldModel(nn.Module):
    """
    Complete World Model with encoder, RSSM core, and decoders
    """

    def __init__(self, obs_dim: int, action_dim: int, state_dim: int = 30,
                 hidden_dim: int = 200, embed_dim: int = 1024):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

        self.rssm = RSSMCore(state_dim, hidden_dim, action_dim, embed_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim),
        )

        self.reward_model = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.continue_model = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to embedding"""
        return self.encoder(obs)

    def decode(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode state to observation"""
        state_concat = torch.cat([state['stoch'], state['hidden']], dim=1)
        return self.decoder(state_concat)

    def predict_reward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict reward from state"""
        state_concat = torch.cat([state['stoch'], state['hidden']], dim=1)
        return self.reward_model(state_concat)

    def predict_continue(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict episode continuation probability"""
        state_concat = torch.cat([state['stoch'], state['hidden']], dim=1)
        return self.continue_model(state_concat)

    def observe_sequence(self, obs_seq: torch.Tensor,
                        action_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process a sequence of observations and actions
        Returns states, reconstructions, and losses
        """
        batch_size, seq_len = obs_seq.shape[:2]

        state = self.rssm.initial_state(batch_size)

        states = []
        reconstructions = []
        rewards = []
        continues = []
        kl_losses = []

        for t in range(seq_len):
            embed = self.encode(obs_seq[:, t])

            if t == 0:
                action = torch.zeros(batch_size, self.action_dim, device=obs_seq.device)
            else:
                action = action_seq[:, t-1]

            state = self.rssm.observe(embed, action, state)
            states.append(state)

            reconstruction = self.decode(state)
            reward = self.predict_reward(state)
            continue_prob = self.predict_continue(state)

            reconstructions.append(reconstruction)
            rewards.append(reward)
            continues.append(continue_prob)

            if 'posterior_mean' in state:
                kl = self._kl_divergence(
                    state['posterior_mean'], state['posterior_std'],
                    state['prior_mean'], state['prior_std']
                )
                kl_losses.append(kl)

        return {
            'states': states,
            'reconstructions': torch.stack(reconstructions, dim=1),
            'rewards': torch.stack(rewards, dim=1),
            'continues': torch.stack(continues, dim=1),
            'kl_losses': torch.stack(kl_losses, dim=1) if kl_losses else None
        }

    def imagine_sequence(self, initial_state: Dict[str, torch.Tensor],
                        actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Imagine future sequence using world model
        """
        batch_size, seq_len = actions.shape[:2]

        state = {k: v.clone() for k, v in initial_state.items()}

        states = [state]
        rewards = []
        continues = []

        for t in range(seq_len):
            state = self.rssm.imagine(actions[:, t], state)
            states.append(state)

            reward = self.predict_reward(state)
            continue_prob = self.predict_continue(state)

            rewards.append(reward)
            continues.append(continue_prob)

        return {
            'states': states[1:],  # Exclude initial state
            'rewards': torch.stack(rewards, dim=1),
            'continues': torch.stack(continues, dim=1)
        }

    def _kl_divergence(self, mean1: torch.Tensor, std1: torch.Tensor,
                      mean2: torch.Tensor, std2: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between two Gaussian distributions"""
        var1 = std1.pow(2)
        var2 = std2.pow(2)

        kl = (var1 / var2 + (mean2 - mean1).pow(2) / var2 +
              torch.log(std2 / std1) - 1).sum(dim=1, keepdim=True)

        return 0.5 * kl


class MPCPlanner:
    """
    Model Predictive Control planner using Cross Entropy Method
    """

    def __init__(self, world_model: WorldModel, action_dim: int,
                 horizon: int = 12, n_candidates: int = 1000,
                 n_iterations: int = 10, n_elite: int = 100):

        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_iterations = n_iterations
        self.n_elite = n_elite

        self.action_min = -1.0
        self.action_max = 1.0

    def plan(self, initial_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Plan action sequence using CEM
        """
        batch_size = initial_state['hidden'].shape[0]

        mean = torch.zeros(batch_size, self.horizon, self.action_dim, device=initial_state['hidden'].device)
        std = torch.ones(batch_size, self.horizon, self.action_dim, device=initial_state['hidden'].device)

        for iteration in range(self.n_iterations):
            noise = torch.randn(batch_size, self.n_candidates, self.horizon,
                              self.action_dim, device=initial_state['hidden'].device)

            mean_expanded = mean.unsqueeze(1).expand(-1, self.n_candidates, -1, -1)
            std_expanded = std.unsqueeze(1).expand(-1, self.n_candidates, -1, -1)

            actions = mean_expanded + std_expanded * noise
            actions = torch.clamp(actions, self.action_min, self.action_max)

            returns = self._evaluate_sequences(initial_state, actions)

            _, elite_indices = torch.topk(returns, self.n_elite, dim=1)

            for b in range(batch_size):
                elite_actions = actions[b, elite_indices[b]]
                mean[b] = elite_actions.mean(dim=0)
                std[b] = elite_actions.std(dim=0) + 1e-6

        final_noise = torch.randn(batch_size, 1, self.horizon,
                                self.action_dim, device=initial_state['hidden'].device)
        final_actions = mean.unsqueeze(1) + std.unsqueeze(1) * final_noise
        final_actions = torch.clamp(final_actions, self.action_min, self.action_max)

        return final_actions[:, 0, 0]  # First action

    def _evaluate_sequences(self, initial_state: Dict[str, torch.Tensor],
                           actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate action sequences using world model
        """
        batch_size, n_candidates = actions.shape[:2]

        expanded_state = {}
        for key, value in initial_state.items():
            expanded_state[key] = value.unsqueeze(1).expand(
                -1, n_candidates, -1
            ).reshape(batch_size * n_candidates, -1)

        actions_flat = actions.reshape(batch_size * n_candidates, self.horizon, -1)

        with torch.no_grad():
            imagined = self.world_model.imagine_sequence(expanded_state, actions_flat)

        rewards = imagined['rewards']  # [batch*candidates, horizon, 1]
        continues = imagined['continues']  # [batch*candidates, horizon, 1]

        gamma = 0.99
        returns = torch.zeros(batch_size * n_candidates, device=initial_state['hidden'].device)

        for t in range(self.horizon):
            discount = gamma ** t
            continue_discount = torch.prod(continues[:, :t+1], dim=1) if t > 0 else continues[:, 0]
            returns += discount * continue_discount.squeeze() * rewards[:, t].squeeze()

        returns = returns.reshape(batch_size, n_candidates)

        return returns


class ImaginationAugmentedAgent(nn.Module):
    """
    Agent that uses imagination for decision making (I2A style)
    """

    def __init__(self, obs_dim: int, action_dim: int, world_model: WorldModel,
                 planner: MPCPlanner, hidden_dim: int = 256):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.world_model = world_model
        self.planner = planner

        self.model_free_policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.value_function = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.imagination_encoder = nn.Sequential(
            nn.Linear(world_model.state_dim + world_model.hidden_dim + 1, 128), # state + reward
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.combined_policy = nn.Sequential(
            nn.Linear(action_dim + 64, hidden_dim),  # MF action + imagination features
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, obs: torch.Tensor, use_imagination: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining model-free and model-based components
        """
        batch_size = obs.shape[0]

        mf_action = self.model_free_policy(obs)

        value = self.value_function(obs)

        if not use_imagination:
            return {
                'action': mf_action,
                'value': value,
                'imagination_features': None
            }

        with torch.no_grad():
            embed = self.world_model.encode(obs)
            initial_state = self.world_model.rssm.initial_state(batch_size)
            dummy_action = torch.zeros(batch_size, self.action_dim, device=obs.device)
            current_state = self.world_model.rssm.observe(embed, dummy_action, initial_state)

        imagination_features = self._generate_imagination_features(current_state)

        combined_input = torch.cat([mf_action, imagination_features], dim=1)
        final_action = self.combined_policy(combined_input)

        return {
            'action': final_action,
            'value': value,
            'imagination_features': imagination_features,
            'mf_action': mf_action
        }

    def _generate_imagination_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate features from imagined rollouts
        """
        batch_size = state['hidden'].shape[0]
        horizon = 5  # Short imagination horizon

        imagination_actions = torch.randn(batch_size, horizon, self.action_dim, device=state['hidden'].device)
        imagination_actions = torch.clamp(imagination_actions, -1, 1)

        with torch.no_grad():
            imagined = self.world_model.imagine_sequence(state, imagination_actions)

        features = []
        for t in range(horizon):
            state_t = imagined['states'][t]
            reward_t = imagined['rewards'][:, t]

            state_concat = torch.cat([state_t['stoch'], state_t['hidden'], reward_t], dim=1)

            step_features = self.imagination_encoder(state_concat)
            features.append(step_features)

        imagination_features = torch.stack(features, dim=1).mean(dim=1)

        return imagination_features
