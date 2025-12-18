import torch
import torch.nn as nn
import torch.distributions as distributions
from typing import Tuple, Dict, Any
from src.config import Config
from src.utils import CausalGraphLearner
import torch.nn.functional as F

def build_mlp(input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, activation: nn.Module) -> nn.Module:
    """Helper function to build a multi-layer perceptron."""
    layers = []
    if num_layers == 0:
        return nn.Linear(input_dim, output_dim)
    
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(activation())
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class RSSMCore(nn.Module):
    """Recurrent State-Space Model core component."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.gru = nn.GRUCell(config.rssm_stochastic_dim + config.action_dim, config.rssm_deterministic_dim)

        self.obs_encoder = build_mlp(config.obs_dim, config.encoder_hidden_dim, config.encoder_hidden_dim, 1, getattr(nn, config.rssm_activation))
        self.obs_decoder = build_mlp(config.rssm_stochastic_dim + config.rssm_deterministic_dim, config.obs_dim, config.decoder_hidden_dim, 1, getattr(nn, config.rssm_activation))

        self.prior_mlp = build_mlp(config.rssm_deterministic_dim, 2 * config.rssm_stochastic_dim, config.rssm_hidden_layers, 1, getattr(nn, config.rssm_activation))
        self.posterior_mlp = build_mlp(config.rssm_deterministic_dim + config.encoder_hidden_dim, 2 * config.rssm_stochastic_dim, config.rssm_hidden_layers, 1, getattr(nn, config.rssm_activation))
        
        self.reward_predictor = build_mlp(config.rssm_stochastic_dim + config.rssm_deterministic_dim, 1, config.reward_predictor_hidden_dim, 1, getattr(nn, config.rssm_activation))
        self.cost_predictor = build_mlp(config.rssm_stochastic_dim + config.rssm_deterministic_dim, 1, config.cost_predictor_hidden_dim, 1, getattr(nn, config.rssm_activation))

    def forward(self, 
                h_prev: torch.Tensor, 
                z_prev: torch.Tensor, 
                action_prev: torch.Tensor, 
                obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, distributions.Normal, distributions.Normal, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs one step of RSSM dynamics and inference."""
        
        # Recurrent model: h_t = f_mu(h_{t-1}, z_{t-1}, a_{t-1})
        # Ensure action_prev has the correct shape [batch_size, action_dim] if it's 1D
        if action_prev.dim() == 1:
            action_prev = action_prev.unsqueeze(-1) # Add a dimension for action if it's a scalar
        recurrent_input = torch.cat([z_prev, action_prev], dim=-1)
        h_t = self.gru(recurrent_input, h_prev)

        # Prior: p_theta(z_t | h_t)
        prior_params = self.prior_mlp(h_t)
        prior_mean, prior_std = torch.split(prior_params, self.config.rssm_stochastic_dim, dim=-1)
        prior_std = F.softplus(prior_std) + 0.1 # Ensure std is positive
        prior_dist = distributions.Normal(prior_mean, prior_std)
        z_t_prior = prior_dist.rsample() # Sample from prior for dynamics

        # Encoder: q_phi(z_t | h_t, o_t)
        embedded_obs = self.obs_encoder(obs)
        posterior_params = self.posterior_mlp(torch.cat([h_t, embedded_obs], dim=-1))
        posterior_mean, posterior_std = torch.split(posterior_params, self.config.rssm_stochastic_dim, dim=-1)
        posterior_std = F.softplus(posterior_std) + 0.1 # Ensure std is positive
        posterior_dist = distributions.Normal(posterior_mean, posterior_std)
        z_t_posterior = posterior_dist.rsample() # Sample from posterior for inference

        # Reconstruction
        state_features = torch.cat([z_t_posterior, h_t], dim=-1)
        recon_obs = self.obs_decoder(state_features)
        recon_reward = self.reward_predictor(state_features)
        recon_cost = self.cost_predictor(state_features)

        return h_t, z_t_posterior, prior_dist, posterior_dist, recon_obs, recon_reward, recon_cost

    def get_latent_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Combines deterministic and stochastic states for downstream tasks."""
        return torch.cat([z, h], dim=-1)


class CausalWorldModel(nn.Module):
    """Extends RSSMCore to include causal graph learning and integration."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.rssm_core = RSSMCore(config).to(config.device)
        self.causal_graph_learner = CausalGraphLearner(config.causal_graph_num_nodes, config.device)
        # Initialize causal graph (adjacency matrix)
        self.causal_graph_adjacency = torch.eye(config.causal_graph_num_nodes, device=config.device)

        # Additional MLP for causal prior (if causal graph influences prior directly)
        # For now, we'll keep the prior simple and assume causal influence is via learned latent features
        # or more complex interactions. If you want direct causal influence on prior, uncomment below:
        # self.causal_prior_mlp = build_mlp(config.rssm_deterministic_dim + config.causal_graph_num_nodes, 2 * config.rssm_stochastic_dim, config.rssm_hidden_layers, 1, getattr(nn, config.rssm_activation))

    def forward(self,
                h_prev: torch.Tensor,
                z_prev: torch.Tensor,
                action_prev: torch.Tensor,
                obs: torch.Tensor,
                update_causal_graph: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, distributions.Normal, distributions.Normal, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        h_t, z_t_posterior, prior_dist, posterior_dist, recon_obs, recon_reward, recon_cost = \
            self.rssm_core(h_prev, z_prev, action_prev, obs)
        
        causal_regularization_loss = torch.tensor(0.0, device=self.config.device) # Initialize to zero

        if update_causal_graph:
            # In a full implementation, you'd extract latent features and learn the graph more rigorously.
            # For now, let's pass a dummy latent_data or integrate causal learning more deeply.
            # For causal learning, we often need a sequence of latent states.
            # Here, we'll just use the current posterior mean as a stand-in for latent_data
            # This is a simplification and would need a proper sequence of latent observations
            # for effective causal discovery.
            dummy_latent_data = posterior_dist.mean # Or collect a batch of z_t_posterior over time
            self.causal_graph_adjacency = self.causal_graph_learner.learn_causal_graph(
                dummy_latent_data.unsqueeze(0), # Add batch dimension
                self.causal_graph_adjacency
            )
            # A simplistic causal regularization: encourage sparsity in the causal graph
            causal_regularization_loss = self.causal_graph_adjacency.sum() * self.config.causal_graph_complexity_penalty

        return h_t, z_t_posterior, prior_dist, posterior_dist, recon_obs, recon_reward, recon_cost, causal_regularization_loss

    def imagine_step(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a step in imagination space, predicting next state, reward, and cost without observation."""
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        recurrent_input = torch.cat([z, action], dim=-1)
        h_next = self.rssm_core.gru(recurrent_input, h)
        
        # Prior prediction for imagined z_next
        prior_params = self.rssm_core.prior_mlp(h_next)
        prior_mean, prior_std = torch.split(prior_params, self.config.rssm_stochastic_dim, dim=-1)
        prior_std = F.softplus(prior_std) + 0.1
        prior_dist = distributions.Normal(prior_mean, prior_std)
        z_next = prior_dist.rsample()

        state_features_next = torch.cat([z_next, h_next], dim=-1)
        imagined_reward = self.rssm_core.reward_predictor(state_features_next)
        imagined_cost = self.rssm_core.cost_predictor(state_features_next)

        return h_next, z_next, imagined_reward, imagined_cost, prior_dist


class PolicyNetwork(nn.Module):
    """Implements the actor-critic policy network, trained on imagined experiences."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        input_dim = config.rssm_stochastic_dim + config.rssm_deterministic_dim
        
        self.actor_mlp = build_mlp(input_dim, config.action_dim * 2, config.actor_hidden_dim, config.actor_layers, nn.ReLU)
        self.critic_mlp = build_mlp(input_dim, 1, config.critic_hidden_dim, config.critic_layers, nn.ReLU)
    
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> Tuple[distributions.Normal, torch.Tensor]:
        latent_features = torch.cat([z, h], dim=-1)
        
        # Actor output
        action_params = self.actor_mlp(latent_features)
        action_mean, action_std = torch.split(action_params, self.config.action_dim, dim=-1)
        action_std = 0.1 + 0.9 * torch.sigmoid(action_std)  # Scale std to be between 0.1 and 1.0
        action_dist = distributions.Normal(action_mean, action_std)
        
        # Critic output
        value = self.critic_mlp(latent_features)
        
        return action_dist, value


class SafetyCritic(nn.Module):
    """Predicts the expected cumulative cost (safety violation) for constrained policy optimization."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        input_dim = config.rssm_stochastic_dim + config.rssm_deterministic_dim
        self.cost_critic_mlp = build_mlp(input_dim, 1, config.critic_hidden_dim, config.critic_layers, nn.ReLU)
    
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        latent_features = torch.cat([z, h], dim=-1)
        predicted_cost = self.cost_critic_mlp(latent_features)
        return predicted_cost
















