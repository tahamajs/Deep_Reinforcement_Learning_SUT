import torch
import torch.nn as nn
import torch.distributions as dist
from src.config import WorldModelConfig, ManagerConfig, WorkerConfig

class Encoder(nn.Module):
    """
    Encoder network to map high-dimensional observations to a latent representation.
    """
    def __init__(self, observation_shape: tuple, latent_dim: int, hidden_dim: int):
        super().__init__()
        # Example for visual observations (3, 64, 64) -> latent_dim
        # You would typically use CNNs here
        if len(observation_shape) == 3: # (C, H, W) for images
            self.cnn = nn.Sequential(
                nn.Conv2d(observation_shape[0], 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten()
            )
            # Calculate flattened size based on input image size
            dummy_input = torch.zeros(1, *observation_shape)
            with torch.no_grad():
                flattened_size = self.cnn(dummy_input).shape[1]
            self.fc = nn.Linear(flattened_size, hidden_dim)
        else: # For vector observations
            self.fc = nn.Sequential(
                nn.Linear(observation_shape[0], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        self.output_fc = nn.Linear(hidden_dim, latent_dim) # Output to latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 5: # (B, T, C, H, W) -> (B*T, C, H, W)
            batch_size, seq_len = obs.shape[0], obs.shape[1]
            obs = obs.view(batch_size * seq_len, *obs.shape[2:])
            features = self.fc(self.cnn(obs)) if hasattr(self, 'cnn') else self.fc(obs)
            return self.output_fc(features).view(batch_size, seq_len, -1)
        else: # (B, C, H, W) or (B, D_o)
            features = self.fc(self.cnn(obs)) if hasattr(self, 'cnn') else self.fc(obs)
            return self.output_fc(features)

class Decoder(nn.Module):
    """
    Decoder network to reconstruct observations from latent representation.
    """
    def __init__(self, observation_shape: tuple, latent_dim: int, hidden_dim: int):
        super().__init__()
        if len(observation_shape) == 3: # (C, H, W)
            self.fc = nn.Linear(latent_dim, hidden_dim)
            self.deconv_input_size = 256 * 2 * 2 # Based on inverse of encoder
            self.deconv_fc = nn.Linear(hidden_dim, self.deconv_input_size)
            self.decnn = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, observation_shape[0], kernel_size=6, stride=2),
            )
        else: # For vector observations
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, observation_shape[0])
            )

    def forward(self, latent: torch.Tensor) -> dist.Distribution:
        if len(latent.shape) == 3: # (B, T, latent_dim) -> (B*T, latent_dim)
            batch_size, seq_len = latent.shape[0], latent.shape[1]
            latent = latent.view(batch_size * seq_len, -1)
            if hasattr(self, 'decnn'):
                features = self.fc(latent)
                features = self.deconv_fc(features)
                features = features.view(-1, 256, 2, 2) # Reshape to (B*T, C, H, W)
                return dist.Independent(dist.Normal(self.decnn(features), 1), 3).base_dist
            else:
                return dist.Independent(dist.Normal(self.fc(latent), 1), 1).base_dist
        else:
            if hasattr(self, 'decnn'):
                features = self.fc(latent)
                features = self.deconv_fc(features)
                features = features.view(-1, 256, 2, 2)
                return dist.Independent(dist.Normal(self.decnn(features), 1), 3).base_dist
            else:
                return dist.Independent(dist.Normal(self.fc(latent), 1), 1).base_dist

class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM) for learning world dynamics.
    Combines a Prior (Dynamics) Model and a Posterior (Representation) Model.
    """
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.action_dim = config.action_dim
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.rssm_type = config.rssm_type

        # Prior (Dynamics) Model: (h_{t-1}, a_{t-1}) -> h_t, s_t
        self.prior_rnn = nn.GRUCell(self.latent_dim + self.action_dim, self.hidden_dim)
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim) # mean and std
        )

        # Posterior (Representation) Model: (h_t, o_t_features) -> s_t
        self.posterior_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim), # hidden_dim from RNN + hidden_dim from observation features
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim) # mean and std
        )

    def forward(self, prev_latent: torch.Tensor, prev_action: torch.Tensor, prev_hidden: torch.Tensor) -> tuple[dist.Distribution, torch.Tensor]:
        """
        Predicts the prior latent state and updates the recurrent hidden state.
        For the full forward pass including posterior, refer to WorldModel class.
        """
        x = torch.cat([prev_latent, prev_action], dim=-1)
        hidden = self.prior_rnn(x, prev_hidden)
        prior_mean_std = self.prior_mlp(hidden)
        prior_mean, prior_std = prior_mean_std.chunk(2, dim=-1)
        prior_dist = dist.Normal(prior_mean, nn.functional.softplus(prior_std) + 1e-5)
        return prior_dist, hidden

    def posterior(self, hidden: torch.Tensor, obs_features: torch.Tensor) -> dist.Distribution:
        """
        Computes the posterior latent state given the recurrent hidden state and observation features.
        """
        x = torch.cat([hidden, obs_features], dim=-1)
        posterior_mean_std = self.posterior_mlp(x)
        posterior_mean, posterior_std = posterior_mean_std.chunk(2, dim=-1)
        posterior_dist = dist.Normal(posterior_mean, nn.functional.softplus(posterior_std) + 1e-5)
        return posterior_dist

class RewardModel(nn.Module):
    """
    Reward Model to predict rewards from latent states.
    """
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent: torch.Tensor) -> dist.Distribution:
        # Reward is typically a scalar. We use a Normal distribution for flexibility
        return dist.Independent(dist.Normal(self.model(latent), 1), 1).base_dist

class WorldModel(nn.Module):
    """
    Integrated World Model combining Encoder, RSSM, RewardModel, and Decoder.
    """
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.observation_shape, config.hidden_dim, config.hidden_dim) # Encoder output to hidden_dim for posterior
        self.rssm = RSSM(config)
        self.reward_model = RewardModel(config.latent_dim, config.hidden_dim)
        self.decoder = Decoder(config.observation_shape, config.latent_dim, config.hidden_dim)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, prev_states: tuple) -> tuple:
        """
        Processes a sequence of observations and actions through the world model.
        Returns priors, posteriors, and reconstructed observations/rewards.

        Args:
            observations (torch.Tensor): (B, T, C, H, W) or (B, T, D_o)
            actions (torch.Tensor): (B, T, D_a)
            prev_states (tuple): (prev_latent, prev_hidden) from last step/episode
        """
        batch_size, seq_len = observations.shape[0], observations.shape[1]
        prev_latent, prev_hidden = prev_states

        priors = []
        posteriors = []
        rec_observations = []
        predicted_rewards = []

        obs_features = self.encoder(observations)

        for t in range(seq_len):
            # Prior step
            prior_dist, prev_hidden = self.rssm(prev_latent, actions[:, t], prev_hidden)
            prior_sample = prior_dist.sample()

            # Posterior step
            posterior_dist = self.rssm.posterior(prev_hidden, obs_features[:, t])
            posterior_sample = posterior_dist.sample()

            # Update latent and hidden for next step
            prev_latent = posterior_sample

            priors.append(prior_dist)
            posteriors.append(posterior_dist)
            rec_observations.append(self.decoder(posterior_sample))
            predicted_rewards.append(self.reward_model(posterior_sample))

        return (torch.stack(priors), torch.stack(posteriors),
                torch.stack(rec_observations), torch.stack(predicted_rewards))

    def get_latent(self, observation: torch.Tensor, prev_latent: torch.Tensor, prev_action: torch.Tensor, prev_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a single observation to its latent posterior and updates recurrent state.
        """
        obs_feature = self.encoder(observation)
        prior_dist, hidden = self.rssm(prev_latent, prev_action, prev_hidden)
        posterior_dist = self.rssm.posterior(hidden, obs_feature)
        latent_sample = posterior_dist.sample()
        return latent_sample, hidden

    def imagine_dynamics(self, start_latent: torch.Tensor, start_hidden: torch.Tensor, actions: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Imagine trajectories using the learned dynamics model.

        Args:
            start_latent (torch.Tensor): Initial latent state.
            start_hidden (torch.Tensor): Initial recurrent hidden state.
            actions (torch.Tensor): Sequence of actions to apply in imagination (H_M or N length).

        Returns:
            tuple: List of imagined latent states, hidden states, and predicted rewards.
        """
        imagined_latents = []
        imagined_hiddens = []
        imagined_rewards = []

        current_latent = start_latent
        current_hidden = start_hidden

        for action in actions:
            prior_dist, current_hidden = self.rssm(current_latent, action.unsqueeze(0), current_hidden)
            current_latent = prior_dist.sample()

            imagined_latents.append(current_latent)
            imagined_hiddens.append(current_hidden)
            imagined_rewards.append(self.reward_model(current_latent))

        return imagined_latents, imagined_hiddens, imagined_rewards


class ManagerActor(nn.Module):
    """
    Manager's Actor network to predict goals.
    """
    def __init__(self, config: ManagerConfig, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.goal_dim) # Output a goal vector
        )

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        # Manager outputs a goal in the latent space
        return self.model(latent_state)

class ManagerCritic(nn.Module):
    """
    Manager's Critic network to estimate the value of states.
    """
    def __init__(self, config: ManagerConfig, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output a single value
        )

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        return self.model(latent_state)

class WorkerActor(nn.Module):
    """
    Worker's Actor network to predict primitive actions.
    """
    def __init__(self, config: WorkerConfig, latent_dim: int, goal_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Output action logits or continuous action parameters
        )
        self.action_dim = action_dim

    def forward(self, latent_state: torch.Tensor, goal: torch.Tensor) -> dist.Distribution:
        x = torch.cat([latent_state, goal], dim=-1)
        # For discrete actions, this would be logits, for continuous, mean/std
        action_output = self.model(x)
        if self.action_dim == 1: # Assuming continuous scalar action for now
            return dist.Normal(action_output, 0.1) # Simple normal distribution
        else: # Assuming discrete actions with logits
            return dist.Categorical(logits=action_output)

class WorkerCritic(nn.Module):
    """
    Worker's Critic network to estimate the value of states conditioned on a goal.
    """
    def __init__(self, config: WorkerConfig, latent_dim: int, goal_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output a single value
        )

    def forward(self, latent_state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent_state, goal], dim=-1)
        return self.model(x)
