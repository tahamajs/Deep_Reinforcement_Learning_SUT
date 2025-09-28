"""
Robust Reinforcement Learning Agents

This module implements domain randomization and adversarial training agents
for robust reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DomainRandomizationAgent:
    """RL agent trained with domain randomization for robustness."""

    def __init__(self, obs_dim, action_dim, lr=3e-4):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.policy_network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        ).to(device)

        self.value_network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        ).to(device)

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        self.gamma = 0.99
        self.clip_ratio = 0.2

        self.policy_losses = []
        self.value_losses = []
        self.environment_diversity = []

    def get_action(self, observation):
        """Get action from policy with exploration."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
            action_probs = self.policy_network(obs_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.value_network(obs_tensor)

        return action.item(), log_prob.item(), value.item()

    def update(self, trajectories):
        """Update agent using PPO with domain randomization data."""
        if not trajectories:
            return None

        all_obs, all_actions, all_rewards, all_log_probs, all_values = (
            [],
            [],
            [],
            [],
            [],
        )
        environment_params = []

        for trajectory in trajectories:
            obs, actions, rewards, log_probs, values, env_params = zip(*trajectory)
            all_obs.extend(obs)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_log_probs.extend(log_probs)
            all_values.extend(values)
            environment_params.extend(env_params)

        observations = torch.FloatTensor(all_obs).to(device)
        actions = torch.LongTensor(all_actions).to(device)
        old_log_probs = torch.FloatTensor(all_log_probs).to(device)

        returns = self.compute_returns(trajectories)
        advantages = returns - torch.FloatTensor(all_values).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):  # Multiple epochs
            action_probs = self.policy_network(observations)
            action_dist = Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                * advantages
            )

            policy_loss = -torch.min(surr1, surr2).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(), max_norm=0.5
            )
            self.policy_optimizer.step()

            new_values = self.value_network(observations).squeeze()
            value_loss = F.mse_loss(new_values, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_network.parameters(), max_norm=0.5
            )
            self.value_optimizer.step()

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())

        unique_sizes = len(
            set([params["environment_size"] for params in environment_params])
        )
        avg_noise = np.mean([params["noise_level"] for params in environment_params])
        self.environment_diversity.append(
            {"unique_sizes": unique_sizes, "avg_noise": avg_noise}
        )

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "environment_diversity": unique_sizes,
        }

    def compute_returns(self, trajectories):
        """Compute returns for all trajectories."""
        all_returns = []

        for trajectory in trajectories:
            rewards = [step[2] for step in trajectory]
            returns = []
            G = 0

            for reward in reversed(rewards):
                G = reward + self.gamma * G
                returns.insert(0, G)

            all_returns.extend(returns)

        return torch.FloatTensor(all_returns).to(device)


class AdversarialRobustAgent:
    """RL agent trained with adversarial perturbations."""

    def __init__(self, obs_dim, action_dim, lr=3e-4, adversarial_strength=0.1):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.adversarial_strength = adversarial_strength

        self.policy_network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        ).to(device)

        self.value_network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        self.gamma = 0.99

        self.robust_losses = []
        self.adversarial_losses = []
        self.perturbation_norms = []

    def generate_adversarial_observation(self, observation):
        """Generate adversarial perturbation using FGSM."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
        obs_tensor.requires_grad_(True)

        action_probs = self.policy_network(obs_tensor)

        entropy_loss = -(action_probs * torch.log(action_probs + 1e-8)).sum()

        entropy_loss.backward()

        with torch.no_grad():
            gradient = obs_tensor.grad.data
            perturbation = self.adversarial_strength * torch.sign(gradient)
            adversarial_obs = obs_tensor + perturbation

            adversarial_obs = torch.clamp(adversarial_obs, -2.0, 2.0)

            self.perturbation_norms.append(torch.norm(perturbation).item())

        return adversarial_obs.squeeze().cpu().numpy()

    def get_action(self, observation, use_adversarial=True):
        """Get action with optional adversarial robustness."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
            action_probs = self.policy_network(obs_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.value_network(obs_tensor)

        original_action = action.item()

        if use_adversarial:
            adversarial_obs = self.generate_adversarial_observation(observation)
            with torch.no_grad():
                adv_obs_tensor = (
                    torch.FloatTensor(adversarial_obs).unsqueeze(0).to(device)
                )
                adv_action_probs = self.policy_network(adv_obs_tensor)
                adv_action_dist = Categorical(adv_action_probs)
                adv_action = adv_action_dist.sample()

            return original_action, log_prob.item(), value.item()

        return original_action, log_prob.item(), value.item()

    def update(self, trajectories):
        """Update with adversarial training."""
        if not trajectories:
            return None

        all_obs, all_actions, all_rewards, all_log_probs, all_values = (
            [],
            [],
            [],
            [],
            [],
        )

        for trajectory in trajectories:
            obs, actions, rewards, log_probs, values, _ = zip(*trajectory)
            all_obs.extend(obs)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_log_probs.extend(log_probs)
            all_values.extend(values)

        observations = torch.FloatTensor(all_obs).to(device)
        actions = torch.LongTensor(all_actions).to(device)

        all_returns = []
        for trajectory in trajectories:
            rewards = [step[2] for step in trajectory]
            returns = []
            G = 0
            for reward in reversed(rewards):
                G = reward + self.gamma * G
                returns.insert(0, G)
            all_returns.extend(returns)

        returns = torch.FloatTensor(all_returns).to(device)
        values = torch.FloatTensor(all_values).to(device)
        advantages = returns - values

        action_probs = self.policy_network(observations)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)

        adversarial_loss = 0
        for i in range(min(32, len(observations))):  # Subsample for efficiency
            obs = observations[i]

            obs_adv = obs.clone().detach()
            obs_adv.requires_grad_(True)

            action_probs_adv = self.policy_network(obs_adv.unsqueeze(0))
            entropy = -(action_probs_adv * torch.log(action_probs_adv + 1e-8)).sum()

            grad = torch.autograd.grad(entropy, obs_adv, create_graph=True)[0]
            perturbation = self.adversarial_strength * torch.sign(grad)
            obs_adversarial = obs + perturbation

            action_probs_original = self.policy_network(obs.unsqueeze(0))
            action_probs_adversarial = self.policy_network(obs_adversarial.unsqueeze(0))

            kl_loss = F.kl_div(
                torch.log(action_probs_adversarial + 1e-8),
                action_probs_original,
                reduction="batchmean",
            )
            adversarial_loss += kl_loss

        adversarial_loss /= min(32, len(observations))

        total_policy_loss = policy_loss + 0.1 * adversarial_loss

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.5)
        self.value_optimizer.step()

        self.robust_losses.append(total_policy_loss.item())
        self.adversarial_losses.append(adversarial_loss.item())

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "adversarial_loss": adversarial_loss.item(),
            "total_loss": total_policy_loss.item(),
        }
