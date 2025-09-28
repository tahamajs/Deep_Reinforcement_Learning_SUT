"""
Dreamer Agent for Planning in Latent Space
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from .latent_actor import LatentActor
from .latent_critic import LatentCritic


class DreamerAgent:
    """Dreamer-style agent for planning in latent space"""

    def __init__(
        self,
        world_model,
        state_dim,
        action_dim,
        device,
        actor_lr=8e-5,
        critic_lr=8e-5,
        gamma=0.99,
        lambda_=0.95,
        imagination_horizon=15,
    ):

        self.world_model = world_model
        self.device = device
        self.gamma = gamma
        self.lambda_ = lambda_
        self.imagination_horizon = imagination_horizon

        # Actor and critic networks
        self.actor = LatentActor(state_dim, action_dim).to(device)
        self.critic = LatentCritic(state_dim).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Training statistics
        self.stats = {
            "actor_loss": [],
            "critic_loss": [],
            "imagination_reward": [],
            "policy_entropy": [],
        }

    def imagine_trajectories(self, initial_states, batch_size=50):
        """Generate imagined trajectories using world model"""
        horizon = self.imagination_horizon

        # Storage for trajectory
        states = [initial_states]
        actions = []
        rewards = []
        log_probs = []
        values = []

        current_state = initial_states

        for t in range(horizon):
            # Sample action from current policy
            action, log_prob = self.actor.sample(current_state)
            value = self.critic(current_state)

            # Store
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

            # Predict next state and reward using world model
            if hasattr(self.world_model, "dynamics"):
                # Simple world model
                if self.world_model.dynamics.stochastic:
                    next_state, _, _ = self.world_model.dynamics(current_state, action)
                else:
                    next_state = self.world_model.dynamics(current_state, action)
                reward = self.world_model.reward_model(current_state, action)
            else:
                # RSSM world model
                batch_size = current_state.shape[0]
                h_dim = self.world_model.deter_dim
                z_dim = self.world_model.stoch_dim

                # Split state into h and z components
                h = current_state[:, :h_dim]
                z = current_state[:, h_dim : h_dim + z_dim]

                # Imagination step
                h, z, _ = self.world_model.imagine(h, z, action)
                next_state = torch.cat([h, z], dim=-1)
                reward = self.world_model.predict_reward(h, z)

            states.append(next_state)
            rewards.append(reward)
            current_state = next_state

        # Convert to tensors
        states = torch.stack(states[:-1])  # Exclude last state
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        # Final value for bootstrapping
        final_value = self.critic(states[-1])

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "log_probs": log_probs,
            "values": values,
            "final_value": final_value,
        }

    def compute_returns_and_advantages(self, trajectory):
        """Compute returns and advantages using GAE"""
        rewards = trajectory["rewards"]
        values = trajectory["values"]
        final_value = trajectory["final_value"]

        # Compute returns
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        last_return = final_value
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * last_return

            delta = (
                rewards[t]
                + self.gamma * (final_value if t == len(rewards) - 1 else values[t + 1])
                - values[t]
            )
            advantages[t] = delta + self.gamma * self.lambda_ * last_advantage

            last_return = returns[t]
            last_advantage = advantages[t]

        return returns, advantages

    def update_actor_critic(self, trajectory):
        """Update actor and critic networks"""
        states = trajectory["states"]
        actions = trajectory["actions"]
        log_probs = trajectory["log_probs"]

        # Reshape for processing
        states = states.view(-1, states.shape[-1])
        actions = actions.view(-1, actions.shape[-1])
        log_probs = log_probs.view(-1)

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(trajectory)
        returns = returns.view(-1)
        advantages = advantages.view(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update critic
        self.critic_optimizer.zero_grad()
        values_pred = self.critic(states)
        critic_loss = F.mse_loss(values_pred, returns)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()

        # Recompute log probs for current policy
        action_mean, action_std = self.actor(states)
        dist = Normal(action_mean, action_std)

        # Handle tanh transformation
        raw_actions = torch.atanh(
            torch.clamp(actions / self.actor.action_range, -0.999, 0.999)
        )
        new_log_probs = dist.log_prob(raw_actions).sum(dim=-1)
        new_log_probs -= (
            2 * (np.log(2) - raw_actions - F.softplus(-2 * raw_actions))
        ).sum(dim=-1)

        # Actor loss (policy gradient with advantages)
        actor_loss = -(new_log_probs * advantages.detach()).mean()

        # Add entropy regularization
        entropy = dist.entropy().sum(dim=-1).mean()
        actor_loss -= 0.001 * entropy

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # Record statistics
        self.stats["actor_loss"].append(actor_loss.item())
        self.stats["critic_loss"].append(critic_loss.item())
        self.stats["imagination_reward"].append(trajectory["rewards"].mean().item())
        self.stats["policy_entropy"].append(entropy.item())

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages.mean().item(),
        }

    def train_step(self, initial_states):
        """Single training step"""
        # Generate imagined trajectories
        trajectory = self.imagine_trajectories(initial_states)

        # Update networks
        losses = self.update_actor_critic(trajectory)

        return losses
