"""Recurrent Meta-RL (RL²) implementation."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np
from .config import RL2Config
from .utils import compute_gae_returns


class RL2Policy(nn.Module):
    """Recurrent Meta-RL (RL²) Policy."""
    def __init__(self, config: RL2Config):
        super().__init__()
        self.config = config
        self.discrete = True  # Assuming discrete actions for CartPole

        # Input dimension: obs + prev_action + prev_reward + done
        input_dim = config.obs_dim + config.action_dim + 1 + 1

        # Recurrent encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_lstm_layers,
            batch_first=True
        )

        # Policy head
        if self.discrete:
            self.policy_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, config.action_dim)
            )
        else:
            self.policy_mean = nn.Sequential(
                nn.Linear(config.hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, config.action_dim)
            )
            self.policy_logstd = nn.Parameter(torch.zeros(config.action_dim))

        # Value head
        self.value = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor,
                prev_reward: torch.Tensor, done: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim)
            prev_action: (batch, action_dim) - one-hot for discrete
            prev_reward: (batch,)
            done: (batch,)
            hidden: tuple of (h, c) each (num_layers, batch, hidden_dim)

        Returns:
            For discrete: logits, value, hidden_new
        """
        batch_size = obs.shape[0]

        # Concatenate inputs
        x = torch.cat([
            obs,
            prev_action,
            prev_reward.unsqueeze(-1),
            done.unsqueeze(-1).float()
        ], dim=-1)

        # LSTM forward
        x = x.unsqueeze(1)  # Add sequence dimension
        output, hidden_new = self.lstm(x, hidden)
        output = output.squeeze(1)

        # Value output
        value = self.value(output)

        if self.discrete:
            logits = self.policy_head(output)
            return logits, value, hidden_new
        else:
            mean = self.policy_mean(output)
            std = torch.exp(self.policy_logstd).expand_as(mean)
            return mean, std, value, hidden_new

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state for new task."""
        return (
            torch.zeros(self.config.num_lstm_layers, batch_size,
                       self.config.hidden_dim).to(device),
            torch.zeros(self.config.num_lstm_layers, batch_size,
                       self.config.hidden_dim).to(device)
        )

    def sample_action(self, obs: torch.Tensor, prev_action: torch.Tensor,
                     prev_reward: torch.Tensor, done: torch.Tensor,
                     hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """Sample action from policy."""
        if self.discrete:
            logits, value, hidden_new = self.forward(
                obs, prev_action, prev_reward, done, hidden
            )
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            mean, std, value, hidden_new = self.forward(
                obs, prev_action, prev_reward, done, hidden
            )
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob, value, hidden_new


class RL2Trainer:
    """Trainer for RL²."""
    def __init__(self, config: RL2Config):
        self.config = config
        self.policy = RL2Policy(config)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)

    def collect_trajectory_rl2(self, env, max_steps: int = 200) -> Dict[str, torch.Tensor]:
        """Collect trajectory with recurrent policy."""
        obs = env.reset()
        hidden = self.policy.init_hidden()

        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        prev_action = torch.zeros(self.config.action_dim)  # One-hot for discrete
        prev_reward = 0.0
        done = False

        for _ in range(max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            prev_action_tensor = prev_action.unsqueeze(0)
            prev_reward_tensor = torch.FloatTensor([prev_reward])
            done_tensor = torch.FloatTensor([done])

            action, log_prob, value, hidden = self.policy.sample_action(
                obs_tensor, prev_action_tensor, prev_reward_tensor, done_tensor, hidden
            )

            next_obs, reward, done, _ = env.step(action.item())

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            obs = next_obs
            prev_action = F.one_hot(action, self.config.action_dim).float()
            prev_reward = reward

            if done:
                break

        return {
            'states': torch.FloatTensor(states),
            'actions': torch.stack(actions),
            'rewards': torch.tensor(rewards),
            'log_probs': torch.stack(log_probs),
            'values': torch.stack(values),
            'dones': torch.tensor(dones)
        }

    def train_on_task(self, task_env, num_episodes: int = 5):
        """Train on a single task."""
        for episode in range(num_episodes):
            trajectory = self.collect_trajectory_rl2(task_env)

            returns, advantages = compute_gae_returns(
                trajectory['rewards'].tolist(), trajectory['values'],
                trajectory['dones'].tolist(), self.config.gamma, self.config.lam
            )

            # PPO-style update
            for _ in range(self.config.ppo_epochs):  # PPO epochs
                logits, values, _ = self.policy(
                    trajectory['states'],
                    F.one_hot(trajectory['actions'], self.config.action_dim).float(),
                    trajectory['rewards'].unsqueeze(-1),
                    trajectory['dones'].unsqueeze(-1).float(),
                    self.policy.init_hidden(trajectory['states'].shape[0])
                )

                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(trajectory['actions'])

                ratio = torch.exp(new_log_probs - trajectory['log_probs'])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), returns)
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def meta_train_step(self, task_envs: List[Any]) -> float:
        """Meta-training step across tasks."""
        total_loss = 0
        for task_env in task_envs:
            self.train_on_task(task_env, num_episodes=self.config.num_episodes_per_task)
            # Evaluate
            trajectory = self.collect_trajectory_rl2(task_env)
            total_loss += -trajectory['rewards'].sum()
        return total_loss / len(task_envs)

    def train(self, task_distribution, num_meta_iterations: int = 50, meta_batch_size: int = 5) -> List[float]:
        """Train the meta-learner."""
        losses = []
        for iteration in range(num_meta_iterations):
            task_batch = task_distribution.sample(meta_batch_size)
            loss = self.meta_train_step([task.env for task in task_batch])
            losses.append(loss.item())

            if (iteration + 1) % 10 == 0:
                print(f"RL² Iteration {iteration+1}/{num_meta_iterations}, Loss: {loss.item():.4f}")
        return losses