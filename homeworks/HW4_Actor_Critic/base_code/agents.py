import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# Small utility functions
def weights_init_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class ReplayMemory:
    """Simple replay buffer for off-policy methods."""

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0
        if seed is not None:
            random.seed(seed)

    def push(self, state, action, reward, next_state, done) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Twin Q-network used in SAC. Returns Q1 and Q2 estimates.
    """

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int) -> None:
        super(QNetwork, self).__init__()
        input_dim = int(num_inputs + num_actions)
        # Q1
        self.q1_l1 = nn.Linear(input_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        # Q2
        self.q2_l1 = nn.Linear(input_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_l1(x))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_l1(x))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_out(q2)
        return q1, q2


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing and action scaling."""

    def __init__(
        self, num_inputs: int, num_actions: int, hidden_dim: int, action_space
    ) -> None:
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # Action scaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            action_high = torch.FloatTensor(action_space.high)
            action_low = torch.FloatTensor(action_space.low)
            self.action_scale = (action_high - action_low) / 2.0
            self.action_bias = (action_high + action_low) / 2.0

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale.to(x_t.device) + self.action_bias.to(
            x_t.device
        )
        # log_prob before and after tanh
        log_prob = normal.log_prob(x_t)
        # correction for Tanh squashing
        log_prob -= torch.log(
            self.action_scale.to(x_t.device) * (1 - y_t.pow(2)) + epsilon
        )
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale.to(
            mean.device
        ) + self.action_bias.to(mean.device)
        return action, log_prob, mean_action


class DeterministicPolicy(nn.Module):
    """Deterministic policy for DDPG-style algorithms (with noise for exploration)."""

    def __init__(
        self, num_inputs: int, num_actions: int, hidden_dim: int, action_space=None
    ) -> None:
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            action_high = torch.FloatTensor(action_space.high)
            action_low = torch.FloatTensor(action_space.low)
            self.action_scale = (action_high - action_low) / 2.0
            self.action_bias = (action_high + action_low) / 2.0

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale.to(
            x.device
        ) + self.action_bias.to(x.device)
        return mean

    def sample(
        self, state: torch.Tensor, noise_scale: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action with exploration noise, zero log_prob, and deterministic mean."""
        mean = self.forward(state)
        noise = torch.randn_like(mean) * noise_scale
        action = mean + noise
        # Clip to action bounds (if scale is tensor)
        return action, torch.tensor(0.0), mean


class SAC(object):
    """Soft Actor-Critic implementation wrapping networks and update logic."""

    def __init__(self, num_inputs: int, action_space, config: dict) -> None:
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.alpha = config.get("alpha", 0.2)
        self.policy_type = config["policy"]
        self.target_update_interval = config["target_update_interval"]
        self.automatic_entropy_tuning = config.get("automatic_entropy_tuning", False)
        self.device = torch.device("cuda" if config.get("cuda", False) else "cpu")

        self.critic = QNetwork(
            num_inputs, action_space.shape[0], config["hidden_size"]
        ).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=config["lr"])

        self.critic_target = QNetwork(
            num_inputs, action_space.shape[0], config["hidden_size"]
        ).to(self.device)
        self.hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target entropy default: -|A|
            if self.automatic_entropy_tuning:
                self.target_entropy = -float(action_space.shape[0])
                # log_alpha is the learnable parameter
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=config["lr"])
                self.alpha = self.log_alpha.exp().item()
            else:
                self.target_entropy = -float(action_space.shape[0])
                self.log_alpha = None

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], config["hidden_size"], action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=config["lr"])
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], config["hidden_size"], action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=config["lr"])

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state_t)
            return action.detach().cpu().numpy()[0]
        else:
            # deterministic mean action
            with torch.no_grad():
                if isinstance(self.policy, GaussianPolicy):
                    _, _, mean = self.policy.sample(state_t)
                    return mean.detach().cpu().numpy()[0]
                else:
                    mean = self.policy.forward(state_t)
                    return mean.detach().cpu().numpy()[0]

    def update_parameters(self, memory: ReplayMemory, batch_size: int, updates: int):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = (
            memory.sample(batch_size)
        )
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Compute target Q value
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # Q-function loss
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Policy loss
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_tlogs = self.alpha
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = self.alpha

        # Soft update of target network
        if updates % self.target_update_interval == 0:
            self.soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs,
        )

    # Soft and hard updates
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def evaluate_policy(agent: SAC, env, eval_episodes: int = 5) -> float:
    """Run `eval_episodes` episodes using the agent in deterministic mode and return average reward."""
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            ep_reward += reward
            state = next_state
        avg_reward += ep_reward
    avg_reward /= float(eval_episodes)
    return avg_reward





