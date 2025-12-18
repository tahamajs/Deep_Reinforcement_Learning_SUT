"""
Advanced Policy Gradient Algorithms
CA4: Policy Gradient Methods and Neural Networks in RL - Advanced Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal, MultivariateNormal
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math
import copy
from collections import deque
import random


class TRPOAgent:
    """Trust Region Policy Optimization Agent"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        kl_target: float = 0.01,
        max_kl: float = 0.02,
        cg_iters: int = 10,
        damping: float = 0.1,
    ):
        """Initialize TRPO agent

        Args:
            state_size: State space dimension
            action_size: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            kl_target: Target KL divergence
            max_kl: Maximum KL divergence
            cg_iters: Conjugate gradient iterations
            damping: Damping coefficient
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.kl_target = kl_target
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.damping = damping

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Training statistics
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.kl_divergences = []

    def get_action_and_value(
        self, state: np.ndarray
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Get action and value estimate

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Policy forward pass
        logits = self.policy_net(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Value forward pass
        value = self.value_net(state_tensor).squeeze()

        return action.item(), log_prob, value

    def compute_gae_advantages(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        next_value: torch.Tensor = None,
        lambda_gae: float = 0.95,
    ) -> List[torch.Tensor]:
        """Compute Generalized Advantage Estimation

        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value of next state
            lambda_gae: GAE lambda parameter

        Returns:
            List of advantage estimates
        """
        advantages = []
        gae = 0

        if next_value is None:
            next_value = torch.tensor(0.0)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * lambda_gae * gae
            advantages.insert(0, gae)
            next_value = values[t]

        return advantages

    def conjugate_gradient(self, A_func, b, x=None):
        """Conjugate gradient algorithm for solving Ax = b

        Args:
            A_func: Function that computes matrix-vector product Ax
            b: Right-hand side vector
            x: Initial guess

        Returns:
            Solution vector x
        """
        if x is None:
            x = torch.zeros_like(b)

        r = b - A_func(x)
        p = r.clone()

        for _ in range(self.cg_iters):
            Ap = A_func(p)
            alpha = torch.dot(r, r) / torch.dot(p, Ap)
            x += alpha * p
            r_new = r - alpha * Ap

            if torch.norm(r_new) < 1e-10:
                break

            beta = torch.dot(r_new, r_new) / torch.dot(r, r)
            p = r_new + beta * p
            r = r_new

        return x

    def fisher_vector_product(
        self, states: torch.Tensor, vector: torch.Tensor
    ) -> torch.Tensor:
        """Compute Fisher information matrix-vector product

        Args:
            states: Batch of states
            vector: Vector to multiply

        Returns:
            Fisher-vector product
        """
        logits = self.policy_net(states)
        probs = F.softmax(logits, dim=-1)

        # Compute KL divergence gradient
        kl_div = torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        kl_grad = torch.autograd.grad(
            kl_div, self.policy_net.parameters(), create_graph=True, retain_graph=True
        )

        # Compute Fisher-vector product
        kl_grad_flat = torch.cat([g.view(-1) for g in kl_grad])
        fisher_vector_product = torch.autograd.grad(
            kl_grad_flat,
            self.policy_net.parameters(),
            grad_outputs=vector,
            retain_graph=True,
        )

        return torch.cat([g.view(-1) for g in fisher_vector_product])

    def update_policy(
        self,
        states: List[np.ndarray],
        actions: List[int],
        advantages: List[torch.Tensor],
        old_log_probs: List[torch.Tensor],
    ) -> Dict[str, float]:
        """Update policy using TRPO

        Args:
            states: List of states
            actions: List of actions
            advantages: List of advantages
            old_log_probs: List of old log probabilities

        Returns:
            Dictionary of loss values
        """
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        advantages_tensor = torch.stack(advantages)
        old_log_probs_tensor = torch.stack(old_log_probs)

        # Compute current policy log probabilities
        logits = self.policy_net(states_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions_tensor)

        # Compute policy gradient
        policy_loss = -(new_log_probs * advantages_tensor).mean()
        policy_grad = torch.autograd.grad(
            policy_loss,
            self.policy_net.parameters(),
            create_graph=True,
            retain_graph=True,
        )
        policy_grad_flat = torch.cat([g.view(-1) for g in policy_grad])

        # Compute natural gradient using conjugate gradient
        def A_func(x):
            return self.fisher_vector_product(states_tensor, x) + self.damping * x

        natural_grad = self.conjugate_gradient(A_func, policy_grad_flat)

        # Compute step size using line search
        step_size = 1.0
        for _ in range(10):
            # Apply step
            old_params = [p.clone() for p in self.policy_net.parameters()]
            flat_params = torch.cat([p.view(-1) for p in self.policy_net.parameters()])
            new_params = flat_params + step_size * natural_grad

            # Update parameters
            idx = 0
            for param in self.policy_net.parameters():
                param_size = param.numel()
                param.data = new_params[idx : idx + param_size].view(param.size())
                idx += param_size

            # Compute KL divergence
            new_logits = self.policy_net(states_tensor)
            new_probs = F.softmax(new_logits, dim=-1)
            kl_div = torch.sum(
                probs * torch.log(probs / (new_probs + 1e-8) + 1e-8), dim=-1
            ).mean()

            if kl_div <= self.max_kl:
                break
            else:
                # Restore old parameters
                for param, old_param in zip(self.policy_net.parameters(), old_params):
                    param.data = old_param.data
                step_size *= 0.5

        # Store statistics
        self.policy_losses.append(policy_loss.item())
        self.kl_divergences.append(kl_div.item())

        return {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_div.item(),
            "step_size": step_size,
        }

    def train_episode(self, env) -> Tuple[float, int]:
        """Train for one episode

        Args:
            env: Environment

        Returns:
            Tuple of (total_reward, episode_length)
        """
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        states = []
        actions = []
        rewards = []
        values = []
        old_log_probs = []

        total_reward = 0
        episode_length = 0

        while True:
            action, log_prob, value = self.get_action_and_value(state)
            next_state, reward, done, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            old_log_probs.append(log_prob)

            state = next_state
            total_reward += reward
            episode_length += 1

            if done or truncated:
                break

        # Compute advantages using GAE
        advantages = self.compute_gae_advantages(rewards, values)

        # Update policy
        policy_losses = self.update_policy(states, actions, advantages, old_log_probs)

        # Update value function
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns_tensor = torch.FloatTensor(returns)
        values_tensor = torch.stack(values)
        value_loss = F.mse_loss(values_tensor, returns_tensor)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.value_losses.append(value_loss.item())
        self.episode_rewards.append(total_reward)

        return total_reward, episode_length

    def train(
        self, env, num_episodes: int = 1000, print_every: int = 100
    ) -> Dict[str, List]:
        """Train the agent

        Args:
            env: Environment
            num_episodes: Number of episodes
            print_every: Print frequency

        Returns:
            Training results
        """
        scores = []

        for episode in range(num_episodes):
            total_reward, episode_length = self.train_episode(env)
            scores.append(total_reward)

            if (episode + 1) % print_every == 0:
                avg_score = np.mean(scores[-print_every:])
                avg_policy_loss = (
                    np.mean(self.policy_losses[-print_every:])
                    if self.policy_losses
                    else 0
                )
                avg_value_loss = (
                    np.mean(self.value_losses[-print_every:])
                    if self.value_losses
                    else 0
                )
                avg_kl_div = (
                    np.mean(self.kl_divergences[-print_every:])
                    if self.kl_divergences
                    else 0
                )

                print(
                    f"Episode {episode + 1:4d} | Avg Score: {avg_score:7.2f} | "
                    f"Policy Loss: {avg_policy_loss:8.4f} | "
                    f"Value Loss: {avg_value_loss:8.4f} | "
                    f"KL Div: {avg_kl_div:8.4f}"
                )

        return {
            "scores": scores,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "kl_divergences": self.kl_divergences,
            "episode_rewards": self.episode_rewards,
        }


class SACAgent:
    """Soft Actor-Critic Agent for Continuous Control"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.0003,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        hidden_size: int = 256,
        target_entropy: float = None,
    ):
        """Initialize SAC agent

        Args:
            state_size: State space dimension
            action_size: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update parameter
            alpha: Temperature parameter
            hidden_size: Hidden layer size
            target_entropy: Target entropy for automatic temperature tuning
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        if target_entropy is None:
            self.target_entropy = -action_size
        else:
            self.target_entropy = target_entropy

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size * 2),  # mean and log_std
        )

        # Critic networks (Q-functions)
        self.critic1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Target networks
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Temperature parameter (learnable)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.episode_rewards = []

    def get_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Get action from policy

        Args:
            state: Current state
            evaluate: Whether to use deterministic action

        Returns:
            Tuple of (action, log_prob)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get policy parameters
        policy_output = self.actor(state_tensor)
        mean, log_std = policy_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        if evaluate:
            # Deterministic action for evaluation
            action = torch.tanh(mean)
        else:
            # Stochastic action for training
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action.detach().numpy(), log_prob if not evaluate else None

    def update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Update critic networks

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        """
        with torch.no_grad():
            # Get next actions and log probabilities
            next_policy_output = self.actor(next_states)
            next_mean, next_log_std = next_policy_output.chunk(2, dim=-1)
            next_log_std = torch.clamp(next_log_std, -20, 2)
            next_std = torch.exp(next_log_std)

            next_normal = Normal(next_mean, next_std)
            next_x_t = next_normal.rsample()
            next_actions = torch.tanh(next_x_t)
            next_log_probs = next_normal.log_prob(next_x_t) - torch.log(
                1 - next_actions.pow(2) + 1e-6
            )
            next_log_probs = next_log_probs.sum(dim=-1, keepdim=True)

            # Compute target Q-values
            target_q1 = self.target_critic1(
                torch.cat([next_states, next_actions], dim=-1)
            )
            target_q2 = self.target_critic2(
                torch.cat([next_states, next_actions], dim=-1)
            )
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q

        # Current Q-values
        current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=-1))

        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)

    def update_actor(self, states: torch.Tensor):
        """Update actor network

        Args:
            states: Batch of states
        """
        # Get policy parameters
        policy_output = self.actor(states)
        mean, log_std = policy_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # Sample actions
        normal = Normal(mean, std)
        x_t = normal.rsample()
        actions = torch.tanh(x_t)
        log_probs = normal.log_prob(x_t) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        # Compute Q-values
        q1 = self.critic1(torch.cat([states, actions], dim=-1))
        q2 = self.critic2(torch.cat([states, actions], dim=-1))
        q_values = torch.min(q1, q2)

        # Actor loss
        actor_loss = (self.alpha * log_probs - q_values).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor_losses.append(actor_loss.item())

        # Update temperature
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()
        self.alpha_losses.append(alpha_loss.item())

    def soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def train_episode(self, env) -> Tuple[float, int]:
        """Train for one episode

        Args:
            env: Environment

        Returns:
            Tuple of (total_reward, episode_length)
        """
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        episode_length = 0

        # Experience buffer for this episode
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        while True:
            action, _ = self.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)

            state = next_state
            total_reward += reward
            episode_length += 1

            if done or truncated:
                break

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

        # Update networks
        self.update_critics(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )
        self.update_actor(states_tensor)
        self.soft_update()

        self.episode_rewards.append(total_reward)

        return total_reward, episode_length

    def train(
        self, env, num_episodes: int = 1000, print_every: int = 100
    ) -> Dict[str, List]:
        """Train the agent

        Args:
            env: Environment
            num_episodes: Number of episodes
            print_every: Print frequency

        Returns:
            Training results
        """
        scores = []

        for episode in range(num_episodes):
            total_reward, episode_length = self.train_episode(env)
            scores.append(total_reward)

            if (episode + 1) % print_every == 0:
                avg_score = np.mean(scores[-print_every:])
                avg_actor_loss = (
                    np.mean(self.actor_losses[-print_every:])
                    if self.actor_losses
                    else 0
                )
                avg_critic_loss = (
                    np.mean(self.critic_losses[-print_every:])
                    if self.critic_losses
                    else 0
                )
                avg_alpha_loss = (
                    np.mean(self.alpha_losses[-print_every:])
                    if self.alpha_losses
                    else 0
                )

                print(
                    f"Episode {episode + 1:4d} | Avg Score: {avg_score:7.2f} | "
                    f"Actor Loss: {avg_actor_loss:8.4f} | "
                    f"Critic Loss: {avg_critic_loss:8.4f} | "
                    f"Alpha Loss: {avg_alpha_loss:8.4f}"
                )

        return {
            "scores": scores,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "alpha_losses": self.alpha_losses,
            "episode_rewards": self.episode_rewards,
        }


class DDPGAgent:
    """Deep Deterministic Policy Gradient Agent"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr_actor: float = 0.001,
        lr_critic: float = 0.002,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_std: float = 0.1,
        hidden_size: int = 256,
    ):
        """Initialize DDPG agent

        Args:
            state_size: State space dimension
            action_size: Action space dimension
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            tau: Soft update parameter
            noise_std: Noise standard deviation
            hidden_size: Hidden layer size
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std

        # Actor networks
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),
        )

        self.target_actor = copy.deepcopy(self.actor)

        # Critic networks
        self.critic = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.target_critic = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        self.episode_rewards = []

    def get_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Get action from policy

        Args:
            state: Current state
            add_noise: Whether to add noise for exploration

        Returns:
            Action array
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()

        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)

        return action

    def update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Update critic network

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        """
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.target_actor(next_states)
            # Compute target Q-values
            target_q = self.target_critic(
                torch.cat([next_states, next_actions], dim=-1)
            )
            target_q = rewards + self.gamma * (1 - dones) * target_q

        # Current Q-values
        current_q = self.critic(torch.cat([states, actions], dim=-1))

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic_losses.append(critic_loss.item())

    def update_actor(self, states: torch.Tensor):
        """Update actor network

        Args:
            states: Batch of states
        """
        # Get actions from current actor
        actions = self.actor(states)

        # Compute Q-values
        q_values = self.critic(torch.cat([states, actions], dim=-1))

        # Actor loss (negative Q-values to maximize)
        actor_loss = -q_values.mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor_losses.append(actor_loss.item())

    def soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def train_episode(self, env) -> Tuple[float, int]:
        """Train for one episode

        Args:
            env: Environment

        Returns:
            Tuple of (total_reward, episode_length)
        """
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        episode_length = 0

        # Experience buffer for this episode
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        while True:
            action = self.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)

            state = next_state
            total_reward += reward
            episode_length += 1

            if done or truncated:
                break

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

        # Update networks
        self.update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )
        self.update_actor(states_tensor)
        self.soft_update()

        self.episode_rewards.append(total_reward)

        return total_reward, episode_length

    def train(
        self, env, num_episodes: int = 1000, print_every: int = 100
    ) -> Dict[str, List]:
        """Train the agent

        Args:
            env: Environment
            num_episodes: Number of episodes
            print_every: Print frequency

        Returns:
            Training results
        """
        scores = []

        for episode in range(num_episodes):
            total_reward, episode_length = self.train_episode(env)
            scores.append(total_reward)

            if (episode + 1) % print_every == 0:
                avg_score = np.mean(scores[-print_every:])
                avg_actor_loss = (
                    np.mean(self.actor_losses[-print_every:])
                    if self.actor_losses
                    else 0
                )
                avg_critic_loss = (
                    np.mean(self.critic_losses[-print_every:])
                    if self.critic_losses
                    else 0
                )

                print(
                    f"Episode {episode + 1:4d} | Avg Score: {avg_score:7.2f} | "
                    f"Actor Loss: {avg_actor_loss:8.4f} | "
                    f"Critic Loss: {avg_critic_loss:8.4f}"
                )

        return {
            "scores": scores,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "episode_rewards": self.episode_rewards,
        }


