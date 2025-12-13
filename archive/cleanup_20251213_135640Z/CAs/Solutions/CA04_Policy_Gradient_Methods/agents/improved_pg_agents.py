"""
Improved Policy Gradient Agents
CA4: Policy Gradient Methods and Neural Networks in RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math


class ImprovedREINFORCEAgent:
    """Enhanced REINFORCE with advanced techniques"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        baseline: bool = True,
        entropy_coef: float = 0.01,
        gradient_clip: float = 1.0,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
    ):
        """Initialize improved REINFORCE agent

        Args:
            state_size: State space dimension
            action_size: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            baseline: Whether to use baseline
            entropy_coef: Entropy regularization coefficient
            gradient_clip: Gradient clipping value
            use_gae: Whether to use Generalized Advantage Estimation
            gae_lambda: GAE lambda parameter
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.baseline = baseline
        self.entropy_coef = entropy_coef
        self.gradient_clip = gradient_clip
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        # Enhanced policy network with batch normalization
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_size),
        )

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        if baseline:
            self.value_net = nn.Sequential(
                nn.Linear(state_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
            )
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.reset_episode()

        # Training statistics
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

    def reset_episode(self):
        """Reset episode-specific storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Get action from policy

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get action probabilities
        logits = self.policy_net(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Get value estimate if using baseline
        value = None
        if self.baseline:
            value = self.value_net(state_tensor).squeeze()

        return action.item(), log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor = None,
    ):
        """Store transition for episode

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate (optional)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)

    def compute_gae_advantages(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        next_value: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        """Compute Generalized Advantage Estimation

        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value of next state (optional)

        Returns:
            List of advantage estimates
        """
        advantages = []
        gae = 0

        if next_value is None:
            next_value = torch.tensor(0.0)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            next_value = values[t]

        return advantages

    def compute_returns(self, rewards: List[float]) -> List[torch.Tensor]:
        """Compute discounted returns

        Args:
            rewards: List of rewards

        Returns:
            List of returns
        """
        returns = []
        G = 0

        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, torch.tensor(G))

        return returns

    def update_policy(self) -> Dict[str, float]:
        """Update policy using improved REINFORCE

        Returns:
            Dictionary of loss values
        """
        if len(self.rewards) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}

        # Compute returns and advantages
        returns = self.compute_returns(self.rewards)
        returns = torch.stack(returns)

        if self.use_gae and self.baseline and len(self.values) > 0:
            advantages = self.compute_gae_advantages(self.rewards, self.values)
            advantages = torch.stack(advantages)
        elif self.baseline and len(self.values) > 0:
            values = torch.stack(self.values)
            advantages = returns - values
        else:
            advantages = returns

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        policy_loss = 0
        entropy_loss = 0

        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss += -log_prob * advantage.detach()

            # Entropy bonus
            if self.entropy_coef > 0:
                probs = F.softmax(
                    self.policy_net(torch.FloatTensor(self.states[0]).unsqueeze(0)),
                    dim=-1,
                )
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                entropy_loss += -self.entropy_coef * entropy

        # Update policy network
        self.optimizer.zero_grad()
        total_policy_loss = policy_loss + entropy_loss
        total_policy_loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), self.gradient_clip
            )

        self.optimizer.step()

        # Value loss (if using baseline)
        value_loss = 0
        if self.baseline and len(self.values) > 0:
            values = torch.stack(self.values)
            value_loss = F.mse_loss(values, returns.detach())

            self.value_optimizer.zero_grad()
            value_loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.gradient_clip
                )

            self.value_optimizer.step()

        # Store statistics
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(
            value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss
        )
        self.entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": (
                value_loss.item()
                if isinstance(value_loss, torch.Tensor)
                else value_loss
            ),
            "entropy_loss": entropy_loss.item(),
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

        self.reset_episode()
        total_reward = 0
        episode_length = 0

        while True:
            action, log_prob, value = self.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            self.store_transition(state, action, reward, log_prob, value)

            state = next_state
            total_reward += reward
            episode_length += 1

            if done or truncated:
                break

        # Update policy
        losses = self.update_policy()

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
                avg_entropy_loss = (
                    np.mean(self.entropy_losses[-print_every:])
                    if self.entropy_losses
                    else 0
                )

                print(
                    f"Episode {episode + 1:4d} | Avg Score: {avg_score:7.2f} | "
                    f"Policy Loss: {avg_policy_loss:8.4f} | "
                    f"Value Loss: {avg_value_loss:8.4f} | "
                    f"Entropy Loss: {avg_entropy_loss:8.4f}"
                )

        return {
            "scores": scores,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropy_losses": self.entropy_losses,
            "episode_rewards": self.episode_rewards,
        }


class ImprovedActorCriticAgent:
    """Enhanced Actor-Critic with advanced techniques"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr_actor: float = 0.001,
        lr_critic: float = 0.005,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        gradient_clip: float = 1.0,
        use_target_network: bool = False,
        tau: float = 0.005,
    ):
        """Initialize improved Actor-Critic agent

        Args:
            state_size: State space dimension
            action_size: Action space dimension
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            gradient_clip: Gradient clipping value
            use_target_network: Whether to use target network
            tau: Soft update parameter
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.gradient_clip = gradient_clip
        self.use_target_network = use_target_network
        self.tau = tau

        # Enhanced actor network
        self.actor = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_size),
        )

        # Enhanced critic network
        self.critic = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Target networks for stability
        if use_target_network:
            self.target_critic = nn.Sequential(
                nn.Linear(state_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
            )
            self.target_critic.load_state_dict(self.critic.state_dict())

        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.td_errors = []
        self.episode_rewards = []

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

        # Actor forward pass
        logits = self.actor(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic forward pass
        value = self.critic(state_tensor).squeeze()

        return action.item(), log_prob, value

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> Dict[str, float]:
        """Update actor and critic networks

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            value: Value estimate

        Returns:
            Dictionary of loss values
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Compute TD target
        if done:
            td_target = torch.tensor(reward, dtype=torch.float32)
        else:
            if self.use_target_network:
                next_value = self.target_critic(next_state_tensor).squeeze()
            else:
                next_value = self.critic(next_state_tensor).squeeze()
            td_target = reward + self.gamma * next_value.detach()

        # Critic update
        td_error = td_target - value
        critic_loss = F.mse_loss(value, td_target.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)

        self.critic_optimizer.step()

        # Actor update with entropy bonus
        actor_loss = -log_prob * td_error.detach()

        # Add entropy bonus
        if self.entropy_coef > 0:
            probs = F.softmax(self.actor(state_tensor), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            actor_loss -= self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)

        self.actor_optimizer.step()

        # Soft update target network
        if self.use_target_network:
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

        # Store statistics
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.td_errors.append(abs(td_error.item()))

        if self.entropy_coef > 0:
            self.entropy_losses.append(entropy.item())

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_error": td_error.item(),
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

        total_reward = 0
        episode_length = 0

        while True:
            action, log_prob, value = self.get_action_and_value(state)
            next_state, reward, done, truncated, _ = env.step(action)

            losses = self.update(
                state, action, reward, next_state, done or truncated, log_prob, value
            )

            state = next_state
            total_reward += reward
            episode_length += 1

            if done or truncated:
                break

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
                avg_td_error = (
                    np.mean(self.td_errors[-print_every:]) if self.td_errors else 0
                )

                print(
                    f"Episode {episode + 1:4d} | Avg Score: {avg_score:7.2f} | "
                    f"Actor Loss: {avg_actor_loss:8.4f} | "
                    f"Critic Loss: {avg_critic_loss:8.4f} | "
                    f"TD Error: {avg_td_error:6.3f}"
                )

        return {
            "scores": scores,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "entropy_losses": self.entropy_losses,
            "td_errors": self.td_errors,
            "episode_rewards": self.episode_rewards,
        }


class PPOAgent:
    """Proximal Policy Optimization Agent"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.0003,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
    ):
        """Initialize PPO agent

        Args:
            state_size: State space dimension
            action_size: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            clip_ratio: PPO clipping ratio
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm
            ppo_epochs: Number of PPO epochs per update
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        # Actor-Critic network
        self.actor_critic = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size + 1),  # +1 for value
        )

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.episode_rewards = []

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

        output = self.actor_critic(state_tensor)
        action_logits = output[:, :-1]
        value = output[:, -1]

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value.squeeze()

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        output = self.actor_critic(states)
        action_logits = output[:, :-1]
        values = output[:, -1]

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy

    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        old_log_probs: List[torch.Tensor],
        values: List[torch.Tensor],
        dones: List[bool],
    ) -> Dict[str, float]:
        """PPO update

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            old_log_probs: List of old log probabilities
            values: List of old values
            dones: List of done flags

        Returns:
            Dictionary of loss values
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.stack(old_log_probs)
        old_values_tensor = torch.stack(values)

        # Compute returns and advantages
        returns = []
        advantages = []
        G = 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                G = rewards[i]
            else:
                G = rewards[i] + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        advantages = returns - old_values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Get current policy outputs
            log_probs, current_values, entropy = self.evaluate_actions(
                states_tensor, actions_tensor
            )

            # Compute policy loss
            ratio = torch.exp(log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss
            value_loss = F.mse_loss(current_values, returns)

            # Compute entropy loss
            entropy_loss = -entropy.mean()

            # Total loss
            total_loss = (
                actor_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            # Update
            self.optimizer.zero_grad()
            total_loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )

            self.optimizer.step()

        # Store statistics
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(value_loss.item())
        self.entropy_losses.append(entropy_loss.item())

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
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
        old_log_probs = []
        values = []
        dones = []

        total_reward = 0
        episode_length = 0

        while True:
            action, log_prob, value = self.get_action_and_value(state)
            next_state, reward, done, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(log_prob)
            values.append(value)
            dones.append(done or truncated)

            state = next_state
            total_reward += reward
            episode_length += 1

            if done or truncated:
                break

        # Update policy
        losses = self.update(states, actions, rewards, old_log_probs, values, dones)

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
                avg_entropy_loss = (
                    np.mean(self.entropy_losses[-print_every:])
                    if self.entropy_losses
                    else 0
                )

                print(
                    f"Episode {episode + 1:4d} | Avg Score: {avg_score:7.2f} | "
                    f"Actor Loss: {avg_actor_loss:8.4f} | "
                    f"Critic Loss: {avg_critic_loss:8.4f} | "
                    f"Entropy Loss: {avg_entropy_loss:8.4f}"
                )

        return {
            "scores": scores,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "entropy_losses": self.entropy_losses,
            "episode_rewards": self.episode_rewards,
        }
