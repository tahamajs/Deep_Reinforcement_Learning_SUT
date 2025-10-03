"""
Reinforcement Learning Algorithms for Policy Gradients
CA4: Policy Gradient Methods and Neural Networks in RL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


try:
    from .policies import (
        PolicyNetwork,
        ValueNetwork,
        AdvancedActorCritic,
        ContinuousActorCriticAgent,
    )
except ImportError:
    from .policies import (
        PolicyNetwork,
        ValueNetwork,
        AdvancedActorCritic,
        ContinuousActorCriticAgent,
    )


class REINFORCEAgent:
    """REINFORCE agent with baseline"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        baseline: bool = True,
    ):
        """Initialize REINFORCE agent

        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            lr: Learning rate
            gamma: Discount factor
            baseline: Whether to use baseline for variance reduction
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.baseline = baseline

        self.policy_net = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        if baseline:
            self.value_net = ValueNetwork(state_size)

            for param in self.value_net.parameters():
                param.requires_grad_(True)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.reset_episode()

        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []

    def reset_episode(self):
        """Reset episode-specific storage"""
        self.log_probs = []
        self.rewards = []
        self.states = []
        self.actions = []

    def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using current policy

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_probability)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.policy_net.sample_action(state_tensor)

    def store_transition(
        self, state: np.ndarray, action: int, reward: float, log_prob: torch.Tensor
    ):
        """Store transition for episode

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns

        Args:
            rewards: List of rewards from episode

        Returns:
            Tensor of discounted returns
        """
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def compute_baselines(
        self, states: List[np.ndarray], requires_grad: bool = False
    ) -> torch.Tensor:
        """Compute baseline values for states

        Args:
            states: List of states from episode
            requires_grad: Whether the baselines should require gradients

        Returns:
            Tensor of baseline values
        """
        if not self.baseline:
            return torch.zeros(len(states))

        state_tensors = torch.FloatTensor(np.array(states))
        if requires_grad:
            baselines = self.value_net(state_tensors).squeeze()
        else:
            with torch.no_grad():
                baselines = self.value_net(state_tensors).squeeze()
        return baselines

    def update_policy(self) -> Tuple[float, Optional[float]]:
        """Update policy using REINFORCE algorithm

        Returns:
            Tuple of (policy_loss, value_loss)
        """
        if len(self.rewards) == 0:
            return 0.0, None

        returns = self.compute_returns(self.rewards)
        baselines = self.compute_baselines(self.states)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, G, baseline in zip(self.log_probs, returns, baselines):
            advantage = G - baseline
            policy_loss.append(-log_prob * advantage)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        value_loss = None
        if self.baseline:

            state_tensors = torch.FloatTensor(np.array(self.states))
            baselines = self.value_net(state_tensors).squeeze()
            value_targets = returns.detach()

            self.value_optimizer.zero_grad()
            value_loss = F.mse_loss(baselines, value_targets)

            value_loss = value_loss.item()

        self.episode_rewards.append(sum(self.rewards))
        self.episode_lengths.append(len(self.rewards))
        self.policy_losses.append(policy_loss.item())

        return policy_loss.item(), value_loss

    def train_episode(self, env) -> Tuple[float, int]:
        """Train for one episode

        Args:
            env: Environment to train in

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
            action, log_prob = self.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            self.store_transition(state, action, reward, log_prob)

            state = next_state
            total_reward += reward
            episode_length += 1

            if done or truncated:
                break

        policy_loss, value_loss = self.update_policy()

        return total_reward, episode_length

    def train(
        self, env, num_episodes: int = 1000, print_every: int = 100
    ) -> Dict[str, List]:
        """Train REINFORCE agent

        Args:
            env: Environment to train in
            num_episodes: Number of episodes to train
            print_every: Print progress every N episodes

        Returns:
            Dictionary with training results
        """
        scores = []

        for episode in range(num_episodes):
            total_reward, episode_length = self.train_episode(env)
            scores.append(total_reward)

            if (episode + 1) % print_every == 0:
                avg_score = np.mean(scores[-print_every:])
                print(f"Episode {episode + 1:4d} | " f"Avg Score: {avg_score:7.2f}")

        return {
            "scores": scores,
            "episode_lengths": self.episode_lengths,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses if self.baseline else None,
        }


class ActorCriticAgent:
    """Actor-Critic agent with separate networks"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr_actor: float = 0.001,
        lr_critic: float = 0.005,
        gamma: float = 0.99,
    ):
        """Initialize Actor-Critic agent

        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma

        self.actor = PolicyNetwork(state_size, action_size)
        self.critic = ValueNetwork(state_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.actor_losses = []
        self.critic_losses = []
        self.episode_rewards = []
        self.td_errors = []

    def get_action_and_value(
        self, state: np.ndarray
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Get action from actor and value from critic

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        action_probs = self.actor.get_action_probs(state_tensor)
        value = self.critic(state_tensor).squeeze()

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

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
    ) -> Tuple[float, float, float]:
        """Update actor and critic networks

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            value: Value estimate of current state

        Returns:
            Tuple of (actor_loss, critic_loss, td_error)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        if done:
            td_target = torch.tensor(reward, dtype=torch.float32)
        else:
            next_value = self.critic(next_state_tensor).squeeze()
            td_target = reward + self.gamma * next_value.detach()

        td_error = td_target - value

        critic_loss = F.mse_loss(value, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss = -log_prob * td_error.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.td_errors.append(abs(td_error.item()))

        return actor_loss.item(), critic_loss.item(), td_error.item()

    def train_episode(self, env) -> Tuple[float, int]:
        """Train for one episode

        Args:
            env: Environment to train in

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

            actor_loss, critic_loss, td_error = self.update(
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
        """Train Actor-Critic agent

        Args:
            env: Environment to train in
            num_episodes: Number of episodes to train
            print_every: Print progress every N episodes

        Returns:
            Dictionary with training results
        """
        scores = []

        for episode in range(num_episodes):
            total_reward, episode_length = self.train_episode(env)
            scores.append(total_reward)

            if (episode + 1) % print_every == 0:
                avg_score = np.mean(scores[-print_every:])
                avg_actor_loss = np.mean(self.actor_losses[-episode_length:])
                avg_critic_loss = np.mean(self.critic_losses[-episode_length:])
                avg_td_error = np.mean(self.td_errors[-episode_length:])

                print(
                    f"Episode {episode + 1:4d} | "
                    f"Avg Score: {avg_score:7.2f} | "
                    f"Actor Loss: {avg_actor_loss:8.4f} | "
                    f"Critic Loss: {avg_critic_loss:8.4f} | "
                    f"TD Error: {avg_td_error:6.3f}"
                )

        return {
            "scores": scores,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "td_errors": self.td_errors,
            "episode_rewards": self.episode_rewards,
        }


class ContinuousActorCriticTrainer:
    """Trainer for continuous action spaces"""

    def __init__(
        self,
        agent: ContinuousActorCriticAgent,
        lr_policy: float = 0.001,
        lr_value: float = 0.001,
        gamma: float = 0.99,
    ):
        """Initialize continuous trainer

        Args:
            agent: Continuous actor-critic agent
            lr_policy: Learning rate for policy
            lr_value: Learning rate for value function
            gamma: Discount factor
        """
        self.agent = agent
        self.gamma = gamma

        self.policy_optimizer = optim.Adam(
            self.agent.policy_net.parameters(), lr=lr_policy
        )
        self.value_optimizer = optim.Adam(
            self.agent.value_net.parameters(), lr=lr_value
        )

        self.policy_losses = []
        self.value_losses = []
        self.episode_rewards = []

    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> Tuple[float, float]:
        """Update networks for continuous control

        Args:
            state: Current state tensor
            action: Action tensor
            reward: Reward received
            next_state: Next state tensor
            done: Whether episode is done

        Returns:
            Tuple of (policy_loss, value_loss)
        """

        if done:
            td_target = reward
        else:
            next_value = self.agent.value_net(next_state).squeeze()
            td_target = reward + self.gamma * next_value.detach()

        value = self.agent.value_net(state).squeeze()
        td_error = td_target - value

        value_loss = F.mse_loss(value, td_target.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.value_net.parameters(), 1.0)
        self.value_optimizer.step()

        log_prob, entropy, _ = self.agent.evaluate_action(state, action)
        policy_loss = -(log_prob * td_error.detach() + 0.01 * entropy)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())

        return policy_loss.item(), value_loss.item()

    def train_episode(self, env) -> Tuple[float, int]:
        """Train for one episode in continuous environment

        Args:
            env: Continuous environment

        Returns:
            Tuple of (total_reward, episode_length)
        """
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        state = torch.FloatTensor(state)
        total_reward = 0
        episode_length = 0

        while True:
            action, log_prob = self.agent.get_action(state.unsqueeze(0))
            action_tensor = torch.FloatTensor(action)

            next_state, reward, done, truncated, _ = env.step(action)

            next_state_tensor = torch.FloatTensor(next_state)

            policy_loss, value_loss = self.update(
                state.unsqueeze(0),
                action_tensor.unsqueeze(0),
                reward,
                next_state_tensor.unsqueeze(0),
                done or truncated,
            )

            state = next_state_tensor
            total_reward += reward
            episode_length += 1

            if done or truncated:
                break

        self.episode_rewards.append(total_reward)
        return total_reward, episode_length


def create_agent(
    algorithm: str,
    state_size: int,
    action_size: int,
    continuous: bool = False,
    **kwargs,
) -> Any:
    """Factory function to create RL agent

    Args:
        algorithm: Type of algorithm ('reinforce', 'actor_critic')
        state_size: Dimension of state space
        action_size: Dimension of action space
        continuous: Whether action space is continuous
        **kwargs: Additional arguments

    Returns:
        Agent instance
    """
    if algorithm.lower() == "reinforce":
        return REINFORCEAgent(state_size, action_size, **kwargs)
    elif algorithm.lower() == "actor_critic":
        if continuous:
            agent = ContinuousActorCriticAgent(state_size, action_size, **kwargs)
            trainer = ContinuousActorCriticTrainer(agent, **kwargs)
            return trainer
        else:
            return ActorCriticAgent(state_size, action_size, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def compare_algorithms(
    env,
    algorithms: List[str],
    state_size: int,
    action_size: int,
    num_episodes: int = 300,
    **kwargs,
) -> Dict[str, Dict]:
    """Compare different algorithms

    Args:
        env: Environment to test in
        algorithms: List of algorithm names
        state_size: Dimension of state space
        action_size: Dimension of action space
        num_episodes: Number of episodes per algorithm
        **kwargs: Additional arguments for agents

    Returns:
        Dictionary with results for each algorithm
    """
    results = {}

    for alg in algorithms:
        print(f"Training {alg}...")
        agent = create_agent(alg, state_size, action_size, **kwargs)

        if hasattr(agent, "train"):
            results[alg] = agent.train(env, num_episodes, print_every=50)
        else:

            scores = []
            for episode in range(num_episodes):
                total_reward, _ = agent.train_episode(env)
                scores.append(total_reward)
                if (episode + 1) % 50 == 0:
                    avg_score = np.mean(scores[-50:])
                    print(f"Episode {episode + 1:4d} | Avg Score: {avg_score:7.2f}")

            results[alg] = {
                "scores": scores,
                "policy_losses": agent.policy_losses,
                "value_losses": agent.value_losses,
                "episode_rewards": agent.episode_rewards,
            }

        print(f"✓ {alg} training completed")

    return results
