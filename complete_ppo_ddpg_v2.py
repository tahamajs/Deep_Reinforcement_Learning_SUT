"""
Complete Implementation of PPO and DDPG Algorithms
This file contains all the code implementations for the notebook
"""

import random
from collections import deque
from typing import Deque, Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.distributions import Normal
import copy

# ============================================================================
# SECTION 1: Utility Functions
# ============================================================================

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Initialize uniform parameters on the single layer.
    
    Args:
        layer: Neural network layer to initialize
        init_w: Initialization range [-init_w, init_w]
        
    Returns:
        Initialized layer
    """
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


# ============================================================================
# SECTION 2: PPO Networks
# ============================================================================

class PPOActor(nn.Module):
    """PPO Actor Network - Outputs stochastic policy parameters"""
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize PPO Actor Network.
        
        Args:
            in_dim: Dimension of input (state space)
            out_dim: Dimension of output (action space)
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(PPOActor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared hidden layers
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        
        # Separate output heads for mean and log_std
        self.mu_layer = nn.Linear(128, out_dim)
        self.log_std_layer = nn.Linear(128, out_dim)
        
        # Initialize output layers with uniform distribution
        init_layer_uniform(self.mu_layer)
        init_layer_uniform(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (mean, std) for the action distribution
        """
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        
        # Mean of the action distribution (bounded by tanh)
        mu = torch.tanh(self.mu_layer(x))
        
        # Log standard deviation (clamped for stability)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mu, std


class Critic(nn.Module):
    """Critic Network - Estimates state value function V(s)"""
    
    def __init__(self, in_dim: int):
        """Initialize Critic Network.
        
        Args:
            in_dim: Dimension of input (state space)
        """
        super(Critic, self).__init__()
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        
        # Initialize output layer
        init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to estimate state value.
        
        Args:
            state: Input state tensor
            
        Returns:
            Estimated value V(s)
        """
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        
        return value


# ============================================================================
# SECTION 3: DDPG Networks
# ============================================================================

class DDPGActor(nn.Module):
    """DDPG Actor Network - Outputs deterministic policy"""
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int,
        init_w: float = 3e-3,
    ):
        """Initialize DDPG Actor Network.
        
        Args:
            in_dim: Dimension of input (state space)
            out_dim: Dimension of output (action space)
            init_w: Initialization range for output layer
        """
        super(DDPGActor, self).__init__()
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_dim)
        
        # Initialize output layer uniformly
        init_layer_uniform(self.out, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate deterministic action.
        
        Args:
            state: Input state tensor
            
        Returns:
            Deterministic action (bounded by tanh)
        """
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = torch.tanh(self.out(x))
        
        return action

    
class DDPGCritic(nn.Module):
    """DDPG Critic Network - Estimates action-value function Q(s,a)"""
    
    def __init__(
        self, 
        in_dim: int, 
        init_w: float = 3e-3,
    ):
        """Initialize DDPG Critic Network.
        
        Args:
            in_dim: Dimension of input (state_dim + action_dim)
            init_w: Initialization range for output layer
        """
        super(DDPGCritic, self).__init__()
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        
        # Initialize output layer uniformly
        init_layer_uniform(self.out, init_w)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass to estimate Q-value.
        
        Args:
            state: Input state tensor
            action: Input action tensor
            
        Returns:
            Estimated Q(s,a) value
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        
        return value


# ============================================================================
# SECTION 4: PPO Helper Functions
# ============================================================================

def ppo_iter(
    epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Yield mini-batches for PPO training.
    
    Args:
        epoch: Number of epochs to iterate
        mini_batch_size: Size of each mini-batch
        states, actions, values, log_probs, returns, advantages: Training data
        
    Yields:
        Mini-batches of training data
    """
    batch_size = states.size(0)
    
    for _ in range(epoch):
        # Generate random indices for mini-batch sampling
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            
            yield (
                states[rand_ids, :],
                actions[rand_ids, :],
                values[rand_ids, :],
                log_probs[rand_ids, :],
                returns[rand_ids, :],
                advantages[rand_ids, :],
            )


# ============================================================================
# SECTION 5: PPO Agent
# ============================================================================

class PPOAgent:
    """PPO Agent for continuous control tasks."""
    
    def __init__(
        self,
        env: gym.Env,
        batch_size: int,
        gamma: float,
        tau: float,
        epsilon: float,
        epoch: int,
        rollout_len: int,
        entropy_weight: float,
    ):
        """Initialize PPO Agent.
        
        Args:
            env: OpenAI Gym environment
            batch_size: Mini-batch size for training
            gamma: Discount factor
            tau: GAE parameter (lambda)
            epsilon: PPO clipping parameter
            epoch: Number of epochs per update
            rollout_len: Number of steps before update
            entropy_weight: Coefficient for entropy regularization
        """
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.rollout_len = rollout_len
        self.entropy_weight = entropy_weight
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.actor = PPOActor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Memory for rollouts
        self.states: List = []
        self.actions: List = []
        self.rewards: List = []
        self.values: List = []
        self.log_probs: List = []
        self.dones: List = []
        
        # Training statistics
        self.total_step = 1
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            mu, std = self.actor(state)
            value = self.critic(state)
        
        if self.is_test:
            # Use mean action during testing
            action = mu
        else:
            # Sample from policy during training
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Store transition
            self.states.append(state.cpu().numpy())
            self.actions.append(action.cpu().numpy())
            self.values.append(value.cpu().numpy())
            self.log_probs.append(log_prob.cpu().numpy())
        
        return action.cpu().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take action in environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        next_state, reward, done, _ = self.env.step(action)
        
        if not self.is_test:
            self.rewards.append(reward)
            self.dones.append(done)
        
        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update actor and critic networks.
        
        Args:
            next_state: Next state after rollout
            
        Returns:
            Tuple of (actor_loss, critic_loss)
        """
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).unsqueeze(-1).to(self.device)
        
        # Compute returns and advantages using GAE
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state_tensor).cpu().detach().numpy()
        
        returns = []
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value * (1 - dones[i])
            else:
                next_val = values[i + 1] * (1 - dones[i])
            
            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * self.tau * (1 - dones[i]) * gae
            
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        returns = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        values = torch.FloatTensor(values).unsqueeze(-1).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        actor_losses = []
        critic_losses = []
        
        for state, action, old_value, old_log_prob, return_, advantage in ppo_iter(
            self.epoch,
            self.batch_size,
            states,
            actions,
            values,
            log_probs,
            returns,
            advantages,
        ):
            # Actor loss (PPO clip objective)
            mu, std = self.actor(state)
            dist = Normal(mu, std)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            ratio = torch.exp(log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            entropy = dist.entropy().sum(dim=-1).mean()
            actor_loss = actor_loss - self.entropy_weight * entropy
            
            # Critic loss
            value = self.critic(state)
            critic_loss = F.mse_loss(value, return_)
            
            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        # Clear memory
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []
        
        return np.mean(actor_losses), np.mean(critic_losses)

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent.
        
        Args:
            num_frames: Total number of frames to train
            plotting_interval: Interval for plotting
        """
        self.is_test = False
        
        state = self.env.reset()
        actor_losses, critic_losses, scores = [], [], []
        score = 0
        
        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            
            state = next_state
            score += reward
            self.total_step += 1
            
            # Update model
            if self.total_step % self.rollout_len == 0:
                actor_loss, critic_loss = self.update_model(next_state)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            # Episode ended
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0
            
            # Plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, actor_losses, critic_losses)
        
        self.env.close()

    def test(self) -> List:
        """Test the agent and record frames.
        
        Returns:
            List of frames for visualization
        """
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        frames = []
        
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            
            state = next_state
            score += reward
        
        print(f"Test Score: {score:.2f}")
        self.env.close()
        
        return frames

    def _plot(
        self,
        frame_idx: int,
        scores: List[float],
        actor_losses: List[float],
        critic_losses: List[float],
    ):
        """Plot training progress.
        
        Args:
            frame_idx: Current frame index
            scores: List of episode scores
            actor_losses: List of actor losses
            critic_losses: List of critic losses
        """
        clear_output(True)
        plt.figure(figsize=(20, 5))
        
        plt.subplot(131)
        plt.title(f"Frame {frame_idx}. Score: {np.mean(scores[-10:]):.2f}")
        plt.plot(scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        
        plt.subplot(132)
        plt.title(f"Actor Loss")
        plt.plot(actor_losses)
        plt.xlabel("Update")
        plt.ylabel("Loss")
        
        plt.subplot(133)
        plt.title(f"Critic Loss")
        plt.plot(critic_losses)
        plt.xlabel("Update")
        plt.ylabel("Loss")
        
        plt.show()


# ============================================================================
# SECTION 6: DDPG Agent
# ============================================================================

class DDPGAgent:
    """DDPG Agent for continuous control tasks."""
    
    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        ou_noise_theta: float,
        ou_noise_sigma: float,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 10000,
    ):
        """Initialize DDPG Agent.
        
        Args:
            env: OpenAI Gym environment
            memory_size: Replay buffer size
            batch_size: Mini-batch size for training
            ou_noise_theta: OU noise theta parameter
            ou_noise_sigma: OU noise sigma parameter
            gamma: Discount factor
            tau: Soft update parameter
            initial_random_steps: Steps of random exploration
        """
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.initial_random_steps = initial_random_steps
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Actor networks
        self.actor = DDPGActor(obs_dim, action_dim).to(self.device)
        self.actor_target = DDPGActor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks
        self.critic = DDPGCritic(obs_dim + action_dim).to(self.device)
        self.critic_target = DDPGCritic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Replay buffer
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        
        # Noise process
        self.noise = OUNoise(action_dim, theta=ou_noise_theta, sigma=ou_noise_sigma)
        
        # Training statistics
        self.transition = []
        self.total_step = 0
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Initial random exploration
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                selected_action = self.actor(state_tensor).cpu().numpy()
        
        # Add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        
        self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take action in environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        next_state, reward, done, _ = self.env.step(action)
        
        if not self.is_test:
            self.transition.extend([reward, next_state, done])
            self.memory.store(*self.transition)
        
        return next_state, reward, done

    def update_model(self) -> Tuple[float, float]:
        """Update actor and critic networks.
        
        Returns:
            Tuple of (actor_loss, critic_loss)
        """
        # Sample batch from memory
        samples = self.memory.sample_batch()
        states = torch.FloatTensor(samples["obs"]).to(self.device)
        actions = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        next_states = torch.FloatTensor(samples["next_obs"]).to(self.device)
        dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        
        # Compute target Q-value
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Critic loss
        q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(q_values, target_q_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss (policy gradient)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._target_soft_update()
        
        return actor_loss.item(), critic_loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent.
        
        Args:
            num_frames: Total number of frames to train
            plotting_interval: Interval for plotting
        """
        self.is_test = False
        
        state = self.env.reset()
        actor_losses, critic_losses, scores = [], [], []
        score = 0
        
        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            
            state = next_state
            score += reward
            self.total_step += 1
            
            # Update model
            if len(self.memory) >= self.batch_size and self.total_step > self.initial_random_steps:
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            # Episode ended
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0
                self.noise.reset()
            
            # Plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, actor_losses, critic_losses)
        
        self.env.close()

    def test(self) -> List:
        """Test the agent and record frames.
        
        Returns:
            List of frames for visualization
        """
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        frames = []
        
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            
            state = next_state
            score += reward
        
        print(f"Test Score: {score:.2f}")
        self.env.close()
        
        return frames

    def _target_soft_update(self):
        """Soft update target networks."""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        actor_losses: List[float], 
        critic_losses: List[float], 
    ):
        """Plot training progress.
        
        Args:
            frame_idx: Current frame index
            scores: List of episode scores
            actor_losses: List of actor losses
            critic_losses: List of critic losses
        """
        clear_output(True)
        plt.figure(figsize=(20, 5))
        
        plt.subplot(131)
        plt.title(f"Frame {frame_idx}. Score: {np.mean(scores[-10:]):.2f}")
        plt.plot(scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        
        plt.subplot(132)
        plt.title(f"Actor Loss")
        plt.plot(actor_losses)
        plt.xlabel("Update")
        plt.ylabel("Loss")
        
        plt.subplot(133)
        plt.title(f"Critic Loss")
        plt.plot(critic_losses)
        plt.xlabel("Update")
        plt.ylabel("Loss")
        
        plt.show()


# ============================================================================
# SECTION 7: Action Normalizer
# ============================================================================

class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate actions from [-1, 1] to [low, high]."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Transform action from [-1, 1] to environment range.
        
        Args:
            action: Action in range [-1, 1]
            
        Returns:
            Action in environment range [low, high]
        """
        low = self.action_space.low
        high = self.action_space.high
        
        # Scale from [-1, 1] to [low, high]
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Transform action from environment range to [-1, 1].
        
        Args:
            action: Action in environment range [low, high]
            
        Returns:
            Action in range [-1, 1]
        """
        low = self.action_space.low
        high = self.action_space.high
        
        # Scale from [low, high] to [-1, 1]
        action = 2.0 * (action - low) / (high - low) - 1.0
        action = np.clip(action, -1.0, 1.0)
        
        return action

