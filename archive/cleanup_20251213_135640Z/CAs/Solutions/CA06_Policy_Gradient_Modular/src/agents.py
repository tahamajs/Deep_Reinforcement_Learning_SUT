import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, List

from src.model import PolicyNetwork, ValueNetwork, ContinuousPolicyNetwork # Import networks
from src.config import Config # Import Config

class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = Config.REINFORCE_LR,
        gamma: float = Config.REINFORCE_GAMMA,
        hidden_dim: int = Config.HIDDEN_DIM,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = Config.DEVICE

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def store_transition(self, log_prob: torch.Tensor, reward: float):
        """Store transition for later policy update"""
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update_policy(self) -> float:
        """Update policy using REINFORCE algorithm"""
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize

        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        policy_loss = torch.cat(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear episode data
        self.log_probs = []
        self.rewards = []

        return policy_loss.item()


class REINFORCEBaselineAgent:
    """REINFORCE with Baseline Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_policy: float = Config.REINFORCE_BASELINE_LR_POLICY,
        lr_value: float = Config.REINFORCE_BASELINE_LR_VALUE,
        gamma: float = Config.REINFORCE_BASELINE_GAMMA,
        hidden_dim: int = Config.HIDDEN_DIM,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = Config.DEVICE

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)

        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.states: List[np.ndarray] = []

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def store_transition(
        self, state: np.ndarray, log_prob: torch.Tensor, reward: float
    ):
        """Store transition for later policy update"""
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update_policy(self) -> Tuple[float, float]:
        """Update policy and value function using REINFORCE with baseline"""
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns).to(self.device)

        # Calculate value function targets and advantages
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        values = self.value_net(states).squeeze()

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update value function
        value_loss = F.mse_loss(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)

        policy_loss = torch.cat(policy_loss).sum()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Clear episode data
        self.log_probs = []
        self.rewards = []
        self.states = []

        return policy_loss.item(), value_loss.item()


class ActorCriticAgent:
    """Actor-Critic Agent with TD learning"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = Config.ACTOR_CRITIC_LR_ACTOR,
        lr_critic: float = Config.ACTOR_CRITIC_LR_CRITIC,
        gamma: float = Config.ACTOR_CRITIC_GAMMA,
        hidden_dim: int = Config.HIDDEN_DIM,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = Config.DEVICE

        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: torch.Tensor,
    ) -> Tuple[float, float]:
        """Update actor and critic using TD learning"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Calculate TD target and advantage
        value = self.critic(state)
        next_value = self.critic(next_state) if not done else torch.tensor([[0.0]]).to(self.device)

        td_target = reward + self.gamma * next_value
        advantage = td_target - value

        # Update critic
        critic_loss = F.mse_loss(value, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -log_prob * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = Config.PPO_LR,
        gamma: float = Config.PPO_GAMMA,
        eps_clip: float = Config.PPO_EPS_CLIP,
        k_epochs: int = Config.PPO_K_EPOCHS,
        hidden_dim: int = Config.HIDDEN_DIM,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = Config.DEVICE

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.device) # Separate value network for PPO
        self.policy_old = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()), lr=lr
        )
        self.mse_loss = F.mse_loss # Changed from nn.MSELoss() to F.mse_loss

        self.memory: List[Tuple] = []

    def select_action(
        self, state: np.ndarray
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action and return log prob and state value"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_old(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Use the value network for state value prediction
        state_value = self.value_net(state)

        return action.item(), log_prob, state_value

    def store_transition(self, transition: Tuple):
        """Store transition in memory"""
        self.memory.append(transition)

    def update(self) -> Tuple[float, float]:
        """Update policy using PPO"""
        # Convert memory to tensors
        states = torch.FloatTensor([t[0] for t in self.memory]).to(self.device)
        actions = torch.LongTensor([t[1] for t in self.memory]).to(self.device)
        log_probs_old = torch.stack([t[2] for t in self.memory]).to(self.device)
        rewards = torch.FloatTensor([t[3] for t in self.memory]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in self.memory]).to(self.device)

        # Calculate discounted rewards
        discounted_rewards = []
        reward = 0
        for reward_t, done in zip(reversed(rewards), reversed(dones)):
            if done:
                reward = 0
            reward = reward_t + self.gamma * reward
            discounted_rewards.insert(0, reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        # Update policy for K epochs
        total_policy_loss = 0
        total_value_loss = 0

        for _ in range(self.k_epochs):
            # Get current policy probabilities
            probs = self.policy(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            state_values = self.value_net(states).squeeze() # Use value network

            # Calculate ratios and surrogate losses
            ratios = torch.exp(log_probs - log_probs_old.squeeze()) # .squeeze() to match dimensions
            advantages = discounted_rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.mse_loss(state_values, discounted_rewards)

            # Update
            self.optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        self.memory = []

        return total_policy_loss / self.k_epochs, total_value_loss / self.k_epochs


class ContinuousPPOAgent:
    """PPO Agent for Continuous Action Spaces"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = Config.CONTINUOUS_PPO_LR,
        gamma: float = Config.CONTINUOUS_PPO_GAMMA,
        eps_clip: float = Config.CONTINUOUS_PPO_EPS_CLIP,
        k_epochs: int = Config.CONTINUOUS_PPO_K_EPOCHS,
        hidden_dim: int = Config.HIDDEN_DIM,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = Config.DEVICE

        self.policy = ContinuousPolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old = ContinuousPolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory: List[Tuple] = []

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Select action from continuous policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob = self.policy.sample(state)
        return action.detach().cpu().numpy().flatten(), log_prob

    def store_transition(self, transition: Tuple):
        """Store transition in memory"""
        self.memory.append(transition)

    def update(self) -> float:
        """Update policy using PPO for continuous actions"""
        # Convert memory to tensors
        states = torch.FloatTensor([t[0] for t in self.memory]).to(self.device)
        actions = torch.FloatTensor([t[1] for t in self.memory]).to(self.device)
        log_probs_old = torch.stack([t[2] for t in self.memory]).to(self.device)
        rewards = torch.FloatTensor([t[3] for t in self.memory]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in self.memory]).to(self.device)

        # Calculate discounted rewards
        discounted_rewards = []
        reward = 0
        for reward_t, done in zip(reversed(rewards), reversed(dones)):
            if done:
                reward = 0
            reward = reward_t + self.gamma * reward
            discounted_rewards.insert(0, reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        # Update policy for K epochs
        total_loss = 0

        for _ in range(self.k_epochs):
            # Get current policy distribution
            mean, log_std = self.policy(states)
            std = log_std.exp()
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

            # Calculate ratios and surrogate losses
            ratios = torch.exp(log_probs - log_probs_old)
            advantages = discounted_rewards.unsqueeze(
                -1
            )  # Add dimension for broadcasting

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = -torch.min(surr1, surr2).mean()

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        self.memory = []

        return total_loss / self.k_epochs
