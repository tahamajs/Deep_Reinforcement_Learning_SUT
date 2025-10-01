import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Normal
from ..utils.utils import device


class ContinuousActorNetwork(nn.Module):
    """Continuous policy network using Gaussian distribution"""

    def __init__(self, state_dim, action_dim, hidden_dim=128, action_bound=1.0):
        super(ContinuousActorNetwork, self).__init__()
        self.action_bound = action_bound

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean head
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        
        # Log standard deviation (learnable parameter or network)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize to small values for stability
        self.log_std_head.weight.data.uniform_(-1e-3, 1e-3)
        self.log_std_head.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, state):
        """Forward pass returns mean and std of action distribution"""
        shared_features = self.shared(state)
        
        # Mean of action distribution
        mean = self.mean_head(shared_features)
        mean = torch.tanh(mean) * self.action_bound  # Bound actions
        
        # Standard deviation (always positive)
        log_std = self.log_std_head(shared_features)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Numerical stability
        std = torch.exp(log_std)

        return mean, std

    def sample_action(self, state):
        """Sample action from the policy"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
        
        return action, log_prob, mean, std
    
    def get_log_prob(self, state, action):
        """Get log probability of an action"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return log_prob


class ContinuousCriticNetwork(nn.Module):
    """Continuous value network"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(ContinuousCriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state).squeeze()


class ContinuousREINFORCEAgent:
    """REINFORCE for continuous action spaces"""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, action_bound=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.action_bound = action_bound
        
        # Policy network
        self.policy = ContinuousActorNetwork(state_dim, action_dim, action_bound=action_bound).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Episode storage
        self.episode_log_probs = []
        self.episode_rewards = []

        # Training history
        self.episode_rewards_history = []
        self.policy_losses = []

    def select_action(self, state):
        """Select continuous action"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action, log_prob, mean, std = self.policy.sample_action(state)
        
        self.episode_log_probs.append(log_prob)

        return action.cpu().numpy()[0]

    def store_reward(self, reward):
        """Store reward"""
        self.episode_rewards.append(reward)

    def calculate_returns(self):
        """Calculate discounted returns"""
        returns = []
        discounted_sum = 0

        for reward in reversed(self.episode_rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.FloatTensor(returns).to(device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update_policy(self):
        """Update policy using REINFORCE"""
        if len(self.episode_log_probs) == 0:
            return

        returns = self.calculate_returns()

        # Calculate policy loss
        policy_loss = []
        for log_prob, G_t in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * G_t)

        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.policy_losses.append(policy_loss.item())

        # Clear episode data
        self.episode_log_probs.clear()
        self.episode_rewards.clear()

    def train_episode(self, env, max_steps=1000):
        """Train for one episode"""
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.store_reward(reward)
            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        self.update_policy()
        self.episode_rewards_history.append(total_reward)

        return total_reward, steps

    def evaluate(self, env, num_episodes=10):
        """Evaluate current policy"""
        self.policy.eval()
        rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0

            for _ in range(1000):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    mean, _ = self.policy(state_tensor)
                    action = mean.cpu().numpy()[0]  # Use mean for deterministic evaluation

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

                if done:
                    break

                state = next_state

            rewards.append(total_reward)

        self.policy.train()

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards)
        }


class ContinuousActorCriticAgent:
    """Actor-Critic for continuous actions"""
    
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, 
                 gamma=0.99, action_bound=1.0, entropy_coeff=0.01):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        
        # Networks
        self.actor = ContinuousActorNetwork(state_dim, action_dim, action_bound=action_bound).to(device)
        self.critic = ContinuousCriticNetwork(state_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Training history
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state):
        """Select action and get value estimate"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        action, log_prob, mean, std = self.actor.sample_action(state)
        value = self.critic(state)
        
        # Calculate entropy for exploration bonus
        entropy = -0.5 * (torch.log(2 * np.pi * std**2) + 1).sum()
        
        return action.cpu().numpy()[0], log_prob, value, entropy
    
    def update(self, state, action, reward, next_state, done, log_prob, value, entropy):
        """Update actor and critic"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        # Calculate TD target and error
        with torch.no_grad():
            next_value = self.critic(next_state) if not done else 0
            td_target = reward + self.gamma * next_value
            td_error = td_target - value
        
        # Update critic
        value_pred = self.critic(state)
        critic_loss = F.mse_loss(value_pred, td_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -log_prob * td_error.detach() - self.entropy_coeff * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
    
    def train_episode(self, env, max_steps=1000):
        """Train for one episode"""
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action, log_prob, value, entropy = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.update(state, action, reward, next_state, done, log_prob, value, entropy)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
            
            state = next_state
        
        self.episode_rewards.append(total_reward)
        return total_reward, steps
    
    def evaluate(self, env, num_episodes=10):
        """Evaluate current policy"""
        self.actor.eval()
        self.critic.eval()
        rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            for _ in range(1000):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    mean, _ = self.actor(state_tensor)
                    action = mean.cpu().numpy()[0]
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            rewards.append(total_reward)
        
        self.actor.train()
        self.critic.train()
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards)
        }


class ContinuousControlAnalyzer:
    """Analyze continuous control methods"""
    
    def compare_continuous_methods(self, env_name="Pendulum-v1", num_episodes=300):
        """Compare continuous control methods"""

        print("=" * 70)
        print(f"Continuous Control Methods Comparison - {env_name}")
        print("=" * 70)

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Get action bounds
        if hasattr(env.action_space, 'high'):
        action_bound = float(env.action_space.high[0])
        else:
            action_bound = 1.0
        
        # Create agents
        agents = {
            'REINFORCE': ContinuousREINFORCEAgent(
                state_dim, action_dim, lr=1e-3, action_bound=action_bound
            ),
            'Actor-Critic': ContinuousActorCriticAgent(
                state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, action_bound=action_bound
            )
        }
        
        results = {}
        
        for name, agent in agents.items():
            print(f"\nTraining {name}...")

        for episode in range(num_episodes):
            reward, steps = agent.train_episode(env)

                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(agent.episode_rewards[-20:])
                    print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.1f}")
            
            eval_results = agent.evaluate(env, 20)
            
            results[name] = {
                'agent': agent,
                'eval_results': eval_results,
                'final_performance': np.mean(agent.episode_rewards[-20:])
            }

        env.close()

        self._visualize_continuous_comparison(results)
        
        return results
    
    def _visualize_continuous_comparison(self, results):
        """Visualize continuous control comparison"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        colors = ['blue', 'red']

        # Learning curves
        ax = axes[0, 0]
        for i, (name, data) in enumerate(results.items()):
            agent = data['agent']
            rewards = agent.episode_rewards
            
            if len(rewards) > 10:
                smoothed = pd.Series(rewards).rolling(window=20).mean()
                ax.plot(smoothed, label=name, color=colors[i], linewidth=2)
        
        ax.set_title('Learning Curves - Continuous Control')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Final performance
        ax = axes[0, 1]
        method_names = list(results.keys())
        final_performances = [data['final_performance'] for data in results.values()]
        eval_means = [data['eval_results']['mean_reward'] for data in results.values()]
        eval_stds = [data['eval_results']['std_reward'] for data in results.values()]
        
        x = np.arange(len(method_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, final_performances, width, 
                      label='Training', alpha=0.7, color=colors)
        bars2 = ax.bar(x + width/2, eval_means, width, 
                      yerr=eval_stds, label='Evaluation', alpha=0.7)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Average Reward')
        ax.set_title('Final Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names)
                ax.legend()
            ax.grid(True, alpha=0.3)

        # Policy losses
        ax = axes[1, 0]
        for i, (name, data) in enumerate(results.items()):
            agent = data['agent']
            if hasattr(agent, 'policy_losses') and agent.policy_losses:
                losses = agent.policy_losses
                if len(losses) > 20:
                    smoothed = pd.Series(losses).rolling(window=20).mean()
                    ax.plot(smoothed, label=name, color=colors[i], linewidth=2)
            elif hasattr(agent, 'actor_losses') and agent.actor_losses:
                losses = agent.actor_losses
                if len(losses) > 20:
                    smoothed = pd.Series(losses).rolling(window=20).mean()
                    ax.plot(smoothed, label=name, color=colors[i], linewidth=2)
        
        ax.set_title('Policy/Actor Loss Evolution')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
                ax.legend()
            ax.grid(True, alpha=0.3)

        # Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Continuous Control Summary:\n\n"
        for name, data in results.items():
            final_perf = data['final_performance']
            eval_perf = data['eval_results']['mean_reward']
            eval_std = data['eval_results']['std_reward']
            
            summary_text += f"{name}:\n"
            summary_text += f"  Training: {final_perf:.1f}\n"
            summary_text += f"  Evaluation: {eval_perf:.1f} ± {eval_std:.1f}\n\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        plt.show()

        # Print summary
        print("\n" + "=" * 50)
        print("CONTINUOUS CONTROL SUMMARY")
        print("=" * 50)
        
        for name, data in results.items():
            final_perf = data['final_performance']
            eval_perf = data['eval_results']['mean_reward']
            eval_std = data['eval_results']['std_reward']
            
            print(f"\n{name}:")
            print(f"  Final Training Performance: {final_perf:.2f}")
            print(f"  Evaluation Performance: {eval_perf:.2f} ± {eval_std:.2f}")


# Example usage
if __name__ == "__main__":
    analyzer = ContinuousControlAnalyzer()
    results = analyzer.compare_continuous_methods('Pendulum-v1', num_episodes=300)
