import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from .config import Config
from .model import QNetwork
from .utils import ReplayBuffer, set_seed

class DQNAgent:
    """DQN Agent."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(4, 2).to(self.device)  # CartPole state_dim=4, action_dim=2
        self.target_net = QNetwork(4, 2).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(config.memory_size)
        self.epsilon = config.epsilon_start
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def optimize_model(self):
        """Perform one step of optimization."""
        if len(self.memory) < self.config.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values

        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_dqn(config: Config):
    """Train DQN agent."""
    set_seed(config.seed)
    env = gym.make(config.env_name)
    agent = DQNAgent(config)

    episode_rewards = []
    for episode in range(config.num_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(config.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
            total_reward += reward
            if done:
                break
        agent.update_epsilon()
        episode_rewards.append(total_reward)
        if episode % config.target_update == 0:
            agent.update_target_net()
        if episode % 50 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-50:])}")
    env.close()
    return episode_rewards