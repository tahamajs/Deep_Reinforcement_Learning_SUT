import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from .config import Config
from .model import QNetwork
from .utils import ReplayBuffer, set_seed

class DQNAgent:
    """DQN Agent.

    Args:
        config: Experiment config.
        state_dim: Size of the observation vector (default 4 for CartPole).
        action_dim: Number of discrete actions (default 2 for CartPole).
    """

    def __init__(self, config: Config, state_dim: int = 4, action_dim: int = 2):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(config.memory_size)
        self.epsilon = config.epsilon_start
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select action using an epsilon-greedy policy.

        Accepts both raw observation arrays and batched tensors (will take first row).
        """
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                state_t = state
            q_values = self.policy_net(state_t)
            return int(q_values.argmax().item())

    def update_epsilon(self) -> None:
        """Decay epsilon according to the config schedule."""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def optimize_model(self) -> None:
        """Perform one optimization step using a sampled batch from replay."""
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

    def update_target_net(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())


def _unpack_reset(result):
    # Support Gym's old and new reset return signatures
    if isinstance(result, tuple):
        return result[0]
    return result


def _unpack_step(result):
    # Support Gym's old and new step return signatures
    if len(result) == 4:
        next_state, reward, done, info = result
        return next_state, reward, done, info
    elif len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        return next_state, reward, terminated or truncated, info
    else:
        raise RuntimeError("Unrecognized env.step() return signature")


def train_dqn(config: Config):
    """Train DQN agent.

    Returns:
        episode_rewards: List of total reward per episode.
    """
    set_seed(config.seed)
    env = gym.make(config.env_name)
    # infer dims from env
    obs = env.reset()
    obs = _unpack_reset(obs)
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    agent = DQNAgent(config, state_dim=state_dim, action_dim=action_dim)

    episode_rewards = []
    for episode in range(config.num_episodes):
        state = env.reset()
        state = _unpack_reset(state)
        total_reward = 0
        for step in range(config.max_steps):
            action = agent.select_action(state)
            step_result = env.step(action)
            next_state, reward, done, _ = _unpack_step(step_result)
            agent.memory.push(state, action, reward, next_state, bool(done))
            agent.optimize_model()
            state = next_state
            total_reward += float(reward)
            if done:
                break
        agent.update_epsilon()
        episode_rewards.append(total_reward)
        if episode % config.target_update == 0:
            agent.update_target_net()
        if episode % 50 == 0:
            # print progress in a friendly, reproducible way
            recent_mean = float(np.mean(episode_rewards[-50:])) if len(episode_rewards) >= 1 else 0.0
            print(f"Episode {episode}, Average Reward (last 50): {recent_mean:.2f}")
    env.close()
    return episode_rewards