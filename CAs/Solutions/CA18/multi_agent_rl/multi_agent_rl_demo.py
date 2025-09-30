import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_agent_rl.multi_agent_rl import MADDPGAgent, MultiAgentEnvironment
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_multi_agent_environment():
    """Create a cooperative multi-agent environment for demonstration"""

    class CooperativeMultiAgentEnv:
        def __init__(self, n_agents=2, obs_dim=4, action_dim=2):
            self.n_agents = n_agents
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.max_steps = 100

            # Environment dynamics
            self.target_position = np.random.uniform(-1, 1, 2)
            self.agent_positions = None
            self.agent_velocities = None

            self.reset()

        def reset(self):
            # Initialize agent positions randomly
            self.agent_positions = np.random.uniform(-0.5, 0.5, (self.n_agents, 2))
            self.agent_velocities = np.zeros((self.n_agents, 2))
            self.steps = 0

            return self.get_observations()

        def get_observations(self):
            """Get observations for all agents"""
            observations = []

            for i in range(self.n_agents):
                # Agent sees its own position, velocity, target, and other agents
                obs = np.concatenate(
                    [
                        self.agent_positions[i],
                        self.agent_velocities[i],
                        self.target_position,
                        self.agent_positions[
                            (i + 1) % self.n_agents
                        ],  # Other agent position
                    ]
                )
                observations.append(obs)

            return np.array(observations)

        def step(self, actions):
            actions = np.array(actions)
            actions = np.clip(actions, -1, 1)

            # Update agent dynamics
            for i in range(self.n_agents):
                # Action affects velocity
                self.agent_velocities[i] += actions[i] * 0.1
                # Apply friction
                self.agent_velocities[i] *= 0.95
                # Update position
                self.agent_positions[i] += self.agent_velocities[i] * 0.1

                # Keep agents in bounds
                self.agent_positions[i] = np.clip(self.agent_positions[i], -1, 1)

            # Compute rewards
            rewards = []
            for i in range(self.n_agents):
                # Distance to target
                dist_to_target = np.linalg.norm(
                    self.agent_positions[i] - self.target_position
                )

                # Cooperation bonus: closer agents get bonus
                min_dist_to_others = min(
                    [
                        np.linalg.norm(
                            self.agent_positions[i] - self.agent_positions[j]
                        )
                        for j in range(self.n_agents)
                        if j != i
                    ]
                )

                # Reward = -distance + cooperation_bonus
                reward = -dist_to_target - 0.1 * min_dist_to_others
                rewards.append(reward)

            self.steps += 1
            done = self.steps >= self.max_steps

            return self.get_observations(), np.array(rewards), done, {}

    return CooperativeMultiAgentEnv()


def train_maddpg_agents(env, n_episodes=200):
    """Train MADDPG agents in the multi-agent environment"""

    print("🎭 Training MADDPG Agents")

    # Create MADDPG agents
    agents = []
    for i in range(env.n_agents):
        agent = MADDPGAgent(
            agent_idx=i,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            n_agents=env.n_agents,
            hidden_dim=64,
            use_attention=True,
            use_communication=True,
        )
        agents.append(agent)

    episode_rewards = []
    attention_weights_history = []

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        episode_attention = []

        while not done:
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
                action_tensor, attention_weights = agent.act(obs_tensor, explore=True)
                action = action_tensor.cpu().numpy()[0]
                actions.append(action)

                if attention_weights is not None:
                    episode_attention.append(attention_weights.cpu().numpy())

            # Environment step
            next_obs, rewards, done, _ = env.step(actions)

            # Store experiences
            for i, agent in enumerate(agents):
                agent.store_experience(
                    obs[i], actions[i], rewards[i], next_obs[i], done
                )

            # Update agents
            if episode > 10:  # Start updating after some exploration
                for agent in enumerate(agents):
                    agent.update()

            obs = next_obs
            episode_reward += np.mean(rewards)

        episode_rewards.append(episode_reward)
        if episode_attention:
            attention_weights_history.append(np.mean(episode_attention, axis=0))

        if episode % 50 == 0:
            print(f"Episode {episode}: Avg Reward = {episode_reward:.2f}")

    return agents, episode_rewards, attention_weights_history


def demonstrate_attention_mechanism(agents, env):
    """Demonstrate the attention mechanism in multi-agent coordination"""

    print("🔍 Analyzing Attention Mechanism")

    obs = env.reset()
    attention_patterns = []

    for step in range(10):
        actions = []
        step_attention = []

        for i, agent in enumerate(agents):
            obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
            action_tensor, attention_weights = agent.act(obs_tensor, explore=False)
            action = action_tensor.cpu().numpy()[0]
            actions.append(action)

            if attention_weights is not None:
                step_attention.append(attention_weights.cpu().numpy().flatten())

        next_obs, rewards, done, _ = env.step(actions)
        obs = next_obs

        if step_attention:
            attention_patterns.append(np.array(step_attention))

    return attention_patterns


def evaluate_multi_agent_performance(agents, env, n_episodes=10):
    """Evaluate trained multi-agent system"""

    print("📊 Evaluating Multi-Agent Performance")

    evaluation_rewards = []

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            actions = []
            for i, agent in enumerate(agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
                action_tensor, _ = agent.act(obs_tensor, explore=False)
                action = action_tensor.cpu().numpy()[0]
                actions.append(action)

            next_obs, rewards, done, _ = env.step(actions)
            obs = next_obs
            episode_reward += np.mean(rewards)

        evaluation_rewards.append(episode_reward)

    return evaluation_rewards
