"""
Experiment configurations and runners for CA5 Advanced DQN Methods
"""

import os
import json
import time
from typing import Dict, Any, List
import torch
import numpy as np
from datetime import datetime


class ExperimentConfig:
    """Configuration class for experiments"""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save(self, filepath: str):
        """Save configuration to file"""
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from file"""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls(config_dict)


class ExperimentRunner:
    """Run experiments with different configurations"""

    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.results = {}

    def run_experiment(
        self, config: ExperimentConfig, agent_class, env_name: str = "CartPole-v1"
    ) -> Dict[str, Any]:
        """Run a single experiment"""

        print(f"Starting experiment: {config.config.get('name', 'Unnamed')}")
        print(f"Timestamp: {config.timestamp}")

        # Create experiment directory
        exp_dir = os.path.join(self.base_dir, f"exp_{config.timestamp}")
        os.makedirs(exp_dir, exist_ok=True)

        # Save configuration
        config.save(os.path.join(exp_dir, "config.json"))

        # Initialize environment and agent
        import gym

        env = gym.make(env_name)

        agent_config = config.config.get("agent", {})
        agent = agent_class(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            **agent_config,
        )

        # Training parameters
        training_config = config.config.get("training", {})
        num_episodes = training_config.get("num_episodes", 1000)

        # Training loop
        start_time = time.time()
        training_results = self._train_agent(agent, env, num_episodes)
        training_time = time.time() - start_time

        # Evaluation
        eval_results = self._evaluate_agent(agent, env)

        # Compile results
        results = {
            "config": config.config,
            "timestamp": config.timestamp,
            "training_time": training_time,
            "training_results": training_results,
            "evaluation_results": eval_results,
        }

        # Save results
        results_file = os.path.join(exp_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        self.results[config.timestamp] = results

        env.close()
        print(f"Experiment completed in {training_time:.2f} seconds")
        return results

    def _train_agent(self, agent, env, num_episodes: int) -> Dict[str, Any]:
        """Train agent and return training metrics"""

        episode_rewards = []
        episode_lengths = []
        losses = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                agent.replay_buffer.push(state, action, reward, next_state, done)

                if len(agent.replay_buffer) > agent.batch_size:
                    loss = agent.update()
                    losses.append(loss)

                episode_reward += reward
                episode_length += 1
                state = next_state

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Update epsilon
            if hasattr(agent, "epsilon"):
                agent.epsilon = max(
                    agent.epsilon_end, agent.epsilon * agent.epsilon_decay
                )

            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "losses": losses,
            "final_avg_reward": np.mean(episode_rewards[-100:]),
        }

    def _evaluate_agent(self, agent, env, num_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate trained agent"""

        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state, epsilon=0.0)  # No exploration
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state

            episode_rewards.append(episode_reward)

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "episode_rewards": episode_rewards,
        }

    def run_comparison_experiment(
        self,
        configs: List[ExperimentConfig],
        agent_classes: List[Any],
        env_name: str = "CartPole-v1",
    ) -> Dict[str, Any]:
        """Run multiple experiments for comparison"""

        print(f"Running comparison experiment with {len(configs)} configurations")

        comparison_results = {}

        for i, (config, agent_class) in enumerate(zip(configs, agent_classes)):
            print(f"\n--- Experiment {i+1}/{len(configs)} ---")
            results = self.run_experiment(config, agent_class, env_name)
            comparison_results[f"exp_{i+1}"] = results

        # Save comparison results
        comparison_file = os.path.join(self.base_dir, "comparison_results.json")
        with open(comparison_file, "w") as f:
            json.dump(comparison_results, f, indent=2)

        return comparison_results


# Predefined experiment configurations
def get_dqn_configs() -> List[ExperimentConfig]:
    """Get predefined DQN experiment configurations"""

    configs = [
        ExperimentConfig(
            {
                "name": "Vanilla DQN",
                "agent": {
                    "lr": 1e-3,
                    "gamma": 0.99,
                    "epsilon_start": 1.0,
                    "epsilon_end": 0.01,
                    "epsilon_decay": 0.995,
                    "buffer_size": 10000,
                    "batch_size": 32,
                    "target_update_freq": 100,
                },
                "training": {"num_episodes": 1000},
            }
        ),
        ExperimentConfig(
            {
                "name": "Double DQN",
                "agent": {
                    "lr": 1e-3,
                    "gamma": 0.99,
                    "epsilon_start": 1.0,
                    "epsilon_end": 0.01,
                    "epsilon_decay": 0.995,
                    "buffer_size": 10000,
                    "batch_size": 32,
                    "target_update_freq": 100,
                },
                "training": {"num_episodes": 1000},
            }
        ),
        ExperimentConfig(
            {
                "name": "Dueling DQN",
                "agent": {
                    "lr": 1e-3,
                    "gamma": 0.99,
                    "epsilon_start": 1.0,
                    "epsilon_end": 0.01,
                    "epsilon_decay": 0.995,
                    "buffer_size": 10000,
                    "batch_size": 32,
                    "target_update_freq": 100,
                },
                "training": {"num_episodes": 1000},
            }
        ),
        ExperimentConfig(
            {
                "name": "Prioritized DQN",
                "agent": {
                    "lr": 1e-3,
                    "gamma": 0.99,
                    "epsilon_start": 1.0,
                    "epsilon_end": 0.01,
                    "epsilon_decay": 0.995,
                    "buffer_size": 10000,
                    "batch_size": 32,
                    "target_update_freq": 100,
                    "alpha": 0.6,
                    "beta": 0.4,
                },
                "training": {"num_episodes": 1000},
            }
        ),
    ]

    return configs


if __name__ == "__main__":
    print("Experiment module loaded successfully!")


