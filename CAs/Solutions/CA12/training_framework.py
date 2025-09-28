# Multi-Agent Training Framework and Evaluation
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from game_theory import MultiAgentEnvironment


class MultiAgentTrainingOrchestrator:
    """Comprehensive training orchestrator for multi-agent systems."""

    def __init__(self, agents, environment, config=None):
        self.agents = agents  # List of agent instances
        self.environment = environment
        self.n_agents = len(agents)

        # Default configuration
        self.config = {
            "n_episodes": 1000,
            "max_steps": 200,
            "update_frequency": 10,
            "eval_frequency": 50,
            "save_frequency": 100,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.01,  # Soft update parameter
            "lr": 1e-3,
            "buffer_size": 100000,
            "warmup_steps": 1000,
        }
        if config:
            self.config.update(config)

        # Training components
        self.replay_buffers = [
            ReplayBuffer(self.config["buffer_size"]) for _ in range(self.n_agents)
        ]
        self.optimizers = [
            torch.optim.Adam(agent.parameters(), lr=self.config["lr"])
            for agent in self.agents
        ]

        # Training metrics
        self.training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "agent_rewards": [[] for _ in range(self.n_agents)],
            "losses": [[] for _ in range(self.n_agents)],
            "eval_scores": [],
        }

        # Best models tracking
        self.best_models = [agent.state_dict() for agent in self.agents]
        self.best_eval_score = -float("inf")

    def train_episode(self, episode_idx):
        """Train for one episode."""
        # Reset environment
        obs = self.environment.reset()
        episode_rewards = [0] * self.n_agents
        episode_steps = 0

        # Episode loop
        done = False
        while not done and episode_steps < self.config["max_steps"]:
            # Get actions from all agents
            actions = []
            action_log_probs = []

            for i, agent in enumerate(self.agents):
                action, log_prob = agent.select_action(obs[i])
                actions.append(action)
                action_log_probs.append(log_prob)

            # Execute actions in environment
            next_obs, rewards, done, info = self.environment.step(actions)

            # Store transitions
            for i in range(self.n_agents):
                transition = {
                    "obs": obs[i],
                    "action": actions[i],
                    "reward": rewards[i],
                    "next_obs": next_obs[i],
                    "done": done,
                    "log_prob": action_log_probs[i],
                }
                self.replay_buffers[i].push(transition)
                episode_rewards[i] += rewards[i]

            # Update agents
            if (
                episode_idx > self.config["warmup_steps"]
                and episode_steps % self.config["update_frequency"] == 0
            ):
                for i, agent in enumerate(self.agents):
                    loss = self.update_agent(i)
                    self.training_metrics["losses"][i].append(loss)

            obs = next_obs
            episode_steps += 1

        # Record episode metrics
        self.training_metrics["episode_rewards"].append(sum(episode_rewards))
        self.training_metrics["episode_lengths"].append(episode_steps)
        for i in range(self.n_agents):
            self.training_metrics["agent_rewards"][i].append(episode_rewards[i])

        return episode_rewards, episode_steps

    def update_agent(self, agent_idx):
        """Update a specific agent."""
        if len(self.replay_buffers[agent_idx]) < self.config["batch_size"]:
            return 0.0

        # Sample batch
        batch = self.replay_buffers[agent_idx].sample(self.config["batch_size"])

        # Update agent
        loss = self.agents[agent_idx].update(batch)

        return loss

    def evaluate_agents(self, n_eval_episodes=10):
        """Evaluate current agents."""
        eval_rewards = []

        for _ in range(n_eval_episodes):
            obs = self.environment.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < self.config["max_steps"]:
                # Get actions (deterministic for evaluation)
                actions = []
                for agent in self.agents:
                    action = agent.select_action(obs, deterministic=True)
                    actions.append(action)

                next_obs, rewards, done, _ = self.environment.step(actions)
                episode_reward += sum(rewards)
                obs = next_obs
                steps += 1

            eval_rewards.append(episode_reward)

        avg_eval_score = np.mean(eval_rewards)
        self.training_metrics["eval_scores"].append(avg_eval_score)

        # Save best models
        if avg_eval_score > self.best_eval_score:
            self.best_eval_score = avg_eval_score
            self.best_models = [agent.state_dict() for agent in self.agents]
            print(f"New best evaluation score: {avg_eval_score:.2f}")

        return avg_eval_score

    def train(self, n_episodes=None):
        """Main training loop."""
        if n_episodes is None:
            n_episodes = self.config["n_episodes"]

        print("🚀 Starting Multi-Agent Training")
        print(f"Training for {n_episodes} episodes with {self.n_agents} agents")

        for episode in range(n_episodes):
            # Train episode
            episode_rewards, episode_steps = self.train_episode(episode)

            # Periodic evaluation
            if episode % self.config["eval_frequency"] == 0:
                eval_score = self.evaluate_agents()
                print(
                    f"Episode {episode}: Total Reward = {sum(episode_rewards):.2f}, "
                    f"Steps = {episode_steps}, Eval Score = {eval_score:.2f}"
                )
            else:
                print(
                    f"Episode {episode}: Total Reward = {sum(episode_rewards):.2f}, Steps = {episode_steps}"
                )

            # Periodic saving
            if episode % self.config["save_frequency"] == 0:
                self.save_models(f"checkpoint_episode_{episode}")

        print("✅ Training completed!")
        return self.training_metrics

    def save_models(self, filename):
        """Save agent models."""
        checkpoint = {
            "agents": [agent.state_dict() for agent in self.agents],
            "config": self.config,
            "metrics": self.training_metrics,
            "best_models": self.best_models,
            "best_eval_score": self.best_eval_score,
        }
        torch.save(checkpoint, f"{filename}.pth")
        print(f"Models saved to {filename}.pth")

    def load_models(self, filename):
        """Load agent models."""
        checkpoint = torch.load(f"{filename}.pth")
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(checkpoint["agents"][i])
        print(f"Models loaded from {filename}.pth")


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for multi-agent systems."""

    def __init__(self, agents, environments):
        self.agents = agents
        self.environments = (
            environments if isinstance(environments, list) else [environments]
        )

        self.evaluation_metrics = {
            "cooperation_metrics": [],
            "performance_metrics": [],
            "robustness_metrics": [],
            "scalability_metrics": [],
        }

    def evaluate_cooperation(self, n_episodes=50):
        """Evaluate cooperation quality."""
        print("🤝 Evaluating Cooperation")

        cooperation_scores = []

        for env in self.environments:
            env_scores = []

            for _ in range(n_episodes):
                obs = env.reset()
                episode_cooperation = 0
                done = False
                steps = 0

                while not done and steps < 200:
                    actions = [
                        agent.select_action(obs[i], deterministic=True)
                        for i, agent in enumerate(self.agents)
                    ]
                    next_obs, rewards, done, _ = env.step(actions)

                    # Cooperation score based on joint reward vs individual rewards
                    joint_reward = sum(rewards)
                    individual_rewards = rewards

                    # Simple cooperation metric: joint reward bonus
                    cooperation_bonus = max(
                        0, joint_reward - sum(individual_rewards) * 0.8
                    )
                    episode_cooperation += cooperation_bonus

                    obs = next_obs
                    steps += 1

                env_scores.append(episode_cooperation / steps)

            cooperation_scores.append(np.mean(env_scores))

        avg_cooperation = np.mean(cooperation_scores)
        self.evaluation_metrics["cooperation_metrics"].append(avg_cooperation)

        print(f"Average cooperation score: {avg_cooperation:.3f}")
        return avg_cooperation

    def evaluate_performance(self, n_episodes=50):
        """Evaluate performance across different scenarios."""
        print("📊 Evaluating Performance")

        performance_scores = []

        for env_idx, env in enumerate(self.environments):
            env_scores = []

            for _ in range(n_episodes):
                obs = env.reset()
                episode_reward = 0
                done = False
                steps = 0

                while not done and steps < 200:
                    actions = [
                        agent.select_action(obs[i], deterministic=True)
                        for i, agent in enumerate(self.agents)
                    ]
                    next_obs, rewards, done, _ = env.step(actions)

                    episode_reward += sum(rewards)
                    obs = next_obs
                    steps += 1

                env_scores.append(episode_reward)

            avg_env_score = np.mean(env_scores)
            performance_scores.append(avg_env_score)
            print(f"Environment {env_idx}: Average score = {avg_env_score:.3f}")

        overall_performance = np.mean(performance_scores)
        self.evaluation_metrics["performance_metrics"].append(overall_performance)

        print(f"Overall performance score: {overall_performance:.3f}")
        return overall_performance

    def evaluate_robustness(self, noise_levels=[0.0, 0.1, 0.2, 0.5]):
        """Evaluate robustness to environmental noise."""
        print("🛡️  Evaluating Robustness")

        robustness_scores = []

        for noise_level in noise_levels:
            # Add noise to environment
            noisy_env = self.add_environment_noise(self.environments[0], noise_level)

            scores = []
            for _ in range(20):  # Fewer episodes for robustness testing
                obs = noisy_env.reset()
                episode_reward = 0
                done = False
                steps = 0

                while not done and steps < 200:
                    actions = [
                        agent.select_action(obs[i], deterministic=True)
                        for i, agent in enumerate(self.agents)
                    ]
                    next_obs, rewards, done, _ = noisy_env.step(actions)

                    episode_reward += sum(rewards)
                    obs = next_obs
                    steps += 1

                scores.append(episode_reward)

            avg_score = np.mean(scores)
            robustness_scores.append(avg_score)
            print(f"Noise level {noise_level}: Average score = {avg_score:.3f}")

        # Robustness metric: performance degradation with noise
        baseline_score = robustness_scores[0]
        max_noise_score = robustness_scores[-1]
        robustness_metric = (
            max_noise_score / baseline_score if baseline_score > 0 else 0
        )

        self.evaluation_metrics["robustness_metrics"].append(robustness_metric)

        print(f"Robustness metric: {robustness_metric:.3f}")
        return robustness_metric

    def evaluate_scalability(self, agent_counts=[2, 4, 6, 8]):
        """Evaluate scalability with different numbers of agents."""
        print("📈 Evaluating Scalability")

        scalability_scores = []

        for n_agents in agent_counts:
            # Create scaled environment and agents
            scaled_env = self.scale_environment(self.environments[0], n_agents)
            scaled_agents = (
                self.agents[:n_agents]
                if n_agents <= len(self.agents)
                else self.agents * (n_agents // len(self.agents))
                + self.agents[: n_agents % len(self.agents)]
            )

            scores = []
            for _ in range(20):
                obs = scaled_env.reset()
                episode_reward = 0
                done = False
                steps = 0

                while not done and steps < 200:
                    actions = [
                        agent.select_action(obs[i], deterministic=True)
                        for i, agent in enumerate(scaled_agents)
                    ]
                    next_obs, rewards, done, _ = scaled_env.step(actions)

                    episode_reward += sum(rewards)
                    obs = next_obs
                    steps += 1

                scores.append(episode_reward)

            avg_score = np.mean(scores)
            scalability_scores.append(avg_score)
            print(f"{n_agents} agents: Average score = {avg_score:.3f}")

        # Scalability metric: how performance scales with agent count
        scalability_metric = np.polyfit(agent_counts, scalability_scores, 1)[
            0
        ]  # Linear trend

        self.evaluation_metrics["scalability_metrics"].append(scalability_metric)

        print(f"Scalability metric (slope): {scalability_metric:.3f}")
        return scalability_metric

    def add_environment_noise(self, env, noise_level):
        """Add noise to environment observations."""

        class NoisyEnvironment:
            def __init__(self, base_env, noise_level):
                self.base_env = base_env
                self.noise_level = noise_level

            def reset(self):
                return self.base_env.reset()

            def step(self, actions):
                next_obs, rewards, done, info = self.base_env.step(actions)
                # Add noise to observations
                noisy_obs = next_obs + torch.randn_like(next_obs) * self.noise_level
                return noisy_obs, rewards, done, info

        return NoisyEnvironment(env, noise_level)

    def scale_environment(self, env, n_agents):
        """Scale environment to different number of agents."""

        # This is a simplified scaling - in practice, you'd need environment-specific scaling
        class ScaledEnvironment:
            def __init__(self, base_env, n_agents):
                self.base_env = base_env
                self.n_agents = n_agents

            def reset(self):
                obs = self.base_env.reset()
                # Repeat or truncate observations to match agent count
                if len(obs) < self.n_agents:
                    obs = obs.repeat(self.n_agents // len(obs) + 1, 1)[: self.n_agents]
                else:
                    obs = obs[: self.n_agents]
                return obs

            def step(self, actions):
                # Truncate actions if too many
                actions = actions[: len(self.base_env.reset())]
                next_obs, rewards, done, info = self.base_env.step(actions)

                # Scale rewards and observations
                if len(next_obs) < self.n_agents:
                    next_obs = next_obs.repeat(self.n_agents // len(next_obs) + 1, 1)[
                        : self.n_agents
                    ]
                    rewards = rewards.repeat(self.n_agents // len(rewards) + 1)[
                        : self.n_agents
                    ]
                else:
                    next_obs = next_obs[: self.n_agents]
                    rewards = rewards[: self.n_agents]

                return next_obs, rewards, done, info

        return ScaledEnvironment(env, n_agents)

    def run_comprehensive_evaluation(self):
        """Run all evaluation metrics."""
        print("🔬 Running Comprehensive Evaluation")

        cooperation_score = self.evaluate_cooperation()
        performance_score = self.evaluate_performance()
        robustness_score = self.evaluate_robustness()
        scalability_score = self.evaluate_scalability()

        summary = {
            "cooperation": cooperation_score,
            "performance": performance_score,
            "robustness": robustness_score,
            "scalability": scalability_score,
            "overall_score": np.mean(
                [
                    cooperation_score,
                    performance_score,
                    robustness_score,
                    scalability_score,
                ]
            ),
        }

        print("\n📋 Evaluation Summary:")
        for metric, score in summary.items():
            print(f"  {metric.capitalize()}: {score:.3f}")

        return summary


# Utility classes
class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        # Convert to tensors
        obs = torch.stack([t["obs"] for t in batch])
        actions = torch.stack([t["action"] for t in batch])
        rewards = torch.stack([t["reward"] for t in batch])
        next_obs = torch.stack([t["next_obs"] for t in batch])
        dones = torch.stack([t["done"] for t in batch])

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }

    def __len__(self):
        return len(self.buffer)


# Demonstration functions
def demonstrate_training_orchestrator():
    """Demonstrate training orchestrator."""
    print("🎯 Training Orchestrator Demo")

    # Create simple agents and environment
    class SimpleAgent(nn.Module):
        def __init__(self, obs_dim=10, action_dim=4):
            super().__init__()
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.policy = nn.Sequential(nn.Linear(obs_dim, action_dim), nn.Tanh())

        def select_action(self, obs, deterministic=False):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = self.policy(obs_tensor)
            if deterministic:
                return action
            else:
                # Add noise for stochasticity
                noise = torch.randn_like(action) * 0.1
                return action + noise, torch.tensor(0.0)  # Dummy log_prob

        def update(self, batch):
            # Simple policy gradient update
            return torch.tensor(0.1)  # Dummy loss

    agents = [SimpleAgent() for _ in range(3)]
    env = MultiAgentEnvironment(n_agents=3, state_dim=10, action_dim=4)

    orchestrator = MultiAgentTrainingOrchestrator(agents, env, {"n_episodes": 10})

    # Run short training
    metrics = orchestrator.train(n_episodes=5)

    print(f"Training completed. Episodes: {len(metrics['episode_rewards'])}")
    print(f"Average episode reward: {np.mean(metrics['episode_rewards']):.3f}")

    return orchestrator


def demonstrate_comprehensive_evaluation():
    """Demonstrate comprehensive evaluation."""
    print("\n🔬 Comprehensive Evaluation Demo")

    # Create simple setup
    class SimpleAgent(nn.Module):
        def __init__(self):
            super().__init__()

        def select_action(self, obs, deterministic=False):
            return torch.zeros(2)  # Return zero vector for 2D action space

    agents = [SimpleAgent() for _ in range(2)]
    env = MultiAgentEnvironment(n_agents=2, state_dim=4, action_dim=2)

    evaluator = ComprehensiveEvaluator(agents, [env])

    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()

    print(f"Evaluation completed. Overall score: {results['overall_score']:.3f}")

    return evaluator


# Run demonstrations
if __name__ == "__main__":
    print("🎓 Multi-Agent Training Framework and Evaluation")
    training_demo = demonstrate_training_orchestrator()
    eval_demo = demonstrate_comprehensive_evaluation()
    print("\n🚀 Training framework and evaluation ready!")
    print("✅ Multi-agent training orchestrator and comprehensive evaluation implemented!")
