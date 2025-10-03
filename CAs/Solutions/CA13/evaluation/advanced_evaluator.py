"""Advanced evaluation framework for reinforcement learning methods."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy

from environments.grid_world import SimpleGridWorld
from agents.model_free import DQNAgent
from agents.sample_efficient import SampleEfficientAgent, DataAugmentationDQN
from agents.hierarchical import OptionsCriticAgent
from buffers.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from models.world_model import VariationalWorldModel


class AdvancedRLEvaluator:
    """Comprehensive evaluation framework for advanced RL methods."""

    def __init__(
        self,
        environments,
        agents,
        metrics=["reward", "sample_efficiency", "robustness"],
    ):
        self.environments = environments
        self.agents = agents
        self.metrics = metrics
        self.results = {}

        self.num_trials = 5
        self.num_episodes = 300
        self.evaluation_interval = 50

    def evaluate_sample_efficiency(self, agent, env, convergence_threshold=0.8):
        """Measure episodes to convergence."""
        max_rewards = []
        convergence_episodes = []

        for trial in range(self.num_trials):
            episode_rewards = []

            if hasattr(agent, "reset"):
                agent.reset()

            for episode in range(self.num_episodes):
                state = env.reset()
                episode_reward = 0

                for step in range(100):
                    if hasattr(agent, "act"):
                        if "Options" in str(type(agent)):
                            action, _ = agent.act(state)
                        else:
                            action = agent.act(state)
                    else:
                        action = env.action_space.sample()

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward

                    if hasattr(agent, "replay_buffer"):
                        agent.replay_buffer.push(
                            state, action, reward, next_state, done
                        )
                        if len(agent.replay_buffer) > 32:
                            if hasattr(agent, "update"):
                                agent.update(32)

                    if done:
                        break

                    state = next_state

                episode_rewards.append(episode_reward)

                if len(episode_rewards) >= 20:
                    recent_performance = np.mean(episode_rewards[-20:])
                    if recent_performance >= convergence_threshold * np.max(
                        episode_rewards[: max(1, episode - 20)]
                    ):
                        convergence_episodes.append(episode)
                        break

            max_rewards.append(np.max(episode_rewards))
            if not convergence_episodes or len(convergence_episodes) <= trial:
                convergence_episodes.append(self.num_episodes)

        return {
            "convergence_episodes": np.mean(convergence_episodes),
            "convergence_std": np.std(convergence_episodes),
            "max_reward": np.mean(max_rewards),
            "max_reward_std": np.std(max_rewards),
        }

    def evaluate_transfer_capability(self, agent, source_env, target_envs):
        """Evaluate transfer learning capability."""
        source_performance = []
        state = source_env.reset()

        for episode in range(100):  # Limited training
            episode_reward = 0
            for step in range(50):
                action = (
                    agent.act(state)
                    if hasattr(agent, "act")
                    else source_env.action_space.sample()
                )
                next_state, reward, done, _ = source_env.step(action)
                episode_reward += reward

                if hasattr(agent, "replay_buffer"):
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    if len(agent.replay_buffer) > 32 and hasattr(agent, "update"):
                        agent.update(32)

                if done:
                    break
                state = next_state

            source_performance.append(episode_reward)

        transfer_results = {}
        for target_name, target_env in target_envs.items():
            target_rewards = []

            for episode in range(20):  # Quick evaluation
                state = target_env.reset()
                episode_reward = 0

                for step in range(50):
                    action = (
                        agent.act(state)
                        if hasattr(agent, "act")
                        else target_env.action_space.sample()
                    )
                    next_state, reward, done, _ = target_env.step(action)
                    episode_reward += reward

                    if done:
                        break
                    state = next_state

                target_rewards.append(episode_reward)

            transfer_results[target_name] = {
                "mean_reward": np.mean(target_rewards),
                "std_reward": np.std(target_rewards),
            }

        return {
            "source_performance": np.mean(source_performance[-20:]),
            "transfer_results": transfer_results,
        }

    def comprehensive_evaluation(self):
        """Run comprehensive evaluation across all agents and environments."""
        print("🔬 Starting Comprehensive Evaluation...")

        for agent_name, agent in self.agents.items():
            print(f"\n📊 Evaluating {agent_name}...")
            self.results[agent_name] = {}

            if "sample_efficiency" in self.metrics:
                env = (
                    self.environments[0]
                    if self.environments
                    else SimpleGridWorld(size=5)
                )
                efficiency_results = self.evaluate_sample_efficiency(agent, env)
                self.results[agent_name]["sample_efficiency"] = efficiency_results
                print(
                    f"  Sample Efficiency: {efficiency_results['convergence_episodes']:.1f} ± {efficiency_results['convergence_std']:.1f} episodes"
                )

            if "transfer" in self.metrics and len(self.environments) > 1:
                source_env = self.environments[0]
                target_envs = {
                    f"env_{i}": env for i, env in enumerate(self.environments[1:])
                }
                transfer_results = self.evaluate_transfer_capability(
                    agent, source_env, target_envs
                )
                self.results[agent_name]["transfer"] = transfer_results
                print(
                    f"  Transfer Capability: Source performance {transfer_results['source_performance']:.2f}"
                )

        return self.results

    def generate_report(self):
        """Generate comprehensive evaluation report."""
        if not self.results:
            self.comprehensive_evaluation()

        print("\n" + "=" * 60)
        print("🏆 COMPREHENSIVE EVALUATION REPORT")
        print("=" * 60)

        if any("sample_efficiency" in results for results in self.results.values()):
            print("\n📈 Sample Efficiency Ranking:")
            efficiency_scores = []
            for agent_name, results in self.results.items():
                if "sample_efficiency" in results:
                    score = results["sample_efficiency"]["convergence_episodes"]
                    efficiency_scores.append((agent_name, score))

            efficiency_scores.sort(key=lambda x: x[1])  # Lower is better
            for rank, (agent_name, score) in enumerate(efficiency_scores, 1):
                print(f"  {rank}. {agent_name}: {score:.1f} episodes to convergence")

        print("\n🎯 Final Performance Comparison:")
        performance_scores = []
        for agent_name, results in self.results.items():
            if "sample_efficiency" in results:
                score = results["sample_efficiency"]["max_reward"]
                performance_scores.append((agent_name, score))

        performance_scores.sort(key=lambda x: x[1], reverse=True)  # Higher is better
        for rank, (agent_name, score) in enumerate(performance_scores, 1):
            print(f"  {rank}. {agent_name}: {score:.2f} max reward")

        print("\n💡 Method Recommendations:")

        if efficiency_scores:
            best_efficiency = efficiency_scores[0][0]
            print(f"  • Best Sample Efficiency: {best_efficiency}")

        if performance_scores:
            best_performance = performance_scores[0][0]
            print(f"  • Best Final Performance: {best_performance}")

        print("\n🔧 Implementation Guidelines:")
        print("  • Use prioritized replay for sample efficiency")
        print("  • Apply data augmentation for robustness")
        print("  • Consider world models for planning tasks")
        print("  • Employ hierarchical methods for long-horizon problems")
        print("  • Leverage transfer learning for related domains")


class IntegratedAdvancedAgent:
    """Agent integrating multiple advanced RL techniques."""

    def __init__(self, state_dim, action_dim, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        default_config = {
            "use_prioritized_replay": True,
            "use_auxiliary_tasks": True,
            "use_data_augmentation": True,
            "use_world_model": False,
            "use_hierarchical": False,
            "lr": 1e-3,
            "buffer_size": 10000,
        }
        self.config = {**default_config, **(config or {})}

        self._initialize_components()

        self.training_stats = {
            "episode_rewards": [],
            "losses": [],
            "sample_efficiency": [],
            "component_usage": {},
        }

    def _initialize_components(self):
        """Initialize RL components based on configuration."""
        if self.config["use_auxiliary_tasks"]:
            self.network = DataAugmentationDQN(self.state_dim, self.action_dim)
        else:
            self.network = DQNAgent(self.state_dim, self.action_dim).network

        self.target_network = copy.deepcopy(self.network)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config["lr"])

        if self.config["use_prioritized_replay"]:
            self.replay_buffer = PrioritizedReplayBuffer(self.config["buffer_size"])
        else:
            self.replay_buffer = ReplayBuffer(self.config["buffer_size"])

        if self.config["use_world_model"]:
            self.world_model = VariationalWorldModel(self.state_dim, self.action_dim)

        if self.config["use_hierarchical"]:
            self.hierarchical_agent = OptionsCriticAgent(
                self.state_dim, self.action_dim
            )

        self.gamma = 0.99
        self.update_count = 0
        self.target_update_freq = 100

    def act(self, state, epsilon=0.1):
        """Select action using integrated approach."""
        if self.config["use_hierarchical"]:
            action, option = self.hierarchical_agent.act(state)
            self.training_stats["component_usage"]["hierarchical"] = (
                self.training_stats["component_usage"].get("hierarchical", 0) + 1
            )
            return action

        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            if self.config["use_data_augmentation"] and np.random.random() < 0.1:
                state_tensor = self.network.apply_augmentation(state_tensor, "noise")

            q_values = self.network(state_tensor)
            if isinstance(q_values, tuple):
                q_values = q_values[0]  # Extract Q-values from auxiliary network

            return q_values.argmax().item()

    def update(self, batch_size=32):
        """Update agent using integrated advanced techniques."""
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            sample_result = self.replay_buffer.sample(batch_size)
            if sample_result is None:
                return None
            experiences, indices, weights = sample_result
        else:
            batch = self.replay_buffer.sample(batch_size)
            if batch is None:
                return None
            experiences = batch
            weights = torch.ones(batch_size)
            indices = None

        states, actions, rewards, next_states, dones = experiences
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        weights = (
            torch.FloatTensor(weights)
            if not isinstance(weights, torch.Tensor)
            else weights
        )

        if self.config["use_data_augmentation"]:
            aug_type = np.random.choice(["noise", "dropout", "scaling"])
            states = self.network.apply_augmentation(states, aug_type)
            next_states = self.network.apply_augmentation(next_states, aug_type)
            self.training_stats["component_usage"]["augmentation"] = (
                self.training_stats["component_usage"].get("augmentation", 0) + 1
            )

        if self.config["use_auxiliary_tasks"]:
            current_q_values, reward_pred, next_state_pred = self.network(
                states, actions
            )
            current_q_values = current_q_values.gather(
                1, actions.unsqueeze(1)
            ).squeeze()
        else:
            current_q_values = (
                self.network(states).gather(1, actions.unsqueeze(1)).squeeze()
            )

        with torch.no_grad():
            if self.config["use_auxiliary_tasks"] and hasattr(
                self.target_network, "forward"
            ):
                next_q_values = self.target_network(next_states)
                if isinstance(next_q_values, tuple):
                    next_q_values = next_q_values[0]
            else:
                next_q_values = self.target_network(next_states)

            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (~dones))

        td_errors = (current_q_values - target_q_values).detach()
        q_loss = (
            weights * F.mse_loss(current_q_values, target_q_values, reduction="none")
        ).mean()

        total_loss = q_loss

        if self.config["use_auxiliary_tasks"]:
            aux_reward_loss = F.mse_loss(reward_pred.squeeze(), rewards)
            aux_dynamics_loss = F.mse_loss(next_state_pred, next_states)
            total_loss += 0.1 * aux_reward_loss + 0.1 * aux_dynamics_loss
            self.training_stats["component_usage"]["auxiliary"] = (
                self.training_stats["component_usage"].get("auxiliary", 0) + 1
            )

        if self.config["use_world_model"]:
            world_model_loss = self.world_model.compute_loss(
                states, actions, next_states
            )
            total_loss += 0.1 * world_model_loss
            self.training_stats["component_usage"]["world_model"] = (
                self.training_stats["component_usage"].get("world_model", 0) + 1
            )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.numpy())

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.training_stats["losses"].append(total_loss.item())

        return {"total_loss": total_loss.item(), "q_loss": q_loss.item()}
