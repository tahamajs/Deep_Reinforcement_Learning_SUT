"""
Advanced RL Concepts and Implementations
مفاهیم پیشرفته یادگیری تقویتی

This module contains advanced RL concepts including:
- Transfer Learning and Domain Adaptation
- Curriculum Learning
- Multi-Task Learning
- Continual Learning
- Meta-Learning Applications
- Causal Inference in RL
- Quantum Machine Learning
- Neurosymbolic Reasoning
- Federated Learning
- Explainable AI
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import copy
from typing import Dict, List, Tuple, Any, Optional
import math
import random
from collections import deque
import pickle
import os


class TransferLearningAgent:
    """Transfer Learning Agent for domain adaptation."""

    def __init__(self, source_state_dim, target_state_dim, action_dim, lr=3e-4):
        self.source_state_dim = source_state_dim
        self.target_state_dim = target_state_dim
        self.action_dim = action_dim

        # Source domain policy (frozen)
        self.source_policy = nn.Sequential(
            nn.Linear(source_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )

        # Target domain adaptation layers
        self.domain_adapter = nn.Sequential(
            nn.Linear(target_state_dim, source_state_dim),
            nn.ReLU(),
            nn.Linear(source_state_dim, source_state_dim),
        )

        # Fine-tuning layers
        self.fine_tuning_layers = nn.ModuleList(
            [
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Softmax(dim=-1),
            ]
        )

        # Freeze source policy
        for param in self.source_policy.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(
            list(self.domain_adapter.parameters())
            + list(self.fine_tuning_layers.parameters()),
            lr=lr,
        )

        self.transfer_losses = []
        self.adaptation_losses = []

    def adapt_to_target_domain(self, target_data, num_epochs=100):
        """Adapt to target domain using transfer learning."""
        for epoch in range(num_epochs):
            total_loss = 0

            for batch in target_data:
                states, actions, rewards = batch

                # Domain adaptation
                adapted_states = self.domain_adapter(states)

                # Get source policy output
                with torch.no_grad():
                    source_output = self.source_policy(adapted_states)

                # Fine-tune for target domain
                target_output = self.fine_tuning_layers[0](adapted_states)
                for layer in self.fine_tuning_layers[1:]:
                    target_output = layer(target_output)

                # Transfer loss (knowledge distillation)
                transfer_loss = F.kl_div(
                    torch.log(target_output + 1e-8),
                    source_output,
                    reduction="batchmean",
                )

                # Policy loss
                log_probs = torch.log(
                    target_output.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8
                )
                policy_loss = -log_probs.mean()

                # Total loss
                loss = transfer_loss + policy_loss
                total_loss += loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.transfer_losses.append(total_loss / len(target_data))

            if epoch % 20 == 0:
                print(
                    f"Transfer Learning Epoch {epoch}: Loss = {total_loss / len(target_data):.4f}"
                )

    def get_action(self, state):
        """Get action using adapted policy."""
        with torch.no_grad():
            # Adapt state to source domain
            adapted_state = self.domain_adapter(state.unsqueeze(0))

            # Get policy output
            output = self.fine_tuning_layers[0](adapted_state)
            for layer in self.fine_tuning_layers[1:]:
                output = layer(output)

            action_dist = Categorical(output)
            action = action_dist.sample()
            return action.item()


class CurriculumLearningAgent:
    """Curriculum Learning Agent with progressive difficulty."""

    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )

        # Curriculum scheduler
        self.curriculum_levels = [
            {"difficulty": 0.1, "success_threshold": 0.8},
            {"difficulty": 0.3, "success_threshold": 0.7},
            {"difficulty": 0.5, "success_threshold": 0.6},
            {"difficulty": 0.7, "success_threshold": 0.5},
            {"difficulty": 1.0, "success_threshold": 0.4},
        ]

        self.current_level = 0
        self.level_performance = deque(maxlen=100)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def update_curriculum(self, performance):
        """Update curriculum based on performance."""
        self.level_performance.append(performance)

        if len(self.level_performance) >= 50:
            avg_performance = np.mean(list(self.level_performance))
            threshold = self.curriculum_levels[self.current_level]["success_threshold"]

            if (
                avg_performance >= threshold
                and self.current_level < len(self.curriculum_levels) - 1
            ):
                self.current_level += 1
                print(f"Curriculum advanced to level {self.current_level}")
                self.level_performance.clear()

    def get_current_difficulty(self):
        """Get current difficulty level."""
        return self.curriculum_levels[self.current_level]["difficulty"]

    def get_action(self, state):
        """Get action from policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            return action.item()

    def update(self, trajectories):
        """Update policy."""
        if not trajectories:
            return None

        total_loss = 0

        for trajectory in trajectories:
            states, actions, rewards = zip(*trajectory)

            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)

            action_probs = self.policy(states_tensor)
            log_probs = torch.log(
                action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8
            )

            # Curriculum-weighted loss
            difficulty = self.get_current_difficulty()
            loss = -(log_probs * rewards).mean() * difficulty

            total_loss += loss.item()

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss


class MultiTaskLearningAgent:
    """Multi-Task Learning Agent."""

    def __init__(self, state_dim, action_dim, num_tasks, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks

        # Shared feature extractor
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim),
                    nn.Softmax(dim=-1),
                )
                for _ in range(num_tasks)
            ]
        )

        # Task selector
        self.task_selector = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_tasks), nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(
            list(self.shared_encoder.parameters())
            + list(self.task_heads.parameters())
            + list(self.task_selector.parameters()),
            lr=lr,
        )

        self.task_performances = {i: deque(maxlen=100) for i in range(num_tasks)}

    def get_action(self, state, task_id=None):
        """Get action for specific task."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Get shared features
            shared_features = self.shared_encoder(state_tensor)

            if task_id is None:
                # Auto-select task
                task_probs = self.task_selector(shared_features)
                task_id = torch.multinomial(task_probs, 1).item()

            # Get task-specific action
            action_probs = self.task_heads[task_id](shared_features)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()

            return action.item(), task_id

    def update(self, task_data):
        """Update multi-task agent."""
        total_loss = 0

        for task_id, trajectories in task_data.items():
            if not trajectories:
                continue

            task_loss = 0

            for trajectory in trajectories:
                states, actions, rewards = zip(*trajectory)

                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions)

                # Get shared features
                shared_features = self.shared_encoder(states_tensor)

                # Get task-specific output
                action_probs = self.task_heads[task_id](shared_features)
                log_probs = torch.log(
                    action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8
                )

                # Task-specific loss
                loss = -(log_probs * rewards).mean()
                task_loss += loss.item()

            total_loss += task_loss
            self.task_performances[task_id].append(task_loss)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss


class ContinualLearningAgent:
    """Continual Learning Agent with catastrophic forgetting prevention."""

    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Main policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )

        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)

        # Elastic Weight Consolidation (EWC) parameters
        self.ewc_lambda = 1000
        self.fisher_information = {}
        self.optimal_params = {}

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.task_count = 0

    def compute_fisher_information(self, task_data):
        """Compute Fisher Information Matrix for EWC."""
        fisher_info = {}

        for name, param in self.policy.named_parameters():
            fisher_info[name] = torch.zeros_like(param)

        for trajectory in task_data:
            states, actions, rewards = zip(*trajectory)

            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)

            # Compute gradients
            self.optimizer.zero_grad()

            action_probs = self.policy(states_tensor)
            log_probs = torch.log(
                action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8
            )
            loss = -log_probs.mean()

            loss.backward()

            # Accumulate Fisher information
            for name, param in self.policy.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data**2

        # Average over trajectories
        for name in fisher_info:
            fisher_info[name] /= len(task_data)

        return fisher_info

    def consolidate_knowledge(self, task_data):
        """Consolidate knowledge using EWC."""
        # Compute Fisher information
        self.fisher_information = self.compute_fisher_information(task_data)

        # Store optimal parameters
        self.optimal_params = {}
        for name, param in self.policy.named_parameters():
            self.optimal_params[name] = param.data.clone()

        self.task_count += 1
        print(f"Knowledge consolidated for task {self.task_count}")

    def ewc_loss(self):
        """Compute EWC regularization loss."""
        ewc_loss = 0

        for name, param in self.policy.named_parameters():
            if name in self.fisher_information:
                fisher_info = self.fisher_information[name]
                optimal_param = self.optimal_params[name]

                ewc_loss += (fisher_info * (param - optimal_param) ** 2).sum()

        return self.ewc_lambda * ewc_loss

    def get_action(self, state):
        """Get action from policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            return action.item()

    def update(self, trajectories):
        """Update continual learning agent."""
        if not trajectories:
            return None

        # Store experiences
        for trajectory in trajectories:
            self.experience_buffer.append(trajectory)

        # Sample from experience buffer
        if len(self.experience_buffer) > 32:
            sampled_experiences = random.sample(self.experience_buffer, 32)
        else:
            sampled_experiences = list(self.experience_buffer)

        total_loss = 0

        for trajectory in sampled_experiences:
            states, actions, rewards = zip(*trajectory)

            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)

            action_probs = self.policy(states_tensor)
            log_probs = torch.log(
                action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8
            )

            # Policy loss
            policy_loss = -(log_probs * rewards).mean()

            # EWC loss
            ewc_loss = self.ewc_loss()

            # Total loss
            loss = policy_loss + ewc_loss
            total_loss += loss.item()

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss


class ExplainableRLAgent:
    """Explainable RL Agent with attention mechanisms."""

    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Attention-based policy
        self.attention = nn.MultiheadAttention(state_dim, num_heads=4, batch_first=True)

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )

        # Value function for explanations
        self.value_function = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.optimizer = optim.Adam(
            list(self.attention.parameters())
            + list(self.policy.parameters())
            + list(self.value_function.parameters()),
            lr=lr,
        )

        self.attention_weights_history = []

    def get_action_with_explanation(self, state):
        """Get action with attention-based explanation."""
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            )  # [1, 1, state_dim]

            # Self-attention
            attended_state, attention_weights = self.attention(
                state_tensor, state_tensor, state_tensor
            )

            # Store attention weights for explanation
            self.attention_weights_history.append(
                attention_weights.squeeze().cpu().numpy()
            )

            # Get policy output
            policy_input = attended_state.squeeze(0)  # [1, state_dim]
            action_probs = self.policy(policy_input)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()

            # Get value for explanation
            value = self.value_function(policy_input)

            return (
                action.item(),
                attention_weights.squeeze().cpu().numpy(),
                value.item(),
            )

    def get_action(self, state):
        """Get action without explanation."""
        action, _, _ = self.get_action_with_explanation(state)
        return action

    def generate_explanation(self, state, action):
        """Generate explanation for action choice."""
        action_probs, attention_weights, value = self.get_action_with_explanation(state)

        explanation = {
            "action": action,
            "action_probability": action_probs[action],
            "attention_weights": attention_weights,
            "state_value": value,
            "confidence": float(action_probs.max()),
            "attention_focus": float(attention_weights.max()),
        }

        return explanation

    def update(self, trajectories):
        """Update explainable agent."""
        if not trajectories:
            return None

        total_loss = 0

        for trajectory in trajectories:
            states, actions, rewards = zip(*trajectory)

            states_tensor = torch.FloatTensor(states).unsqueeze(
                0
            )  # [1, seq_len, state_dim]
            actions_tensor = torch.LongTensor(actions)

            # Attention
            attended_states, _ = self.attention(
                states_tensor, states_tensor, states_tensor
            )

            # Policy loss
            policy_outputs = []
            for i in range(attended_states.size(1)):
                policy_output = self.policy(attended_states[:, i, :])
                policy_outputs.append(policy_output)

            policy_outputs = torch.stack(policy_outputs, dim=1).squeeze(
                0
            )  # [seq_len, action_dim]
            log_probs = torch.log(
                policy_outputs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8
            )
            policy_loss = -(log_probs * rewards).mean()

            # Value loss
            value_outputs = []
            for i in range(attended_states.size(1)):
                value_output = self.value_function(attended_states[:, i, :])
                value_outputs.append(value_output)

            value_outputs = torch.stack(value_outputs, dim=1).squeeze(0)  # [seq_len, 1]
            value_loss = F.mse_loss(value_outputs.squeeze(), torch.FloatTensor(rewards))

            # Total loss
            loss = policy_loss + value_loss
            total_loss += loss.item()

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss


class AdaptiveMetaLearningAgent:
    """Adaptive Meta-Learning Agent with dynamic adaptation."""

    def __init__(self, state_dim, action_dim, meta_lr=3e-4, inner_lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr

        # Meta-network
        self.meta_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )

        # Adaptive learning rate network
        self.lr_network = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        self.meta_optimizer = optim.Adam(
            list(self.meta_network.parameters()) + list(self.lr_network.parameters()),
            lr=meta_lr,
        )

        self.task_networks = {}
        self.adaptation_history = []

    def adapt_to_task(self, task_id, support_data, num_steps=5):
        """Adapt to task with adaptive learning rate."""
        if task_id not in self.task_networks:
            self.task_networks[task_id] = copy.deepcopy(self.meta_network)

        task_network = self.task_networks[task_id]

        # Compute adaptive learning rate
        if support_data:
            sample_state = support_data[0][0]  # First state from first trajectory
            state_tensor = torch.FloatTensor(sample_state).unsqueeze(0)
            adaptive_lr = self.lr_network(state_tensor).item() * self.inner_lr
        else:
            adaptive_lr = self.inner_lr

        task_optimizer = optim.SGD(task_network.parameters(), lr=adaptive_lr)

        # Inner loop adaptation
        for step in range(num_steps):
            task_optimizer.zero_grad()

            total_loss = 0
            for trajectory in support_data:
                states, actions, rewards = zip(*trajectory)
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions)

                action_probs = task_network(states_tensor)
                log_probs = torch.log(
                    action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8
                )
                loss = -log_probs.mean()
                total_loss += loss

            total_loss.backward()
            task_optimizer.step()

        # Store adaptation info
        self.adaptation_history.append(
            {
                "task_id": task_id,
                "adaptive_lr": adaptive_lr,
                "final_loss": total_loss.item(),
            }
        )

        return task_network

    def meta_update(self, tasks_data):
        """Meta-update with adaptive learning rates."""
        meta_loss = 0

        for task_id, (support_data, query_data) in tasks_data.items():
            # Adapt to task
            adapted_network = self.adapt_to_task(task_id, support_data)

            # Evaluate on query data
            query_states, query_actions, query_rewards = query_data
            query_states_tensor = torch.FloatTensor(query_states)
            query_actions_tensor = torch.LongTensor(query_actions)

            query_action_probs = adapted_network(query_states_tensor)
            query_log_probs = torch.log(
                query_action_probs.gather(
                    1, query_actions_tensor.unsqueeze(1)
                ).squeeze()
                + 1e-8
            )

            meta_loss += -query_log_probs.mean()

        # Meta-gradient update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def get_action(self, state, task_id=None):
        """Get action from meta-network or task-specific network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            if task_id and task_id in self.task_networks:
                action_probs = self.task_networks[task_id](state_tensor)
            else:
                action_probs = self.meta_network(state_tensor)

            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            return action.item()


class AdvancedRLExperimentManager:
    """Advanced RL Experiment Manager for complex experiments."""

    def __init__(self):
        self.experiments = {}
        self.results = {}
        self.configurations = {}

    def create_experiment(self, experiment_name, config):
        """Create new experiment."""
        self.experiments[experiment_name] = {
            "config": config,
            "agents": {},
            "environments": {},
            "results": {},
            "status": "created",
        }
        self.configurations[experiment_name] = config

    def add_agent(self, experiment_name, agent_name, agent_class, agent_params):
        """Add agent to experiment."""
        if experiment_name in self.experiments:
            self.experiments[experiment_name]["agents"][agent_name] = {
                "class": agent_class,
                "params": agent_params,
                "instance": None,
            }

    def add_environment(self, experiment_name, env_name, env_class, env_params):
        """Add environment to experiment."""
        if experiment_name in self.experiments:
            self.experiments[experiment_name]["environments"][env_name] = {
                "class": env_class,
                "params": env_params,
                "instance": None,
            }

    def run_experiment(self, experiment_name, num_episodes=1000):
        """Run experiment."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        experiment = self.experiments[experiment_name]
        experiment["status"] = "running"

        results = {}

        for agent_name, agent_info in experiment["agents"].items():
            agent_class = agent_info["class"]
            agent_params = agent_info["params"]

            # Create agent instance
            agent = agent_class(**agent_params)
            experiment["agents"][agent_name]["instance"] = agent

            agent_results = {}

            for env_name, env_info in experiment["environments"].items():
                env_class = env_info["class"]
                env_params = env_info["params"]

                # Create environment instance
                env = env_class(**env_params)
                experiment["environments"][env_name]["instance"] = env

                # Run training
                episode_rewards = []
                episode_lengths = []

                for episode in range(num_episodes):
                    state = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    done = False

                    while not done:
                        action = agent.get_action(state)
                        next_state, reward, done, info = env.step(action)

                        episode_reward += reward
                        episode_length += 1

                        # Update agent (if applicable)
                        if hasattr(agent, "update"):
                            # Collect trajectory for batch update
                            pass

                        state = next_state

                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)

                    if episode % 100 == 0:
                        avg_reward = np.mean(episode_rewards[-100:])
                        print(
                            f"Agent: {agent_name}, Env: {env_name}, Episode: {episode}, Avg Reward: {avg_reward:.3f}"
                        )

                agent_results[env_name] = {
                    "episode_rewards": episode_rewards,
                    "episode_lengths": episode_lengths,
                    "final_performance": np.mean(episode_rewards[-100:]),
                }

            results[agent_name] = agent_results

        experiment["results"] = results
        experiment["status"] = "completed"

        return results

    def save_experiment(self, experiment_name, filepath):
        """Save experiment results."""
        if experiment_name in self.experiments:
            with open(filepath, "wb") as f:
                pickle.dump(self.experiments[experiment_name], f)

    def load_experiment(self, filepath):
        """Load experiment results."""
        with open(filepath, "rb") as f:
            experiment_data = pickle.load(f)

        experiment_name = f"loaded_experiment_{len(self.experiments)}"
        self.experiments[experiment_name] = experiment_data

        return experiment_name

    def compare_experiments(self, experiment_names):
        """Compare multiple experiments."""
        comparison_results = {}

        for exp_name in experiment_names:
            if exp_name in self.experiments:
                comparison_results[exp_name] = self.experiments[exp_name]["results"]

        return comparison_results

    def generate_report(self, experiment_name):
        """Generate comprehensive experiment report."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        experiment = self.experiments[experiment_name]
        results = experiment["results"]

        report = {
            "experiment_name": experiment_name,
            "config": experiment["config"],
            "status": experiment["status"],
            "summary": {},
            "detailed_results": results,
        }

        # Generate summary statistics
        for agent_name, agent_results in results.items():
            agent_summary = {}

            for env_name, env_results in agent_results.items():
                rewards = env_results["episode_rewards"]
                lengths = env_results["episode_lengths"]

                agent_summary[env_name] = {
                    "final_performance": env_results["final_performance"],
                    "average_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "average_length": np.mean(lengths),
                    "convergence_episode": self._find_convergence_episode(rewards),
                    "success_rate": self._calculate_success_rate(rewards),
                }

            report["summary"][agent_name] = agent_summary

        return report

    def _find_convergence_episode(self, rewards, window=100, threshold=0.95):
        """Find episode where performance converges."""
        if len(rewards) < window:
            return len(rewards)

        for i in range(window, len(rewards)):
            recent_performance = np.mean(rewards[i - window : i])
            overall_performance = np.mean(rewards)

            if recent_performance >= threshold * overall_performance:
                return i

        return len(rewards)

    def _calculate_success_rate(self, rewards, success_threshold=None):
        """Calculate success rate."""
        if success_threshold is None:
            success_threshold = np.mean(rewards) + np.std(rewards)

        successful_episodes = sum(1 for r in rewards if r >= success_threshold)
        return successful_episodes / len(rewards)
