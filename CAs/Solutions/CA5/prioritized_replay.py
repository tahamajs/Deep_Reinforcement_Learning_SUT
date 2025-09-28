"""
Prioritized Experience Replay Implementation
==========================================

This module implements Prioritized Experience Replay (PER) which
improves sample efficiency by prioritizing important experiences.

Key Features:
- SumTree data structure for efficient prioritized sampling
- Proportional prioritization with importance sampling
- PrioritizedReplayBuffer class
- PrioritizedDQNAgent implementation
- Performance analysis and comparison tools

Author: CA5 Implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dqn_base import DQNAgent, ReplayBuffer, device
import random
import warnings

warnings.filterwarnings("ignore")


class SumTree:
    """SumTree data structure for efficient prioritized sampling"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Retrieve sample from tree"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Get total priority sum"""
        return self.tree[0]

    def add(self, p, data):
        """Add new experience with priority"""
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """Update priority of existing experience"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Get experience and its tree index"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer using SumTree"""

    def __init__(
        self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-6, epsilon=1e-6
    ):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # Small constant to avoid zero priority
        self.max_priority = 1.0

        self.tree = SumTree(capacity)
        self.experience_count = 0

    def add(self, experience):
        """Add experience with max priority"""
        self.tree.add(self.max_priority, experience)
        self.experience_count = min(self.experience_count + 1, self.capacity)

    def sample(self, batch_size):
        """Sample batch with prioritized probabilities"""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)

            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(
            self.experience_count * sampling_probabilities, -self.beta
        )
        is_weights /= is_weights.max()  # Normalize

        # Increase beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, idxs, is_weights

    def update_priorities(self, idxs, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.experience_count


class PrioritizedDQNAgent(DQNAgent):
    """DQN agent with Prioritized Experience Replay"""

    def __init__(self, state_size, action_size, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.agent_type = "Prioritized DQN"

        # Override memory with prioritized buffer
        self.memory = PrioritizedReplayBuffer(
            capacity=(
                self.memory.capacity if hasattr(self.memory, "capacity") else 10000
            ),
            alpha=0.6,  # Priority exponent
            beta=0.4,  # Initial importance sampling weight
            beta_increment=1e-6,
        )

        # Additional tracking for analysis
        self.priorities = []
        self.is_weights = []
        self.td_errors = []

    def train_step(self):
        """Prioritized DQN training step"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch with priorities
        experiences, idxs, is_weights = self.memory.sample(self.batch_size)
        batch = self.experience_to_batch(experiences)

        states, actions, rewards, next_states, dones = batch

        # Convert importance sampling weights to tensor
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute TD errors
        td_errors = target_q_values - current_q_values
        td_errors_np = td_errors.detach().cpu().numpy().flatten()

        # Update priorities
        self.memory.update_priorities(idxs, td_errors_np)

        # Compute weighted loss
        loss = (
            is_weights * F.mse_loss(current_q_values, target_q_values, reduction="none")
        ).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Store training metrics
        self.losses.append(loss.item())
        avg_q_value = current_q_values.mean().item()
        avg_target = target_q_values.mean().item()

        self.q_values.append(avg_q_value)
        self.priorities.extend(td_errors_np)
        self.is_weights.extend(is_weights.cpu().numpy().flatten())
        self.td_errors.extend(td_errors_np)

        return loss.item()


class PERAnalysis:
    """Analyze Prioritized Experience Replay behavior and effectiveness"""

    def __init__(self):
        self.results = {}

    def analyze_priority_distribution(self, agent, num_samples=1000):
        """Analyze how priorities evolve during training"""
        print("Analyzing Priority Distribution...")

        priorities_over_time = []
        td_errors_over_time = []

        # Sample priorities at different training stages
        for i in range(0, len(agent.priorities), len(agent.priorities) // 10):
            if i + num_samples < len(agent.priorities):
                priorities_over_time.append(agent.priorities[i : i + num_samples])
                td_errors_over_time.append(agent.td_errors[i : i + num_samples])

        return {
            "priorities_over_time": priorities_over_time,
            "td_errors_over_time": td_errors_over_time,
            "final_priorities": (
                agent.priorities[-num_samples:]
                if len(agent.priorities) >= num_samples
                else agent.priorities
            ),
            "final_td_errors": (
                agent.td_errors[-num_samples:]
                if len(agent.td_errors) >= num_samples
                else agent.td_errors
            ),
        }

    def visualize_per_behavior(self, agent):
        """Visualize PER behavior and priority dynamics"""
        analysis_data = self.analyze_priority_distribution(agent)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Priority distribution over time
        if analysis_data["priorities_over_time"]:
            for i, priorities in enumerate(analysis_data["priorities_over_time"]):
                axes[0, 0].hist(
                    priorities, bins=30, alpha=0.5, label=f"Stage {i+1}", density=True
                )
            axes[0, 0].set_title("Priority Distribution Evolution")
            axes[0, 0].set_xlabel("Priority")
            axes[0, 0].set_ylabel("Density")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. TD error vs Priority correlation
        if analysis_data["final_priorities"] and analysis_data["final_td_errors"]:
            axes[0, 1].scatter(
                analysis_data["final_td_errors"],
                analysis_data["final_priorities"],
                alpha=0.6,
                color="blue",
                s=10,
            )
            axes[0, 1].set_title("TD Error vs Priority Correlation")
            axes[0, 1].set_xlabel("TD Error")
            axes[0, 1].set_ylabel("Priority")
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Importance sampling weights distribution
        if hasattr(agent, "is_weights") and agent.is_weights:
            axes[0, 2].hist(agent.is_weights[-1000:], bins=30, alpha=0.7, color="green")
            axes[0, 2].set_title("Importance Sampling Weights Distribution")
            axes[0, 2].set_xlabel("IS Weight")
            axes[0, 2].set_ylabel("Frequency")
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Priority evolution
        if agent.priorities:
            window = 100
            rolling_priorities = []
            for i in range(window, len(agent.priorities)):
                rolling_priorities.append(np.mean(agent.priorities[i - window : i]))

            axes[1, 0].plot(rolling_priorities, color="red", linewidth=2)
            axes[1, 0].set_title(f"Rolling Average Priority (Window={window})")
            axes[1, 0].set_xlabel("Training Step")
            axes[1, 0].set_ylabel("Average Priority")
            axes[1, 0].grid(True, alpha=0.3)

        # 5. TD error evolution
        if agent.td_errors:
            window = 100
            rolling_td_errors = []
            for i in range(window, len(agent.td_errors)):
                rolling_td_errors.append(
                    np.mean(np.abs(agent.td_errors[i - window : i]))
                )

            axes[1, 1].plot(rolling_td_errors, color="purple", linewidth=2)
            axes[1, 1].set_title(f"Rolling Average |TD Error| (Window={window})")
            axes[1, 1].set_xlabel("Training Step")
            axes[1, 1].set_ylabel("Average |TD Error|")
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Sampling efficiency
        if hasattr(agent, "memory") and hasattr(agent.memory, "tree"):
            # Show how the tree evolves (simplified view)
            tree_values = agent.memory.tree.tree[:100]  # First 100 nodes
            axes[1, 2].bar(
                range(len(tree_values)), tree_values, alpha=0.7, color="orange"
            )
            axes[1, 2].set_title("SumTree Structure (First 100 Nodes)")
            axes[1, 2].set_xlabel("Tree Index")
            axes[1, 2].set_ylabel("Priority Sum")
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_per_vs_uniform(self, uniform_agent, per_agent, env, num_episodes=300):
        """Compare Prioritized vs Uniform Experience Replay"""
        print("Comparing Prioritized vs Uniform Experience Replay...")

        # Train both agents
        print("Training Uniform Replay DQN...")
        uniform_scores, _ = uniform_agent.train(
            env, num_episodes, print_every=num_episodes // 5
        )

        print("Training Prioritized Replay DQN...")
        per_scores, _ = per_agent.train(
            env, num_episodes, print_every=num_episodes // 5
        )

        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        episodes = range(len(uniform_scores))

        # 1. Learning curves
        axes[0, 0].plot(
            episodes, uniform_scores, color="red", label="Uniform Replay", linewidth=2
        )
        axes[0, 0].plot(
            episodes, per_scores, color="blue", label="Prioritized Replay", linewidth=2
        )
        axes[0, 0].set_title("Learning Curves Comparison")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Episode Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Sample efficiency (reward per training step)
        # Approximate training steps (assuming similar batch sizes)
        training_steps = np.arange(len(uniform_scores)) * 100  # Rough estimate

        axes[0, 1].plot(
            training_steps,
            uniform_scores,
            color="red",
            label="Uniform Replay",
            linewidth=2,
        )
        axes[0, 1].plot(
            training_steps,
            per_scores,
            color="blue",
            label="Prioritized Replay",
            linewidth=2,
        )
        axes[0, 1].set_title("Sample Efficiency Comparison")
        axes[0, 1].set_xlabel("Training Steps (approx)")
        axes[0, 1].set_ylabel("Episode Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Final performance comparison
        final_window = 50
        uniform_final = uniform_scores[-final_window:]
        per_final = per_scores[-final_window:]

        axes[1, 0].boxplot(
            [uniform_final, per_final], labels=["Uniform Replay", "Prioritized Replay"]
        )
        axes[1, 0].set_title(f"Final Performance (Last {final_window} Episodes)")
        axes[1, 0].set_ylabel("Episode Reward")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Convergence analysis
        convergence_threshold = np.mean(per_scores[-50:]) * 0.9

        uniform_convergence = next(
            (
                i
                for i, score in enumerate(uniform_scores)
                if score >= convergence_threshold
            ),
            len(uniform_scores),
        )
        per_convergence = next(
            (i for i, score in enumerate(per_scores) if score >= convergence_threshold),
            len(per_scores),
        )

        axes[1, 1].bar(
            ["Uniform Replay", "Prioritized Replay"],
            [uniform_convergence, per_convergence],
            color=["red", "blue"],
            alpha=0.7,
        )
        axes[1, 1].set_title("Convergence Speed")
        axes[1, 1].set_ylabel("Episodes to Convergence")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\\nPER vs Uniform Replay Summary:")
        print("=" * 40)
        print(
            f"Uniform Final Avg: {np.mean(uniform_final):.2f} ± {np.std(uniform_final):.2f}"
        )
        print(f"PER Final Avg: {np.mean(per_final):.2f} ± {np.std(per_final):.2f}")
        print(f"Improvement: {np.mean(per_final) - np.mean(uniform_final):.2f}")
        print(f"Convergence Speedup: {uniform_convergence - per_convergence} episodes")

        return uniform_scores, per_scores


# Example usage and demonstration
if __name__ == "__main__":
    print("Prioritized Experience Replay Implementation")
    print("=" * 50)

    # Test SumTree
    tree = SumTree(10)
    for i in range(10):
        tree.add(i + 1, f"data_{i}")
    print(f"SumTree total: {tree.total()}")

    # Test Prioritized Replay Buffer
    buffer = PrioritizedReplayBuffer(1000)
    for i in range(100):
        buffer.add((i, i + 1, i + 2, i + 3, i + 4))  # Dummy experience

    batch, idxs, weights = buffer.sample(32)
    print(f"Sampled batch size: {len(batch)}")
    print(f"Sample indices: {idxs[:5]}...")
    print(f"IS weights shape: {weights.shape}")

    # Test Prioritized DQN Agent
    agent = PrioritizedDQNAgent(state_size=4, action_size=2)
    print(f"Prioritized DQN Agent created: {agent.agent_type}")

    print("\\n✓ Prioritized Experience Replay implementation complete")
    print("✓ SumTree data structure ready")
    print("✓ PER analysis framework available")
