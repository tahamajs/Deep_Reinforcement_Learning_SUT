import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import TabularModel
from environments.environments import SimpleGridWorld


class MCTSNode:
    """Node in MCTS tree"""

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = {}  # Action -> child node mapping
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = None  # Will be set when expanded

    def is_fully_expanded(self):
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0

    def is_terminal(self):
        """Check if this is a terminal state"""
        return (
            len(self.children) == 0
            and self.visits > 0
            and self.untried_actions is not None
            and len(self.untried_actions) == 0
        )

    def get_ucb_value(self, exploration_weight=1.0):
        """Calculate UCB value for node selection"""
        if self.visits == 0:
            return float("inf")

        exploitation = self.total_reward / self.visits
        exploration = exploration_weight * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def select_child(self, exploration_weight=1.0):
        """Select child with highest UCB value"""
        return max(
            self.children.values(),
            key=lambda child: child.get_ucb_value(exploration_weight),
        )

    def expand(self, action, new_state):
        """Expand node by adding a child"""
        if action in self.untried_actions:
            self.untried_actions.remove(action)

        child = MCTSNode(state=new_state, parent=self, action=action)
        self.children[action] = child
        return child

    def update(self, reward):
        """Update node statistics"""
        self.visits += 1
        self.total_reward += reward

    def get_best_action(self):
        """Get the action leading to the most visited child"""
        if not self.children:
            return None
        return max(self.children.keys(), key=lambda a: self.children[a].visits)


class MCTS:
    """Monte Carlo Tree Search implementation"""

    def __init__(self, model, num_actions, exploration_weight=1.0, max_depth=50):
        self.model = model
        self.num_actions = num_actions
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.gamma = 0.95  # Discount factor

    def search(self, root_state, num_simulations=1000):
        """Perform MCTS to find best action"""
        root = MCTSNode(root_state)
        root.untried_actions = list(range(self.num_actions))

        for _ in range(num_simulations):
            leaf = self._select_leaf(root)

            if leaf.untried_actions and len(leaf.untried_actions) > 0:
                action = np.random.choice(leaf.untried_actions)
                next_state, reward, done = self._simulate_step(leaf.state, action)
                child = leaf.expand(action, next_state)
                child.untried_actions = (
                    list(range(self.num_actions)) if not done else []
                )
                leaf = child

            simulation_reward = self._simulate_rollout(leaf.state)

            self._backpropagate(leaf, simulation_reward)

        return root.get_best_action(), root

    def _select_leaf(self, node):
        """Select leaf node using UCB"""
        while node.is_fully_expanded() and node.children:
            node = node.select_child(self.exploration_weight)
        return node

    def _simulate_step(self, state, action):
        """Simulate one step using the model"""
        if hasattr(self.model, "predict"):
            next_state, reward = self.model.predict(state, action)
            done = False
        else:
            next_state, reward = self.model.sample_transition(state, action)
            done = False  # Assume not done for simulation

        return next_state, reward, done

    def _simulate_rollout(self, state, max_depth=None):
        """Perform random rollout from state"""
        if max_depth is None:
            max_depth = self.max_depth

        total_reward = 0.0
        current_state = state
        discount = 1.0

        for depth in range(max_depth):
            action = np.random.randint(self.num_actions)
            next_state, reward, done = self._simulate_step(current_state, action)

            total_reward += discount * reward
            discount *= self.gamma

            if done:
                break

            current_state = next_state

        return total_reward

    def _backpropagate(self, node, reward):
        """Backpropagate reward up the tree"""
        while node is not None:
            node.update(reward)
            node = node.parent
            reward *= self.gamma  # Discount for parent nodes


class MCTSAgent:
    """Agent using MCTS for planning"""

    def __init__(
        self,
        model,
        num_states,
        num_actions,
        num_simulations=1000,
        exploration_weight=1.0,
    ):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.mcts = MCTS(model, num_actions, exploration_weight)
        self.num_simulations = num_simulations

        self.search_times = []
        self.tree_sizes = []
        self.episode_rewards = []

    def select_action(self, state, deterministic=False):
        """Select action using MCTS"""
        start_time = time.time()

        best_action, root = self.mcts.search(state, self.num_simulations)

        search_time = time.time() - start_time
        tree_size = self._count_nodes(root)

        self.search_times.append(search_time)
        self.tree_sizes.append(tree_size)

        return (
            best_action
            if best_action is not None
            else np.random.randint(self.num_actions)
        )

    def _count_nodes(self, node):
        """Count total nodes in tree"""
        if not node.children:
            return 1
        return 1 + sum(self._count_nodes(child) for child in node.children.values())

    def train_episode(self, env, max_steps=200):
        """Run episode with MCTS planning"""
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done = env.step(action)

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        self.episode_rewards.append(total_reward)
        return total_reward, steps

    def get_statistics(self):
        """Get performance statistics"""
        return {
            "avg_search_time": np.mean(self.search_times) if self.search_times else 0,
            "avg_tree_size": np.mean(self.tree_sizes) if self.tree_sizes else 0,
            "total_searches": len(self.search_times),
            "avg_episode_reward": (
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            ),
        }


def demonstrate_mcts():
    """MCTS Demonstration"""

    print("Monte Carlo Tree Search (MCTS) Demonstration")
    print("=" * 50)

    print("\n1. Setting up environment and learned model...")
    env = SimpleGridWorld(size=6)
    tabular_model = TabularModel(env.num_states, env.num_actions)

    print("Training tabular model...")
    for episode in range(100):
        state = env.reset()
        for step in range(50):
            action = np.random.randint(env.num_actions)
            next_state, reward, done = env.step(action)
            tabular_model.update(state, action, reward, next_state)
            if done:
                break
            state = next_state

    mcts_agent = MCTSAgent(
        model=tabular_model,
        num_states=env.num_states,
        num_actions=env.num_actions,
        num_simulations=200,
        exploration_weight=1.4,
    )

    print(f"Model trained with {np.sum(tabular_model.sa_counts)} transitions")

    print("\n2. Testing MCTS performance...")
    n_test_episodes = 20
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_test_episodes):
        reward, length = mcts_agent.train_episode(env, max_steps=100)
        episode_rewards.append(reward)
        episode_lengths.append(length)

        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            avg_length = np.mean(episode_lengths[-5:])
            stats = mcts_agent.get_statistics()
            print(
                f"Episodes {episode-4}-{episode+1}: Avg Reward = {avg_reward:.2f}, "
                f"Avg Length = {avg_length:.1f}, Avg Search Time = {stats['avg_search_time']:.4f}s"
            )

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards, "b-", linewidth=2, label="Episode Reward")
    plt.axhline(
        y=np.mean(episode_rewards),
        color="r",
        linestyle="--",
        alpha=0.7,
        label="Average",
    )
    plt.title("MCTS Episode Performance")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(episode_lengths, "g-", linewidth=2, label="Episode Length")
    plt.axhline(
        y=np.mean(episode_lengths),
        color="r",
        linestyle="--",
        alpha=0.7,
        label="Average",
    )
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps to Goal")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    search_times = mcts_agent.search_times
    plt.plot(search_times, "purple", linewidth=2, label="Search Time")
    plt.axhline(
        y=np.mean(search_times), color="r", linestyle="--", alpha=0.7, label="Average"
    )
    plt.title("MCTS Search Times")
    plt.xlabel("Search Number")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    tree_sizes = mcts_agent.tree_sizes
    plt.plot(tree_sizes, "orange", linewidth=2, label="Tree Size")
    plt.axhline(
        y=np.mean(tree_sizes), color="r", linestyle="--", alpha=0.7, label="Average"
    )
    plt.title("MCTS Tree Sizes")
    plt.xlabel("Search Number")
    plt.ylabel("Number of Nodes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    if len(search_times) > 0 and len(tree_sizes) > 0:
        plt.scatter(tree_sizes, search_times, alpha=0.6, c="red", s=30)
        if len(tree_sizes) > 1:
            z = np.polyfit(tree_sizes, search_times, 1)
            p = np.poly1d(z)
            plt.plot(
                sorted(tree_sizes), p(sorted(tree_sizes)), "r--", alpha=0.8, linewidth=2
            )
    plt.title("Search Time vs Tree Size")
    plt.xlabel("Tree Size (nodes)")
    plt.ylabel("Search Time (seconds)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    random_rewards = []
    for _ in range(n_test_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(100):
            action = np.random.randint(env.num_actions)
            next_state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
            state = next_state
        random_rewards.append(total_reward)

    comparison_data = [episode_rewards, random_rewards]
    labels = ["MCTS", "Random"]
    plt.boxplot(comparison_data, labels=labels)
    plt.title("Performance Comparison")
    plt.ylabel("Episode Reward")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create visualizations directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/mcts_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n3. MCTS Performance Analysis:")
    final_stats = mcts_agent.get_statistics()
    print(
        f"Average Episode Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}"
    )
    print(
        f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    print(f"Average Search Time: {final_stats['avg_search_time']:.4f} seconds")
    print(f"Average Tree Size: {final_stats['avg_tree_size']:.1f} nodes")
    print(f"Total MCTS Searches: {final_stats['total_searches']}")

    print(f"\nRandom Policy Baseline:")
    print(
        f"Average Episode Reward: {np.mean(random_rewards):.3f} ± {np.std(random_rewards):.3f}"
    )

    improvement = (
        (np.mean(episode_rewards) - np.mean(random_rewards))
        / np.mean(random_rewards)
        * 100
    )
    print(f"\nMCTS Improvement over Random: {improvement:.1f}%")

    print(f"\n📊 Key MCTS Insights:")
    print("• MCTS provides sophisticated planning through tree search")
    print("• UCB balances exploration and exploitation in tree nodes")
    print("• Performance scales with number of simulations")
    print("• Computational cost grows with search depth and simulations")
    print("• Effective for discrete action spaces with learned models")


if __name__ == "__main__":
    demonstrate_mcts()
