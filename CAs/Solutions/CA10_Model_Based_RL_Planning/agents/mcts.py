import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
from typing import Dict, Set, Tuple, Any, List, Optional, Union
import gymnasium as gym
import random
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import TabularModel, NeuralModel, device
from environments.environments import SimpleGridWorld


class MCTSNode:
    """Node in the Monte Carlo Tree Search (MCTS) tree.

    Args:
        state (Any): The current state represented by this node.
        parent (Optional[MCTSNode]): The parent node in the MCTS tree. None for the root node.
        action (Optional[int]): The action taken from the parent to reach this state.
        is_terminal (bool): True if this state is a terminal state in the environment.
    """

    def __init__(
        self, state: Any, parent: Optional["MCTSNode"] = None, action: Optional[int] = None, is_terminal: bool = False
    ):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children: Dict[int, "MCTSNode"] = {}  # Action -> child node mapping
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.untried_actions: List[int] = []  # List of actions not yet tried from this node
        self.is_terminal: bool = is_terminal

    def is_fully_expanded(self) -> bool:
        """Checks if all possible actions from this node have been tried (expanded)."""
        return len(self.untried_actions) == 0

    def get_ucb_value(self, exploration_weight: float = 1.0) -> float:
        """Calculates the UCB (Upper Confidence Bound) value for this node.

        Args:
            exploration_weight (float, optional): The exploration constant (c_puct). Defaults to 1.0.

        Returns:
            float: The UCB value, or infinity if the node has not been visited to encourage exploration.

        Raises:
            ValueError: If the parent node has no visits (should not happen for non-root nodes during search).
        """
        if self.visits == 0:
            return float("inf")

        # Ensure parent visits is not zero to avoid log(0)
        if self.parent and self.parent.visits == 0:
            # This case should ideally not be reached if tree is built correctly
            raise ValueError("Parent node must have been visited to calculate UCB.")

        exploitation = self.total_reward / self.visits
        # Add a small epsilon to self.visits in the denominator to prevent division by zero
        # and to parent.visits for log calculation for robustness
        exploration = exploration_weight * np.sqrt(
            np.log(self.parent.visits + 1e-8) / (self.visits + 1e-8)
        )
        return exploitation + exploration

    def select_child(self, exploration_weight: float = 1.0) -> "MCTSNode":
        """Selects the child node with the highest UCB value.

        Args:
            exploration_weight (float, optional): The exploration constant. Defaults to 1.0.

        Returns:
            MCTSNode: The selected child node.

        Raises:
            ValueError: If the node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select child from a node with no children.")
        return max(
            self.children.values(),
            key=lambda child: child.get_ucb_value(exploration_weight),
        )

    def expand(self, action: int, new_state: Any, is_terminal: bool, num_actions: int) -> "MCTSNode":
        """Expands the node by adding a new child node for the given action and state.

        Args:
            action (int): The action that leads to the new state.
            new_state (Any): The state of the new child node.
            is_terminal (bool): True if the new state is a terminal state.
            num_actions (int): Total number of actions in the environment.

        Returns:
            MCTSNode: The newly created child node.
        """
        # Remove the action from untried_actions if it's there
        if action in self.untried_actions:
            self.untried_actions.remove(action)

        child = MCTSNode(state=new_state, parent=self, action=action, is_terminal=is_terminal)
        if not is_terminal:
            child.untried_actions = list(range(num_actions))
        self.children[action] = child
        return child

    def update(self, reward: float):
        """Updates the node's visit count and total reward.

        Args:
            reward (float): The reward to add to the node's total reward.
        """
        self.visits += 1
        self.total_reward += reward

    def get_best_action(self) -> Optional[int]:
        """Gets the action leading to the child with the most visits.

        Returns:
            Optional[int]: The best action, or None if the node has no children.
        """
        if not self.children:
            return None
        return max(self.children.keys(), key=lambda a: self.children[a].visits)


class MCTS:
    """Monte Carlo Tree Search (MCTS) implementation.

    This class performs the full MCTS process (selection, expansion, simulation, backpropagation)
    to determine the best action from a given state.

    Args:
        model (Union[TabularModel, NeuralModel]): The environment model that provides
                                                  transition predictions (e.g., `sample_transition` or `predict`).
        num_actions (int): The total number of actions in the environment.
        exploration_weight (float, optional): The exploration constant (c_puct) for UCB calculation. Defaults to 1.0.
        max_depth (int, optional): The maximum depth for simulation rollouts. Defaults to 50.
        gamma (float, optional): The discount factor. Defaults to 0.95.
    """

    def __init__(
        self,
        model: Union[TabularModel, NeuralModel],
        num_actions: int,
        exploration_weight: float = 1.0,
        max_depth: int = 50,
        gamma: float = 0.95,
    ):
        self.model = model
        self.num_actions = num_actions
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.gamma = gamma
        # Assuming model has state_dim if it's a NeuralModel for one-hot encoding
        self.state_dim = getattr(model, 'num_states', getattr(model, 'state_dim', None))
        if self.state_dim is None and isinstance(model, NeuralModel):
            raise ValueError("NeuralModel requires 'state_dim' attribute.")

    def search(self, root_state: Any, num_simulations: int = 1000) -> Tuple[Optional[int], MCTSNode]:
        """Performs MCTS to find the best action from a given root state.

        Args:
            root_state (Any): The starting state for the MCTS search.
            num_simulations (int, optional): The number of simulations to run. Defaults to 1000.

        Returns:
            Tuple[Optional[int], MCTSNode]: A tuple containing:
                - best_action (Optional[int]): The action leading to the most visited child from the root.
                - root (MCTSNode): The root node of the constructed MCTS tree.
        """
        # Note: The is_terminal for root node must be determined by the actual environment or problem context.
        # For generic model-based MCTS, we might not know if root_state is terminal without env.is_terminal(root_state)
        # We will assume initial state is not terminal for simplicity.
        root = MCTSNode(root_state, is_terminal=False) # Assume non-terminal for initial search
        root.untried_actions = list(range(self.num_actions))

        for _ in range(num_simulations):
            current_node = root
            current_state_in_sim = root_state # State changes during simulation rollout
            path: List[MCTSNode] = [current_node]

            # Selection
            while current_node.is_fully_expanded() and not current_node.is_terminal:
                current_node = current_node.select_child(self.exploration_weight)
                path.append(current_node)
                current_state_in_sim = current_node.state # Update state for next step

            # Expansion
            if not current_node.is_terminal and not current_node.is_fully_expanded():
                action = np.random.choice(current_node.untried_actions)

                # Simulate one step using the model
                next_state, reward, is_terminal_step = self._simulate_step(current_state_in_sim, action) # Pass current_state_in_sim

                child_node = current_node.expand(action, next_state, is_terminal_step, self.num_actions)
                path.append(child_node)
                current_node = child_node # Move to child for rollout

            # Simulation (Rollout)
            # If the current node is terminal, its reward is directly taken, no rollout needed
            simulation_reward = 0.0
            if current_node.is_terminal:
                # Assuming the reward from reaching terminal state is captured when is_terminal was set
                # Or, if simulate_step returns reward, it's already considered in path.
                # For now, let's assume if it's a terminal node, no additional rollout reward.
                simulation_reward = reward # Reward from the last step that led to terminal
            else:
                simulation_reward = self._simulate_rollout(current_node.state)

            # Backpropagation
            self._backpropagate(current_node, simulation_reward)

        return root.get_best_action(), root

    def _select_leaf(self, node: MCTSNode) -> MCTSNode:
        """Selects a leaf node by traversing the tree using the UCB criterion.

        Args:
            node (MCTSNode): The starting node for selection (typically the root).

        Returns:
            MCTSNode: The selected leaf node.
        """
        while node.is_fully_expanded() and not node.is_terminal:
            node = node.select_child(self.exploration_weight)
        return node

    def _simulate_step(self, state: Any, action: int) -> Tuple[Any, float, bool]:
        """Simulates one step using the learned environment model.

        Args:
            state (Any): The current state.
            action (int): The action to take.

        Returns:
            Tuple[Any, float, bool]: A tuple containing:
                - next_state (Any): The predicted next state.
                - reward (float): The predicted reward.
                - done (bool): True if the next state is terminal, False otherwise.

        Note: This method assumes the model can provide next_state, reward, and implicitly if it's terminal.
              For TabularModel, `sample_transition` returns next_state, reward. Done status is not directly predicted.
              For NeuralModel, it predicts next_state and reward. Done status is not directly predicted.
              We will use a heuristic for `done` or assume environment checks (e.g. `is_terminal_state` if available).
              For now, we'll mark as `False` and rely on `max_depth` for rollouts or external checks.
        """
        next_state_val: Any # To hold either int or torch.Tensor
        reward_val: float

        if isinstance(self.model, TabularModel):
            next_state_val, reward_val = self.model.sample_transition(state, action)
            # For TabularModel, 'done' is usually determined by reaching a known terminal state from environment
            # or max steps. For now, assume not done here.
            done = False
            # If SimpleGridWorld is the assumed environment, we need to check its terminal states.
            # This would ideally be passed or wrapped in the model.
            # For this context, we will assume a generic model does not explicitly tell us if state is terminal.
            # If `state` can be compared to `env.terminal_states`, that check can be added here.

        elif isinstance(self.model, NeuralModel):
            # Convert state and action to tensors for NeuralModel
            state_tensor = (
                torch.eye(self.state_dim)[state:state+1].to(device) # For discrete states (one-hot)
                if isinstance(state, int) else torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            )
            action_tensor = torch.tensor([action], dtype=torch.long).to(device) # Assuming discrete action

            pred_next_state, pred_reward = self.model.sample_transition(state_tensor, action_tensor)

            # Convert predictions back to numpy/python types
            next_state_val = np.argmax(pred_next_state.cpu().numpy().squeeze()) if self.state_dim else pred_next_state.cpu().numpy().squeeze()
            reward_val = pred_reward.cpu().item()
            done = False # NeuralModel does not predict done directly, rely on rollout depth or external logic
        else:
            raise TypeError("Unsupported model type for MCTS simulation.")

        return next_state_val, reward_val, done

    def _simulate_rollout(self, state: Any, max_depth: Optional[int] = None) -> float:
        """Performs a random rollout from the given state using the environment model.

        Args:
            state (Any): The starting state for the rollout.
            max_depth (Optional[int]): Maximum depth for the rollout. If None, uses `self.max_depth`.
                                        Defaults to None.

        Returns:
            float: The total discounted reward obtained during the rollout.
        """
        if max_depth is None:
            max_depth = self.max_depth

        total_reward = 0.0
        current_state_in_rollout = state
        discount = 1.0

        for depth in range(max_depth):
            action = np.random.randint(self.num_actions) # Use self.num_actions
            next_state, reward, done = self._simulate_step(current_state_in_rollout, action)

            total_reward += discount * reward
            discount *= self.gamma

            if done:
                break

            current_state_in_rollout = next_state

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagates the reward up the tree, updating node statistics.

        Args:
            node (MCTSNode): The node from which to start backpropagation.
            reward (float): The reward to propagate.
        """
        while node is not None:
            node.update(reward)
            node = node.parent
            reward *= self.gamma  # Discount for parent nodes


class MCTSAgent:
    """Agent using Monte Carlo Tree Search (MCTS) for action selection.

    Args:
        model (Union[TabularModel, NeuralModel]): The environment model.
        num_states (int): The number of states in the environment.
        num_actions (int): The number of actions available to the agent.
        num_simulations (int, optional): Number of MCTS simulations per action selection. Defaults to 1000.
        exploration_weight (float, optional): Exploration constant for UCB. Defaults to 1.0.
    """

    def __init__(
        self,
        model: Union[TabularModel, NeuralModel],
        num_states: int,
        num_actions: int,
        num_simulations: int = 1000,
        exploration_weight: float = 1.0,
    ):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.mcts = MCTS(model, num_actions, exploration_weight) # Pass model directly
        self.num_simulations = num_simulations

        self.search_times: List[float] = []
        self.tree_sizes: List[int] = []
        self.episode_rewards: List[float] = []

    def select_action(self, state: Any, deterministic: bool = False) -> int:
        """Selects an action using MCTS.

        Args:
            state (Any): The current state of the environment.
            deterministic (bool, optional): If True, always selects the action with the most visits.
                                            Defaults to False.

        Returns:
            int: The chosen action.
        """
        start_time = time.time()

        best_action, root = self.mcts.search(state, self.num_simulations)

        search_time = time.time() - start_time
        tree_size = self._count_nodes(root)

        self.search_times.append(search_time)
        self.tree_sizes.append(tree_size)

        if deterministic:
            # For deterministic, get best action based on visit counts
            return root.get_best_action() if root.get_best_action() is not None else np.random.randint(self.num_actions)
        else:
            # Otherwise, use the action selected by MCTS search (which is often based on highest visit count)
            return (
                best_action
                if best_action is not None
                else np.random.randint(self.num_actions) # Fallback to random action
            )

    def _count_nodes(self, node: MCTSNode) -> int:
        """Recursively counts the total number of nodes in the MCTS tree rooted at `node`.

        Args:
            node (MCTSNode): The root of the subtree to count.

        Returns:
            int: The total number of nodes.
        """
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def train_episode(
        self, env: SimpleGridWorld, max_steps: int = 200
    ) -> Tuple[float, int]:
        """Runs a single episode, with the agent selecting actions using MCTS planning.

        Args:
            env (SimpleGridWorld): The environment to interact with.
            max_steps (int, optional): Maximum steps per episode. Defaults to 200.

        Returns:
            Tuple[float, int]: A tuple containing the total reward and number of steps in the episode.
        """
        state, _ = env.reset() # Gymnasium API
        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action) # Gymnasium API
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        self.episode_rewards.append(total_reward)
        return total_reward, steps

    def get_statistics(self) -> Dict[str, Union[float, int]]:
        """Retrieves performance statistics for the MCTS agent.

        Returns:
            Dict[str, Union[float, int]]: A dictionary containing:
                - 'avg_search_time' (float): Average time spent per MCTS search in seconds.
                - 'avg_tree_size' (float): Average number of nodes in the MCTS tree.
                - 'total_searches' (int): Total number of MCTS searches performed.
                - 'avg_episode_reward' (float): Average reward over the last 10 episodes.
        """
        return {
            "avg_search_time": np.mean(self.search_times) if self.search_times else 0.0,
            "avg_tree_size": np.mean(self.tree_sizes) if self.tree_sizes else 0.0,
            "total_searches": len(self.search_times),
            "avg_episode_reward": (
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
            ),
        }


def demonstrate_mcts():
    """Demonstrates the Monte Carlo Tree Search (MCTS) algorithm.

    This function sets up a `SimpleGridWorld` environment, trains a `TabularModel`,
    and then uses an `MCTSAgent` to learn and plan. It visualizes the agent's
    performance, MCTS search characteristics, and compares it against a random policy baseline.
    """
    print("\n--- Monte Carlo Tree Search (MCTS) Demonstration ---")
    print("=" * 50)

    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)

    print("\n1. Setting up environment and training Tabular Model...")
    env = SimpleGridWorld(size=6)
    num_states = env.num_states
    num_actions = env.num_actions
    tabular_model = TabularModel(num_states, num_actions)

    # Collect experience for tabular model
    n_model_train_episodes = 100
    print(f"  Collecting {n_model_train_episodes} episodes of random experience for model training...")
    for episode in range(n_model_train_episodes):
        state, _ = env.reset() # Gymnasium API
        for step in range(50):
            action = np.random.randint(num_actions)
            next_state, reward, terminated, truncated, _ = env.step(action) # Gymnasium API
            done = terminated or truncated
            tabular_model.update(state, action, reward, next_state)
            if done:
                break
            state = next_state
    print(f"  Tabular model trained with {np.sum(tabular_model.sa_counts)} transitions.")

    mcts_agent = MCTSAgent(
        model=tabular_model,
        num_states=num_states,
        num_actions=num_actions,
        num_simulations=200,
        exploration_weight=1.4,
    )

    print("\n2. Testing MCTS agent performance in the environment...")
    n_test_episodes = 20
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    for episode in range(n_test_episodes):
        reward, length = mcts_agent.train_episode(env, max_steps=100)
        episode_rewards.append(reward)
        episode_lengths.append(length)

        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            avg_length = np.mean(episode_lengths[-5:])
            stats = mcts_agent.get_statistics()
            print(
                f"  Episodes {episode-4}-{episode+1}: Avg Reward = {avg_reward:.2f}, "
                f"Avg Length = {avg_length:.1f}, Avg Search Time = {stats['avg_search_time']:.4f}s"
            )

    print("\n3. Generating Visualizations...")
    plt.figure(figsize=(18, 12))

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
    plt.plot(search_times, "purple", linewidth=2, label="Search Time per Step")
    plt.axhline(
        y=np.mean(search_times), color="r", linestyle="--", alpha=0.7, label="Average"
    )
    plt.title("MCTS Search Times")
    plt.xlabel("Search Step")
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
    plt.xlabel("Search Step")
    plt.ylabel("Number of Nodes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    if len(search_times) > 0 and len(tree_sizes) > 0:
        # Ensure inputs to polyfit are not empty
        if len(tree_sizes) > 1 and len(search_times) > 1: # Need at least 2 points for polyfit
            z = np.polyfit(tree_sizes, search_times, 1)
            p = np.poly1d(z)
            plt.plot(
                sorted(tree_sizes), p(sorted(tree_sizes)), "r--", alpha=0.8, linewidth=2
            )
        plt.scatter(tree_sizes, search_times, alpha=0.6, c="red", s=30)
    plt.title("Search Time vs Tree Size")
    plt.xlabel("Tree Size (nodes)")
    plt.ylabel("Search Time (seconds)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    random_rewards: List[float] = []
    print(f"  Collecting {n_test_episodes} episodes of random policy baseline...")
    for _ in range(n_test_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        for step in range(100):
            action = np.random.randint(env.num_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
            state = next_state
        random_rewards.append(total_reward)
    print(f"  Random policy baseline collected. Average reward: {np.mean(random_rewards):.3f}")

    comparison_data = [episode_rewards, random_rewards]
    labels = ["MCTS", "Random"]
    plt.boxplot(comparison_data, labels=labels)
    plt.title("Performance Comparison")
    plt.ylabel("Episode Reward")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/mcts_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n4. MCTS Performance Analysis:")
    final_stats = mcts_agent.get_statistics()
    print(
        f"  Average Episode Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}"
    )
    print(
        f"  Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    print(f"  Average Search Time: {final_stats['avg_search_time']:.4f} seconds")
    print(f"  Average Tree Size: {final_stats['avg_tree_size']:.1f} nodes")
    print(f"  Total MCTS Searches: {final_stats['total_searches']}")

    print(f"\n  Random Policy Baseline:")
    print(
        f"  Average Episode Reward: {np.mean(random_rewards):.3f} ± {np.std(random_rewards):.3f}"
    )

    # Calculate improvement, handle division by zero if random_rewards is all zeros
    mean_random_reward = np.mean(random_rewards)
    if mean_random_reward != 0:
        improvement = (
            (np.mean(episode_rewards) - mean_random_reward)
            / mean_random_reward
            * 100
        )
        print(f"  MCTS Improvement over Random: {improvement:.1f}%")
    else:
        print(f"  Cannot calculate improvement over random policy as baseline reward is zero.")

    print("\n📊 Key MCTS Insights:")
    print("• MCTS provides sophisticated planning through tree search.")
    print("• UCB balances exploration and exploitation in tree nodes.")
    print("• Performance scales with number of simulations (computational cost trade-off).")
    print("• Computational cost grows with search depth and simulations.")
    print("• Effective for discrete action spaces with models that can simulate transitions.")

    print("\n✅ Monte Carlo Tree Search (MCTS) demonstration complete!")
    print("📊 Check 'visualizations/mcts_analysis.png' for plots.")
    print("""
Next: Model Predictive Control (MPC) implementation.
Explore `agents/mpc.py` for its implementation.
""")

if __name__ == "__main__":
    demonstrate_mcts()
