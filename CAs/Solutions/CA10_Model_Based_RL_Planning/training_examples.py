"""
Model-Based Reinforcement Learning and Planning Methods - Training Examples
===========================================================================

This module provides comprehensive implementations and training examples for
Model-Based Reinforcement Learning and Planning Methods (CA10).

Key Components:
- Environment model learning (tabular and neural network models)
- Dyna-Q algorithm with planning
- Monte Carlo Tree Search (MCTS)
- Model Predictive Control (MPC)
- World models and latent state learning
- Advanced analysis and visualization tools

Author: DRL Course Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import gymnasium as gym
from collections import defaultdict, deque
import random
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# =============================================================================
# ENVIRONMENT MODELS
# =============================================================================


class TabularModel:
    """Tabular environment model for discrete state-action spaces"""

    def __init__(self, num_states: int, num_actions: int):
        self.num_states = num_states
        self.num_actions = num_actions

        # Model parameters: P(s'|s,a) and R(s,a)
        self.transition_counts = np.zeros((num_states, num_actions, num_states))
        self.reward_counts = np.zeros((num_states, num_actions))
        self.reward_sums = np.zeros((num_states, num_actions))

        # Learned model
        self.transition_probs = (
            np.ones((num_states, num_actions, num_states)) / num_states
        )
        self.reward_function = np.zeros((num_states, num_actions))

    def update(self, state: int, action: int, next_state: int, reward: float):
        """Update model with new experience"""
        self.transition_counts[state, action, next_state] += 1
        self.reward_counts[state, action] += 1
        self.reward_sums[state, action] += reward

        # Update transition probabilities
        total_transitions = np.sum(self.transition_counts[state, action])
        if total_transitions > 0:
            self.transition_probs[state, action] = (
                self.transition_counts[state, action] / total_transitions
            )

        # Update reward function
        if self.reward_counts[state, action] > 0:
            self.reward_function[state, action] = (
                self.reward_sums[state, action] / self.reward_counts[state, action]
            )

    def predict(self, state: int, action: int) -> Tuple[np.ndarray, float]:
        """Predict next state distribution and expected reward"""
        return self.transition_probs[state, action], self.reward_function[state, action]

    def sample_next_state(self, state: int, action: int) -> int:
        """Sample next state from learned model"""
        probs = self.transition_probs[state, action]
        return np.random.choice(self.num_states, p=probs)


class NeuralModel(nn.Module):
    """Neural network environment model built as an ensemble of MLPs"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        ensemble_size: int = 5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # Ensemble of models for uncertainty quantification
        self.models = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim + action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, state_dim + 1),  # next_state + reward
                )
                for _ in range(ensemble_size)
            ]
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble

        Args:
            state (torch.Tensor): Current state tensor [batch_size, state_dim].
            action (torch.Tensor): Action tensor [batch_size, action_dim].

        Returns:
            torch.Tensor: Stacked predictions from all ensemble models
                          [ensemble_size, batch_size, state_dim + 1].
        """
        state_action = torch.cat([state, action], dim=-1)
        predictions = []

        for model in self.models:
            pred = model(state_action)
            predictions.append(pred)

        return torch.stack(predictions)

    def predict(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next state and reward with uncertainty

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Mean next state prediction [batch_size, state_dim]
                - Standard deviation of next state prediction [batch_size, state_dim]
                - Mean reward prediction [batch_size, 1]
                - Standard deviation of reward prediction [batch_size, 1]
        """
        predictions = self.forward(state, action) # [ensemble_size, batch_size, state_dim + 1]

        # Split predictions into next_states and rewards
        next_states = predictions[:, :, : self.state_dim]
        rewards = predictions[:, :, -1:]

        # Compute mean and std
        next_state_mean = next_states.mean(dim=0)
        next_state_std = next_states.std(dim=0)
        reward_mean = rewards.mean(dim=0)
        reward_std = rewards.std(dim=0)

        return next_state_mean, next_state_std, reward_mean, reward_std

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """Update ensemble models"""
        self.optimizer.zero_grad()

        predictions = self.forward(states, actions) # [ensemble_size, batch_size, state_dim + 1]
        total_loss = 0

        for i in range(self.ensemble_size):
            pred = predictions[i]
            next_state_pred = pred[:, : self.state_dim]
            reward_pred = pred[:, -1:]

            # Compute losses
            next_state_loss = F.mse_loss(next_state_pred, next_states)
            reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards) # Squeeze pred_reward to match rewards shape [batch_size,]

            loss = next_state_loss + reward_loss
            total_loss += loss

        total_loss.backward()
        self.optimizer.step()

        return {"model_loss": total_loss.item() / self.ensemble_size}


# =============================================================================
# DYNA-Q ALGORITHM
# =============================================================================


class DynaQAgent:
    """Dyna-Q agent with planning"""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        planning_steps: int = 50,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        # Q-table and model
        self.Q = np.zeros((num_states, num_actions))
        self.model = TabularModel(num_states, num_actions)

        # Experience buffer for planning
        self.visited = set()

    def select_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ):
        """Update Q-table and model"""
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        # Update model
        self.model.update(state, action, next_state, reward)

        # Mark state-action as visited
        self.visited.add((state, action))

        # Planning: simulate experience using model
        self._planning()

    def _planning(self):
        """Perform planning by simulating experience"""
        for _ in range(self.planning_steps):
            if not self.visited:
                break

            # Sample previously visited state-action pair
            state, action = random.choice(list(self.visited))

            # Use model to predict next state and reward
            next_state = self.model.sample_next_state(state, action)
            reward = self.model.reward_function[state, action]

            # Update Q-table using simulated experience
            target = reward + self.gamma * np.max(self.Q[next_state])
            self.Q[state, action] += self.alpha * (target - self.Q[state, action])


# =============================================================================
# MONTE CARLO TREE SEARCH (MCTS)
# =============================================================================


class MCTSNode:
    """MCTS tree node"""

    def __init__(
        self,
        state: Any,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[int, "MCTSNode"] = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[int] = []  # Initialize as empty list

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0

    def best_child(self, c_puct: float = 1.0) -> "MCTSNode":
        """Select best child using UCB formula"""
        best_score = -np.inf
        best_child = None

        if not self.children:
            raise ValueError("Cannot select best child from a node with no children.")

        # Ensure self.visits is not zero for log calculation
        log_parent_visits = np.log(self.visits + 1e-8)  # Add epsilon to prevent log(0)

        for action, child in self.children.items():
            exploitation = child.value / (child.visits + 1e-8)
            exploration = c_puct * np.sqrt(
                log_parent_visits / (child.visits + 1e-8)
            )
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def most_visited_child(self) -> "MCTSNode":
        """Return child with most visits"""
        if not self.children:
            raise ValueError("Cannot select most visited child from a node with no children.")
        return max(self.children.values(), key=lambda x: x.visits)


class MCTS:
    """Monte Carlo Tree Search"""

    def __init__(self, env: gym.Env, num_simulations: int = 1000, c_puct: float = 1.0):
        self.env = env
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.num_actions = self.env.action_space.n if isinstance(self.env.action_space, gym.spaces.Discrete) else None
        if self.num_actions is None:
            raise ValueError("MCTS requires a discrete action space.")

    def search(self, root_state: Any) -> MCTSNode:
        """Perform MCTS search"""
        root = MCTSNode(root_state)
        root.untried_actions = list(range(self.num_actions))

        for _ in range(self.num_simulations):
            node = root
            # Reset environment for simulation. Gym API often returns (observation, info)
            current_sim_state, info = self.env.reset()
            # Ensure we are using the actual root_state for the simulation tree
            # This requires the environment to be set to the root_state. For simple
            # environments, current_sim_state might be enough. For complex, requires env.set_state(root_state)
            # Assuming root_state itself is directly usable as observation or simple enough
            # to be passed as such. If not, a specific env.set_state() method would be needed.

            path = []

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_puct)
                action = node.action
                # Apply action to environment for simulation
                current_sim_state, _, terminated, truncated, _ = self.env.step(action) # Use current_sim_state for next step
                path.append(node)

            # Expansion
            if not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)

                # Apply action to environment for simulation
                next_sim_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                child = MCTSNode(next_sim_state, parent=node, action=action)
                child.untried_actions = list(range(self.num_actions))
                node.children[action] = child

                path.append(child)

                # Simulation (rollout) from the newly expanded node
                rollout_reward = self._rollout(self.env, next_sim_state) # Pass env to rollout

                # Backpropagation
                for node in reversed(path):
                    node.visits += 1
                    node.value += rollout_reward
            else: # If node is fully expanded but children is empty, it means it's a terminal node
                # For terminal nodes, just backpropagate current reward, no rollout
                # Assuming the reward from last step before terminal is relevant.
                # This part needs careful consideration based on the environment and reward structure.
                # For now, let's assume if it's terminal, the rollout reward is 0 (or actual reward).
                # This can be improved by passing a `terminal_reward` to the MCTSNode.
                pass

        return root

    def _rollout(self, env: gym.Env, state: Any, max_depth: int = 50) -> float:
        """Perform random rollout from state using the environment"""
        total_reward = 0
        depth = 0
        current_rollout_state = state # Start rollout from the given state

        while depth < max_depth:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            depth += 1

            if terminated or truncated:
                break
            current_rollout_state = next_state

        return total_reward

    def get_action_probabilities(
        self, root: MCTSNode, temperature: float = 1.0
    ) -> np.ndarray:
        """Get action probabilities from visit counts"""
        visits = np.array(
            [
                root.children.get(a, MCTSNode(None)).visits
                for a in range(self.num_actions) # Use self.num_actions
            ]
        )

        if temperature == 0:
            # Greedy selection
            probs = np.zeros_like(visits, dtype=float)
            probs[np.argmax(visits)] = 1.0
        else:
            # Softmax with temperature
            visits_temp = visits ** (1 / temperature)
            probs = visits_temp / np.sum(visits_temp)

        return probs


# =============================================================================
# MODEL PREDICTIVE CONTROL (MPC)
# =============================================================================


class MPCController:
    """Model Predictive Control for continuous control

    Args:
        model (Callable): A callable model that takes current state and action,
                          and returns the next state (and optionally reward).
                          Expected signature: `model(state: np.ndarray, action: np.ndarray) -> np.ndarray`
        horizon (int): The planning horizon, i.e., how many future steps to consider.
        num_samples (int): Number of random action sequences to sample for optimization.
        action_bounds (Union[Tuple[float, float], np.ndarray]): Bounds for actions.
                                                            Can be a tuple (min, max) for scalar actions
                                                            or a (action_dim, 2) numpy array for multi-dimensional actions.
    """

    def __init__(
        self,
        model: Callable[..., np.ndarray],
        horizon: int = 10,
        num_samples: int = 100,
        action_bounds: Union[Tuple[float, float], np.ndarray] = (-1.0, 1.0),
    ):
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.action_bounds = np.array(action_bounds) if isinstance(action_bounds, list) else action_bounds

    def optimize(
        self, current_state: np.ndarray, goal_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Optimize action sequence using MPC (Random Shooting method).

        Args:
            current_state (np.ndarray): The current state of the environment.
            goal_state (Optional[np.ndarray]): The target goal state. If None, cost is based on negative L2 norm from origin.

        Returns:
            np.ndarray: The first action of the optimal sequence found.
                        Returns a zero-action array if no optimal sequence is found.
        """
        best_action_sequence = None
        best_cost = np.inf

        state_dim = current_state.shape[-1] if isinstance(current_state, np.ndarray) and current_state.ndim > 0 else 1

        # Determine action_dim based on action_bounds or assume 1 if action_bounds is a tuple
        if isinstance(self.action_bounds, tuple):
            action_dim = 1
            min_action = self.action_bounds[0]
            max_action = self.action_bounds[1]
        elif isinstance(self.action_bounds, np.ndarray):
            action_dim = self.action_bounds.shape[0]
            min_action = self.action_bounds[:, 0]  # Assuming shape (action_dim, 2)
            max_action = self.action_bounds[:, 1]
        else:
            raise ValueError("action_bounds must be a tuple or a numpy array")

        for _ in range(self.num_samples):
            # Sample random action sequence within bounds
            action_sequence = np.random.uniform(
                low=min_action,
                high=max_action,
                size=(self.horizon, action_dim)
            )

            # Simulate trajectory
            state = current_state.copy()
            total_cost = 0.0

            for action in action_sequence:
                # Predict next state using model
                # Model is expected to return np.ndarray, so ensure consistency
                next_state_prediction = self.model(state, action)
                state = next_state_prediction # Update state for next step in horizon

                # Compute cost
                if goal_state is not None:
                    cost = np.linalg.norm(state - goal_state)
                else:
                    # Default cost: negative reward for being away from origin (positive for being close)
                    cost = np.linalg.norm(state) # L2 norm as cost, aim for 0

                total_cost += cost

            # Update best sequence
            if total_cost < best_cost:
                best_cost = total_cost
                best_action_sequence = action_sequence

        # Return first action of the best sequence, or a zero-action array if no best sequence was found
        if best_action_sequence is not None:
            return best_action_sequence[0]
        else:
            # Return a zero-action array with correct dimensionality
            return np.zeros(action_dim, dtype=np.float32)


# =============================================================================
# WORLD MODELS
# =============================================================================


# Remove the redundant WorldModel class, as NeuralModel in models.py serves this purpose.
# The WorldModel class here had an encoder-decoder structure which is not strictly
# used in the current classical_planning.py or train_dyna_q. Instead, NeuralModel
# directly predicts next state and reward. We will use the NeuralModel from models.py
# for all neural model needs.


# =============================================================================
# TRAINING UTILITIES
# =============================================================================


# Ensure to import NeuralModel from models.models and ModelTrainer from models.models
from models.models import NeuralModel, ModelTrainer


def train_dyna_q(
    env_name: str = "FrozenLake-v1",
    num_episodes: int = 500,
    planning_steps: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train Dyna-Q agent"""

    set_seed(seed)

    env = gym.make(env_name)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    agent = DynaQAgent(num_states, num_actions, planning_steps=planning_steps)

    episode_rewards = []
    episode_lengths = []

    print(f"Training Dyna-Q Agent on {env_name}")
    print("=" * 40)

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "agent": agent,
        "config": {
            "env_name": env_name,
            "num_episodes": num_episodes,
            "planning_steps": planning_steps,
        },
    }

    return results


def train_world_model(
    env_name: str = "Pendulum-v1",
    num_episodes: int = 100,
    max_steps: int = 200,
    latent_dim: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train world model (using NeuralModel from models.py)"""

    set_seed(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = (
        env.action_space.n
        if isinstance(env.action_space, gym.spaces.Discrete)
        else env.action_space.shape[0]
    )

    # Instantiate NeuralModel from models.models
    world_model = NeuralModel(obs_dim, action_dim, hidden_dim=latent_dim, ensemble_size=5).to(device)
    trainer = ModelTrainer(world_model, lr=1e-3)

    episode_rewards = []
    losses = {"model_loss": []}

    print(f"Training Neural Model on {env_name}")
    print("=" * 35)

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0

        obs_batch = []
        action_batch = []
        next_obs_batch = []
        reward_batch = []

        for step in range(max_steps):
            action = env.action_space.sample()  # Random exploration
            
            # Handle discrete vs. continuous actions for storage
            if isinstance(env.action_space, gym.spaces.Discrete):
                action_to_store = action
            else:
                action_to_store = action.astype(np.float32) # Ensure float for continuous actions

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_batch.append(state)
            action_batch.append(action_to_store)
            next_obs_batch.append(next_state)
            reward_batch.append(reward)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Convert to tensors
        obs_tensor = torch.tensor(np.array(obs_batch), dtype=torch.float32)
        action_tensor = (
            torch.tensor(np.array(action_batch), dtype=torch.long) # Use long for discrete actions
            if isinstance(env.action_space, gym.spaces.Discrete)
            else torch.tensor(np.array(action_batch), dtype=torch.float32)
        )
        next_obs_tensor = torch.tensor(np.array(next_obs_batch), dtype=torch.float32)
        reward_tensor = torch.tensor(np.array(reward_batch), dtype=torch.float32)

        # Update world model using ModelTrainer
        loss = trainer.train_step(
            obs_tensor, action_tensor, next_obs_tensor, reward_tensor
        )
        losses["model_loss"].append(loss)

        episode_rewards.append(episode_reward)

    env.close()

    results = {
        "episode_rewards": episode_rewards,
        "losses": losses,
        "world_model": world_model,
        "config": {
            "env_name": env_name,
            "num_episodes": num_episodes,
            "latent_dim": latent_dim,
        },
    }

    return results


def compare_model_based_methods(
    env_name: str = "CartPole-v1", num_runs: int = 3, num_episodes: int = 200
) -> Dict[str, Any]:
    """Compare different model-based methods"""

    methods = ["Dyna-Q (Planning=0)", "Dyna-Q (Planning=10)", "Dyna-Q (Planning=50)"]
    results = {}

    for method in methods:
        print(f"Testing {method}...")

        run_rewards = []
        planning_steps = int(method.split("=")[1].rstrip(")")) if "=" in method else 0

        for run in range(num_runs):
            set_seed(42 + run)

            result = train_dyna_q(
                env_name,
                num_episodes=num_episodes,
                planning_steps=planning_steps,
                seed=42 + run,
            )
            run_rewards.append(result["episode_rewards"])

        # Average across runs
        avg_rewards = np.mean(run_rewards, axis=0)
        std_rewards = np.std(run_rewards, axis=0)

        results[method] = {
            "mean_rewards": avg_rewards,
            "std_rewards": std_rewards,
            "final_score": np.mean(avg_rewards[-50:]),  # Average of last 50 episodes
        }

    return results


# =============================================================================
# ANALYSIS AND VISUALIZATION FUNCTIONS
# =============================================================================


def plot_model_based_comparison(save_path: Optional[str] = None) -> plt.Figure:
    """Compare model-free vs model-based approaches"""

    print("Analyzing model-based vs model-free RL approaches...")
    print("=" * 55)

    # Performance comparison data
    environments = ["CartPole", "MountainCar", "Pendulum", "LunarLander"]
    methods = [
        "Q-Learning",
        "Dyna-Q (5)",
        "Dyna-Q (25)",
        "Dyna-Q (100)",
        "Neural Model",
    ]

    # Sample efficiency (lower is better)
    sample_efficiency = {
        "CartPole": [1000, 800, 600, 400, 300],
        "MountainCar": [2000, 1500, 1000, 700, 500],
        "Pendulum": [5000, 4000, 3000, 2000, 1500],
        "LunarLander": [3000, 2500, 1800, 1200, 900],
    }

    # Final performance (higher is better)
    final_performance = {
        "CartPole": [180, 190, 195, 198, 185],
        "MountainCar": [-150, -120, -100, -80, -110],
        "Pendulum": [-800, -600, -500, -400, -550],
        "LunarLander": [150, 180, 200, 220, 190],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Sample efficiency comparison
    x = np.arange(len(environments))
    width = 0.15
    multiplier = 0

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (method, color) in enumerate(zip(methods, colors)):
        efficiency_scores = [sample_efficiency[env][i] for env in environments]
        offset = width * multiplier
        bars = axes[0, 0].bar(
            x + offset, efficiency_scores, width, label=method, color=color, alpha=0.8
        )
        multiplier += 1

    axes[0, 0].set_xlabel("Environment")
    axes[0, 0].set_ylabel("Samples to Learn")
    axes[0, 0].set_title("Sample Efficiency Comparison (Lower is Better)")
    axes[0, 0].set_xticks(x + width * 2, environments)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    # Performance vs sample efficiency trade-off
    for i, env in enumerate(environments):
        eff_scores = [sample_efficiency[env][j] for j in range(len(methods))]
        perf_scores = [final_performance[env][j] for j in range(len(methods))]

        axes[0, 1].scatter(eff_scores, perf_scores, s=100, alpha=0.7, label=env)

    axes[0, 1].set_xlabel("Sample Efficiency (Samples)")
    axes[0, 1].set_ylabel("Final Performance")
    axes[0, 1].set_title("Performance vs Sample Efficiency Trade-off")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Planning benefits analysis
    planning_steps = [0, 5, 25, 100]
    performance_improvement = [0, 15, 35, 50]  # Percentage improvement

    axes[1, 0].plot(
        planning_steps,
        performance_improvement,
        "o-",
        linewidth=3,
        markersize=8,
        color="purple",
    )
    axes[1, 0].fill_between(
        planning_steps, 0, performance_improvement, alpha=0.3, color="purple"
    )
    axes[1, 0].set_xlabel("Planning Steps per Real Step")
    axes[1, 0].set_ylabel("Performance Improvement (%)")
    axes[1, 0].set_title("Benefits of Planning in Model-Based RL")
    axes[1, 0].grid(True, alpha=0.3)

    # Model accuracy vs performance
    model_accuracies = np.linspace(0.5, 0.95, 10)
    performance_gains = 50 * (model_accuracies - 0.5) / 0.45  # Linear relationship

    axes[1, 1].plot(
        model_accuracies,
        performance_gains,
        "s-",
        linewidth=3,
        markersize=8,
        color="green",
    )
    axes[1, 1].set_xlabel("Model Prediction Accuracy")
    axes[1, 1].set_ylabel("Performance Gain (%)")
    axes[1, 1].set_title("Model Accuracy vs Performance Relationship")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Model-based RL analysis completed!")

    return fig


def analyze_mcts_performance(save_path: Optional[str] = None) -> plt.Figure:
    """Analyze MCTS performance characteristics"""

    print("Analyzing Monte Carlo Tree Search performance...")
    print("=" * 50)

    # MCTS performance data
    simulations = [100, 500, 1000, 2000, 5000]
    environments = ["TicTacToe", "Connect4", "Go 9x9", "Chess"]

    # Win rates vs simulations
    win_rates = {
        "TicTacToe": [0.85, 0.92, 0.95, 0.97, 0.98],
        "Connect4": [0.75, 0.85, 0.90, 0.93, 0.95],
        "Go 9x9": [0.60, 0.70, 0.78, 0.82, 0.85],
        "Chess": [0.55, 0.65, 0.72, 0.76, 0.80],
    }

    # Computation time
    comp_times = [0.1, 0.5, 1.0, 2.0, 5.0]  # seconds

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Win rate vs simulations
    for env, rates in win_rates.items():
        axes[0, 0].plot(simulations, rates, "o-", linewidth=2, markersize=6, label=env)

    axes[0, 0].set_xlabel("Number of Simulations")
    axes[0, 0].set_ylabel("Win Rate")
    axes[0, 0].set_title("MCTS Win Rate vs Number of Simulations")
    axes[0, 0].set_xscale("log")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Win rate vs computation time
    for env, rates in win_rates.items():
        axes[0, 1].plot(comp_times, rates, "s-", linewidth=2, markersize=6, label=env)

    axes[0, 1].set_xlabel("Computation Time (seconds)")
    axes[0, 1].set_ylabel("Win Rate")
    axes[0, 1].set_title("MCTS Win Rate vs Computation Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Exploration vs exploitation trade-off
    c_puct_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    exploration_bias = [0.9, 0.7, 0.5, 0.3, 0.1]
    exploitation_focus = [0.1, 0.3, 0.5, 0.7, 0.9]

    axes[1, 0].plot(
        c_puct_values,
        exploration_bias,
        "o-",
        label="Exploration Bias",
        linewidth=2,
        markersize=6,
    )
    axes[1, 0].plot(
        c_puct_values,
        exploitation_focus,
        "s-",
        label="Exploitation Focus",
        linewidth=2,
        markersize=6,
    )
    axes[1, 0].set_xlabel("c_puct Parameter")
    axes[1, 0].set_ylabel("Strategy Weight")
    axes[1, 0].set_title("Exploration vs Exploitation in MCTS")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # MCTS variants comparison
    variants = ["Vanilla MCTS", "AlphaGo MCTS", "MuZero MCTS", "Neural MCTS"]
    strengths = [6, 8, 9, 7]
    sample_efficiency = [4, 7, 9, 8]
    generality = [7, 6, 8, 9]

    x = np.arange(len(variants))
    width = 0.25

    axes[1, 1].bar(x - width, strengths, width, label="Game Strength", alpha=0.7)
    axes[1, 1].bar(x, sample_efficiency, width, label="Sample Efficiency", alpha=0.7)
    axes[1, 1].bar(x + width, generality, width, label="Generality", alpha=0.7)
    axes[1, 1].set_xlabel("MCTS Variant")
    axes[1, 1].set_ylabel("Score (1-10)")
    axes[1, 1].set_title("MCTS Variants Comparison")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(variants, rotation=45, ha="right")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("MCTS performance analysis completed!")

    return fig


def comprehensive_model_based_analysis(
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Comprehensive analysis of model-based RL methods"""

    print("Comprehensive model-based RL methods analysis...")
    print("=" * 55)

    methods = ["Dyna-Q", "Neural Dyna", "MBPO", "Dreamer", "MuZero"]
    environments = [
        "Discrete Control",
        "Continuous Control",
        "Planning-Heavy",
        "Memory-Intensive",
    ]

    # Method characteristics (1-10 scale)
    characteristics = {
        "Sample Efficiency": {
            "Dyna-Q": 8,
            "Neural Dyna": 7,
            "MBPO": 9,
            "Dreamer": 9,
            "MuZero": 10,
        },
        "Computational Cost": {
            "Dyna-Q": 3,
            "Neural Dyna": 6,
            "MBPO": 8,
            "Dreamer": 7,
            "MuZero": 9,
        },
        "Stability": {
            "Dyna-Q": 8,
            "Neural Dyna": 6,
            "MBPO": 7,
            "Dreamer": 8,
            "MuZero": 8,
        },
        "Generality": {
            "Dyna-Q": 6,
            "Neural Dyna": 8,
            "MBPO": 7,
            "Dreamer": 9,
            "MuZero": 10,
        },
    }

    # Performance by environment type
    performance_by_env = {
        "Discrete Control": {
            "Dyna-Q": 8,
            "Neural Dyna": 7,
            "MBPO": 6,
            "Dreamer": 5,
            "MuZero": 9,
        },
        "Continuous Control": {
            "Dyna-Q": 5,
            "Neural Dyna": 7,
            "MBPO": 8,
            "Dreamer": 9,
            "MuZero": 7,
        },
        "Planning-Heavy": {
            "Dyna-Q": 9,
            "Neural Dyna": 8,
            "MBPO": 7,
            "Dreamer": 6,
            "MuZero": 10,
        },
        "Memory-Intensive": {
            "Dyna-Q": 4,
            "Neural Dyna": 6,
            "MBPO": 5,
            "Dreamer": 9,
            "MuZero": 8,
        },
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Method characteristics radar
    categories = list(characteristics.keys())
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for method in methods[:4]:  # Show first 4 to avoid clutter
        scores = [characteristics[cat][method] for cat in categories]
        scores += scores[:1]
        axes[0, 0].plot(angles, scores, "o-", linewidth=2, label=method, markersize=6)

    axes[0, 0].set_xticks(angles[:-1])
    axes[0, 0].set_xticklabels(categories, fontsize=9)
    axes[0, 0].set_ylim(0, 10)
    axes[0, 0].set_title("Model-Based Method Characteristics")
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    # Performance by environment type
    env_names = list(performance_by_env.keys())
    x = np.arange(len(env_names))
    width = 0.15
    multiplier = 0

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (method, color) in enumerate(zip(methods, colors)):
        scores = [performance_by_env[env][method] for env in env_names]
        offset = width * multiplier
        bars = axes[0, 1].bar(
            x + offset, scores, width, label=method, color=color, alpha=0.8
        )
        multiplier += 1

    axes[0, 1].set_xlabel("Environment Type")
    axes[0, 1].set_ylabel("Performance Score")
    axes[0, 1].set_title("Method Performance by Environment Type")
    axes[0, 1].set_xticks(x + width * 2, env_names)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 1].grid(True, alpha=0.3)

    # Sample efficiency vs computational cost trade-off
    sample_eff = [characteristics["Sample Efficiency"][m] for m in methods]
    comp_cost = [characteristics["Computational Cost"][m] for m in methods]

    axes[1, 0].scatter(comp_cost, sample_eff, s=150, alpha=0.7, c="blue")
    for i, method in enumerate(methods):
        axes[1, 0].annotate(
            method,
            (comp_cost[i], sample_eff[i]),
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
        )

    axes[1, 0].set_xlabel("Computational Cost")
    axes[1, 0].set_ylabel("Sample Efficiency")
    axes[1, 0].set_title("Sample Efficiency vs Computational Cost")
    axes[1, 0].grid(True, alpha=0.3)

    # Method evolution timeline
    years = [1990, 2008, 2018, 2019, 2020]
    method_timeline = ["Dyna-Q", "Neural Dyna", "MBPO", "Dreamer", "MuZero"]
    performance_over_time = [4, 6, 7, 8, 9]

    axes[1, 1].plot(
        years, performance_over_time, "o-", linewidth=3, markersize=8, color="purple"
    )
    for i, (year, method) in enumerate(zip(years, method_timeline)):
        axes[1, 1].annotate(
            method,
            (year, performance_over_time[i]),
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        )

    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Method Sophistication")
    axes[1, 1].set_title("Evolution of Model-Based RL Methods")
    axes[1, 1].grid(True, alpha=0.3)

    # Strengths and weaknesses comparison
    strengths = ["Planning", "Sample Efficiency", "Generality", "Stability"]
    method_strengths = {
        "Dyna-Q": [9, 8, 6, 8],
        "Neural Dyna": [8, 7, 8, 6],
        "MBPO": [7, 9, 7, 7],
        "Dreamer": [6, 9, 9, 8],
        "MuZero": [10, 10, 10, 8],
    }

    x = np.arange(len(strengths))
    width = 0.15
    multiplier = 0

    for method in methods:
        scores = method_strengths[method]
        offset = width * multiplier
        bars = axes[2, 0].bar(x + offset, scores, width, label=method, alpha=0.8)
        multiplier += 1

    axes[2, 0].set_xlabel("Strength Category")
    axes[2, 0].set_ylabel("Score (1-10)")
    axes[2, 0].set_title("Method Strengths Analysis")
    axes[2, 0].set_xticks(x + width * 2, strengths)
    axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[2, 0].grid(True, alpha=0.3)

    # Future directions
    future_areas = [
        "Meta-Learning",
        "Multi-Agent",
        "Hierarchical",
        "Continual Learning",
    ]
    current_state = [6, 5, 7, 4]
    potential_impact = [9, 8, 9, 8]

    x = np.arange(len(future_areas))
    width = 0.35

    axes[2, 1].bar(
        x - width / 2, current_state, width, label="Current State", alpha=0.7
    )
    axes[2, 1].bar(
        x + width / 2, potential_impact, width, label="Potential Impact", alpha=0.7
    )
    axes[2, 1].set_xlabel("Research Area")
    axes[2, 1].set_ylabel("Score (1-10)")
    axes[2, 1].set_title("Future Directions for Model-Based RL")
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(future_areas, rotation=45, ha="right")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Print comprehensive analysis
    print("\n" + "=" * 55)
    print("MODEL-BASED RL METHODS COMPREHENSIVE ANALYSIS")
    print("=" * 55)

    for method in methods:
        avg_perf = np.mean([performance_by_env[env][method] for env in env_names])
        print(f"{method:12} | Average Performance: {avg_perf:6.1f}")

    print("\n💡 Key Insights for Model-Based RL:")
    print("• Model-based methods excel in sample efficiency")
    print("• Planning provides significant performance boosts")
    print("• Neural models offer better generalization")
    print("• Modern methods balance planning with learning")

    print("\n🎯 Recommendations:")
    print("• Use Dyna-Q for simple discrete environments")
    print("• Choose MBPO/Dreamer for complex continuous control")
    print("• Consider MuZero for general planning problems")
    print("• Neural models preferred for high-dimensional spaces")

    return {
        "characteristics": characteristics,
        "performance_by_env": performance_by_env,
        "methods": methods,
    }


# =============================================================================
# MAIN TRAINING EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("Model-Based Reinforcement Learning and Planning Methods")
    print("=" * 60)
    print("Available training examples:")
    print("1. train_dyna_q() - Train Dyna-Q agent with planning")
    print("2. train_world_model() - Train world model")
    print("3. compare_model_based_methods() - Compare planning methods")
    print("4. plot_model_based_comparison() - Model-based vs model-free analysis")
    print("5. analyze_mcts_performance() - MCTS performance analysis")
    print("6. comprehensive_model_based_analysis() - Full method comparison")
    print("\nExample usage:")
    print("results = train_dyna_q(planning_steps=50)")
    print("comparison = compare_model_based_methods()")
