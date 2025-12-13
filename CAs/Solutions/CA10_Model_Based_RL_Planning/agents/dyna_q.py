import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys
import os
from typing import Dict, Set, Tuple, Any, List, Union
import gymnasium as gym

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import TabularModel
from environments.environments import SimpleGridWorld, BlockingMaze


class DynaQAgent:
    """Dyna-Q Agent implementing integrated planning and learning.

    This agent combines direct Q-learning updates from real environment interactions
    with simulated planning steps using a learned tabular model of the environment.

    Args:
        num_states (int): The number of states in the environment.
        num_actions (int): The number of actions available to the agent.
        alpha (float, optional): The learning rate for Q-value updates. Defaults to 0.1.
        gamma (float, optional): The discount factor. Defaults to 0.95.
        epsilon (float, optional): The exploration rate for ε-greedy policy. Defaults to 0.1.
        planning_steps (int, optional): The number of planning steps to perform per real interaction.
                                      Defaults to 5.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        planning_steps: int = 5,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        self.Q = np.zeros((num_states, num_actions), dtype=np.float32)

        # Use TabularModel for environment model
        self.model = TabularModel(num_states, num_actions)
        self.visited_state_actions: Set[Tuple[int, int]] = set()  # Track visited (s,a) pairs

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.planning_updates: int = 0
        self.direct_updates: int = 0

    def select_action(self, state: int) -> int:
        """Selects an action using an ε-greedy policy.

        Args:
            state (int): The current state.

        Returns:
            int: The selected action.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def update_q_function(
        self, state: int, action: int, reward: float, next_state: int
    ) -> float:
        """Performs a Q-learning update for a given experience.

        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state observed.

        Returns:
            float: The Temporal Difference (TD) error for the update.
        """
        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error

    def update_model(self, state: int, action: int, reward: float, next_state: int):
        """Updates the internal tabular environment model with new experience.

        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state observed.
        """
        self.model.update(state, action, next_state, reward)
        self.visited_state_actions.add((state, action))

    def planning_update(self):
        """Performs `planning_steps` simulated planning updates using the learned model.
        Each update randomly samples a previously visited state-action pair and uses
        the model to simulate a transition, then performs a Q-learning update.
        """
        if len(self.visited_state_actions) == 0:
            return

        for _ in range(self.planning_steps):
            state, action = random.choice(list(self.visited_state_actions))

            # Use TabularModel to sample next state and get expected reward
            next_state, reward = self.model.sample_transition(state, action)

            self.update_q_function(state, action, reward, next_state)
            self.planning_updates += 1

    def train_episode(
        self, env: gym.Env, max_steps: int = 200
    ) -> Tuple[float, int]:
        """Trains the agent for a single episode.

        Args:
            env (gym.Env): The environment to interact with.
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

            self.update_q_function(state, action, reward, next_state)
            self.direct_updates += 1

            self.update_model(state, action, reward, next_state)

            self.planning_update()

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)

        return total_reward, steps

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Retrieves various training statistics from the agent.

        Returns:
            Dict[str, Union[int, float]]: A dictionary containing:
                - 'direct_updates' (int): Number of updates from real experience.
                - 'planning_updates' (int): Number of updates from simulated experience.
                - 'model_size' (int): Number of (s,a) pairs stored in the model.
                - 'avg_episode_reward' (float): Average reward over the last 10 episodes.
        """
        return {
            "direct_updates": self.direct_updates,
            "planning_updates": self.planning_updates,
            "model_size": len(self.visited_state_actions),
            "avg_episode_reward": (
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
            ),
        }


class DynaQPlusAgent(DynaQAgent):
    """Dyna-Q+ Agent, an extension of Dyna-Q with an exploration bonus for states/actions
    that have not been visited for a long time. This encourages exploration in changing environments.

    Args:
        num_states (int): The number of states in the environment.
        num_actions (int): The number of actions available to the agent.
        alpha (float, optional): The learning rate for Q-value updates. Defaults to 0.1.
        gamma (float, optional): The discount factor. Defaults to 0.95.
        epsilon (float, optional): The exploration rate for ε-greedy policy. Defaults to 0.1.
        planning_steps (int, optional): The number of planning steps to perform per real interaction.
                                      Defaults to 5.
        kappa (float, optional): The exploration bonus weight. Defaults to 0.001.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        planning_steps: int = 5,
        kappa: float = 0.001,
    ):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon, planning_steps)

        self.kappa = kappa  # Exploration bonus weight
        self.last_visit_time: Dict[Tuple[int, int], int] = {}  # Track when each (s,a) was last tried
        self.current_time: int = 0

    def update_q_function(
        self, state: int, action: int, reward: float, next_state: int, is_real_experience: bool = True
    ) -> float:
        """Performs an enhanced Q-learning update with an exploration bonus.

        If `is_real_experience` is False (i.e., planning), an exploration bonus is added
        to the reward based on the time since the (state, action) pair was last visited.

        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state observed.
            is_real_experience (bool, optional): True if this is a real experience, False if simulated.
                                                Defaults to True.

        Returns:
            float: The Temporal Difference (TD) error for the update.
        """
        if is_real_experience:
            self.last_visit_time[(state, action)] = self.current_time
            self.current_time += 1

        exploration_bonus = 0.0
        if not is_real_experience and (state, action) in self.last_visit_time:
            time_since_visit = self.current_time - self.last_visit_time[(state, action)]
            exploration_bonus = self.kappa * np.sqrt(time_since_visit)

        td_target = reward + exploration_bonus + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

        return td_error

    def planning_update(self):
        """Performs `planning_steps` simulated planning updates for Dyna-Q+.

        Similar to Dyna-Q, but uses the `update_q_function` with `is_real_experience=False`
        to incorporate the exploration bonus.
        """
        if len(self.visited_state_actions) == 0:
            return

        for _ in range(self.planning_steps):
            state, action = random.choice(list(self.visited_state_actions))

            # Use TabularModel to sample next state and get expected reward
            next_state, reward = self.model.sample_transition(state, action)

            self.update_q_function(
                state, action, reward, next_state, is_real_experience=False
            )
            self.planning_updates += 1

    def train_episode(
        self, env: gym.Env, max_steps: int = 200
    ) -> Tuple[float, int]:
        """Trains the Dyna-Q+ agent for a single episode.

        Args:
            env (gym.Env): The environment to interact with.
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

            self.update_q_function(
                state, action, reward, next_state, is_real_experience=True
            )
            self.direct_updates += 1

            self.update_model(state, action, reward, next_state)
            self.planning_update()

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)

        return total_reward, steps


def demonstrate_dyna_q():
    """Demonstrates the Dyna-Q and Dyna-Q+ algorithms through training and visualization.

    This function compares the learning performance and sample efficiency of Q-Learning
    (Dyna-Q with 0 planning steps), Dyna-Q with varying planning steps, and Dyna-Q+
    in both a stable `SimpleGridWorld` and a changing `BlockingMaze` environment.
    Visualizations are generated to illustrate key insights.
    """
    print("\n--- Dyna-Q Algorithm Demonstration ---")
    print("=" * 50)

    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1. Training on Simple GridWorld
    print("\n1. Training Dyna-Q agents on Simple GridWorld:")
    simple_env = SimpleGridWorld(size=5)

    agents: Dict[str, DynaQAgent] = {
        "Q-Learning (n=0)": DynaQAgent(
            simple_env.num_states, simple_env.num_actions, planning_steps=0, epsilon=0.1
        ),
        "Dyna-Q (n=5)": DynaQAgent(
            simple_env.num_states, simple_env.num_actions, planning_steps=5, epsilon=0.1
        ),
        "Dyna-Q (n=50)": DynaQAgent(
            simple_env.num_states, simple_env.num_actions, planning_steps=50, epsilon=0.1
        ),
        "Dyna-Q+ (n=5)": DynaQPlusAgent(
            simple_env.num_states, simple_env.num_actions, planning_steps=5, kappa=0.001, epsilon=0.1
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}
    n_episodes_simple = 200

    for name, agent in agents.items():
        print(f"\n  Training {name}...")
        # Reset agent's internal state for each training run
        agent.Q = np.zeros((simple_env.num_states, simple_env.num_actions), dtype=np.float32)
        agent.model = TabularModel(simple_env.num_states, simple_env.num_actions) # Re-init model
        agent.visited_state_actions = set()
        agent.episode_rewards = []
        agent.episode_lengths = []
        agent.direct_updates = 0
        agent.planning_updates = 0

        if isinstance(agent, DynaQPlusAgent):
            agent.last_visit_time = {}
            agent.current_time = 0

        for episode in range(n_episodes_simple):
            reward, _ = agent.train_episode(simple_env, max_steps=100)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(agent.episode_rewards[-10:])
                stats = agent.get_statistics()
                print(
                    f"    Episode {episode+1}: Avg Reward = {avg_reward:.3f}, "
                    f"Direct Updates = {stats['direct_updates']}, "
                    f"Planning Updates = {stats['planning_updates']}"
                )

        results[name] = {
            "episode_rewards": agent.episode_rewards.copy(),
            "statistics": agent.get_statistics(),
        }

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    colors = ["blue", "red", "green", "orange"]
    for i, (name, data) in enumerate(results.items()):
        rewards = np.array(data["episode_rewards"])
        smoothed = pd.Series(rewards).rolling(window=10, min_periods=1).mean()
        plt.plot(smoothed, label=name, color=colors[i], linewidth=2)

    plt.title("Simple GridWorld: Learning Performance Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    agent_names = list(results.keys())
    direct_updates = [
        results[name]["statistics"]["direct_updates"] for name in agent_names
    ]
    planning_updates = [
        results[name]["statistics"]["planning_updates"] for name in agent_names
    ]

    x = np.arange(len(agent_names))
    width = 0.35

    plt.bar(x - width / 2, direct_updates, width, label="Direct Updates", alpha=0.7)
    plt.bar(x + width / 2, planning_updates, width, label="Planning Updates", alpha=0.7)

    plt.title("Simple GridWorld: Update Statistics")
    plt.xlabel("Agent")
    plt.ylabel("Number of Updates")
    plt.xticks(x, agent_names, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Testing on Blocking Maze (Environment Change)
    print("\n2. Testing Dyna-Q agents on Blocking Maze (Environment Change at Episode 100):")
    maze_env = BlockingMaze(size=5, change_episode=100)

    maze_agents: Dict[str, DynaQAgent] = {
        "Dyna-Q (n=50)": DynaQAgent(
            maze_env.num_states, maze_env.num_actions, planning_steps=50, epsilon=0.1
        ),
        "Dyna-Q+ (n=50)": DynaQPlusAgent(
            maze_env.num_states, maze_env.num_actions, planning_steps=50, kappa=0.01, epsilon=0.1
        ),
    }

    maze_results: Dict[str, Dict[str, List[Union[float, int]]]] = {}
    n_episodes_maze = 300

    for name, agent in maze_agents.items():
        print(f"\n  Training {name} on Blocking Maze...")
        # Reset agent's internal state for each training run
        agent.Q = np.zeros((maze_env.num_states, maze_env.num_actions), dtype=np.float32)
        agent.model = TabularModel(maze_env.num_states, maze_env.num_actions) # Re-init model
        agent.visited_state_actions = set()
        agent.episode_rewards = []
        agent.episode_lengths = []
        agent.direct_updates = 0
        agent.planning_updates = 0

        if isinstance(agent, DynaQPlusAgent):
            agent.last_visit_time = {}
            agent.current_time = 0

        # Need to manually reset episode_count for the environment for each agent run
        # The BlockingMaze environment uses episode_count to trigger changes
        maze_env.reset_episode_count()

        for episode in range(n_episodes_maze):
            # Increment episode count for the environment to trigger changes correctly
            maze_env.episode_count = episode
            reward, steps = agent.train_episode(maze_env, max_steps=3000)

            if (episode + 1) % 50 == 0 or episode == maze_env.change_episode -1 or episode == maze_env.change_episode:
                print(f"    Episode {episode+1}: Reward = {reward:.1f}, Steps = {steps}")

        maze_results[name] = {
            "episode_rewards": agent.episode_rewards.copy(),
            "episode_lengths": agent.episode_lengths.copy(),
        }

    plt.subplot(2, 2, 3)
    for name, data in maze_results.items():
        rewards = np.array(data["episode_rewards"])
        smoothed = pd.Series(rewards).rolling(window=20, min_periods=1).mean()
        plt.plot(smoothed, label=name, linewidth=2)

    plt.axvline(
        x=maze_env.change_episode, color="red", linestyle="--", alpha=0.7, label="Environment Change"
    )
    plt.title("Blocking Maze: Performance Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    for name, data in maze_results.items():
        lengths = np.array(data["episode_lengths"])
        smoothed = pd.Series(lengths).rolling(window=20, min_periods=1).mean()
        plt.plot(smoothed, label=name, linewidth=2)

    plt.axvline(
        x=maze_env.change_episode, color="red", linestyle="--", alpha=0.7, label="Environment Change"
    )
    plt.title("Blocking Maze: Episode Length (Steps to Goal)")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/dyna_q_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n3. Key Insights from Dyna-Q Experiments:")
    print("\n  Simple GridWorld Results:")
    for name, data in results.items():
        # Ensure there are enough rewards for a meaningful average
        if len(data["episode_rewards"]) >= 20:
            final_performance = np.mean(data["episode_rewards"][-20:])
        else:
            final_performance = np.mean(data["episode_rewards"]) if data["episode_rewards"] else 0.0

        stats = data["statistics"]
        # Avoid division by zero
        efficiency = stats["planning_updates"] / max(stats["direct_updates"], 1) if stats["direct_updates"] > 0 else 0.0
        print(
            f"    {name}: Final Performance = {final_performance:.3f}, "
            f"Planning Efficiency = {efficiency:.1f}x"
        )

    print("\n  Blocking Maze Results (Adaptability):")
    for name, data in maze_results.items():
        # Ensure enough data points for before and after change
        change_episode = maze_env.change_episode
        if len(data["episode_rewards"]) > change_episode + 40:
            before_change = np.mean(data["episode_rewards"][change_episode - 20 : change_episode])
            after_change = np.mean(data["episode_rewards"][change_episode + 20 : change_episode + 40])
            # Adaptation speed: difference between after-change performance and the lowest point right after change
            min_after_change = np.min(data["episode_rewards"][change_episode : change_episode + 20])
            adaptation_speed = after_change - min_after_change
        else:
            before_change = np.mean(data["episode_rewards"][max(0, change_episode - 20) : change_episode]) if data["episode_rewards"] else 0.0
            after_change = np.mean(data["episode_rewards"][change_episode:]) if data["episode_rewards"] else 0.0
            adaptation_speed = after_change - before_change

        print(
            f"    {name}: Performance before change = {before_change:.3f}, "
            f"after change = {after_change:.3f}, adaptation = {adaptation_speed:.3f}"
        )

    print("\n📊 Key Takeaways:")
    print("• Dyna-Q achieves better sample efficiency through planning.")
    print("• More planning steps generally improve performance.")
    print("• Dyna-Q+ adapts better to sudden environment changes due to its exploration bonus.")
    print("• Model-based methods excel when the environment model is accurate and stable.")

    print("\n✅ Dyna-Q algorithm demonstration complete!")
    print("📊 Check 'visualizations/dyna_q_comparison.png' for plots.")
    print("""
Next: Monte Carlo Tree Search (MCTS) implementation.
Explore `agents/mcts.py` for its implementation.
""")

if __name__ == "__main__":
    demonstrate_dyna_q()
