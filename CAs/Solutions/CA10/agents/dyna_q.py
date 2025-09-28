import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..models.models import TabularModel
from ..environments.environments import SimpleGridWorld, BlockingMaze
import random


class DynaQAgent:
    """Dyna-Q Agent implementing integrated planning and learning"""

    def __init__(
        self,
        num_states,
        num_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        planning_steps=5,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        self.Q = np.zeros((num_states, num_actions))

        self.model = {}  # Dictionary to store (s,a) -> (r, s') mappings
        self.visited_state_actions = set()  # Track visited (s,a) pairs

        self.episode_rewards = []
        self.episode_lengths = []
        self.planning_updates = 0
        self.direct_updates = 0

    def select_action(self, state):
        """ε-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def update_q_function(self, state, action, reward, next_state):
        """Q-learning update"""
        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error

    def update_model(self, state, action, reward, next_state):
        """Update internal model with new experience"""
        self.model[(state, action)] = (reward, next_state)
        self.visited_state_actions.add((state, action))

    def planning_update(self):
        """Perform planning updates using learned model"""
        if len(self.visited_state_actions) == 0:
            return

        for _ in range(self.planning_steps):
            state, action = random.choice(list(self.visited_state_actions))

            if (state, action) in self.model:
                reward, next_state = self.model[(state, action)]

                self.update_q_function(state, action, reward, next_state)
                self.planning_updates += 1

    def train_episode(self, env, max_steps=200):
        """Train for one episode"""
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done = env.step(action)

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

    def get_statistics(self):
        """Get training statistics"""
        return {
            "direct_updates": self.direct_updates,
            "planning_updates": self.planning_updates,
            "model_size": len(self.model),
            "avg_episode_reward": (
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            ),
        }


class DynaQPlusAgent(DynaQAgent):
    """Dyna-Q+ with exploration bonus for changed environments"""

    def __init__(
        self,
        num_states,
        num_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        planning_steps=5,
        kappa=0.001,
    ):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon, planning_steps)

        self.kappa = kappa  # Exploration bonus weight
        self.last_visit_time = {}  # Track when each (s,a) was last tried
        self.current_time = 0

    def update_q_function(
        self, state, action, reward, next_state, is_real_experience=True
    ):
        """Enhanced Q-learning update with exploration bonus"""

        if is_real_experience:
            self.last_visit_time[(state, action)] = self.current_time
            self.current_time += 1

        exploration_bonus = 0
        if not is_real_experience and (state, action) in self.last_visit_time:
            time_since_visit = self.current_time - self.last_visit_time[(state, action)]
            exploration_bonus = self.kappa * np.sqrt(time_since_visit)

        td_target = reward + exploration_bonus + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

        return td_error

    def planning_update(self):
        """Planning update with exploration bonus"""
        if len(self.visited_state_actions) == 0:
            return

        for _ in range(self.planning_steps):
            state, action = random.choice(list(self.visited_state_actions))

            if (state, action) in self.model:
                reward, next_state = self.model[(state, action)]

                self.update_q_function(
                    state, action, reward, next_state, is_real_experience=False
                )
                self.planning_updates += 1

    def train_episode(self, env, max_steps=200):
        """Training episode with proper experience tracking"""
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done = env.step(action)

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
    """Comprehensive Dyna-Q Demonstration"""

    print("Dyna-Q Algorithm Demonstration")
    print("=" * 50)

    agents = {
        "Q-Learning": DynaQAgent(25, 4, planning_steps=0),  # No planning
        "Dyna-Q (n=5)": DynaQAgent(25, 4, planning_steps=5),
        "Dyna-Q (n=50)": DynaQAgent(25, 4, planning_steps=50),
        "Dyna-Q+ (n=5)": DynaQPlusAgent(25, 4, planning_steps=5, kappa=0.001),
    }

    print("\n1. Training on Simple GridWorld:")
    simple_env = SimpleGridWorld(size=5)

    results = {}
    n_episodes = 200

    for name, agent in agents.items():
        print(f"\nTraining {name}...")
        episode_rewards = []

        for episode in range(n_episodes):
            reward, _ = agent.train_episode(simple_env, max_steps=100)
            episode_rewards.append(reward)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                stats = agent.get_statistics()
                print(
                    f"  Episode {episode+1}: Avg Reward = {avg_reward:.3f}, "
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
        rewards = data["episode_rewards"]
        smoothed = pd.Series(rewards).rolling(window=10).mean()
        plt.plot(smoothed, label=name, color=colors[i], linewidth=2)

    plt.title("Learning Performance Comparison")
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

    plt.title("Update Statistics")
    plt.xlabel("Agent")
    plt.ylabel("Number of Updates")
    plt.xticks(x, agent_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    print("\n2. Testing on Blocking Maze (Environment Change):")
    maze_env = BlockingMaze(change_episode=100)

    maze_agents = {
        "Dyna-Q": DynaQAgent(
            maze_env.num_states, maze_env.num_actions, planning_steps=50
        ),
        "Dyna-Q+": DynaQPlusAgent(
            maze_env.num_states, maze_env.num_actions, planning_steps=50, kappa=0.01
        ),
    }

    maze_results = {}
    n_episodes = 300

    for name, agent in maze_agents.items():
        print(f"\nTraining {name} on Blocking Maze...")
        maze_env.episode_count = 0

        for episode in range(n_episodes):
            reward, steps = agent.train_episode(maze_env, max_steps=3000)

            if episode in [50, 99, 150, 200, 250]:
                print(f"  Episode {episode+1}: Reward = {reward:.1f}, Steps = {steps}")

        maze_results[name] = {
            "episode_rewards": agent.episode_rewards.copy(),
            "episode_lengths": agent.episode_lengths.copy(),
        }

    plt.subplot(2, 2, 3)
    for name, data in maze_results.items():
        rewards = data["episode_rewards"]
        smoothed = pd.Series(rewards).rolling(window=20).mean()
        plt.plot(smoothed, label=name, linewidth=2)

    plt.axvline(
        x=100, color="red", linestyle="--", alpha=0.7, label="Environment Change"
    )
    plt.title("Blocking Maze Performance")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    for name, data in maze_results.items():
        lengths = data["episode_lengths"]
        smoothed = pd.Series(lengths).rolling(window=20).mean()
        plt.plot(smoothed, label=name, linewidth=2)

    plt.axvline(
        x=100, color="red", linestyle="--", alpha=0.7, label="Environment Change"
    )
    plt.title("Episode Length (Steps to Goal)")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("visualizations/dyna_q_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n3. Key Insights from Dyna-Q Experiments:")
    print("\nSimple GridWorld Results:")
    for name, data in results.items():
        final_performance = np.mean(data["episode_rewards"][-20:])
        stats = data["statistics"]
        efficiency = stats["planning_updates"] / max(stats["direct_updates"], 1)
        print(
            f"  {name}: Final Performance = {final_performance:.3f}, "
            f"Planning Efficiency = {efficiency:.1f}x"
        )

    print("\nBlocking Maze Results (Adaptability):")
    for name, data in maze_results.items():
        before_change = np.mean(data["episode_rewards"][80:100])
        after_change = np.mean(data["episode_rewards"][120:140])
        adaptation_speed = after_change - min(data["episode_rewards"][100:120])

        print(
            f"  {name}: Performance before change = {before_change:.3f}, "
            f"after change = {after_change:.3f}, adaptation = {adaptation_speed:.3f}"
        )

    print("\n📊 Key Takeaways:")
    print("• Dyna-Q achieves better sample efficiency through planning")
    print("• More planning steps generally improve performance")
    print("• Dyna-Q+ adapts better to environment changes")
    print("• Model-based methods excel when environment is stable")


if __name__ == "__main__":
    demonstrate_dyna_q()
