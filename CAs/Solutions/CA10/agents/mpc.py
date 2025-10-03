import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import NeuralModel, ModelTrainer, device
from environments.environments import SimpleGridWorld


class MPCController:
    """Model Predictive Control for RL"""

    def __init__(
        self,
        model,
        num_actions,
        num_states,
        horizon=10,
        num_samples=100,
        temperature=1.0,
        elite_ratio=0.1,
    ):
        self.model = model
        self.num_actions = num_actions
        self.state_dim = num_states
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.elite_ratio = elite_ratio
        self.elite_size = max(1, int(num_samples * elite_ratio))

        self.optimization_costs = []
        self.episode_rewards = []
        self.model = model
        self.num_actions = num_actions
        self.state_dim = num_states
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.elite_ratio = elite_ratio
        self.elite_size = max(1, int(num_samples * elite_ratio))

        self.optimization_costs = []
        self.episode_rewards = []

    def cross_entropy_optimization(self, initial_state):
        """Cross-Entropy Method for action sequence optimization"""
        action_means = np.zeros((self.horizon, self.num_actions))
        action_stds = np.ones((self.horizon, self.num_actions))

        best_cost = float("inf")
        best_actions = None

        for iteration in range(5):  # Number of CEM iterations
            action_sequences = []
            costs = []

            for _ in range(self.num_samples):
                actions = []
                for h in range(self.horizon):
                    probs = np.exp(action_means[h] / self.temperature)
                    probs = probs / np.sum(probs)
                    action = np.random.choice(self.num_actions, p=probs)
                    actions.append(action)

                action_sequences.append(actions)
                cost = self.evaluate_sequence(initial_state, actions)
                costs.append(cost)

            elite_indices = np.argsort(costs)[: self.elite_size]
            elite_actions = [action_sequences[i] for i in elite_indices]

            for h in range(self.horizon):
                elite_actions_h = [seq[h] for seq in elite_actions]
                for a in range(self.num_actions):
                    count = elite_actions_h.count(a)
                    action_means[h, a] = count / len(elite_actions_h)

                action_means[h] = np.log(action_means[h] + 1e-8)

            min_cost = min(costs)
            if min_cost < best_cost:
                best_cost = min_cost
                best_actions = action_sequences[costs.index(min_cost)]

        self.optimization_costs.append(best_cost)
        return best_actions

    def random_shooting(self, initial_state):
        """Random shooting optimization"""
        best_cost = float("inf")
        best_actions = None

        for _ in range(self.num_samples):
            actions = [np.random.randint(self.num_actions) for _ in range(self.horizon)]

            cost = self.evaluate_sequence(initial_state, actions)

            if cost < best_cost:
                best_cost = cost
                best_actions = actions

        self.optimization_costs.append(best_cost)
        return best_actions

    def evaluate_sequence(self, initial_state, actions):
        """Evaluate cost of action sequence using model"""
        state = initial_state
        total_cost = 0.0
        discount = 1.0

        state_onehot = np.zeros(self.state_dim)
        state_onehot[state] = 1

        for action in actions:
            state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
            action_tensor = torch.LongTensor([action]).to(device)

            next_state_pred, reward_pred = self.model.forward(
                state_tensor, action_tensor
            )

            next_state_logits = next_state_pred.squeeze()
            next_state = torch.argmax(next_state_logits).item()

            reward = reward_pred.squeeze().item()

            cost = -reward
            total_cost += discount * cost
            discount *= 0.95  # Discount factor

            state = next_state
            state_onehot = np.zeros(self.state_dim)
            state_onehot[state] = 1

        return total_cost

    def select_action(self, state, method="cross_entropy"):
        """Select action using MPC"""
        if method == "cross_entropy":
            action_sequence = self.cross_entropy_optimization(state)
        else:
            action_sequence = self.random_shooting(state)

        return (
            action_sequence[0]
            if action_sequence
            else np.random.randint(self.num_actions)
        )


class MPCAgent:
    """RL Agent using MPC for control"""

    def __init__(
        self, model, num_states, num_actions, horizon=10, method="cross_entropy"
    ):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.controller = MPCController(model, num_actions, num_states, horizon=horizon)
        self.method = method

        self.episode_rewards = []
        self.planning_costs = []

    def train_episode(self, env, max_steps=200):
        """Run episode with MPC planning"""
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = self.controller.select_action(state, self.method)
            next_state, reward, done = env.step(action)

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        self.episode_rewards.append(total_reward)
        if self.controller.optimization_costs:
            self.planning_costs.extend(self.controller.optimization_costs)

        return total_reward, steps

    def get_statistics(self):
        """Get performance statistics"""
        return {
            "avg_episode_reward": (
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            ),
            "avg_planning_cost": (
                np.mean(self.planning_costs[-10:]) if self.planning_costs else 0
            ),
            "total_episodes": len(self.episode_rewards),
        }


def demonstrate_mpc():
    """MPC Demonstration"""

    print("Model Predictive Control (MPC) Demonstration")
    print("=" * 50)

    print("\n1. Setting up MPC with learned model...")
    env = SimpleGridWorld(size=5)

    n_episodes = 1000
    experience_data = []

    for episode in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            action = np.random.randint(env.num_actions)  # Random policy
            next_state, reward, done = env.step(action)

            experience_data.append((state, action, next_state, reward))

            state = next_state

    print(f"Collected {len(experience_data)} transitions")

    states = np.array([exp[0] for exp in experience_data])
    actions = np.array([exp[1] for exp in experience_data])
    next_states = np.array([exp[2] for exp in experience_data])
    rewards = np.array([exp[3] for exp in experience_data])

    states_onehot = np.eye(env.num_states)[states]
    next_states_onehot = np.eye(env.num_states)[next_states]

    neural_model = NeuralModel(env.num_states, env.num_actions, hidden_dim=64).to(
        device
    )

    trainer = ModelTrainer(neural_model, lr=1e-3)
    print("Training neural model for MPC...")
    trainer.train_batch(
        (states_onehot, actions, next_states_onehot, rewards), epochs=50, batch_size=64
    )

    agents = {
        "MPC-CEM": MPCAgent(
            neural_model,
            env.num_states,
            env.num_actions,
            horizon=8,
            method="cross_entropy",
        ),
        "MPC-RS": MPCAgent(
            neural_model,
            env.num_states,
            env.num_actions,
            horizon=8,
            method="random_shooting",
        ),
    }

    print("\n2. Testing MPC performance...")
    n_episodes = 15
    results = {}

    for name, agent in agents.items():
        print(f"\nTesting {name}...")
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            reward, length = agent.train_episode(env, max_steps=100)
            episode_rewards.append(reward)
            episode_lengths.append(length)

            if (episode + 1) % 5 == 0:
                avg_reward = np.mean(episode_rewards[-5:])
                print(
                    f"  Episodes {episode-4}-{episode+1}: Avg Reward = {avg_reward:.2f}"
                )

        results[name] = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "statistics": agent.get_statistics(),
        }

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    for name, data in results.items():
        rewards = data["episode_rewards"]
        plt.plot(rewards, linewidth=2, label=name, marker="o", markersize=4)

    plt.title("MPC Performance Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    for name, data in results.items():
        lengths = data["episode_lengths"]
        plt.plot(lengths, linewidth=2, label=name, marker="s", markersize=4)

    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps to Goal")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    reward_data = [results[name]["episode_rewards"] for name in results.keys()]
    labels = list(results.keys())
    plt.boxplot(reward_data, labels=labels)
    plt.title("Reward Distribution")
    plt.ylabel("Episode Reward")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    if "MPC-CEM" in results:
        agent = agents["MPC-CEM"]
        if agent.planning_costs:
            plt.plot(agent.planning_costs, "purple", linewidth=2, alpha=0.7)
            plt.axhline(
                y=np.mean(agent.planning_costs),
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Mean: {np.mean(agent.planning_costs):.2f}",
            )
            plt.title("MPC-CEM Planning Costs")
            plt.xlabel("Planning Step")
            plt.ylabel("Cost")
            plt.legend()
            plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    print("\n3. Analyzing effect of planning horizon...")
    horizon_results = {}
    horizons = [3, 5, 8, 12]

    for h in horizons:
        agent = MPCAgent(
            neural_model,
            env.num_states,
            env.num_actions,
            horizon=h,
            method="cross_entropy",
        )
        rewards = []

        for episode in range(5):  # Quick test
            reward, _ = agent.train_episode(env, max_steps=100)
            rewards.append(reward)

        horizon_results[h] = np.mean(rewards)

    horizons_list = list(horizon_results.keys())
    performance_list = list(horizon_results.values())
    plt.bar(
        horizons_list, performance_list, alpha=0.7, color="skyblue", edgecolor="black"
    )
    plt.title("Performance vs Planning Horizon")
    plt.xlabel("Planning Horizon")
    plt.ylabel("Average Episode Reward")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    method_names = list(results.keys())
    avg_rewards = [np.mean(results[name]["episode_rewards"]) for name in method_names]
    avg_lengths = [np.mean(results[name]["episode_lengths"]) for name in method_names]

    x = np.arange(len(method_names))
    width = 0.35

    plt.bar(x - width / 2, avg_rewards, width, label="Avg Reward", alpha=0.7)
    plt.bar(
        x + width / 2,
        [l / 10 for l in avg_lengths],
        width,
        label="Avg Length/10",
        alpha=0.7,
    )

    plt.title("MPC Method Comparison")
    plt.xlabel("Method")
    plt.ylabel("Performance")
    plt.xticks(x, method_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create visualizations directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/mpc_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n4. MPC Analysis Summary:")
    for name, data in results.items():
        stats = data["statistics"]
        print(f"\n{name}:")
        print(
            f"  Average Episode Reward: {np.mean(data['episode_rewards']):.3f} ± {np.std(data['episode_rewards']):.3f}"
        )
        print(
            f"  Average Episode Length: {np.mean(data['episode_lengths']):.1f} ± {np.std(data['episode_lengths']):.1f}"
        )
        if stats["avg_planning_cost"] > 0:
            print(f"  Average Planning Cost: {stats['avg_planning_cost']:.3f}")

    print(f"\nHorizon Analysis:")
    for h, perf in horizon_results.items():
        print(f"  Horizon {h}: {perf:.3f} average reward")

    print(f"\n📊 Key MPC Insights:")
    print("• MPC provides principled planning with explicit horizons")
    print("• Cross-Entropy Method often outperforms random shooting")
    print("• Longer horizons generally improve performance but increase computation")
    print("• MPC naturally handles constraints and can incorporate uncertainty")
    print("• Effective for continuous control and discrete planning problems")


if __name__ == "__main__":
    demonstrate_mpc()
