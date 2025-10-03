"""
Causal reasoning demonstrations for CA8
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Handle both relative and absolute imports
try:
    from ..agents.causal_discovery import CausalGraph, CausalDiscovery
    from ..agents.causal_rl_agent import CausalRLAgent
except ImportError:
    from agents.causal_discovery import CausalGraph, CausalDiscovery
    from agents.causal_rl_agent import CausalRLAgent


def demonstrate_causal_graph():
    """Demonstrate causal graph construction and manipulation"""
    print("=== Causal Graph Demonstration ===")

    # Create a causal graph
    variables = ["A", "B", "C", "D"]
    causal_graph = CausalGraph(variables)

    # Add edges
    causal_graph.add_edge("A", "B")
    causal_graph.add_edge("A", "C")
    causal_graph.add_edge("B", "D")
    causal_graph.add_edge("C", "D")

    print(f"Variables: {variables}")
    print(f"Graph structure: {causal_graph}")
    print(f"Is DAG: {causal_graph.is_dag()}")
    print(f"Topological order: {causal_graph.get_topological_order()}")
    print(f"Parents of D: {causal_graph.get_parents('D')}")
    print(f"Children of A: {causal_graph.get_children('A')}")
    print(f"Ancestors of D: {causal_graph.get_ancestors('D')}")
    print(f"Descendants of A: {causal_graph.get_descendants('A')}")

    return causal_graph


def demonstrate_causal_discovery():
    """Demonstrate causal discovery from data"""
    print("=== Causal Discovery Demonstration ===")

    np.random.seed(42)
    n_samples = 1000
    n_vars = 4

    # Generate data with known causal structure
    A = np.random.normal(0, 1, n_samples)
    C = A + np.random.normal(0, 0.5, n_samples)
    B = A + np.random.normal(0, 0.5, n_samples)
    D = B + C + np.random.normal(0, 0.5, n_samples)

    data = np.column_stack([A, B, C, D])
    var_names = ["A", "B", "C", "D"]

    print("Generated data with true causal structure: A -> B, A -> C, B -> D, C -> D")

    algorithms = {
        "PC Algorithm": CausalDiscovery.pc_algorithm,
        "GES Algorithm": CausalDiscovery.ges_algorithm,
        "LiNGAM": CausalDiscovery.lingam_algorithm,
    }

    discovered_graphs = {}

    for name, algorithm in algorithms.items():
        try:
            graph = algorithm(data, var_names)
            discovered_graphs[name] = graph
            print(f"\n{name} discovered structure:")
            print(graph)
        except Exception as e:
            print(f"\n{name} failed: {e}")

    return discovered_graphs


def demonstrate_causal_rl():
    """Demonstrate causal RL agent on a simple environment"""
    print("=== Causal RL Agent Demonstration ===")

    class SimpleGridWorld:
        """Simple grid world for testing"""

        def __init__(self, size=5):
            self.size = size
            self.state_dim = 4  # pos_x, pos_y, distance, reward
            self.action_dim = 4  # up, down, left, right

        def reset(self):
            self.pos = np.random.randint(0, self.size, 2)
            center = np.array([self.size // 2, self.size // 2])
            self.distance = np.linalg.norm(self.pos - center)
            self.current_reward = 0.0  # Placeholder
            state = np.array(
                [self.pos[0], self.pos[1], self.distance, self.current_reward]
            )
            return state.astype(float), {}

        def step(self, action):
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
            new_pos = self.pos + np.array(moves[action])

            new_pos = np.clip(new_pos, 0, self.size - 1)
            self.pos = new_pos

            center = np.array([self.size // 2, self.size // 2])
            self.distance = np.linalg.norm(self.pos - center)
            reward = -self.distance / (self.size * np.sqrt(2))
            self.current_reward = reward

            state = np.array(
                [self.pos[0], self.pos[1], self.distance, self.current_reward]
            )
            return state.astype(float), reward, False, False, {}

    env = SimpleGridWorld()

    # Define causal structure
    variables = ["pos_x", "pos_y", "distance", "reward"]
    causal_graph = CausalGraph(variables)
    causal_graph.add_edge("pos_x", "distance")
    causal_graph.add_edge("pos_y", "distance")
    causal_graph.add_edge("distance", "reward")

    print(f"Environment causal graph: {causal_graph}")

    # Create and train agent
    agent = CausalRLAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        causal_graph=causal_graph,
        lr=1e-3,
    )

    print("\nTraining Causal RL Agent...")
    rewards = []

    for episode in range(100):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(20):
            action, _ = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.update([state], [action], [reward], [next_state], [done])

            episode_reward += reward
            state = next_state

            if done:
                break

        rewards.append(episode_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            print(f"Episode {episode+1:3d} | Avg Reward: {avg_reward:.3f}")

    # Test causal interventions
    print("\nTesting causal interventions...")
    center = np.array([env.size // 2, env.size // 2])
    test_pos = np.array([2.0, 2.0])
    test_distance = np.linalg.norm(test_pos - center)
    test_state = np.array([test_pos[0], test_pos[1], test_distance, 0.0])

    original_action, _ = agent.select_action(test_state, deterministic=True)
    print(f"Original state {test_state}: Action {original_action}")

    intervention = {"pos_x": 0.0, "pos_y": 0.0}  # Move to corner
    intervened_state = agent.perform_intervention(test_state, intervention)
    intervened_action, _ = agent.select_action(intervened_state, deterministic=True)
    print(f"After intervention {intervention}: Action {intervened_action}")

    return {
        "agent": agent,
        "environment": env,
        "rewards": rewards,
        "causal_graph": causal_graph,
    }
