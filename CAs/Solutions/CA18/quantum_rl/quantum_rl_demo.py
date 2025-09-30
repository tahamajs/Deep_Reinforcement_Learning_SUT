import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantum_rl.quantum_rl import (
    QuantumState,
    QuantumGate,
    QuantumCircuit,
    VariationalQuantumCircuit,
    QuantumQLearning,
    QuantumActorCritic,
    QuantumEnvironment,
)
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_quantum_environment():
    """Create a simple environment suitable for quantum RL demonstration"""

    class QuantumInspiredEnvironment:
        def __init__(self, n_qubits=2):
            self.n_qubits = n_qubits
            self.state_dim = 2**n_qubits  # Full quantum state representation
            self.action_dim = n_qubits  # Actions are rotations on each qubit
            self.max_steps = 50

            # Initialize quantum state (simplified classical representation)
            self.reset()

        def reset(self):
            # Start with |00...0> state
            self.quantum_state = np.zeros(2**self.n_qubits)
            self.quantum_state[0] = 1.0

            self.steps = 0
            return self.quantum_state.copy()

        def step(self, action):
            # Apply quantum gates (simplified)
            action = np.clip(action, -np.pi, np.pi)

            # Apply rotation gates
            for i, theta in enumerate(action):
                # Simplified single-qubit rotation (affects computational basis)
                self.apply_rotation(i, theta)

            # Add some quantum-inspired dynamics
            self.apply_quantum_noise()

            # Compute reward based on quantum state properties
            reward = self.compute_quantum_reward()

            self.steps += 1
            done = self.steps >= self.max_steps

            return self.quantum_state.copy(), reward, done, {}

        def apply_rotation(self, qubit_idx, theta):
            """Apply a rotation gate to a specific qubit"""
            cos_theta = np.cos(theta / 2)
            sin_theta = np.sin(theta / 2)

            # Simplified single-qubit rotation (affects computational basis)
            new_state = np.zeros_like(self.quantum_state)

            for i in range(len(self.quantum_state)):
                bit = (i >> qubit_idx) & 1
                if bit == 0:
                    # |0> -> cos(θ/2)|0> - sin(θ/2)|1>
                    new_state[i] += cos_theta * self.quantum_state[i]
                    flipped_i = i | (1 << qubit_idx)
                    new_state[flipped_i] -= sin_theta * self.quantum_state[i]
                else:
                    # |1> -> sin(θ/2)|0> + cos(θ/2)|1>
                    flipped_i = i & ~(1 << qubit_idx)
                    new_state[flipped_i] += sin_theta * self.quantum_state[i]
                    new_state[i] += cos_theta * self.quantum_state[i]

            self.quantum_state = new_state

        def apply_quantum_noise(self):
            """Add quantum-inspired noise"""
            # Dephasing-like noise
            noise_strength = 0.01
            phase_noise = np.random.normal(0, noise_strength, len(self.quantum_state))
            self.quantum_state *= np.exp(1j * phase_noise)
            self.quantum_state = np.real(self.quantum_state)  # Keep real for simplicity

        def compute_quantum_reward(self):
            """Compute reward based on quantum state properties"""
            # Reward for creating superposition and entanglement-like states
            probabilities = np.abs(self.quantum_state) ** 2

            # Entropy (reward for mixed states)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

            # Reward for having multiple non-zero amplitudes
            n_nonzero = np.sum(probabilities > 0.01)

            # Penalty for being in computational basis states
            basis_penalty = -np.sum(probabilities[[0, -1]] ** 2)

            reward = entropy + 0.1 * n_nonzero + basis_penalty

            return reward

    return QuantumInspiredEnvironment()


def demonstrate_quantum_circuit():
    """Demonstrate basic quantum circuit operations"""

    print("🔬 Quantum Circuit Demonstration")

    # Create quantum circuit
    circuit = QuantumCircuit(n_qubits=2)

    print("Initial state: |00>")
    print(f"State vector: {circuit.get_state_vector()}")

    # Apply Hadamard gates
    circuit.apply_gate(QuantumGate.hadamard(), 0)
    circuit.apply_gate(QuantumGate.hadamard(), 1)

    print("\nAfter H⊗H: (|00> + |01> + |10> + |11>)/2")
    print(f"State vector: {circuit.get_state_vector()}")

    # Apply CNOT gate
    circuit.apply_cnot(0, 1)

    print("\nAfter CNOT(0,1): Creates entanglement")
    print(f"State vector: {circuit.get_state_vector()}")

    # Measure
    measurements = []
    for _ in range(1000):
        result = circuit.measure()
        measurements.append(result)

    print(f"\nMeasurement statistics (1000 shots):")
    unique, counts = np.unique(measurements, return_counts=True)
    for state, count in zip(unique, counts):
        print(f"  |{state:02b}>: {count/10:.1f}%")

    return circuit


def train_quantum_q_learning(env, n_episodes=200):
    """Train quantum-enhanced Q-learning"""

    print("🧠 Training Quantum Q-Learning Agent")

    # Create quantum Q-learning agent
    agent = QuantumQLearning(
        n_qubits=env.n_qubits,
        action_dim=env.action_dim,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
    )

    episode_rewards = []
    exploration_rates = []

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action using quantum state
            action = agent.select_action(obs)

            # Take action in environment
            next_obs, reward, done, _ = env.step(action)

            # Update quantum Q-function
            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward

        episode_rewards.append(episode_reward)
        exploration_rates.append(agent.exploration_rate)

        # Decay exploration
        agent.exploration_rate *= agent.exploration_decay

        if episode % 50 == 0:
            print(
                f"Episode {episode}: Reward={episode_reward:.2f}, "
                f"Exploration={agent.exploration_rate:.3f}"
            )

    return agent, episode_rewards, exploration_rates


def demonstrate_quantum_actor_critic(env):
    """Demonstrate quantum actor-critic algorithm"""

    print("🎭 Training Quantum Actor-Critic Agent")

    # Create quantum actor-critic agent
    agent = QuantumActorCritic(
        n_qubits=env.n_qubits,
        action_dim=env.action_dim,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        learning_rate=1e-3,
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    n_episodes = 100
    episode_rewards = []

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        log_probs = []
        values = []
        rewards = []

        while not done:
            obs_tensor = torch.FloatTensor(obs).to(device)

            # Get action and value from quantum actor-critic
            action_dist, value = agent(obs_tensor)
            action = action_dist.sample()
            next_obs, reward, done, _ = env.step(action.cpu().numpy())

            log_probs.append(action_dist.log_prob(action))
            values.append(value)
            rewards.append(reward)

            obs = next_obs
            episode_reward += reward

        # Compute returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(device)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)

        advantages = returns - values

        # Compute losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        total_loss = actor_loss + critic_loss

        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        episode_rewards.append(episode_reward)

        if episode % 20 == 0:
            print(
                f"Episode {episode}: Reward={episode_reward:.2f}, "
                f"Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}"
            )

    return agent, episode_rewards
