"""
Federated Reinforcement Learning Demo Functions

This module provides demonstration functions for federated RL algorithms,
showing how to train agents in a privacy-preserving distributed manner.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy
from collections import defaultdict

from .federated_rl import (
    FederatedRLClient,
    FederatedRLServer,
    SimpleAgent,
    FederatedEnvironment,
)


def create_federated_environment(n_clients: int = 3) -> List[FederatedEnvironment]:
    """
    Create federated environments for each client

    Args:
        n_clients: Number of federated clients

    Returns:
        List of environments, one per client
    """
    environments = []
    for client_id in range(n_clients):
        # Each client gets a slightly different environment
        env = FederatedEnvironment(
            state_dim=4,
            action_dim=2,
            reward_bias=client_id * 0.1,  # Different reward distributions
        )
        environments.append(env)

    print(f"✅ Created {n_clients} federated environments")
    return environments


def demonstrate_federated_learning(
    n_clients: int = 3,
    n_rounds: int = 10,
    local_epochs: int = 5,
    episodes_per_client: int = 20,
) -> Dict[str, List[float]]:
    """
    Demonstrate federated reinforcement learning

    Args:
        n_clients: Number of federated clients
        n_rounds: Number of communication rounds
        local_epochs: Local training epochs per client
        episodes_per_client: Episodes per client per round

    Returns:
        Dictionary containing training history
    """
    print(f"\n🌐 Federated RL Demonstration")
    print(f"Clients: {n_clients}, Rounds: {n_rounds}")

    # Create environments
    environments = create_federated_environment(n_clients)

    # Create global model
    global_model = SimpleAgent(state_dim=4, action_dim=2, hidden_dim=64)

    # Create server
    server = FederatedRLServer(
        global_model=global_model, n_clients=n_clients, aggregation_method="fedavg"
    )

    # Create clients
    clients = []
    for i in range(n_clients):
        client = FederatedRLClient(
            client_id=i,
            local_model=copy.deepcopy(global_model),
            environment=environments[i],
            learning_rate=1e-3,
        )
        clients.append(client)

    # Training history
    history = {
        "global_rewards": [],
        "client_rewards": [[] for _ in range(n_clients)],
        "communication_rounds": [],
    }

    # Federated training loop
    for round_idx in range(n_rounds):
        print(f"\n📡 Round {round_idx + 1}/{n_rounds}")

        # Local training at each client
        client_updates = []
        round_rewards = []

        for client_idx, client in enumerate(clients):
            # Download global model
            client.download_model(server.get_global_model())

            # Local training
            rewards = client.local_training(
                n_episodes=episodes_per_client, n_epochs=local_epochs
            )

            # Get model update
            update = client.get_model_update()
            client_updates.append(update)

            # Track rewards
            avg_reward = np.mean(rewards)
            round_rewards.append(avg_reward)
            history["client_rewards"][client_idx].append(avg_reward)

            print(f"  Client {client_idx}: Avg Reward = {avg_reward:.2f}")

        # Server aggregation
        server.aggregate_updates(client_updates)

        # Evaluate global model
        global_model = server.get_global_model()
        global_reward = evaluate_global_model(global_model, environments[0])
        history["global_rewards"].append(global_reward)
        history["communication_rounds"].append(round_idx)

        print(f"  Global Model Reward: {global_reward:.2f}")

    return history


def evaluate_global_model(
    model: nn.Module, environment: FederatedEnvironment, n_episodes: int = 10
) -> float:
    """
    Evaluate the global model performance

    Args:
        model: Global model to evaluate
        environment: Environment for evaluation
        n_episodes: Number of evaluation episodes

    Returns:
        Average reward over evaluation episodes
    """
    rewards = []

    for _ in range(n_episodes):
        state = environment.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = model(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()

            # Take action
            next_state, reward, done, _ = environment.step(action)
            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)

    return np.mean(rewards)


def demonstrate_privacy_preservation(
    n_clients: int = 3,
    use_differential_privacy: bool = True,
    epsilon: float = 1.0,
    delta: float = 1e-5,
) -> Dict[str, float]:
    """
    Demonstrate privacy-preserving federated learning

    Args:
        n_clients: Number of clients
        use_differential_privacy: Whether to use DP
        epsilon: Privacy budget
        delta: Privacy parameter

    Returns:
        Dictionary with privacy metrics
    """
    print(f"\n🔒 Privacy-Preserving Federated RL")
    print(f"Differential Privacy: {use_differential_privacy}")

    if use_differential_privacy:
        print(f"Privacy Parameters: ε={epsilon}, δ={delta}")

    # Create environment and models
    environments = create_federated_environment(n_clients)
    global_model = SimpleAgent(state_dim=4, action_dim=2, hidden_dim=64)

    # Create privacy-aware server
    server = FederatedRLServer(
        global_model=global_model,
        n_clients=n_clients,
        aggregation_method="fedavg",
        use_differential_privacy=use_differential_privacy,
        epsilon=epsilon,
        delta=delta,
    )

    # Create clients with privacy settings
    clients = []
    for i in range(n_clients):
        client = FederatedRLClient(
            client_id=i,
            local_model=copy.deepcopy(global_model),
            environment=environments[i],
            learning_rate=1e-3,
            use_differential_privacy=use_differential_privacy,
            clip_norm=1.0,  # Gradient clipping for DP
        )
        clients.append(client)

    # Run one round of training
    client_updates = []
    for client in clients:
        client.download_model(server.get_global_model())
        rewards = client.local_training(n_episodes=10, n_epochs=3)
        update = client.get_model_update()
        client_updates.append(update)

    # Aggregate with privacy
    server.aggregate_updates(client_updates)

    # Compute privacy metrics
    privacy_metrics = {
        "epsilon_used": epsilon if use_differential_privacy else 0.0,
        "delta": delta if use_differential_privacy else 0.0,
        "noise_scale": server.get_noise_scale() if use_differential_privacy else 0.0,
        "privacy_guarantee": "DP-SGD" if use_differential_privacy else "None",
    }

    print(f"\n📊 Privacy Metrics:")
    for key, value in privacy_metrics.items():
        print(f"  {key}: {value}")

    return privacy_metrics


def demonstrate_communication_efficiency(
    n_clients: int = 5, compression_rates: List[float] = [0.1, 0.5, 1.0]
) -> Dict[str, List[float]]:
    """
    Demonstrate communication-efficient federated learning

    Args:
        n_clients: Number of clients
        compression_rates: List of compression rates to test

    Returns:
        Dictionary with communication costs and performance
    """
    print(f"\n📡 Communication Efficiency Demonstration")

    results = {
        "compression_rates": compression_rates,
        "communication_costs": [],
        "model_performance": [],
    }

    # Create base setup
    environments = create_federated_environment(n_clients)
    base_model = SimpleAgent(state_dim=4, action_dim=2, hidden_dim=64)

    for compression_rate in compression_rates:
        print(f"\n🗜️ Testing compression rate: {compression_rate}")

        # Create server with compression
        server = FederatedRLServer(
            global_model=copy.deepcopy(base_model),
            n_clients=n_clients,
            aggregation_method="fedavg",
            compression_rate=compression_rate,
        )

        # Create clients
        clients = []
        for i in range(n_clients):
            client = FederatedRLClient(
                client_id=i,
                local_model=copy.deepcopy(base_model),
                environment=environments[i],
                learning_rate=1e-3,
                compression_rate=compression_rate,
            )
            clients.append(client)

        # Run training
        total_communication = 0
        for round_idx in range(5):  # 5 rounds
            client_updates = []

            for client in clients:
                client.download_model(server.get_global_model())
                client.local_training(n_episodes=10, n_epochs=2)
                update = client.get_model_update()
                client_updates.append(update)

                # Calculate communication cost
                comm_cost = client.get_communication_cost()
                total_communication += comm_cost

            server.aggregate_updates(client_updates)

        # Evaluate final model
        final_performance = evaluate_global_model(
            server.get_global_model(), environments[0], n_episodes=20
        )

        results["communication_costs"].append(total_communication)
        results["model_performance"].append(final_performance)

        print(f"  Total Communication: {total_communication:.2f} MB")
        print(f"  Final Performance: {final_performance:.2f}")

    return results


def create_heterogeneous_clients(
    n_clients: int = 5, heterogeneity_level: str = "medium"
) -> Tuple[List[FederatedRLClient], List[FederatedEnvironment]]:
    """
    Create clients with heterogeneous data distributions

    Args:
        n_clients: Number of clients
        heterogeneity_level: Level of data heterogeneity ('low', 'medium', 'high')

    Returns:
        Tuple of (clients, environments)
    """
    print(f"\n🌍 Creating Heterogeneous Federated Setup")
    print(f"Heterogeneity Level: {heterogeneity_level}")

    heterogeneity_map = {"low": 0.1, "medium": 0.5, "high": 1.0}

    variance = heterogeneity_map.get(heterogeneity_level, 0.5)

    # Create diverse environments
    environments = []
    for i in range(n_clients):
        env = FederatedEnvironment(
            state_dim=4,
            action_dim=2,
            reward_bias=i * variance,
            transition_noise=0.1 * variance,
        )
        environments.append(env)

    # Create clients
    base_model = SimpleAgent(state_dim=4, action_dim=2, hidden_dim=64)
    clients = []

    for i in range(n_clients):
        client = FederatedRLClient(
            client_id=i,
            local_model=copy.deepcopy(base_model),
            environment=environments[i],
            learning_rate=1e-3 * (1 + i * 0.1),  # Different learning rates
        )
        clients.append(client)

    print(f"✅ Created {n_clients} heterogeneous clients")
    return clients, environments


if __name__ == "__main__":
    print("🚀 Federated RL Demo Module")

    # Run basic demonstration
    history = demonstrate_federated_learning(
        n_clients=3, n_rounds=5, local_epochs=3, episodes_per_client=10
    )

    print("\n📈 Training Summary:")
    print(f"Final Global Reward: {history['global_rewards'][-1]:.2f}")
    print(f"Client Reward Improvement: ", end="")
    for i, client_rewards in enumerate(history["client_rewards"]):
        improvement = client_rewards[-1] - client_rewards[0]
        print(f"Client {i}: {improvement:+.2f} ", end="")
    print()

    # Demonstrate privacy
    privacy_metrics = demonstrate_privacy_preservation(
        n_clients=3, use_differential_privacy=True, epsilon=1.0
    )

    # Demonstrate communication efficiency
    comm_results = demonstrate_communication_efficiency(
        n_clients=3, compression_rates=[0.1, 0.5, 1.0]
    )
