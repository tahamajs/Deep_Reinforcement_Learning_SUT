import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_rl.causal_rl import (
    CausalGraph,
    CausalDiscovery,
    CausalWorldModel,
    InterventionalDataset,
    CausalPolicyGradient,
)
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_causal_environment():
    """Create a simple environment with causal structure for demonstration"""

    class CausalEnvironment:
        def __init__(self):
            # Causal graph: X -> Y -> Z, with confounding
            self.state_dim = 3  # [X, Y, Z]
            self.action_dim = 1
            self.max_steps = 100

            # Causal parameters
            self.x_noise = 0.1
            self.y_noise = 0.05
            self.z_noise = 0.02

            self.reset()

        def reset(self):
            # Initialize with causal dependencies
            self.x = np.random.normal(0, 1)
            self.y = self.x + np.random.normal(0, self.y_noise)
            self.z = self.y + np.random.normal(0, self.z_noise)

            self.steps = 0
            return np.array([self.x, self.y, self.z])

        def step(self, action):
            action = np.clip(action, -1, 1)[0]

            # Causal transitions
            self.x += action * 0.1 + np.random.normal(0, self.x_noise)
            self.y = self.x + action * 0.05 + np.random.normal(0, self.y_noise)
            self.z = self.y + np.random.normal(0, self.z_noise)

            # Reward based on causal understanding
            reward = -abs(self.z) - 0.1 * abs(action)

            self.steps += 1
            done = self.steps >= self.max_steps

            return np.array([self.x, self.y, self.z]), reward, done, {}

    return CausalEnvironment()


def demonstrate_causal_discovery(env, n_samples=1000):
    """Demonstrate causal structure learning from observational data"""

    print("🔍 Learning Causal Structure from Observational Data")

    # Collect observational data
    observations = []
    actions = []

    obs = env.reset()
    for _ in range(n_samples):
        action = np.random.uniform(-1, 1, 1)
        next_obs, reward, done, _ = env.step(action)

        observations.append(obs)
        actions.append(action)

        obs = next_obs
        if done:
            obs = env.reset()

    observations = np.array(observations)
    actions = np.array(actions)

    # Create causal discovery object
    causal_discovery = CausalDiscovery(variables=["X", "Y", "Z", "A"], alpha=0.05)

    # Prepare data for causal discovery
    data = np.column_stack([observations, actions])

    # Learn causal graph
    graph = causal_discovery.discover_structure(data)

    print(f"Discovered causal graph with {len(graph.edges)} edges:")
    for edge in graph.edges:
        print(f"  {edge[0]} → {edge[1]}")

    return graph, data


def train_causal_world_model(env, graph, data, n_epochs=100):
    """Train a world model that respects causal structure"""

    print("🏗️ Training Causal World Model")

    # Create causal world model
    world_model = CausalWorldModel(
        graph=graph, obs_dim=3, action_dim=1, hidden_dim=64, latent_dim=16
    )

    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)

    losses = {"total": [], "reconstruction": [], "causal": []}

    batch_size = 32
    n_batches = len(data) // batch_size

    for epoch in range(n_epochs):
        epoch_losses = {"total": 0, "reconstruction": 0, "causal": 0}

        # Shuffle data
        indices = np.random.permutation(len(data))

        for i in range(n_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            batch_data = data[batch_indices]

            obs_batch = torch.FloatTensor(batch_data[:, :3]).to(device)
            action_batch = torch.FloatTensor(batch_data[:, 3:]).to(device)

            # Forward pass
            output = world_model(obs_batch, action_batch)

            # Losses
            recon_loss = F.mse_loss(output["reconstruction"], obs_batch)
            causal_loss = output["causal_constraint"]
            total_loss = recon_loss + 0.1 * causal_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_losses["total"] += total_loss.item()
            epoch_losses["reconstruction"] += recon_loss.item()
            epoch_losses["causal"] += causal_loss.item()

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
            losses[key].append(epoch_losses[key])

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}: Total={epoch_losses['total']:.4f}, "
                f"Recon={epoch_losses['reconstruction']:.4f}, "
                f"Causal={epoch_losses['causal']:.4f}"
            )

    return world_model, losses


def demonstrate_interventional_reasoning(world_model, env):
    """Demonstrate interventional reasoning capabilities"""

    print("🔬 Demonstrating Interventional Reasoning")

    # Test interventions on different variables
    interventions = {
        "X": lambda obs: np.array([1.0, obs[1], obs[2]]),  # Force X to 1.0
        "Y": lambda obs: np.array([obs[0], 0.5, obs[2]]),  # Force Y to 0.5
        "Z": lambda obs: np.array([obs[0], obs[1], -0.2]),  # Force Z to -0.2
    }

    results = {}

    for var, intervention_fn in interventions.items():
        print(f"\nIntervening on {var}:")

        # Get baseline prediction
        obs = env.reset()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action_tensor = torch.zeros(1, 1).to(device)

        with torch.no_grad():
            baseline_pred = (
                world_model(obs_tensor, action_tensor)["reconstruction"]
                .cpu()
                .numpy()[0]
            )

        # Apply intervention
        intervened_obs = intervention_fn(obs)
        intervened_tensor = torch.FloatTensor(intervened_obs).unsqueeze(0).to(device)

        with torch.no_grad():
            intervened_pred = (
                world_model.predict_intervention(intervened_tensor, action_tensor)
                .cpu()
                .numpy()[0]
            )

        print(f"  Baseline: {baseline_pred}")
        print(f"  After intervention: {intervened_pred}")
        print(f"  Change: {intervened_pred - baseline_pred}")

        results[var] = {
            "baseline": baseline_pred,
            "intervened": intervened_pred,
            "change": intervened_pred - baseline_pred,
        }

    return results
