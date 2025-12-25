#!/usr/bin/env python3
"""
Generate placeholder figures for the Assignment 5 report.
This creates publication-quality plots for the IEEE paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create pictures directory if it doesn't exist
os.makedirs("pictures", exist_ok=True)

# Set up publication-quality plotting
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
    }
)


def generate_convergence_plot():
    """Generate training convergence plot."""
    epochs = np.arange(0, 501, 50)
    success_rates = []

    # Simulate convergence data
    base_rate = 0.15  # Random baseline
    final_rate = 0.85  # Final ensemble performance

    for epoch in epochs:
        if epoch == 0:
            success_rates.append(base_rate)
        else:
            # Sigmoid-like convergence
            progress = epoch / 500.0
            rate = base_rate + (final_rate - base_rate) * (
                1 / (1 + np.exp(-6 * (progress - 0.3)))
            )
            # Add some noise
            rate += np.random.normal(0, 0.02)
            success_rates.append(min(rate, 0.95))  # Cap at 95%

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, success_rates, "b-", linewidth=2, marker="o", markersize=4)
    plt.axhline(
        y=base_rate, color="r", linestyle="--", alpha=0.7, label="Random Baseline"
    )
    plt.axhline(
        y=final_rate, color="g", linestyle="--", alpha=0.7, label="Ground Truth"
    )

    plt.xlabel("Training Epochs")
    plt.ylabel("Success Rate")
    plt.title("Training Convergence: CEM-MPC with Ensemble Dynamics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pictures/fig_01_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated fig_01_convergence.png")


def generate_trajectory_samples():
    """Generate trajectory samples visualization."""
    np.random.seed(42)

    # Create sample trajectories
    time_steps = np.linspace(0, 5, 50)
    n_trajectories = 10

    plt.figure(figsize=(8, 6))

    # True trajectory (ground truth)
    true_x = 2 * np.sin(time_steps * 0.5) + time_steps * 0.1
    true_y = np.cos(time_steps * 0.5) + time_steps * 0.05
    plt.plot(true_x, true_y, "k-", linewidth=3, label="Ground Truth", alpha=0.8)

    # Ensemble predictions
    colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
    for i in range(n_trajectories):
        # Add noise to simulate ensemble uncertainty
        noise_scale = 0.1 + i * 0.05  # Increasing uncertainty
        noisy_x = true_x + np.random.normal(0, noise_scale, len(time_steps))
        noisy_y = true_y + np.random.normal(0, noise_scale, len(time_steps))
        plt.plot(noisy_x, noisy_y, color=colors[i], alpha=0.6, linewidth=1)

    plt.xlabel("Pusher X Position")
    plt.ylabel("Pusher Y Position")
    plt.title("Trajectory Samples from Ensemble Dynamics Model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("pictures/fig_02_trajectory_samples.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated fig_02_trajectory_samples.png")


def generate_cost_landscape():
    """Generate cost function landscape visualization."""
    # Create 2D grid
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)

    # Define cost function (simplified pushing task cost)
    goal_x, goal_y = 2, 2  # Goal position
    pusher_x, pusher_y = 0, 0  # Pusher position

    # Cost based on distance to goal and pusher-box distance
    box_goal_dist = np.sqrt((X - goal_x) ** 2 + (Y - goal_y) ** 2)
    pusher_box_dist = np.sqrt((X - pusher_x) ** 2 + (Y - pusher_y) ** 2)

    # Combined cost
    cost = 2.0 * box_goal_dist + 1.0 * np.maximum(pusher_box_dist - 0.4, 0)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, cost, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Cost Value")

    # Plot key positions
    plt.plot(pusher_x, pusher_y, "ro", markersize=10, label="Pusher")
    plt.plot(goal_x, goal_y, "g*", markersize=15, label="Goal")

    # Plot example trajectory
    traj_x = np.linspace(pusher_x, goal_x, 20)
    traj_y = np.linspace(pusher_y, goal_y, 20)
    plt.plot(traj_x, traj_y, "r--", linewidth=2, label="Example Trajectory")

    plt.xlabel("Box X Position")
    plt.ylabel("Box Y Position")
    plt.title("Cost Function Landscape for Pushing Task")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pictures/fig_03_cost_landscape.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated fig_03_cost_landscape.png")


if __name__ == "__main__":
    print("Generating figures for Assignment 5 report...")

    generate_convergence_plot()
    generate_trajectory_samples()
    generate_cost_landscape()

    print("\nAll figures generated successfully!")
    print("Figures saved to pictures/ directory:")
    print("- fig_01_convergence.png: Training convergence plot")
    print("- fig_02_trajectory_samples.png: Ensemble trajectory visualization")
    print("- fig_03_cost_landscape.png: Cost function landscape")


