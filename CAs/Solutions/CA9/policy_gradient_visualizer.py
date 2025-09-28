import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation
import os
from utils import device

VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)


class PolicyGradientVisualizer:
    """Enhanced visualizer for policy gradient methods with complex visualizations"""

    def demonstrate_policy_gradient_intuition(self):
        """Demonstrate the intuition behind policy gradients with enhanced visualizations"""

        print("=" * 70)
        print("Enhanced Policy Gradient Intuition")
        print("=" * 70)

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :2])
        states = np.linspace(0, 10, 100)
        theta_values = [0.5, 1.0, 1.5, 2.0]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for theta, color in zip(theta_values, colors):
            logits = theta * np.sin(states) + 0.5 * theta * states
            probabilities = 1 / (1 + np.exp(-logits))
            ax1.plot(
                states,
                probabilities,
                label=f"θ={theta}",
                color=color,
                linewidth=3,
                alpha=0.8,
            )

        ax1.set_title(
            "Policy Parameterization: π(a=1|s; θ)", fontsize=14, fontweight="bold"
        )
        ax1.set_xlabel("State", fontsize=12)
        ax1.set_ylabel("P(action=1)", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor("#f8f9fa")

        ax2 = fig.add_subplot(gs[0, 2:])
        actions = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        log_probs = np.array([-0.8, -0.2, -1.2, -0.1, -0.3, -0.9, -0.15, -1.0])
        returns = np.array([10, 15, 5, 20, 18, 3, 22, 8])

        score_values = log_probs * returns

        bars = ax2.bar(
            range(len(actions)),
            score_values,
            color=["#d62728" if r < np.mean(returns) else "#2ca02c" for r in returns],
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        ax2_twin = ax2.twinx()
        ax2_twin.plot(
            range(len(actions)),
            returns,
            "o-",
            color="#ff7f0e",
            linewidth=2,
            label="Returns",
            alpha=0.7,
        )
        ax2_twin.set_ylabel("Returns", color="#ff7f0e", fontsize=12)

        ax2.set_title(
            "Score Function: ∇log π(a|s) × Return", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Time Step", fontsize=12)
        ax2.set_ylabel("Score × Return", fontsize=12)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=1)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor("#f8f9fa")

        for i, (bar, ret) in enumerate(zip(bars, returns)):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + (2 if height > 0 else -3),
                f"R={ret}",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        ax3 = fig.add_subplot(gs[1, :2])
        iterations = np.arange(200)
        true_gradient = 2.5

        np.random.seed(42)
        noisy_estimates = true_gradient + np.random.normal(0, 1.0, len(iterations))

        window_size = 20
        rolling_mean = pd.Series(noisy_estimates).rolling(window=window_size).mean()
        rolling_std = pd.Series(noisy_estimates).rolling(window=window_size).std()

        ax3.plot(
            iterations,
            noisy_estimates,
            alpha=0.3,
            color="#1f77b4",
            linewidth=1,
            label="Noisy Estimates",
        )
        ax3.plot(
            iterations,
            rolling_mean,
            color="#ff7f0e",
            linewidth=3,
            label=f"Rolling Mean ({window_size})",
        )
        ax3.fill_between(
            iterations,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.3,
            color="#ff7f0e",
            label="±1 Std Dev",
        )
        ax3.axhline(
            y=true_gradient,
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label="True Gradient",
        )

        ax3.set_title(
            "Gradient Estimation with Confidence Intervals",
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_xlabel("Training Iteration", fontsize=12)
        ax3.set_ylabel("Gradient Estimate", fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor("#f8f9fa")

        ax4 = fig.add_subplot(gs[1, 2:])
        training_steps = [0, 25, 50, 100, 200, 500]
        state_range = np.linspace(0, 5, 50)

        for i, step in enumerate(training_steps):
            theta = 0.1 + 0.9 * (1 - np.exp(-step / 200))
            policy_probs = 1 / (1 + np.exp(-(theta * state_range - 2)))

            alpha = 0.3 + 0.7 * (i / len(training_steps))
            linewidth = 1 + 2 * (i / len(training_steps))
            ax4.plot(
                state_range,
                policy_probs,
                label=f"Step {step}",
                alpha=alpha,
                linewidth=linewidth,
            )

        ax4.set_title(
            "Policy Evolution During Training", fontsize=14, fontweight="bold"
        )
        ax4.set_xlabel("State", fontsize=12)
        ax4.set_ylabel("P(action=1)", fontsize=12)
        ax4.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor("#f8f9fa")

        ax5 = fig.add_subplot(gs[2, :2])
        ax5.text(
            0.1,
            0.9,
            "Policy Gradient Theorem Derivation:",
            fontsize=14,
            fontweight="bold",
        )
        ax5.text(0.1, 0.8, "J(θ) = E[∑ ∇_θ log π_θ(a_t|s_t) · G_t]", fontsize=12)
        ax5.text(0.1, 0.7, "Where:", fontsize=12, fontweight="bold")
        ax5.text(0.15, 0.65, "• J(θ): Expected return", fontsize=10)
        ax5.text(0.15, 0.60, "• ∇_θ log π_θ: Score function", fontsize=10)
        ax5.text(0.15, 0.55, "• G_t: Return from time t", fontsize=10)
        ax5.text(0.15, 0.50, "• E[·]: Expectation over trajectories", fontsize=10)

        ax5.text(
            0.1,
            0.35,
            r"$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6f3ff"),
        )

        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis("off")
        ax5.set_title("Mathematical Foundation", fontsize=14, fontweight="bold")

        ax6 = fig.add_subplot(gs[2, 2:])
        variance_levels = [
            "High Variance\n(MC Returns)",
            "Medium Variance\n(n-step)",
            "Low Variance\n(TD Target)",
        ]
        variance_values = [1.0, 0.6, 0.3]
        colors_var = ["#d62728", "#ff7f0e", "#2ca02c"]

        bars_var = ax6.bar(
            variance_levels,
            variance_values,
            color=colors_var,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        for bar, val in zip(bars_var, variance_values):
            ax6.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax6.set_title(
            "Variance in Policy Gradient Estimates", fontsize=14, fontweight="bold"
        )
        ax6.set_ylabel("Relative Variance", fontsize=12)
        ax6.set_ylim(0, 1.2)
        ax6.grid(True, alpha=0.3, axis="y")
        ax6.set_facecolor("#f8f9fa")

        ax7 = fig.add_subplot(gs[3, :2])
        ax7.axis("off")

        comparison_data = [
            ["Algorithm", "Variance", "Bias", "Sample Eff.", "Stability"],
            ["REINFORCE", "High", "Low", "Low", "Low"],
            ["Actor-Critic", "Medium", "Medium", "Medium", "Medium"],
            ["PPO", "Low", "Low", "High", "High"],
            ["TRPO", "Low", "Low", "High", "Very High"],
        ]

        table = ax7.table(
            cellText=comparison_data,
            loc="center",
            cellLoc="center",
            colWidths=[0.2, 0.2, 0.2, 0.2, 0.2],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        for i in range(len(comparison_data[0])):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(color="white", weight="bold")

        ax7.set_title(
            "Policy Gradient Algorithms Comparison",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        ax8 = fig.add_subplot(gs[3, 2:], projection="3d")

        theta1 = np.linspace(-2, 2, 50)
        theta2 = np.linspace(-2, 2, 50)
        THETA1, THETA2 = np.meshgrid(theta1, theta2)

        Z = (
            (THETA1 - 1) ** 2
            + (THETA2 - 0.5) ** 2
            + 0.1 * np.sin(5 * THETA1) * np.cos(3 * THETA2)
        )

        surf = ax8.plot_surface(
            THETA1, THETA2, Z, cmap="viridis", alpha=0.8, linewidth=0, antialiased=True
        )

        ax8.scatter([1], [0.5], [0], color="red", s=100, marker="*", label="Optimal θ")

        ax8.set_xlabel("θ₁", fontsize=10)
        ax8.set_ylabel("θ₂", fontsize=10)
        ax8.set_zlabel("J(θ)", fontsize=10)
        ax8.set_title("Policy Parameter Landscape", fontsize=12, fontweight="bold")
        ax8.legend()

        plt.suptitle(
            "Comprehensive Policy Gradient Intuition",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )
        plt.tight_layout()

        plt.savefig(
            os.path.join(VIS_DIR, "policy_gradient_intuition.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(VIS_DIR, "policy_gradient_intuition.pdf"), bbox_inches="tight"
        )
        plt.show()

        return {
            "policy_params": theta_values,
            "gradient_convergence": (
                rolling_mean.iloc[-1] if not rolling_mean.empty else None
            ),
            "variance_analysis": variance_values,
        }

    def compare_value_vs_policy_methods(self):
        """Compare value-based vs policy-based approaches"""

        print("\n" + "=" * 70)
        print("Value-Based vs Policy-Based Methods Comparison")
        print("=" * 70)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        ax = axes[0, 0]

        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)

        Q1 = X**2 + Y**2 + 0.5 * X * Y  # Q-value for action 1
        Q2 = (X - 1) ** 2 + (Y + 0.5) ** 2  # Q-value for action 2
        value_based_policy = (Q1 > Q2).astype(int)

        im1 = ax.contourf(
            X,
            Y,
            value_based_policy,
            levels=1,
            alpha=0.7,
            colors=["lightblue", "lightcoral"],
        )
        ax.contour(X, Y, Q1 - Q2, levels=[0], colors="black", linewidths=2)
        ax.set_title("Value-Based Policy\n(Deterministic Decision Boundary)")
        ax.set_xlabel("State Dimension 1")
        ax.set_ylabel("State Dimension 2")

        ax = axes[0, 1]

        logits = -0.5 * (X**2 + Y**2) + X - Y
        policy_probs = 1 / (1 + np.exp(-logits))

        im2 = ax.contourf(X, Y, policy_probs, levels=20, cmap="RdBu_r", alpha=0.8)
        plt.colorbar(im2, ax=ax, label="P(action=1)")
        ax.set_title("Policy-Based Method\n(Stochastic Policy)")
        ax.set_xlabel("State Dimension 1")
        ax.set_ylabel("State Dimension 2")

        ax = axes[1, 0]

        discrete_actions = ["Up", "Down", "Left", "Right"]
        value_based_discrete = [0.8, 0.1, 0.05, 0.05]  # Deterministic
        policy_based_discrete = [0.4, 0.3, 0.2, 0.1]  # Stochastic

        x_pos = np.arange(len(discrete_actions))
        width = 0.35

        bars1 = ax.bar(
            x_pos - width / 2,
            value_based_discrete,
            width,
            label="Value-Based",
            alpha=0.7,
            color="blue",
        )
        bars2 = ax.bar(
            x_pos + width / 2,
            policy_based_discrete,
            width,
            label="Policy-Based",
            alpha=0.7,
            color="red",
        )

        ax.set_title("Discrete Action Space")
        ax.set_xlabel("Actions")
        ax.set_ylabel("Probability")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(discrete_actions)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]

        actions = np.linspace(-3, 3, 100)

        discrete_bins = np.linspace(-3, 3, 7)
        discrete_probs = np.zeros_like(actions)
        for i, bin_center in enumerate(discrete_bins):
            mask = np.abs(actions - bin_center) < 0.3
            discrete_probs[mask] = 0.3 - 0.05 * i  # Decreasing probabilities

        continuous_mean = 0.5
        continuous_std = 0.8
        continuous_probs = (1 / np.sqrt(2 * np.pi * continuous_std**2)) * np.exp(
            -0.5 * ((actions - continuous_mean) / continuous_std) ** 2
        )

        ax.plot(
            actions,
            discrete_probs,
            "o-",
            label="Value-Based (Discretized)",
            color="blue",
            alpha=0.7,
            linewidth=2,
        )
        ax.plot(
            actions,
            continuous_probs,
            "-",
            label="Policy-Based (Continuous)",
            color="red",
            linewidth=2,
        )

        ax.set_title("Continuous Action Space")
        ax.set_xlabel("Action Value")
        ax.set_ylabel("Probability Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(VIS_DIR, "value_vs_policy_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(VIS_DIR, "value_vs_policy_comparison.pdf"), bbox_inches="tight"
        )
        plt.show()

        comparison_data = {
            "Aspect": [
                "Action Space",
                "Policy Type",
                "Exploration",
                "Convergence",
                "Sample Efficiency",
                "Stability",
            ],
            "Value-Based": [
                "Better for discrete",
                "Deterministic",
                "ε-greedy",
                "Can be unstable",
                "Generally higher",
                "Can oscillate",
            ],
            "Policy-Based": [
                "Natural for continuous",
                "Stochastic",
                "Built-in",
                "Smoother",
                "Generally lower",
                "More stable",
            ],
        }

        df = pd.DataFrame(comparison_data)
        print("\nDetailed Comparison:")
        print(df.to_string(index=False))

        return comparison_data

    def create_advanced_visualizations(self):
        """Create advanced complex visualizations for policy gradient analysis"""

        print("\n" + "=" * 70)
        print("Advanced Policy Gradient Visualizations")
        print("=" * 70)

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :2], projection="3d")

        theta1 = np.linspace(-3, 3, 50)
        theta2 = np.linspace(-3, 3, 50)
        THETA1, THETA2 = np.meshgrid(theta1, theta2)

        policy_performance = np.exp(-(THETA1**2 + THETA2**2)) + 0.5 * np.sin(
            2 * THETA1
        ) * np.cos(2 * THETA2)

        surf = ax1.plot_surface(
            THETA1,
            THETA2,
            policy_performance,
            cmap="viridis",
            alpha=0.8,
            linewidth=0,
            antialiased=True,
        )

        grad_theta1 = -2 * THETA1 - np.cos(2 * THETA1) * np.cos(2 * THETA2)
        grad_theta2 = -2 * THETA2 + np.sin(2 * THETA1) * np.sin(2 * THETA2)

        sample_indices = np.random.choice(50 * 50, 20, replace=False)
        sample_i, sample_j = np.unravel_index(sample_indices, (50, 50))

        for i, j in zip(sample_i, sample_j):
            ax1.quiver(
                THETA1[i, j],
                THETA2[i, j],
                policy_performance[i, j],
                grad_theta1[i, j],
                grad_theta2[i, j],
                0,
                color="red",
                length=0.1,
                normalize=True,
                alpha=0.7,
            )

        ax1.set_xlabel("θ₁", fontsize=12)
        ax1.set_ylabel("θ₂", fontsize=12)
        ax1.set_zlabel("Policy Performance", fontsize=12)
        ax1.set_title(
            "3D Policy Parameter Landscape with Gradients",
            fontsize=14,
            fontweight="bold",
        )

        ax2 = fig.add_subplot(gs[0, 2:])

        states = np.linspace(0, 10, 100)
        time_steps = [0, 10, 25, 50, 100, 200]

        for t, step in enumerate(time_steps):
            theta = 0.5 + 2.0 * (1 - np.exp(-step / 100))
            policy = 1 / (1 + np.exp(-(theta * (states - 5) / 5)))

            alpha = 0.3 + 0.7 * (t / len(time_steps))
            linewidth = 1 + 3 * (t / len(time_steps))
            ax2.plot(
                states, policy, label=f"t={step}", alpha=alpha, linewidth=linewidth
            )

        ax2.set_title("Policy Evolution Over Time", fontsize=14, fontweight="bold")
        ax2.set_xlabel("State", fontsize=12)
        ax2.set_ylabel("P(action=1)", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor("#f8f9fa")

        ax3 = fig.add_subplot(gs[1, :2])

        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)

        U = -X - 0.5 * np.sin(X) * np.cos(Y)  # ∂J/∂θ₁
        V = -Y + 0.5 * np.cos(X) * np.sin(Y)  # ∂J/∂θ₂

        magnitude = np.sqrt(U**2 + V**2)
        U_norm = U / magnitude
        V_norm = V / magnitude

        ax3.quiver(
            X, Y, U_norm, V_norm, magnitude, cmap="coolwarm", alpha=0.8, scale=20
        )

        trajectory_theta1 = []
        trajectory_theta2 = []
        theta1_traj, theta2_traj = -1.5, 1.5
        learning_rate = 0.1

        for _ in range(50):
            trajectory_theta1.append(theta1_traj)
            trajectory_theta2.append(theta2_traj)

            grad1 = -theta1_traj - 0.5 * np.sin(theta1_traj) * np.cos(theta2_traj)
            grad2 = -theta2_traj + 0.5 * np.cos(theta1_traj) * np.sin(theta2_traj)

            theta1_traj += learning_rate * grad1
            theta2_traj += learning_rate * grad2

        ax3.plot(
            trajectory_theta1,
            trajectory_theta2,
            "r-",
            linewidth=3,
            alpha=0.8,
            label="Optimization Path",
        )
        ax3.scatter(
            trajectory_theta1[0],
            trajectory_theta2[0],
            color="red",
            s=100,
            marker="o",
            label="Start",
        )
        ax3.scatter(
            trajectory_theta1[-1],
            trajectory_theta2[-1],
            color="green",
            s=100,
            marker="*",
            label="Converged",
        )

        ax3.set_title(
            "Gradient Flow Field with Optimization Trajectory",
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_xlabel("θ₁", fontsize=12)
        ax3.set_ylabel("θ₂", fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor("#f8f9fa")

        ax4 = fig.add_subplot(gs[1, 2:])

        methods = [
            "REINFORCE",
            "Baseline\nSubtraction",
            "Actor-Critic\n(1-step)",
            "Actor-Critic\n(n-step)",
            "PPO",
        ]
        environments = ["CartPole", "MountainCar", "Pendulum", "Acrobot", "LunarLander"]

        variance_data = np.random.rand(len(environments), len(methods))
        variance_data = variance_data * np.array(
            [1.0, 0.7, 0.5, 0.3, 0.2]
        )  # Progressive improvement

        sns.heatmap(
            variance_data,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            xticklabels=methods,
            yticklabels=environments,
            ax=ax4,
            cbar_kws={"label": "Relative Variance"},
        )

        ax4.set_title(
            "Variance Reduction Across Methods and Environments",
            fontsize=14,
            fontweight="bold",
        )
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")

        ax5 = fig.add_subplot(gs[2, :2])

        episodes = np.logspace(1, 4, 50)  # 10 to 10,000 episodes

        reinforce_perf = 1 - np.exp(-episodes / 2000)
        actor_critic_perf = 1 - np.exp(-episodes / 800)
        ppo_perf = 1 - np.exp(-episodes / 300)

        ax5.plot(
            episodes, reinforce_perf, label="REINFORCE", linewidth=3, color="#d62728"
        )
        ax5.plot(
            episodes,
            actor_critic_perf,
            label="Actor-Critic",
            linewidth=3,
            color="#ff7f0e",
        )
        ax5.plot(episodes, ppo_perf, label="PPO", linewidth=3, color="#2ca02c")

        ax5.set_xscale("log")
        ax5.set_xlabel("Training Episodes (log scale)", fontsize=12)
        ax5.set_ylabel("Performance (Normalized)", fontsize=12)
        ax5.set_title("Sample Efficiency Comparison", fontsize=14, fontweight="bold")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_facecolor("#f8f9fa")

        ax6 = fig.add_subplot(gs[2, 2:])

        training_runs = 5
        max_episodes = 1000
        episodes_range = np.arange(max_episodes)

        for run in range(training_runs):
            base_perf = 1 - np.exp(-episodes_range / 500)
            noise = np.random.normal(0, 0.1, max_episodes)
            cumulative_noise = np.cumsum(noise) * 0.01
            performance = np.clip(base_perf + cumulative_noise, 0, 1)

            ax6.plot(
                episodes_range,
                performance,
                alpha=0.7,
                linewidth=2,
                label=f"Run {run+1}",
            )

        mean_perf = np.mean(
            [1 - np.exp(-episodes_range / 500) for _ in range(10)], axis=0
        )
        ax6.plot(
            episodes_range, mean_perf, "k--", linewidth=3, label="Mean Performance"
        )

        ax6.set_xlabel("Training Episodes", fontsize=12)
        ax6.set_ylabel("Performance", fontsize=12)
        ax6.set_title(
            "Training Stability Across Multiple Runs", fontsize=14, fontweight="bold"
        )
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_facecolor("#f8f9fa")

        plt.suptitle(
            "Advanced Policy Gradient Analysis", fontsize=16, fontweight="bold", y=0.95
        )
        plt.tight_layout()

        plt.savefig(
            os.path.join(VIS_DIR, "advanced_policy_gradient_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(VIS_DIR, "advanced_policy_gradient_analysis.pdf"),
            bbox_inches="tight",
        )
        plt.show()

        return {
            "policy_landscape": policy_performance,
            "gradient_trajectory": (trajectory_theta1, trajectory_theta2),
            "variance_data": variance_data,
            "sample_efficiency": (reinforce_perf, actor_critic_perf, ppo_perf),
        }
