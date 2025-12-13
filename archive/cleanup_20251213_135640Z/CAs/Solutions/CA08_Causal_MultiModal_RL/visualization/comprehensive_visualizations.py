"""
Comprehensive visualizations for CA8
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any


def comprehensive_causal_multi_modal_comparison(save_path=None):
    """Comprehensive comparison of causal and multi-modal approaches"""

    print("Comprehensive causal and multi-modal comparison...")
    print("=" * 55)

    # Define scenarios and approaches
    scenarios = [
        "Simple Navigation",
        "Complex Manipulation",
        "Multi-Task Learning",
        "Transfer Learning",
        "Robust Decision Making",
    ]

    approaches = ["Standard RL", "Causal RL", "Multi-Modal RL", "Causal Multi-Modal RL"]

    # Simulate performance data
    np.random.seed(42)
    performance_matrix = np.random.rand(len(scenarios), len(approaches)) * 0.4 + 0.3

    # Add realistic patterns
    performance_matrix[0, 3] = 0.95  # Causal multi-modal RL excels at simple navigation
    performance_matrix[1, 2] = 0.88  # Multi-modal RL good at complex manipulation
    performance_matrix[2, 1] = 0.82  # Causal RL good at multi-task learning
    performance_matrix[3, 3] = 0.90  # Causal multi-modal RL good at transfer
    performance_matrix[4, 3] = 0.92  # Causal multi-modal RL good at robustness

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Performance heatmap
    im = axes[0, 0].imshow(
        performance_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1
    )
    axes[0, 0].set_xticks(range(len(approaches)))
    axes[0, 0].set_yticks(range(len(scenarios)))
    axes[0, 0].set_xticklabels(approaches, rotation=45, ha="right")
    axes[0, 0].set_yticklabels(scenarios)
    axes[0, 0].set_title("Performance Across Scenarios")

    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(approaches)):
            text = axes[0, 0].text(
                j,
                i,
                f"{performance_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    plt.colorbar(im, ax=axes[0, 0])

    # Average performance by approach
    avg_performance = np.mean(performance_matrix, axis=0)
    bars = axes[0, 1].bar(approaches, avg_performance, alpha=0.7, edgecolor="black")
    axes[0, 1].set_ylabel("Average Performance")
    axes[0, 1].set_title("Average Performance by Approach")
    axes[0, 1].set_xticklabels(approaches, rotation=45, ha="right")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # Color bars based on performance
    for i, (bar, perf) in enumerate(zip(bars, avg_performance)):
        if perf >= 0.8:
            bar.set_color("green")
        elif perf >= 0.6:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    # Performance variance
    performance_variance = np.var(performance_matrix, axis=0)
    bars = axes[0, 2].bar(
        approaches, performance_variance, alpha=0.7, edgecolor="black", color="purple"
    )
    axes[0, 2].set_ylabel("Performance Variance")
    axes[0, 2].set_title("Performance Consistency")
    axes[0, 2].set_xticklabels(approaches, rotation=45, ha="right")
    axes[0, 2].grid(True, alpha=0.3)

    # Learning curves comparison
    episodes = np.arange(0, 200, 5)
    learning_curves = {}

    for i, approach in enumerate(approaches):
        # Simulate different learning patterns
        if approach == "Standard RL":
            curve = 0.3 + 0.4 * (1 - np.exp(-episodes / 50))
        elif approach == "Causal RL":
            curve = 0.4 + 0.35 * (1 - np.exp(-episodes / 40))
        elif approach == "Multi-Modal RL":
            curve = 0.35 + 0.45 * (1 - np.exp(-episodes / 45))
        else:  # Causal Multi-Modal RL
            curve = 0.5 + 0.4 * (1 - np.exp(-episodes / 35))

        learning_curves[approach] = curve
        axes[1, 0].plot(episodes, curve, label=approach, linewidth=2)

    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Performance")
    axes[1, 0].set_title("Learning Curves Comparison")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # Sample efficiency comparison
    sample_efficiency = []
    for approach in approaches:
        # Calculate episodes to reach 80% of final performance
        final_perf = learning_curves[approach][-1]
        target = 0.8 * final_perf
        episodes_to_target = len(episodes)

        for i, perf in enumerate(learning_curves[approach]):
            if perf >= target:
                episodes_to_target = i + 1
                break

        sample_efficiency.append(episodes_to_target)

    bars = axes[1, 1].bar(
        approaches, sample_efficiency, alpha=0.7, edgecolor="black", color="cyan"
    )
    axes[1, 1].set_ylabel("Episodes to 80% Performance")
    axes[1, 1].set_title("Sample Efficiency")
    axes[1, 1].set_xticklabels(approaches, rotation=45, ha="right")
    axes[1, 1].grid(True, alpha=0.3)

    # Color bars (lower is better for sample efficiency)
    max_episodes = max(sample_efficiency)
    for i, (bar, episodes) in enumerate(zip(bars, sample_efficiency)):
        normalized = 1 - (episodes / max_episodes)
        if normalized >= 0.7:
            bar.set_color("green")
        elif normalized >= 0.4:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    # Feature importance analysis
    features = [
        "Causal Reasoning",
        "Multi-Modal Fusion",
        "Counterfactual Analysis",
        "Cross-Modal Attention",
    ]
    importance_scores = {
        "Standard RL": [0.1, 0.0, 0.0, 0.0],
        "Causal RL": [0.9, 0.0, 0.8, 0.0],
        "Multi-Modal RL": [0.0, 0.9, 0.0, 0.8],
        "Causal Multi-Modal RL": [0.9, 0.9, 0.8, 0.8],
    }

    x = np.arange(len(features))
    width = 0.2

    for i, approach in enumerate(approaches):
        offset = (i - len(approaches) / 2 + 0.5) * width
        bars = axes[1, 2].bar(
            x + offset, importance_scores[approach], width, label=approach, alpha=0.8
        )

        # Color bars based on importance
        for bar in bars:
            height = bar.get_height()
            if height >= 0.7:
                bar.set_color("green")
            elif height >= 0.3:
                bar.set_color("orange")
            else:
                bar.set_color("red")

    axes[1, 2].set_xlabel("Feature")
    axes[1, 2].set_ylabel("Importance Score")
    axes[1, 2].set_title("Feature Importance by Approach")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(features, rotation=45, ha="right")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE CAUSAL MULTI-MODAL COMPARISON")
    print("=" * 60)

    print(f"\nAverage Performance by Approach:")
    for i, approach in enumerate(approaches):
        print(
            f"  {approach}: {avg_performance[i]:.3f} ± {np.sqrt(performance_variance[i]):.3f}"
        )

    print(f"\nSample Efficiency (Episodes to 80% Performance):")
    for i, approach in enumerate(approaches):
        print(f"  {approach}: {sample_efficiency[i]} episodes")

    print(f"\nBest Performing Approach per Scenario:")
    for i, scenario in enumerate(scenarios):
        best_idx = np.argmax(performance_matrix[i, :])
        best_approach = approaches[best_idx]
        best_performance = performance_matrix[i, best_idx]
        print(f"  {scenario}: {best_approach} ({best_performance:.3f})")

    return {
        "scenarios": scenarios,
        "approaches": approaches,
        "performance_matrix": performance_matrix,
        "avg_performance": avg_performance,
        "performance_variance": performance_variance,
        "learning_curves": learning_curves,
        "sample_efficiency": sample_efficiency,
        "importance_scores": importance_scores,
    }


def causal_multi_modal_curriculum_learning():
    """Analyze curriculum learning for causal multi-modal RL"""

    print("Analyzing curriculum learning for causal multi-modal RL...")
    print("=" * 55)

    # Define curriculum stages
    stages = [
        "Basic Navigation",
        "Simple Causal Reasoning",
        "Multi-Modal Integration",
        "Complex Causal Chains",
        "Counterfactual Reasoning",
        "Transfer Learning",
    ]

    # Simulate curriculum learning data
    np.random.seed(42)
    n_episodes_per_stage = 50
    total_episodes = len(stages) * n_episodes_per_stage

    # Performance progression through curriculum
    stage_performances = []
    cumulative_episodes = []

    for i, stage in enumerate(stages):
        # Simulate performance improvement within each stage
        stage_start = i * n_episodes_per_stage
        stage_end = (i + 1) * n_episodes_per_stage

        # Base performance increases with stage complexity
        base_performance = 0.3 + 0.1 * i

        # Within-stage learning curve
        stage_episodes = np.arange(n_episodes_per_stage)
        stage_curve = base_performance + 0.3 * (1 - np.exp(-stage_episodes / 20))

        # Add some noise
        noise = np.random.normal(0, 0.02, n_episodes_per_stage)
        stage_curve += noise

        stage_performances.extend(stage_curve)
        cumulative_episodes.extend(range(stage_start, stage_end))

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Overall curriculum learning curve
    axes[0, 0].plot(
        cumulative_episodes, stage_performances, linewidth=2, color="blue", alpha=0.7
    )

    # Add stage boundaries
    for i in range(len(stages)):
        stage_boundary = i * n_episodes_per_stage
        axes[0, 0].axvline(stage_boundary, color="red", linestyle="--", alpha=0.5)
        axes[0, 0].text(
            stage_boundary,
            0.95,
            stages[i],
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
        )

    # Add moving average
    window_size = 10
    moving_avg = np.convolve(
        stage_performances, np.ones(window_size) / window_size, mode="valid"
    )
    axes[0, 0].plot(
        cumulative_episodes[window_size - 1 :],
        moving_avg,
        linewidth=3,
        color="red",
        label="Moving Average",
    )

    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Performance")
    axes[0, 0].set_title("Curriculum Learning Progress")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # Stage-wise performance comparison
    stage_avg_performance = []
    stage_std_performance = []

    for i in range(len(stages)):
        start_idx = i * n_episodes_per_stage
        end_idx = (i + 1) * n_episodes_per_stage
        stage_data = stage_performances[start_idx:end_idx]

        stage_avg_performance.append(np.mean(stage_data))
        stage_std_performance.append(np.std(stage_data))

    bars = axes[0, 1].bar(
        stages,
        stage_avg_performance,
        yerr=stage_std_performance,
        capsize=5,
        alpha=0.7,
        edgecolor="black",
    )
    axes[0, 1].set_ylabel("Average Performance")
    axes[0, 1].set_title("Performance by Curriculum Stage")
    axes[0, 1].set_xticklabels(stages, rotation=45, ha="right")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # Color bars based on performance
    for i, (bar, perf) in enumerate(zip(bars, stage_avg_performance)):
        if perf >= 0.8:
            bar.set_color("green")
        elif perf >= 0.6:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    # Learning rate analysis
    learning_rates = []
    for i in range(len(stages)):
        start_idx = i * n_episodes_per_stage
        end_idx = (i + 1) * n_episodes_per_stage
        stage_data = stage_performances[start_idx:end_idx]

        # Calculate learning rate as slope of performance curve
        if len(stage_data) > 1:
            x = np.arange(len(stage_data))
            slope, _ = np.polyfit(x, stage_data, 1)
            learning_rates.append(slope)
        else:
            learning_rates.append(0)

    bars = axes[0, 2].bar(
        stages, learning_rates, alpha=0.7, edgecolor="black", color="purple"
    )
    axes[0, 2].set_ylabel("Learning Rate")
    axes[0, 2].set_title("Learning Rate by Stage")
    axes[0, 2].set_xticklabels(stages, rotation=45, ha="right")
    axes[0, 2].grid(True, alpha=0.3)

    # Color bars based on learning rate
    for i, (bar, rate) in enumerate(zip(bars, learning_rates)):
        if rate >= 0.001:
            bar.set_color("green")
        elif rate >= 0.0005:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    # Skill transfer analysis
    transfer_matrix = np.random.rand(len(stages), len(stages)) * 0.3 + 0.1
    np.fill_diagonal(transfer_matrix, 1.0)  # Perfect transfer to self

    # Add realistic transfer patterns
    transfer_matrix[0, 1] = 0.8  # Basic navigation helps with causal reasoning
    transfer_matrix[1, 2] = 0.7  # Causal reasoning helps with multi-modal integration
    transfer_matrix[2, 3] = 0.6  # Multi-modal integration helps with complex chains
    transfer_matrix[3, 4] = 0.5  # Complex chains help with counterfactual reasoning
    transfer_matrix[4, 5] = 0.4  # Counterfactual reasoning helps with transfer

    im = axes[1, 0].imshow(
        transfer_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1
    )
    axes[1, 0].set_xticks(range(len(stages)))
    axes[1, 0].set_yticks(range(len(stages)))
    axes[1, 0].set_xticklabels(stages, rotation=45, ha="right")
    axes[1, 0].set_yticklabels(stages)
    axes[1, 0].set_title("Skill Transfer Matrix")

    # Add text annotations
    for i in range(len(stages)):
        for j in range(len(stages)):
            text = axes[1, 0].text(
                j,
                i,
                f"{transfer_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    plt.colorbar(im, ax=axes[1, 0])

    # Curriculum effectiveness metrics
    metrics = ["Final Performance", "Learning Speed", "Transfer Ability", "Robustness"]
    effectiveness_scores = [0.85, 0.78, 0.72, 0.88]

    bars = axes[1, 1].barh(metrics, effectiveness_scores, alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Effectiveness Score")
    axes[1, 1].set_title("Curriculum Effectiveness Metrics")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)

    # Color bars based on effectiveness
    for i, (bar, score) in enumerate(zip(bars, effectiveness_scores)):
        if score >= 0.8:
            bar.set_color("green")
        elif score >= 0.6:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    # Comparison with non-curriculum learning
    non_curriculum_performance = 0.3 + 0.4 * (
        1 - np.exp(-np.arange(total_episodes) / 100)
    )

    axes[1, 2].plot(
        cumulative_episodes,
        stage_performances,
        linewidth=2,
        label="Curriculum Learning",
        color="blue",
    )
    axes[1, 2].plot(
        cumulative_episodes,
        non_curriculum_performance,
        linewidth=2,
        label="Non-Curriculum Learning",
        color="red",
    )

    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylabel("Performance")
    axes[1, 2].set_title("Curriculum vs Non-Curriculum Learning")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # Print curriculum analysis summary
    print("\n" + "=" * 60)
    print("CURRICULUM LEARNING ANALYSIS")
    print("=" * 60)

    print(f"\nStage-wise Performance:")
    for i, stage in enumerate(stages):
        print(
            f"  {stage}: {stage_avg_performance[i]:.3f} ± {stage_std_performance[i]:.3f}"
        )

    print(f"\nLearning Rates by Stage:")
    for i, stage in enumerate(stages):
        print(f"  {stage}: {learning_rates[i]:.6f}")

    print(f"\nCurriculum Effectiveness Metrics:")
    for i, metric in enumerate(metrics):
        print(f"  {metric}: {effectiveness_scores[i]:.3f}")

    # Calculate overall curriculum benefit
    final_curriculum_perf = stage_performances[-1]
    final_non_curriculum_perf = non_curriculum_performance[-1]
    improvement = (
        (final_curriculum_perf - final_non_curriculum_perf)
        / final_non_curriculum_perf
        * 100
    )

    print(
        f"\nOverall Curriculum Benefit: {improvement:.1f}% improvement over non-curriculum learning"
    )

    return {
        "stages": stages,
        "stage_performances": stage_performances,
        "stage_avg_performance": stage_avg_performance,
        "stage_std_performance": stage_std_performance,
        "learning_rates": learning_rates,
        "transfer_matrix": transfer_matrix,
        "effectiveness_scores": effectiveness_scores,
        "non_curriculum_performance": non_curriculum_performance,
        "improvement": improvement,
    }
