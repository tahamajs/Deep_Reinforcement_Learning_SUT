"""
Multi-modal visualizations for CA8
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any


def plot_multi_modal_attention_patterns(agent=None, save_path=None):
    """Visualize attention patterns across modalities"""

    print("Analyzing multi-modal attention patterns...")
    print("=" * 45)

    # Simulate attention data
    modalities = ["Visual", "Textual", "State", "Action"]
    time_steps = 20

    # Generate attention weights
    np.random.seed(42)
    attention_weights = np.random.rand(time_steps, len(modalities), len(modalities))
    attention_weights = attention_weights / attention_weights.sum(axis=2, keepdims=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Average cross-modal attention matrix
    avg_attention = np.mean(attention_weights, axis=0)
    im = axes[0, 0].imshow(avg_attention, cmap="viridis", aspect="equal")
    axes[0, 0].set_xticks(range(len(modalities)))
    axes[0, 0].set_yticks(range(len(modalities)))
    axes[0, 0].set_xticklabels(modalities)
    axes[0, 0].set_yticklabels(modalities)
    axes[0, 0].set_title("Average Cross-Modal Attention")
    plt.colorbar(im, ax=axes[0, 0])

    # Add text annotations for attention values
    for i in range(len(modalities)):
        for j in range(len(modalities)):
            text = axes[0, 0].text(
                j,
                i,
                f"{avg_attention[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    # Self-attention evolution over time
    for i, modality in enumerate(modalities):
        attention_to_self = attention_weights[:, i, i]
        axes[0, 1].plot(
            attention_to_self, label=f"{modality}→{modality}", linewidth=2, marker="o"
        )

    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Self-Attention Weight")
    axes[0, 1].set_title("Self-Attention Evolution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # Cross-modal attention distribution
    cross_attention = []
    for i in range(len(modalities)):
        for j in range(len(modalities)):
            if i != j:
                weights = attention_weights[:, i, j]
                cross_attention.extend(weights)

    axes[1, 0].hist(
        cross_attention, bins=20, alpha=0.7, edgecolor="black", color="skyblue"
    )
    axes[1, 0].set_xlabel("Cross-Modal Attention Weight")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Cross-Modal Attention Distribution")
    axes[1, 0].grid(True, alpha=0.3)

    # Add statistics
    mean_cross = np.mean(cross_attention)
    std_cross = np.std(cross_attention)
    axes[1, 0].axvline(
        mean_cross,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_cross:.3f}",
    )
    axes[1, 0].axvline(
        mean_cross + std_cross,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"+1σ: {mean_cross + std_cross:.3f}",
    )
    axes[1, 0].axvline(
        mean_cross - std_cross,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"-1σ: {mean_cross - std_cross:.3f}",
    )
    axes[1, 0].legend()

    # Attention pattern entropy over time
    attention_entropy = []
    for t in range(time_steps):
        entropy = 0
        for i in range(len(modalities)):
            for j in range(len(modalities)):
                p = attention_weights[t, i, j]
                if p > 0:
                    entropy -= p * np.log(p)
        attention_entropy.append(entropy)

    axes[1, 1].plot(attention_entropy, linewidth=2, color="purple", marker="s")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Attention Entropy")
    axes[1, 1].set_title("Attention Pattern Entropy")
    axes[1, 1].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(range(time_steps), attention_entropy, 1)
    p = np.poly1d(z)
    axes[1, 1].plot(
        range(time_steps),
        p(range(time_steps)),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Trend: {z[0]:.3f}x + {z[1]:.3f}",
    )
    axes[1, 1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Multi-modal attention pattern analysis completed!")


def multi_modal_fusion_strategy_comparison():
    """Compare different multi-modal fusion strategies"""

    print("Comparing multi-modal fusion strategies...")
    print("=" * 45)

    # Simulate fusion strategy performance data
    strategies = [
        "Early Fusion",
        "Late Fusion",
        "Cross-Modal Attention",
        "Hierarchical Fusion",
    ]
    metrics = ["Accuracy", "Robustness", "Efficiency", "Interpretability"]

    # Performance scores for each strategy-metric combination
    np.random.seed(42)
    performance_data = np.random.rand(len(strategies), len(metrics)) * 0.4 + 0.3

    # Add some realistic patterns
    performance_data[0, 0] = 0.85  # Early fusion good accuracy
    performance_data[1, 1] = 0.90  # Late fusion good robustness
    performance_data[2, 2] = 0.88  # Cross-modal attention good efficiency
    performance_data[3, 3] = 0.92  # Hierarchical fusion good interpretability

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Performance heatmap
    im = axes[0, 0].imshow(
        performance_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1
    )
    axes[0, 0].set_xticks(range(len(metrics)))
    axes[0, 0].set_yticks(range(len(strategies)))
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].set_yticklabels(strategies)
    axes[0, 0].set_title("Fusion Strategy Performance Matrix")

    # Add text annotations
    for i in range(len(strategies)):
        for j in range(len(metrics)):
            text = axes[0, 0].text(
                j,
                i,
                f"{performance_data[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    plt.colorbar(im, ax=axes[0, 0])

    # Radar chart for overall performance
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for i, strategy in enumerate(strategies):
        values = performance_data[i, :].tolist()
        values += values[:1]  # Complete the circle

        axes[0, 1].plot(angles, values, "o-", linewidth=2, label=strategy)
        axes[0, 1].fill(angles, values, alpha=0.25)

    axes[0, 1].set_xticks(angles[:-1])
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Fusion Strategy Radar Chart")
    axes[0, 1].legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    axes[0, 1].grid(True)

    # Bar chart comparison
    x = np.arange(len(strategies))
    width = 0.2

    for i, metric in enumerate(metrics):
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = axes[1, 0].bar(
            x + offset, performance_data[:, i], width, label=metric, alpha=0.8
        )

        # Color bars based on performance
        for bar in bars:
            height = bar.get_height()
            if height >= 0.8:
                bar.set_color("green")
            elif height >= 0.6:
                bar.set_color("orange")
            else:
                bar.set_color("red")

    axes[1, 0].set_xlabel("Fusion Strategy")
    axes[1, 0].set_ylabel("Performance Score")
    axes[1, 0].set_title("Fusion Strategy Comparison")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(strategies, rotation=45, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # Computational complexity vs performance
    complexity = [0.3, 0.5, 0.7, 0.4]  # Simulated complexity scores
    overall_performance = np.mean(performance_data, axis=1)

    scatter = axes[1, 1].scatter(
        complexity,
        overall_performance,
        s=200,
        alpha=0.7,
        c=range(len(strategies)),
        cmap="viridis",
    )

    for i, strategy in enumerate(strategies):
        axes[1, 1].annotate(
            strategy,
            (complexity[i], overall_performance[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    axes[1, 1].set_xlabel("Computational Complexity")
    axes[1, 1].set_ylabel("Overall Performance")
    axes[1, 1].set_title("Complexity vs Performance Trade-off")
    axes[1, 1].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(complexity, overall_performance, 1)
    p = np.poly1d(z)
    axes[1, 1].plot(
        complexity,
        p(complexity),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Trend: {z[0]:.3f}x + {z[1]:.3f}",
    )
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n" + "=" * 50)
    print("MULTI-MODAL FUSION STRATEGY COMPARISON")
    print("=" * 50)

    for i, strategy in enumerate(strategies):
        print(f"\n{strategy}:")
        for j, metric in enumerate(metrics):
            print(f"  {metric}: {performance_data[i, j]:.3f}")
        print(f"  Overall Performance: {overall_performance[i]:.3f}")
        print(f"  Computational Complexity: {complexity[i]:.3f}")

    return {
        "strategies": strategies,
        "metrics": metrics,
        "performance_data": performance_data,
        "overall_performance": overall_performance,
        "complexity": complexity,
    }
