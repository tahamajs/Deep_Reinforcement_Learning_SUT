"""
Results Module

This module contains utilities for analyzing and visualizing experiment results.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os
import pandas as pd
from datetime import datetime


class ResultsAnalyzer:
    """Analyzer for experiment results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.results = {}
        self.load_results()

    def load_results(self):
        """Load all results from the results directory."""
        if not os.path.exists(self.results_dir):
            print(f"Results directory {self.results_dir} does not exist")
            return

        for filename in os.listdir(self.results_dir):
            if filename.endswith("_result.json"):
                experiment_name = filename.replace("_result.json", "")
                filepath = os.path.join(self.results_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        result = json.load(f)
                    self.results[experiment_name] = result
                    print(f"Loaded results for {experiment_name}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        summary = {
            "total_experiments": len(self.results),
            "experiment_names": list(self.results.keys()),
            "experiment_types": {},
            "success_rates": {},
        }

        for exp_name, result in self.results.items():
            # Determine experiment type
            if "foundation" in exp_name.lower():
                exp_type = "Foundation Models"
            elif "neurosymbolic" in exp_name.lower():
                exp_type = "Neurosymbolic RL"
            elif "continual" in exp_name.lower():
                exp_type = "Continual Learning"
            elif "collaboration" in exp_name.lower():
                exp_type = "Human-AI Collaboration"
            elif "computing" in exp_name.lower():
                exp_type = "Advanced Computing"
            elif "deployment" in exp_name.lower():
                exp_type = "Deployment & Ethics"
            elif "environment" in exp_name.lower():
                exp_type = "Environments"
            else:
                exp_type = "Other"

            if exp_type not in summary["experiment_types"]:
                summary["experiment_types"][exp_type] = 0
            summary["experiment_types"][exp_type] += 1

            # Check if experiment was successful
            if "error" not in result:
                summary["success_rates"][exp_name] = True
            else:
                summary["success_rates"][exp_name] = False

        return summary

    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics across experiments."""
        performance_analysis = {
            "model_sizes": {},
            "training_losses": {},
            "evaluation_scores": {},
            "computational_efficiency": {},
        }

        for exp_name, result in self.results.items():
            if "error" in result:
                continue

            # Extract model sizes
            if "model_parameters" in result:
                performance_analysis["model_sizes"][exp_name] = result["model_parameters"]

            # Extract training losses
            if "training_losses" in result:
                performance_analysis["training_losses"][exp_name] = result["training_losses"]

            # Extract evaluation scores
            if "evaluation_scores" in result:
                performance_analysis["evaluation_scores"][exp_name] = result["evaluation_scores"]

            # Extract computational efficiency
            if "inference_time" in result:
                performance_analysis["computational_efficiency"][exp_name] = result["inference_time"]

        return performance_analysis

    def compare_methods(self, metric: str = "performance") -> Dict[str, Any]:
        """Compare different methods on a specific metric."""
        comparison = {
            "methods": [],
            "scores": [],
            "best_method": None,
            "worst_method": None,
        }

        for exp_name, result in self.results.items():
            if "error" in result:
                continue

            # Extract metric value
            if metric in result:
                comparison["methods"].append(exp_name)
                comparison["scores"].append(result[metric])
            elif "performance_metrics" in result and metric in result["performance_metrics"]:
                comparison["methods"].append(exp_name)
                comparison["scores"].append(result["performance_metrics"][metric])

        if comparison["scores"]:
            best_idx = np.argmax(comparison["scores"])
            worst_idx = np.argmin(comparison["scores"])
            comparison["best_method"] = comparison["methods"][best_idx]
            comparison["worst_method"] = comparison["methods"][worst_idx]

        return comparison

    def generate_report(self) -> str:
        """Generate a comprehensive report."""
        report = []
        report.append("# CA16 Experiment Results Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        summary = self.get_experiment_summary()
        report.append("## Summary")
        report.append(f"- Total experiments: {summary['total_experiments']}")
        report.append(f"- Successful experiments: {sum(summary['success_rates'].values())}")
        report.append(f"- Failed experiments: {len(summary['success_rates']) - sum(summary['success_rates'].values())}")
        report.append("")

        # Experiment types
        report.append("## Experiment Types")
        for exp_type, count in summary["experiment_types"].items():
            report.append(f"- {exp_type}: {count}")
        report.append("")

        # Performance analysis
        performance = self.analyze_performance_metrics()
        report.append("## Performance Analysis")
        
        if performance["model_sizes"]:
            report.append("### Model Sizes")
            for exp_name, size in performance["model_sizes"].items():
                report.append(f"- {exp_name}: {size:,} parameters")
            report.append("")

        if performance["evaluation_scores"]:
            report.append("### Evaluation Scores")
            for exp_name, score in performance["evaluation_scores"].items():
                report.append(f"- {exp_name}: {score:.4f}")
            report.append("")

        # Method comparison
        comparison = self.compare_methods()
        if comparison["methods"]:
            report.append("## Method Comparison")
            report.append(f"- Best method: {comparison['best_method']}")
            report.append(f"- Worst method: {comparison['worst_method']}")
            report.append("")

        # Individual experiment details
        report.append("## Individual Experiment Details")
        for exp_name, result in self.results.items():
            report.append(f"### {exp_name}")
            if "error" in result:
                report.append(f"**Status**: Failed")
                report.append(f"**Error**: {result['error']}")
            else:
                report.append(f"**Status**: Success")
                if "duration" in result:
                    report.append(f"**Duration**: {result['duration']:.2f}s")
                if "result" in result:
                    report.append(f"**Result**: {result['result']}")
            report.append("")

        return "\n".join(report)

    def save_report(self, filename: str = "experiment_report.md"):
        """Save the report to a file."""
        report = self.generate_report()
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {filepath}")


class ResultsVisualizer:
    """Visualizer for experiment results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.analyzer = ResultsAnalyzer(results_dir)

    def plot_experiment_summary(self, save: bool = True):
        """Plot summary of all experiments."""
        summary = self.analyzer.get_experiment_summary()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("CA16 Experiment Summary", fontsize=16)
        
        # Plot 1: Experiment types
        exp_types = list(summary["experiment_types"].keys())
        exp_counts = list(summary["experiment_types"].values())
        axes[0, 0].bar(exp_types, exp_counts)
        axes[0, 0].set_title("Experiment Types")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Success rates
        success_rates = [sum(summary["success_rates"].values()), 
                        len(summary["success_rates"]) - sum(summary["success_rates"].values())]
        labels = ["Success", "Failure"]
        axes[0, 1].pie(success_rates, labels=labels, autopct='%1.1f%%')
        axes[0, 1].set_title("Success Rate")
        
        # Plot 3: Model sizes
        performance = self.analyzer.analyze_performance_metrics()
        if performance["model_sizes"]:
            model_names = list(performance["model_sizes"].keys())
            model_sizes = list(performance["model_sizes"].values())
            axes[1, 0].bar(model_names, model_sizes)
            axes[1, 0].set_title("Model Sizes")
            axes[1, 0].set_ylabel("Parameters")
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance comparison
        comparison = self.analyzer.compare_methods()
        if comparison["methods"]:
            axes[1, 1].bar(comparison["methods"], comparison["scores"])
            axes[1, 1].set_title("Method Comparison")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.results_dir, "experiment_summary.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to {filename}")
        
        plt.show()

    def plot_performance_trends(self, save: bool = True):
        """Plot performance trends over time."""
        performance = self.analyzer.analyze_performance_metrics()
        
        if not performance["training_losses"]:
            print("No training loss data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Performance Trends", fontsize=16)
        
        # Plot 1: Training losses
        for exp_name, losses in performance["training_losses"].items():
            if isinstance(losses, list):
                axes[0, 0].plot(losses, label=exp_name)
        axes[0, 0].set_title("Training Losses")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        
        # Plot 2: Model sizes comparison
        if performance["model_sizes"]:
            model_names = list(performance["model_sizes"].keys())
            model_sizes = list(performance["model_sizes"].values())
            axes[0, 1].bar(model_names, model_sizes)
            axes[0, 1].set_title("Model Sizes")
            axes[0, 1].set_ylabel("Parameters")
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Evaluation scores
        if performance["evaluation_scores"]:
            exp_names = list(performance["evaluation_scores"].keys())
            scores = list(performance["evaluation_scores"].values())
            axes[1, 0].bar(exp_names, scores)
            axes[1, 0].set_title("Evaluation Scores")
            axes[1, 0].set_ylabel("Score")
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Computational efficiency
        if performance["computational_efficiency"]:
            exp_names = list(performance["computational_efficiency"].keys())
            times = list(performance["computational_efficiency"].values())
            axes[1, 1].bar(exp_names, times)
            axes[1, 1].set_title("Computational Efficiency")
            axes[1, 1].set_ylabel("Inference Time (s)")
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.results_dir, "performance_trends.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Performance trends plot saved to {filename}")
        
        plt.show()

    def plot_method_comparison(self, metric: str = "performance", save: bool = True):
        """Plot comparison of different methods."""
        comparison = self.analyzer.compare_methods(metric)
        
        if not comparison["methods"]:
            print(f"No data available for metric: {metric}")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create bar plot
        bars = ax.bar(comparison["methods"], comparison["scores"])
        
        # Highlight best and worst methods
        if comparison["best_method"] and comparison["worst_method"]:
            best_idx = comparison["methods"].index(comparison["best_method"])
            worst_idx = comparison["methods"].index(comparison["worst_method"])
            
            bars[best_idx].set_color('green')
            bars[worst_idx].set_color('red')
        
        ax.set_title(f"Method Comparison: {metric.title()}")
        ax.set_ylabel("Score")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, comparison["scores"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.results_dir, f"method_comparison_{metric}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Method comparison plot saved to {filename}")
        
        plt.show()

    def create_dashboard(self, save: bool = True):
        """Create a comprehensive dashboard."""
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle("CA16 Experiment Dashboard", fontsize=20)
        
        # Create subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Experiment types
        ax1 = fig.add_subplot(gs[0, 0])
        summary = self.analyzer.get_experiment_summary()
        exp_types = list(summary["experiment_types"].keys())
        exp_counts = list(summary["experiment_types"].values())
        ax1.bar(exp_types, exp_counts)
        ax1.set_title("Experiment Types")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Success rates
        ax2 = fig.add_subplot(gs[0, 1])
        success_rates = [sum(summary["success_rates"].values()), 
                        len(summary["success_rates"]) - sum(summary["success_rates"].values())]
        labels = ["Success", "Failure"]
        ax2.pie(success_rates, labels=labels, autopct='%1.1f%%')
        ax2.set_title("Success Rate")
        
        # Plot 3: Model sizes
        ax3 = fig.add_subplot(gs[0, 2])
        performance = self.analyzer.analyze_performance_metrics()
        if performance["model_sizes"]:
            model_names = list(performance["model_sizes"].keys())
            model_sizes = list(performance["model_sizes"].values())
            ax3.bar(model_names, model_sizes)
            ax3.set_title("Model Sizes")
            ax3.set_ylabel("Parameters")
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance comparison
        ax4 = fig.add_subplot(gs[0, 3])
        comparison = self.analyzer.compare_methods()
        if comparison["methods"]:
            ax4.bar(comparison["methods"], comparison["scores"])
            ax4.set_title("Method Comparison")
            ax4.set_ylabel("Score")
            ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Training losses
        ax5 = fig.add_subplot(gs[1, :2])
        for exp_name, losses in performance["training_losses"].items():
            if isinstance(losses, list):
                ax5.plot(losses, label=exp_name)
        ax5.set_title("Training Losses")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Loss")
        ax5.legend()
        
        # Plot 6: Evaluation scores
        ax6 = fig.add_subplot(gs[1, 2:])
        if performance["evaluation_scores"]:
            exp_names = list(performance["evaluation_scores"].keys())
            scores = list(performance["evaluation_scores"].values())
            ax6.bar(exp_names, scores)
            ax6.set_title("Evaluation Scores")
            ax6.set_ylabel("Score")
            ax6.tick_params(axis='x', rotation=45)
        
        # Plot 7: Computational efficiency
        ax7 = fig.add_subplot(gs[2, :2])
        if performance["computational_efficiency"]:
            exp_names = list(performance["computational_efficiency"].keys())
            times = list(performance["computational_efficiency"].values())
            ax7.bar(exp_names, times)
            ax7.set_title("Computational Efficiency")
            ax7.set_ylabel("Inference Time (s)")
            ax7.tick_params(axis='x', rotation=45)
        
        # Plot 8: Summary statistics
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')
        
        # Add summary text
        summary_text = f"""
        Total Experiments: {summary['total_experiments']}
        Successful: {sum(summary['success_rates'].values())}
        Failed: {len(summary['success_rates']) - sum(summary['success_rates'].values())}
        
        Best Method: {comparison['best_method'] if comparison['best_method'] else 'N/A'}
        Worst Method: {comparison['worst_method'] if comparison['worst_method'] else 'N/A'}
        """
        
        ax8.text(0.1, 0.5, summary_text, transform=ax8.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        if save:
            filename = os.path.join(self.results_dir, "experiment_dashboard.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {filename}")
        
        plt.show()


def main():
    """Main function for results analysis."""
    print("CA16 Results Analysis")
    print("=" * 50)
    
    # Create analyzer
    analyzer = ResultsAnalyzer("experiment_results")
    
    # Generate report
    print("Generating report...")
    analyzer.save_report()
    
    # Create visualizer
    visualizer = ResultsVisualizer("experiment_results")
    
    # Generate plots
    print("Generating plots...")
    visualizer.plot_experiment_summary()
    visualizer.plot_performance_trends()
    visualizer.plot_method_comparison()
    visualizer.create_dashboard()
    
    print("Results analysis complete!")
    print("Check the 'experiment_results' directory for all outputs.")


if __name__ == "__main__":
    main()
