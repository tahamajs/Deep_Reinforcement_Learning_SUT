"""
utils.py - General utility functions for Temporal Difference Learning projects.

This module contains miscellaneous helper functions for tasks such as
model saving/loading, result exporting, and summary report generation.
"""

import os
import pickle
import json
from typing import Any, Dict
from datetime import datetime

# Assuming BaseAgent is available in the src package
from .agents import BaseAgent

def save_model(model: Any, filename: str, save_dir: str = "models") -> None:
    """
    Saves a Python object (e.g., an agent) to a file using pickle.

    Args:
        model (Any): The object to save.
        filename (str): The name of the file to save the model to (e.g., "agent.pkl").
        save_dir (str): The directory where the model will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {filepath}")

def load_model(filepath: str) -> Any:
    """
    Loads a Python object (e.g., an agent) from a pickle file.

    Args:
        filepath (str): The full path to the model file.

    Returns:
        Any: The loaded object.
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from {filepath}")
    return model

def export_results(results: Dict[str, Any], filename_prefix: str, save_dir: str = "results") -> None:
    """
    Exports experimental results to a JSON file.

    Args:
        results (Dict[str, Any]): A dictionary containing the results.
        filename_prefix (str): A prefix for the filename (e.g., "q_learning_run").
        save_dir (str): The directory where the results will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Results exported to {filepath}")

def create_summary_report(all_results: Dict[str, Any], save_dir: str = "results", filename: str = "summary_report.txt") -> None:
    """
    Creates a human-readable summary report of all experimental results.

    Args:
        all_results (Dict[str, Any]): A dictionary containing results from multiple experiments/agents.
                                      Expected structure: {"AgentName": {"avg_reward": ..., "success_rate": ...}}
        save_dir (str): The directory where the summary report will be saved.
        filename (str): The name of the summary report file.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEMPORAL DIFFERENCE LEARNING - EXPERIMENT SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "-" * 30 + "\n")
        f.write("Individual Algorithm Results:\n")
        f.write("-" * 30 + "\n")

        for agent_name, results in all_results.items():
            f.write(f"  Agent: {agent_name}\n")
            if "avg_reward" in results:
                f.write(f"    Average Reward: {results['avg_reward']:.2f}\n")
            if "std_reward" in results:
                f.write(f"    Std Dev Reward: {results['std_reward']:.2f}\n")
            if "success_rate" in results and isinstance(results['success_rate'], (int, float)):
                f.write(f"    Success Rate: {results['success_rate']*100:.1f}%\n")
            elif "success_rate" in results:
                f.write(f"    Success Rate: {results['success_rate']}\n")
            if "total_episodes" in results:
                f.write(f"    Total Episodes Trained: {results['total_episodes']}\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    print(f"✓ Summary report created at {filepath}")


if __name__ == "__main__":
    # Example Usage:
    print("--- Utility Functions Demo ---")
    
    # Assuming an agent class exists for demonstration
    class DummyAgent:
        def __init__(self, name):
            self.name = name
            self.episode_rewards = [np.random.rand() * 10 for _ in range(50)]
            self.Q = {("s1"): {"a1": 1.0, "a2": 2.0}}
        
        def get_value_function(self): return {"s1": 2.0}
        def get_policy(self): return {"s1": "a2"}
        def evaluate_policy(self, num_episodes=10):
            return {"avg_reward": np.mean(self.episode_rewards[-num_episodes:]), "std_reward": np.std(self.episode_rewards[-num_episodes:]), "success_rate": 0.9}

    dummy_agent = DummyAgent("TestAgent")

    # Save model
    save_model(dummy_agent, "dummy_agent.pkl", save_dir="temp_models")

    # Load model
    loaded_agent = load_model("temp_models/dummy_agent.pkl")
    print(f"Loaded agent name: {loaded_agent.name}")

    # Export results
    results = {
        "algorithm": "DummyRL",
        "avg_reward": 5.5,
        "success_rate": 0.8,
        "params": {"lr": 0.01, "gamma": 0.99}
    }
    export_results(results, "dummy_run", save_dir="temp_results")

    # Create summary report
    all_results = {
        "DummyRL_Agent1": {"avg_reward": 6.0, "std_reward": 1.0, "success_rate": 0.95, "total_episodes": 100},
        "DummyRL_Agent2": {"avg_reward": 5.5, "std_reward": 1.2, "success_rate": 0.90, "total_episodes": 120},
    }
    create_summary_report(all_results, save_dir="temp_results", filename="dummy_summary.txt")

    print("\nAll utility functions demos finished. Check 'temp_models' and 'temp_results' directories.")

    # Clean up dummy directories and files
    import shutil
    shutil.rmtree("temp_models", ignore_errors=True)
    shutil.rmtree("temp_results", ignore_errors=True)
    print("Cleaned up temporary directories.")
