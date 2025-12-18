"""
Experiments Module

This module contains experiment scripts and utilities for testing CA16 components.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import json
import os

# Import CA16 modules
from foundation_models import DecisionTransformer, MultiTaskDecisionTransformer
from neurosymbolic import SymbolicKnowledgeBase, NeurosymbolicPolicy
from continual_learning import ContinualLearningAgent, MAML
from human_ai_collaboration import CollaborativeAgent, PreferenceRewardModel
from environments import SymbolicGridWorld, CollaborativeGridWorld, ContinualLearningEnvironment
from advanced_computing import QuantumInspiredRL, NeuromorphicNetwork
from deployment_ethics import ProductionRLSystem, SafetyMonitor, EthicsChecker


class ExperimentRunner:
    """Base class for running experiments."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Experiment tracking
        self.experiments = []
        self.results = {}
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def run_experiment(self, experiment_name: str, experiment_func, *args, **kwargs):
        """Run an experiment and save results."""
        print(f"Running experiment: {experiment_name}")
        start_time = time.time()
        
        try:
            result = experiment_func(*args, **kwargs)
            end_time = time.time()
            
            # Save results
            self.results[experiment_name] = {
                "result": result,
                "duration": end_time - start_time,
                "timestamp": time.time(),
            }
            
            # Save to file
            self._save_experiment_result(experiment_name, result)
            
            print(f"Experiment {experiment_name} completed in {end_time - start_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"Experiment {experiment_name} failed: {str(e)}")
            self.results[experiment_name] = {
                "error": str(e),
                "duration": time.time() - start_time,
                "timestamp": time.time(),
            }
            return None

    def _save_experiment_result(self, experiment_name: str, result: Any):
        """Save experiment result to file."""
        filename = os.path.join(self.output_dir, f"{experiment_name}_result.json")
        
        # Convert result to serializable format
        if isinstance(result, dict):
            serializable_result = self._make_serializable(result)
        else:
            serializable_result = str(result)
        
        with open(filename, 'w') as f:
            json.dump(serializable_result, f, indent=2)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)

    def plot_results(self, experiment_name: str, data: Dict[str, Any], save: bool = True):
        """Plot experiment results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Experiment Results: {experiment_name}")
        
        # Plot 1: Training curves
        if "training_losses" in data:
            axes[0, 0].plot(data["training_losses"])
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
        
        # Plot 2: Performance metrics
        if "performance_metrics" in data:
            metrics = data["performance_metrics"]
            if isinstance(metrics, dict):
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                axes[0, 1].bar(metric_names, metric_values)
                axes[0, 1].set_title("Performance Metrics")
                axes[0, 1].set_ylabel("Value")
        
        # Plot 3: Comparison plot
        if "comparison_data" in data:
            comparison = data["comparison_data"]
            if isinstance(comparison, dict):
                methods = list(comparison.keys())
                scores = list(comparison.values())
                axes[1, 0].bar(methods, scores)
                axes[1, 0].set_title("Method Comparison")
                axes[1, 0].set_ylabel("Score")
        
        # Plot 4: Statistics
        if "statistics" in data:
            stats = data["statistics"]
            if isinstance(stats, dict):
                stat_names = list(stats.keys())
                stat_values = list(stats.values())
                axes[1, 1].bar(stat_names, stat_values)
                axes[1, 1].set_title("Statistics")
                axes[1, 1].set_ylabel("Value")
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, f"{experiment_name}_plot.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        plt.show()


def foundation_model_experiment():
    """Test foundation models."""
    print("Testing Foundation Models...")
    
    # Create Decision Transformer
    dt = DecisionTransformer(
        state_dim=4,
        action_dim=2,
        hidden_dim=64,
        num_layers=3,
            num_heads=4,
        max_length=100,
        )

    # Test forward pass
    batch_size = 8
    seq_length = 10

    states = torch.randn(batch_size, seq_length, 4)
    actions = torch.randint(0, 2, (batch_size, seq_length))
    returns_to_go = torch.randn(batch_size, seq_length, 1)
    timesteps = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)

    # Forward pass
    action_logits, values = dt(states, actions, returns_to_go, timesteps)

    print(f"Decision Transformer output shapes: {action_logits.shape}, {values.shape}")

    # Test action prediction
    action = dt.get_action(states[0], actions[0], returns_to_go[0], timesteps[0])
    print(f"Predicted action: {action}")

        return {
        "model_parameters": sum(p.numel() for p in dt.parameters()),
        "output_shapes": {"action_logits": list(action_logits.shape), "values": list(values.shape)},
        "predicted_action": action.item() if isinstance(action, torch.Tensor) else action,
    }


def neurosymbolic_experiment():
    """Test neurosymbolic RL."""
    print("Testing Neurosymbolic RL...")
    
    # Create symbolic knowledge base
    kb = SymbolicKnowledgeBase()
    
    # Add some predicates
    kb.add_predicate("at", 2)  # at(agent, location)
    kb.add_predicate("goal", 1)  # goal(location)
    kb.add_predicate("obstacle", 1)  # obstacle(location)
    
    # Add some rules
    kb.add_rule("reachable(X, Y)", ["at(agent, X)", "not obstacle(Y)"], 0.8)
    kb.add_rule("safe_move(X, Y)", ["at(agent, X)", "not obstacle(Y)", "reachable(X, Y)"], 0.9)
    
    # Test inference
    facts = ["at(agent, (0,0))", "goal((5,5))", "obstacle((2,2))"]
    kb.add_facts(facts)
    
    # Forward chaining
    inferred = kb.forward_chaining()
    print(f"Inferred facts: {inferred}")
    
    # Create neurosymbolic policy
    policy = NeurosymbolicPolicy(
        state_dim=4,
            action_dim=4,
        hidden_dim=32,
        symbolic_dim=8,
    )
    
    # Test policy
    state = torch.randn(1, 4)
    action_logits, value = policy(state)
    
    print(f"Neurosymbolic policy output shapes: {action_logits.shape}, {value.shape}")
    
    return {
        "knowledge_base_size": len(kb.predicates),
        "rules_count": len(kb.rules),
        "inferred_facts": len(inferred),
        "policy_parameters": sum(p.numel() for p in policy.parameters()),
    }


def continual_learning_experiment():
    """Test continual learning."""
    print("Testing Continual Learning...")
    
    # Create continual learning agent
    agent = ContinualLearningAgent(
        state_dim=4,
        action_dim=2,
        hidden_dim=64,
        num_tasks=3,
    )
    
    # Test on multiple tasks
    task_performances = {}
    
    for task_id in range(3):
        # Generate mock data for task
        states = torch.randn(100, 4)
        actions = torch.randint(0, 2, (100,))
        rewards = torch.randn(100)
        
        # Train agent on task
        loss = agent.train_step(states, actions, rewards, task_id)
        
        # Evaluate performance
        with torch.no_grad():
            action_probs = agent.get_action_probs(states[:10], task_id)
            value = agent.get_value(states[:10], task_id)
        
        task_performances[task_id] = {
            "loss": loss,
            "action_probs_mean": action_probs.mean().item(),
            "value_mean": value.mean().item(),
        }
    
    # Compute forgetting metrics
    forgetting_metrics = agent.compute_forgetting_metrics()
    
    print(f"Task performances: {task_performances}")
    print(f"Forgetting metrics: {forgetting_metrics}")
    
    return {
        "task_performances": task_performances,
        "forgetting_metrics": forgetting_metrics,
        "agent_parameters": sum(p.numel() for p in agent.parameters()),
    }


def human_ai_collaboration_experiment():
    """Test human-AI collaboration."""
    print("Testing Human-AI Collaboration...")
    
    # Create collaborative agent
    agent = CollaborativeAgent(
        state_dim=4,
        action_dim=2,
        hidden_dim=32,
        confidence_threshold=0.7,
    )
    
    # Create preference model
    preference_model = PreferenceRewardModel(
        state_dim=4,
        action_dim=2,
        hidden_dim=32,
    )
    
    # Test collaboration
    state = torch.randn(1, 4)
    action, confidence = agent.get_action(state)
    
    print(f"Agent action: {action}, confidence: {confidence}")
    
    # Test preference learning
    state1 = torch.randn(1, 4)
    state2 = torch.randn(1, 4)
    action1 = torch.tensor([0])
    action2 = torch.tensor([1])
    preference = torch.tensor([1.0])  # Prefer action1
    
    preference_loss = preference_model.compute_loss(state1, action1, state2, action2, preference)
    
    print(f"Preference loss: {preference_loss.item()}")
    
    return {
        "agent_confidence": confidence,
        "preference_loss": preference_loss.item(),
        "agent_parameters": sum(p.numel() for p in agent.parameters()),
        "preference_model_parameters": sum(p.numel() for p in preference_model.parameters()),
    }


def advanced_computing_experiment():
    """Test advanced computing paradigms."""
    print("Testing Advanced Computing Paradigms...")
    
    # Test quantum-inspired RL
    quantum_rl = QuantumInspiredRL(
        state_dim=4,
        action_dim=2,
        num_qubits=4,
        num_layers=2,
    )
    
    state = torch.randn(1, 4)
    action_logits = quantum_rl(state)
    
    print(f"Quantum RL output shape: {action_logits.shape}")
    
    # Test neuromorphic network
    neuromorphic_net = NeuromorphicNetwork(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        num_layers=2,
        time_steps=5,
    )
    
    input_spikes = torch.randn(1, 4)
    output_rates = neuromorphic_net(input_spikes)
    
    print(f"Neuromorphic network output shape: {output_rates.shape}")
    
    return {
        "quantum_rl_parameters": sum(p.numel() for p in quantum_rl.parameters()),
        "neuromorphic_parameters": sum(p.numel() for p in neuromorphic_net.parameters()),
        "quantum_output_shape": list(action_logits.shape),
        "neuromorphic_output_shape": list(output_rates.shape),
    }


def deployment_ethics_experiment():
    """Test deployment and ethics components."""
    print("Testing Deployment and Ethics...")
    
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    
    # Test production system
    production_system = ProductionRLSystem(model)
    
    # Deploy system
    config = {
        "model_path": "test_model.pth",
        "environment_config": {"state_dim": 4, "action_dim": 2},
        "safety_config": {"threshold": 0.8},
    }
    
    deployed = production_system.deploy(config)
    print(f"System deployed: {deployed}")
    
    # Test inference
    if deployed:
        state = torch.randn(1, 4)
        action, info = production_system.inference(state)
        print(f"Inference result: {action.shape}, info: {info}")
    
    # Test safety monitor
    safety_monitor = SafetyMonitor(
        safety_thresholds={"inference_time": 0.1, "memory_usage": 0.8},
    )
    
    safety_monitor.start_monitoring()
    safety_monitor.update_metrics({"inference_time": 0.05, "memory_usage": 0.6})
    
    safety_report = safety_monitor.get_safety_report()
    print(f"Safety report: {safety_report}")
    
    # Test ethics checker
    ethics_checker = EthicsChecker(
        ethical_guidelines={"bias_threshold": 0.1, "fairness_threshold": 0.8},
    )
    
    predictions = torch.randn(100, 2)
    protected_attributes = torch.randint(0, 2, (100,))
    
    bias_result = ethics_checker.check_bias(predictions, protected_attributes)
    print(f"Bias check result: {bias_result}")
    
    return {
        "system_deployed": deployed,
        "safety_violations": safety_report["total_violations"],
        "bias_score": bias_result["bias_score"],
        "system_status": production_system.get_system_status(),
    }


def environment_experiment():
    """Test custom environments."""
    print("Testing Custom Environments...")
    
    # Test symbolic grid world
    symbolic_env = SymbolicGridWorld(size=5, num_goals=2, num_obstacles=3)
    
    obs, info = symbolic_env.reset()
    print(f"Symbolic environment observation shape: {obs.shape}")
    
    # Test collaborative grid world
    collaborative_env = CollaborativeGridWorld(size=5, num_goals=2, num_obstacles=3)
    
    obs, info = collaborative_env.reset()
    print(f"Collaborative environment observation shape: {obs.shape}")
    
    # Test continual learning environment
    continual_env = ContinualLearningEnvironment(num_tasks=3, state_dim=4, action_dim=2)
    
    obs, info = continual_env.reset()
    print(f"Continual environment observation shape: {obs.shape}")
    
    return {
        "symbolic_env_size": symbolic_env.size,
        "collaborative_env_size": collaborative_env.size,
        "continual_env_tasks": continual_env.num_tasks,
        "observation_shapes": {
            "symbolic": list(obs.shape),
            "collaborative": list(obs.shape),
            "continual": list(obs.shape),
        },
    }


def run_all_experiments():
    """Run all experiments."""
    print("Starting CA16 Comprehensive Experiments...")
    
    runner = ExperimentRunner("experiment_results")
    
    # Run experiments
    experiments = [
        ("foundation_models", foundation_model_experiment),
        ("neurosymbolic", neurosymbolic_experiment),
        ("continual_learning", continual_learning_experiment),
        ("human_ai_collaboration", human_ai_collaboration_experiment),
        ("advanced_computing", advanced_computing_experiment),
        ("deployment_ethics", deployment_ethics_experiment),
        ("environments", environment_experiment),
    ]
    
    all_results = {}
    
    for exp_name, exp_func in experiments:
        result = runner.run_experiment(exp_name, exp_func)
        if result:
            all_results[exp_name] = result
    
    # Generate summary report
    summary = {
        "total_experiments": len(experiments),
        "successful_experiments": len(all_results),
        "failed_experiments": len(experiments) - len(all_results),
        "experiment_results": all_results,
    }
    
    # Save summary
    with open(os.path.join(runner.output_dir, "experiment_summary.json"), 'w') as f:
        json.dump(runner._make_serializable(summary), f, indent=2)
    
    print(f"\nExperiment Summary:")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful_experiments']}")
    print(f"Failed: {summary['failed_experiments']}")
    print(f"Results saved to: {runner.output_dir}")
    
    return summary


if __name__ == "__main__":
    # Run all experiments
    results = run_all_experiments()
    
    print("\nAll experiments completed!")
    print("Check the 'experiment_results' directory for detailed results.")