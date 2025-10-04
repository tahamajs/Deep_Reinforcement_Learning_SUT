"""
Next-Generation Deep Reinforcement Learning Package

This package provides a comprehensive implementation of cutting-edge deep RL paradigms,
including world models, multi-agent systems, causal reasoning, quantum-enhanced learning,
federated learning, and advanced safety techniques.

Modules:
- world_models: World models and imagination-augmented agents
- multi_agent_rl: Multi-agent deep reinforcement learning
- causal_rl: Causal reasoning in reinforcement learning
- quantum_rl: Quantum-enhanced reinforcement learning
- federated_rl: Federated reinforcement learning
- advanced_safety: Advanced safety and robustness techniques
- utils: Utility functions and classes
- environments: Custom RL environments
- experiments: Comprehensive evaluation suites

Usage:
    from ca17 import world_models, multi_agent_rl

    from ca17.world_models import ImaginationAugmentedAgent
    from ca17.environments import ContinuousMountainCar

    from ca17.experiments import WorldModelExperiment, create_default_configs
"""

__version__ = "1.0.0"
__author__ = "CA17 Modular RL Package"
__description__ = "Next-Generation Deep Reinforcement Learning Implementations"

from .world_models import RSSMCore, WorldModel, MPCPlanner, ImaginationAugmentedAgent

from .multi_agent_rl import MADDPGAgent, CommunicationNetwork, PredatorPreyEnvironment

from .causal_rl import (
    CausalGraph,
    PCCausalDiscovery,
    CausalRLAgent,
    CausalBanditEnvironment,
)

from .quantum_rl import (
    QuantumGate,
    QuantumCircuit,
    QuantumRLAgent,
    QuantumControlEnvironment,
)

from .federated_rl import DifferentialPrivacy, FederatedRLClient, FederatedRLServer

from .advanced_safety import (
    SafetyConstraints,
    ConstrainedPolicyOptimization,
    SafetyMonitor,
)

from .utils import ReplayBuffer, Config, plot_learning_curve, set_random_seed

from .environments import ContinuousMountainCar, BaseEnvironment

from .experiments import (
    WorldModelExperiment,
    MultiAgentExperiment,
    ComparativeExperiment,
    create_default_configs,
)

__all__ = [
    "RSSMCore",
    "WorldModel",
    "MPCPlanner",
    "ImaginationAugmentedAgent",
    "MADDPGAgent",
    "CommunicationNetwork",
    "PredatorPreyEnvironment",
    "CausalGraph",
    "PCCausalDiscovery",
    "CausalRLAgent",
    "CausalBanditEnvironment",
    "QuantumGate",
    "QuantumCircuit",
    "QuantumRLAgent",
    "QuantumControlEnvironment",
    "DifferentialPrivacy",
    "FederatedRLClient",
    "FederatedRLServer",
    "SafetyConstraints",
    "ConstrainedPolicyOptimization",
    "SafetyMonitor",
    "ReplayBuffer",
    "Config",
    "plot_learning_curve",
    "set_random_seed",
    "ContinuousMountainCar",
    "BaseEnvironment",
    "WorldModelExperiment",
    "MultiAgentExperiment",
    "ComparativeExperiment",
    "create_default_configs",
]


def run_quick_demo():
    """Run a quick demonstration of the package capabilities"""
    print("🚀 CA17 Next-Generation RL Package Demo")
    print("=" * 50)

    set_random_seed(42)

    configs = create_default_configs()

    print("\n📊 Running World Model Demo...")
    config = configs["world_model"]
    config.n_episodes = 10

    experiment = WorldModelExperiment(config, save_dir="demo_results")
    results = experiment.run_experiment()

    print(f"✅ World Model Demo Results: {results['final_reward']:.2f}")
    experiment.plot_results()

    print("\n✅ Demo completed! Check 'demo_results' directory for outputs.")
    print("\n📚 Available modules:")
    print("   • world_models - World models and imagination")
    print("   • multi_agent_rl - Multi-agent deep RL")
    print("   • causal_rl - Causal reasoning")
    print("   • quantum_rl - Quantum-enhanced RL")
    print("   • federated_rl - Federated learning")
    print("   • advanced_safety - Safety and robustness")
    print("   • utils - Utilities and helpers")
    print("   • environments - Custom environments")
    print("   • experiments - Evaluation suites")


if __name__ == "__main__":
    run_quick_demo()
