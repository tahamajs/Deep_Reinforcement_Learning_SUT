"""
CA18 - Advanced Reinforcement Learning Paradigms

A modular Python package implementing advanced reinforcement learning algorithms
including quantum-enhanced RL, world models, multi-agent systems, causal RL,
federated learning, and advanced safety mechanisms.

This package transforms the monolithic CA18.ipynb notebook into a clean,
maintainable, and reusable codebase following the proven CA17 modular structure.

Modules:
- quantum_rl: Quantum-enhanced reinforcement learning algorithms
- world_models: World model-based RL with RSSM architectures
- multi_agent_rl: Multi-agent deep RL with communication
- causal_rl: Causal reinforcement learning with discovery
- federated_rl: Federated reinforcement learning frameworks
- advanced_safety: Safety constraints and robust policies
- utils: Advanced utilities and data structures
- environments: Test environments for advanced RL paradigms
- experiments: Experiment frameworks and evaluation suites

Usage:
    from CA18 import quantum_rl, causal_rl, experiments

    # Create a quantum RL agent
    agent = quantum_rl.QuantumQLearning(...)

    # Run experiments
    experiment = experiments.QuantumRLExperiment(agent, environment)
    results = experiment.run_experiment()
"""

__version__ = "1.0.0"
__author__ = "CA18 Modular RL Package"

# Import main modules for easy access
from . import quantum_rl
from . import world_models
from . import multi_agent_rl
from . import causal_rl
from . import federated_rl
from . import advanced_safety
from . import utils
from . import environments
from . import experiments

# Make key classes easily accessible at package level
from .quantum_rl import QuantumQLearning, QuantumActorCritic, QuantumEnvironment
from .world_models import RSSMCore, WorldModel, ImaginationAugmentedAgent
from .multi_agent_rl import MADDPGAgent, MultiAgentEnvironment
from .causal_rl import CausalDiscovery, CausalWorldModel, CausalPolicyGradient
from .federated_rl import FederatedRLClient, FederatedRLServer
from .advanced_safety import SafetyConstraints, QuantumConstrainedPolicyOptimization
from .utils import QuantumPrioritizedReplayBuffer, QuantumMetricsTracker
from .environments import QuantumEnvironment as QuantumEnv, CausalBanditEnvironment
from .experiments import ComparativeExperimentRunner

__all__ = [
    # Modules
    "quantum_rl",
    "world_models",
    "multi_agent_rl",
    "causal_rl",
    "federated_rl",
    "advanced_safety",
    "utils",
    "environments",
    "experiments",

    # Key classes
    "QuantumQLearning",
    "QuantumActorCritic",
    "QuantumEnvironment",
    "RSSMCore",
    "WorldModel",
    "ImaginationAugmentedAgent",
    "MADDPGAgent",
    "MultiAgentEnvironment",
    "CausalDiscovery",
    "CausalWorldModel",
    "CausalPolicyGradient",
    "FederatedRLClient",
    "FederatedRLServer",
    "SafetyConstraints",
    "QuantumConstrainedPolicyOptimization",
    "QuantumPrioritizedReplayBuffer",
    "QuantumMetricsTracker",
    "QuantumEnv",
    "CausalBanditEnvironment",
    "ComparativeExperimentRunner",
]

def get_available_modules():
    """Get list of available modules in CA18 package"""
    return [
        "quantum_rl",
        "world_models",
        "multi_agent_rl",
        "causal_rl",
        "federated_rl",
        "advanced_safety",
        "utils",
        "environments",
        "experiments",
    ]

def create_experiment_runner(algorithm_type: str, **kwargs):
    """
    Factory function to create experiment runners for different algorithm types

    Args:
        algorithm_type: Type of algorithm ('quantum', 'causal', 'multi_agent', 'federated')
        **kwargs: Additional arguments for the experiment runner

    Returns:
        Appropriate experiment runner instance
    """
    runners = {
        'quantum': experiments.QuantumRLExperiment,
        'causal': experiments.CausalRLExperiment,
        'multi_agent': experiments.MultiAgentRLExperiment,
        'federated': experiments.FederatedRLExperiment,
    }

    if algorithm_type not in runners:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}. "
                        f"Available types: {list(runners.keys())}")

    return runners[algorithm_type](**kwargs)

def run_quick_test(module_name: str = "quantum_rl"):
    """
    Run a quick test to verify that a module can be imported and basic functionality works

    Args:
        module_name: Name of the module to test
    """
    try:
        if module_name == "quantum_rl":
            from .quantum_rl import QuantumState
            # Test basic quantum state creation
            state = QuantumState(n_qubits=2)
            print(f"✅ {module_name} test passed: Created quantum state with {state.n_qubits} qubits")

        elif module_name == "causal_rl":
            from .causal_rl import CausalGraph
            # Test basic causal graph creation
            graph = CausalGraph(n_variables=3)
            print(f"✅ {module_name} test passed: Created causal graph with {graph.n_variables} variables")

        elif module_name == "world_models":
            from .world_models import RSSMCore
            # Test RSSM core creation
            rssm = RSSMCore(state_dim=10, action_dim=4, hidden_dim=32)
            print(f"✅ {module_name} test passed: Created RSSM core")

        elif module_name == "utils":
            from .utils import QuantumRNG
            # Test quantum RNG
            rng = QuantumRNG()
            random_val = rng.quantum_random()
            print(f"✅ {module_name} test passed: Generated quantum random value {random_val}")

        else:
            print(f"⚠️  No quick test defined for {module_name}")

    except Exception as e:
        print(f"❌ {module_name} test failed: {e}")

def print_package_info():
    """Print information about the CA18 package"""
    print("=" * 60)
    print("CA18 - Advanced Reinforcement Learning Paradigms")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    print("Available Modules:")
    for module in get_available_modules():
        print(f"  - {module}")
    print()
    print("Quick Start:")
    print("  from CA18 import quantum_rl, experiments")
    print("  # Create and run experiments with advanced RL algorithms")
    print("=" * 60)

# Run package info on import
if __name__ != "__main__":
    print("🎯 CA18 Advanced RL Package loaded successfully!")
    print("   Use print_package_info() for detailed information")
    print("   Use run_quick_test() to verify module functionality")