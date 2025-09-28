"""
CA19 Modular: Advanced RL Systems Package

This package provides modular implementations of next-generation reinforcement learning
systems from CA19, including:

- Hybrid Quantum-Classical RL: Fusion of quantum circuits with classical neural networks
- Neuromorphic RL: Brain-inspired learning with spiking neural networks and STDP
- Quantum RL: Quantum-enhanced agents for complex control tasks
- Advanced Environments: Testbeds for evaluating different RL paradigms
- Comprehensive Experiments: Systematic evaluation and comparison frameworks

The modular design enables easy experimentation, comparison, and extension of
cutting-edge RL algorithms.

Usage:
    from ca19_modular import (
        HybridQuantumClassicalAgent,
        NeuromorphicActorCritic,
        QuantumEnhancedAgent,
        SpaceStationEnvironment,
        QuantumNeuromorphicComparison
    )

    # Quick experiment
    config = MissionConfig()
    experiment = QuantumNeuromorphicComparison(config)
    results = experiment.run_comparison_experiment()
"""

__version__ = "1.0.0"
__author__ = "CA19 Research Team"
__description__ = "Modular implementations of advanced RL systems from CA19"

# Core imports for easy access
from .hybrid_quantum_classical_rl import (
    HybridQuantumClassicalAgent,
    QuantumStateSimulator,
    QuantumFeatureMap,
    VariationalQuantumCircuit,
)

from .neuromorphic_rl import (
    NeuromorphicActorCritic,
    SpikingNeuron,
    STDPSynapse,
    SpikingNetwork,
)

from .quantum_rl import (
    QuantumEnhancedAgent,
    QuantumRLCircuit,
    SpaceStationEnvironment,
    MissionTrainer,
)

from .environments import (
    NeuromorphicEnvironment,
    HybridQuantumClassicalEnvironment,
    MetaLearningEnvironment,
    ContinualLearningEnvironment,
    HierarchicalEnvironment,
)

from .experiments import (
    QuantumNeuromorphicComparison,
    AblationStudy,
    ScalabilityAnalysis,
    ExperimentRunner,
    run_quick_comparison,
    run_ablation_study,
    benchmark_scalability,
)

from .utils import (
    MissionConfig,
    PerformanceTracker,
    ExperimentManager,
    create_default_config,
    save_config,
    load_config,
    setup_experiment_logging,
    benchmark_quantum_vs_classical,
)


# Package-level utilities
def get_available_modules():
    """Get list of available modules in the package"""
    return [
        "hybrid_quantum_classical_rl",
        "neuromorphic_rl",
        "quantum_rl",
        "environments",
        "experiments",
        "utils",
    ]


def get_module_info(module_name: str) -> dict:
    """Get information about a specific module"""
    module_info = {
        "hybrid_quantum_classical_rl": {
            "description": "Hybrid quantum-classical reinforcement learning agents",
            "classes": [
                "HybridQuantumClassicalAgent",
                "QuantumStateSimulator",
                "QuantumFeatureMap",
                "VariationalQuantumCircuit",
            ],
            "purpose": "Fusion of quantum circuits with classical neural networks",
        },
        "neuromorphic_rl": {
            "description": "Brain-inspired neuromorphic reinforcement learning",
            "classes": [
                "NeuromorphicActorCritic",
                "SpikingNeuron",
                "STDPSynapse",
                "SpikingNetwork",
            ],
            "purpose": "Event-driven learning with spiking neural networks",
        },
        "quantum_rl": {
            "description": "Quantum-enhanced reinforcement learning",
            "classes": [
                "QuantumEnhancedAgent",
                "QuantumRLCircuit",
                "SpaceStationEnvironment",
                "MissionTrainer",
            ],
            "purpose": "Quantum algorithms for complex control tasks",
        },
        "environments": {
            "description": "Advanced environments for RL evaluation",
            "classes": [
                "NeuromorphicEnvironment",
                "HybridQuantumClassicalEnvironment",
                "MetaLearningEnvironment",
                "ContinualLearningEnvironment",
                "HierarchicalEnvironment",
            ],
            "purpose": "Testbeds for different RL paradigms",
        },
        "experiments": {
            "description": "Experiment frameworks and evaluation tools",
            "classes": [
                "QuantumNeuromorphicComparison",
                "AblationStudy",
                "ScalabilityAnalysis",
                "ExperimentRunner",
            ],
            "purpose": "Systematic evaluation and comparison of RL systems",
        },
        "utils": {
            "description": "Utility functions and configuration management",
            "classes": ["MissionConfig", "PerformanceTracker", "ExperimentManager"],
            "purpose": "Configuration, tracking, and helper utilities",
        },
    }

    return module_info.get(module_name, {"error": f"Module {module_name} not found"})


def create_quick_experiment(agent_type: str = "hybrid", env_type: str = "neuromorphic"):
    """
    Create a quick experiment setup for testing

    Args:
        agent_type: Type of agent ('hybrid', 'neuromorphic', 'quantum')
        env_type: Type of environment ('neuromorphic', 'hybrid', 'meta', 'continual', 'hierarchical')

    Returns:
        Tuple of (agent, environment, config)
    """
    config = MissionConfig()

    # Agent selection
    if agent_type == "hybrid":
        agent = HybridQuantumClassicalAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            quantum_dim=config.n_qubits,
            hidden_dim=config.state_dim,
        )
    elif agent_type == "neuromorphic":
        agent = NeuromorphicActorCritic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            neuron_count=config.state_dim * 2,
            synapse_count=config.state_dim * 4,
        )
    elif agent_type == "quantum":
        agent = QuantumEnhancedAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            quantum_dim=config.n_qubits,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Environment selection
    if env_type == "neuromorphic":
        env = NeuromorphicEnvironment(
            state_dim=config.state_dim, action_dim=config.action_dim
        )
    elif env_type == "hybrid":
        env = HybridQuantumClassicalEnvironment(
            state_dim=config.state_dim * 2, action_dim=config.action_dim * 2
        )
    elif env_type == "meta":
        env = MetaLearningEnvironment(base_state_dim=config.state_dim, num_tasks=5)
    elif env_type == "continual":
        env = ContinualLearningEnvironment(state_dim=config.state_dim, num_phases=3)
    elif env_type == "hierarchical":
        env = HierarchicalEnvironment(state_dim=config.state_dim, num_levels=3)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    return agent, env, config


def run_demo_experiment(
    episodes: int = 10,
    agent_type: str = "hybrid",
    env_type: str = "neuromorphic",
    verbose: bool = True,
):
    """
    Run a quick demo experiment to showcase the modular RL systems

    Args:
        episodes: Number of episodes to run
        agent_type: Type of agent to use
        env_type: Type of environment
        verbose: Whether to print progress

    Returns:
        Dictionary with experiment results
    """
    print(f"🚀 Running CA19 Modular Demo: {agent_type} agent in {env_type} environment")
    print(f"Episodes: {episodes}")

    agent, env, config = create_quick_experiment(agent_type, env_type)
    tracker = PerformanceTracker()

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < config.max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Training step (simplified)
            if hasattr(agent, "train_step"):
                agent.train_step(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1

        # Track progress
        tracker.update_episode(episode_reward, episode_length, info or {})

        if verbose:
            print(
                f"Episode {episode + 1:2d}: Reward = {episode_reward:7.2f}, "
                f"Length = {episode_length:3d}"
            )

    # Final results
    stats = tracker.get_summary_stats()
    print("\n📊 Demo Results:")
    print(f"Average Reward: {stats.get('avg_reward', 0):.2f}")
    print(f"Best Reward: {stats.get('best_reward', 0):.2f}")
    print(f"Average Episode Length: {stats.get('avg_episode_length', 0):.1f}")

    return {
        "agent_type": agent_type,
        "env_type": env_type,
        "episodes": episodes,
        "stats": stats,
        "tracker": tracker,
    }


# Version and compatibility info
def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ["numpy", "torch", "matplotlib", "seaborn"]
    optional_packages = ["qiskit", "scipy"]

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)

    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)

    return {
        "required_missing": missing_required,
        "optional_missing": missing_optional,
        "all_available": len(missing_required) == 0,
    }


def print_package_info():
    """Print information about the CA19 modular package"""
    print("🎯 CA19 Modular: Advanced RL Systems Package")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print()

    # Check dependencies
    deps = check_dependencies()
    if deps["all_available"]:
        print("✅ All required dependencies available")
    else:
        print("⚠️  Missing required dependencies:", deps["required_missing"])

    if deps["optional_missing"]:
        print("ℹ️  Optional dependencies not available:", deps["optional_missing"])

    print()
    print("Available Modules:")
    for module in get_available_modules():
        info = get_module_info(module)
        print(f"  • {module}: {info.get('description', 'N/A')}")

    print()
    print("Quick Start:")
    print("  from ca19_modular import run_demo_experiment")
    print("  results = run_demo_experiment()")


# Make key classes available at package level for convenience
__all__ = [
    # Core agents
    "HybridQuantumClassicalAgent",
    "NeuromorphicActorCritic",
    "QuantumEnhancedAgent",
    # Environments
    "NeuromorphicEnvironment",
    "HybridQuantumClassicalEnvironment",
    "SpaceStationEnvironment",
    # Experiments
    "QuantumNeuromorphicComparison",
    "ExperimentRunner",
    # Utilities
    "MissionConfig",
    "PerformanceTracker",
    "ExperimentManager",
    # Helper functions
    "create_quick_experiment",
    "run_demo_experiment",
    "print_package_info",
    "get_available_modules",
    "get_module_info",
]
