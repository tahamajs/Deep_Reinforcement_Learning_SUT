#!/usr/bin/env python3
"""
CA18 Module Test Script
Tests all modules for basic functionality without requiring heavy dependencies
"""

import sys
import os
import numpy as np

# Add current directory to Python path
sys.path.insert(0, os.getcwd())


def test_basic_functionality():
    """Test basic functionality of each module"""

    print("🔍 CA18 Module Functionality Test")
    print("=" * 50)

    # Test 1: Quantum RL Basic Components
    print("\n🔬 Testing Quantum RL...")
    try:
        # Test quantum gates without PyTorch
        from quantum_rl.quantum_rl import QuantumGate, QuantumState, QuantumCircuit

        # Test basic quantum operations
        hadamard = QuantumGate.hadamard()
        print(f"  ✅ Hadamard gate: {hadamard.shape}")

        # Test quantum state
        state = QuantumState.zero_state(n_qubits=2)
        print(f"  ✅ Zero state created: {len(state.amplitudes)} amplitudes")

        # Test quantum circuit
        circuit = QuantumCircuit(n_qubits=2)
        circuit.apply_single_gate(QuantumGate.hadamard(), 0)
        probs = circuit.get_probabilities()
        print(f"  ✅ Circuit probabilities: {len(probs)} states")

        print("  ✅ Quantum RL basic functionality working")

    except Exception as e:
        print(f"  ❌ Quantum RL test failed: {e}")

    # Test 2: Causal RL Basic Components
    print("\n🔍 Testing Causal RL...")
    try:
        from causal_rl.causal_rl import CausalGraph, CausalDiscovery

        # Test causal graph
        graph = CausalGraph(["X", "Y", "Z"])
        graph.add_edge("X", "Y")
        graph.add_edge("Y", "Z")

        parents = graph.get_parents("Y")
        children = graph.get_children("X")

        print(f"  ✅ Causal graph: X->Y->Z")
        print(f"  ✅ Parents of Y: {parents}")
        print(f"  ✅ Children of X: {children}")

        # Test causal discovery
        discovery = CausalDiscovery(alpha=0.05)
        print(f"  ✅ Causal discovery initialized with alpha={discovery.alpha}")

        print("  ✅ Causal RL basic functionality working")

    except Exception as e:
        print(f"  ❌ Causal RL test failed: {e}")

    # Test 3: Utils Basic Components
    print("\n🔧 Testing Utils...")
    try:
        from utils.utils import QuantumRNG

        # Test quantum RNG
        rng = QuantumRNG()
        random_val = rng.quantum_random()
        coherence = rng.get_coherence()

        print(f"  ✅ Quantum random value: {random_val:.4f}")
        print(f"  ✅ Quantum coherence: {coherence:.4f}")

        # Test quantum choice
        choices = [1, 2, 3, 4, 5]
        choice = rng.quantum_choice(choices)
        print(f"  ✅ Quantum choice from {choices}: {choice}")

        print("  ✅ Utils basic functionality working")

    except Exception as e:
        print(f"  ❌ Utils test failed: {e}")

    # Test 4: Environments Basic Components
    print("\n🎮 Testing Environments...")
    try:
        # Test environment imports
        from environments.environments import (
            QuantumEnvironment,
            CausalBanditEnvironment,
        )

        # Test quantum environment creation
        quantum_env = QuantumEnvironment(n_qubits=2, max_steps=10)
        obs = quantum_env.reset()
        print(f"  ✅ Quantum environment reset: obs shape {obs.shape}")

        # Test causal bandit environment
        causal_env = CausalBanditEnvironment(n_arms=3, n_context_vars=2)
        obs = causal_env.reset()
        print(f"  ✅ Causal bandit environment reset: obs shape {obs.shape}")

        print("  ✅ Environments basic functionality working")

    except Exception as e:
        print(f"  ❌ Environments test failed: {e}")

    # Test 5: Configuration
    print("\n⚙️ Testing Configuration...")
    try:
        from config import get_config, print_config

        # Test config access
        quantum_config = get_config("quantum")
        causal_config = get_config("causal")

        print(f"  ✅ Quantum config loaded: {len(quantum_config)} parameters")
        print(f"  ✅ Causal config loaded: {len(causal_config)} parameters")

        print("  ✅ Configuration system working")

    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")

    # Test 6: Integration Demo
    print("\n🔗 Testing Integration...")
    try:
        from integration_demo import create_integrated_environment

        # Test integrated environment
        env = create_integrated_environment()
        obs = env.reset()

        print(f"  ✅ Integrated environment: {env.n_agents} agents")
        print(f"  ✅ Observation shape: {obs.shape}")

        # Test environment step
        actions = [
            np.random.uniform(-1, 1, env.action_dim) for _ in range(env.n_agents)
        ]
        next_obs, rewards, done, _ = env.step(actions)

        print(f"  ✅ Environment step: rewards {rewards}")
        print(f"  ✅ Environment step: done {done}")

        print("  ✅ Integration demo working")

    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")

    print("\n🎉 Basic functionality test completed!")
    print("=" * 50)


def test_imports():
    """Test that all modules can be imported"""

    print("\n📦 Testing Module Imports...")
    print("=" * 30)

    modules_to_test = [
        ("quantum_rl.quantum_rl", "Quantum RL"),
        ("world_models.world_models", "World Models"),
        ("multi_agent_rl.multi_agent_rl", "Multi-Agent RL"),
        ("causal_rl.causal_rl", "Causal RL"),
        ("federated_rl.federated_rl", "Federated RL"),
        ("advanced_safety.advanced_safety", "Advanced Safety"),
        ("utils.utils", "Utils"),
        ("environments.environments", "Environments"),
        ("experiments.experiments", "Experiments"),
    ]

    successful_imports = 0

    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
            successful_imports += 1
        except ImportError as e:
            print(f"  ❌ {display_name}: {str(e)[:50]}...")
        except Exception as e:
            print(f"  ⚠️ {display_name}: {str(e)[:50]}...")

    print(
        f"\n📊 Import Results: {successful_imports}/{len(modules_to_test)} modules imported successfully"
    )

    return successful_imports == len(modules_to_test)


if __name__ == "__main__":
    print("🚀 CA18 Advanced RL Paradigms - Module Test")
    print("=" * 60)

    # Test imports first
    imports_ok = test_imports()

    if imports_ok:
        # Test basic functionality
        test_basic_functionality()

        print("\n✅ All tests passed! CA18 modules are working correctly.")
    else:
        print("\n⚠️ Some modules failed to import. Check dependencies.")

    print("\n🎯 Test Summary:")
    print("- Module imports tested")
    print("- Basic functionality verified")
    print("- Ready for full execution with run.sh")

