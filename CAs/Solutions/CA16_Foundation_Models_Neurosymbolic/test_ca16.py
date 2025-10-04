"""
CA16 Comprehensive Test Script

This script tests all CA16 components and generates comprehensive results.
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
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import CA16 modules
try:
    from foundation_models import DecisionTransformer, MultiTaskDecisionTransformer
    from neurosymbolic import SymbolicKnowledgeBase, NeurosymbolicPolicy
    from continual_learning import ContinualLearningAgent, MAML
    from human_ai_collaboration import CollaborativeAgent, PreferenceRewardModel
    from environments import SymbolicGridWorld, CollaborativeGridWorld, ContinualLearningEnvironment
    from advanced_computing import QuantumInspiredRL, NeuromorphicNetwork
    from deployment_ethics import ProductionRLSystem, SafetyMonitor, EthicsChecker
    from visualizations import create_attention_heatmap, plot_training_dynamics, create_performance_comparison
    print("All CA16 modules imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available. Continuing with available modules...")

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def test_foundation_models():
    """Test foundation models."""
    print("\n" + "="*50)
    print("Testing Foundation Models")
    print("="*50)
    
    results = {}
    
    try:
        # Test Decision Transformer
        print("Testing Decision Transformer...")
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
        
        action_logits, values = dt(states, actions, returns_to_go, timesteps)
        
        print(f"✓ Decision Transformer output shapes: {action_logits.shape}, {values.shape}")
        
        # Test action prediction
        action = dt.get_action(states[0], actions[0], returns_to_go[0], timesteps[0])
        print(f"✓ Predicted action: {action}")
        
        results["decision_transformer"] = {
            "model_parameters": sum(p.numel() for p in dt.parameters()),
            "output_shapes": {"action_logits": list(action_logits.shape), "values": list(values.shape)},
            "predicted_action": action.item() if isinstance(action, torch.Tensor) else action,
        }
        
    except Exception as e:
        print(f"✗ Decision Transformer test failed: {e}")
        results["decision_transformer"] = {"error": str(e)}
    
    try:
        # Test Multi-Task Decision Transformer
        print("Testing Multi-Task Decision Transformer...")
        mtdt = MultiTaskDecisionTransformer(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            max_length=100,
            num_tasks=3,
        )
        
        # Test forward pass
        task_id = 1
        action_logits, values = mtdt(states, actions, returns_to_go, timesteps, task_id)
        
        print(f"✓ Multi-Task Decision Transformer output shapes: {action_logits.shape}, {values.shape}")
        
        results["multi_task_decision_transformer"] = {
            "model_parameters": sum(p.numel() for p in mtdt.parameters()),
            "output_shapes": {"action_logits": list(action_logits.shape), "values": list(values.shape)},
        }
        
    except Exception as e:
        print(f"✗ Multi-Task Decision Transformer test failed: {e}")
        results["multi_task_decision_transformer"] = {"error": str(e)}
    
    return results


def test_neurosymbolic_rl():
    """Test neurosymbolic RL."""
    print("\n" + "="*50)
    print("Testing Neurosymbolic RL")
    print("="*50)
    
    results = {}
    
    try:
        # Test Symbolic Knowledge Base
        print("Testing Symbolic Knowledge Base...")
        kb = SymbolicKnowledgeBase()
        
        # Add predicates
        kb.add_predicate("at", 2)  # at(agent, location)
        kb.add_predicate("goal", 1)  # goal(location)
        kb.add_predicate("obstacle", 1)  # obstacle(location)
        
        # Add rules
        kb.add_rule("reachable(X, Y)", ["at(agent, X)", "not obstacle(Y)"], 0.8)
        kb.add_rule("safe_move(X, Y)", ["at(agent, X)", "not obstacle(Y)", "reachable(X, Y)"], 0.9)
        
        # Test inference
        facts = ["at(agent, (0,0))", "goal((5,5))", "obstacle((2,2))"]
        kb.add_facts(facts)
        
        # Forward chaining
        inferred = kb.forward_chaining()
        print(f"✓ Inferred facts: {len(inferred)}")
        
        results["symbolic_knowledge_base"] = {
            "predicates_count": len(kb.predicates),
            "rules_count": len(kb.rules),
            "inferred_facts_count": len(inferred),
        }
        
    except Exception as e:
        print(f"✗ Symbolic Knowledge Base test failed: {e}")
        results["symbolic_knowledge_base"] = {"error": str(e)}
    
    try:
        # Test Neurosymbolic Policy
        print("Testing Neurosymbolic Policy...")
        policy = NeurosymbolicPolicy(
            state_dim=4,
            action_dim=4,
            hidden_dim=32,
            symbolic_dim=8,
        )
        
        # Test policy
        state = torch.randn(1, 4)
        action_logits, value = policy(state)
        
        print(f"✓ Neurosymbolic policy output shapes: {action_logits.shape}, {value.shape}")
        
        results["neurosymbolic_policy"] = {
            "model_parameters": sum(p.numel() for p in policy.parameters()),
            "output_shapes": {"action_logits": list(action_logits.shape), "value": list(value.shape)},
        }
        
    except Exception as e:
        print(f"✗ Neurosymbolic Policy test failed: {e}")
        results["neurosymbolic_policy"] = {"error": str(e)}
    
    return results


def test_continual_learning():
    """Test continual learning."""
    print("\n" + "="*50)
    print("Testing Continual Learning")
    print("="*50)
    
    results = {}
    
    try:
        # Test Continual Learning Agent
        print("Testing Continual Learning Agent...")
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
        
        print(f"✓ Task performances computed for {len(task_performances)} tasks")
        print(f"✓ Forgetting metrics: {forgetting_metrics}")
        
        results["continual_learning_agent"] = {
            "task_performances": task_performances,
            "forgetting_metrics": forgetting_metrics,
            "model_parameters": sum(p.numel() for p in agent.parameters()),
        }
        
    except Exception as e:
        print(f"✗ Continual Learning Agent test failed: {e}")
        results["continual_learning_agent"] = {"error": str(e)}
    
    try:
        # Test MAML
        print("Testing MAML...")
        maml = MAML(
            model=nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2)),
            inner_lr=0.01,
            meta_lr=0.001,
            adaptation_steps=5,
        )
        
        # Generate mock meta-tasks
        meta_tasks = []
        for _ in range(3):
            task = {
                "states": torch.randn(50, 4),
                "actions": torch.randint(0, 2, (50,)),
                "rewards": torch.randn(50),
            }
            meta_tasks.append(task)
        
        # Test meta-update
        meta_loss_info = maml.meta_update(meta_tasks, meta_tasks)
        
        print(f"✓ MAML meta-update completed")
        print(f"✓ Meta loss: {meta_loss_info['meta_loss']:.4f}")
        
        results["maml"] = {
            "meta_loss": meta_loss_info["meta_loss"],
            "avg_adaptation_loss": meta_loss_info["avg_adaptation_loss"],
            "model_parameters": sum(p.numel() for p in maml.model.parameters()),
        }
        
    except Exception as e:
        print(f"✗ MAML test failed: {e}")
        results["maml"] = {"error": str(e)}
    
    return results


def test_human_ai_collaboration():
    """Test human-AI collaboration."""
    print("\n" + "="*50)
    print("Testing Human-AI Collaboration")
    print("="*50)
    
    results = {}
    
    try:
        # Test Collaborative Agent
        print("Testing Collaborative Agent...")
        agent = CollaborativeAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=32,
            confidence_threshold=0.7,
        )
        
        # Test collaboration
        state = torch.randn(1, 4)
        action, confidence = agent.get_action(state)
        
        print(f"✓ Agent action: {action}, confidence: {confidence}")
        
        results["collaborative_agent"] = {
            "agent_confidence": confidence,
            "model_parameters": sum(p.numel() for p in agent.parameters()),
        }
        
    except Exception as e:
        print(f"✗ Collaborative Agent test failed: {e}")
        results["collaborative_agent"] = {"error": str(e)}
    
    try:
        # Test Preference Reward Model
        print("Testing Preference Reward Model...")
        preference_model = PreferenceRewardModel(
            state_dim=4,
            action_dim=2,
            hidden_dim=32,
        )
        
        # Test preference learning
        state1 = torch.randn(1, 4)
        state2 = torch.randn(1, 4)
        action1 = torch.tensor([0])
        action2 = torch.tensor([1])
        preference = torch.tensor([1.0])  # Prefer action1
        
        preference_loss = preference_model.compute_loss(state1, action1, state2, action2, preference)
        
        print(f"✓ Preference loss: {preference_loss.item():.4f}")
        
        results["preference_reward_model"] = {
            "preference_loss": preference_loss.item(),
            "model_parameters": sum(p.numel() for p in preference_model.parameters()),
        }
        
    except Exception as e:
        print(f"✗ Preference Reward Model test failed: {e}")
        results["preference_reward_model"] = {"error": str(e)}
    
    return results


def test_environments():
    """Test custom environments."""
    print("\n" + "="*50)
    print("Testing Custom Environments")
    print("="*50)
    
    results = {}
    
    try:
        # Test Symbolic Grid World
        print("Testing Symbolic Grid World...")
        symbolic_env = SymbolicGridWorld(size=5, num_goals=2, num_obstacles=3)
        
        obs, info = symbolic_env.reset()
        print(f"✓ Symbolic environment observation shape: {obs.shape}")
        
        # Test step
        action = 0
        obs, reward, done, truncated, info = symbolic_env.step(action)
        print(f"✓ Step completed: reward={reward}, done={done}")
        
        results["symbolic_grid_world"] = {
            "observation_shape": list(obs.shape),
            "reward": reward,
            "done": done,
        }
        
    except Exception as e:
        print(f"✗ Symbolic Grid World test failed: {e}")
        results["symbolic_grid_world"] = {"error": str(e)}
    
    try:
        # Test Collaborative Grid World
        print("Testing Collaborative Grid World...")
        collaborative_env = CollaborativeGridWorld(size=5, num_goals=2, num_obstacles=3)
        
        obs, info = collaborative_env.reset()
        print(f"✓ Collaborative environment observation shape: {obs.shape}")
        
        # Test step
        action = 0
        obs, reward, done, truncated, info = collaborative_env.step(action)
        print(f"✓ Step completed: reward={reward}, done={done}")
        
        results["collaborative_grid_world"] = {
            "observation_shape": list(obs.shape),
            "reward": reward,
            "done": done,
        }
        
    except Exception as e:
        print(f"✗ Collaborative Grid World test failed: {e}")
        results["collaborative_grid_world"] = {"error": str(e)}
    
    try:
        # Test Continual Learning Environment
        print("Testing Continual Learning Environment...")
        continual_env = ContinualLearningEnvironment(num_tasks=3, state_dim=4, action_dim=2)
        
        obs, info = continual_env.reset()
        print(f"✓ Continual environment observation shape: {obs.shape}")
        
        # Test step
        action = 0
        obs, reward, done, truncated, info = continual_env.step(action)
        print(f"✓ Step completed: reward={reward}, done={done}")
        
        results["continual_learning_environment"] = {
            "observation_shape": list(obs.shape),
            "reward": reward,
            "done": done,
        }
        
    except Exception as e:
        print(f"✗ Continual Learning Environment test failed: {e}")
        results["continual_learning_environment"] = {"error": str(e)}
    
    return results


def test_advanced_computing():
    """Test advanced computing paradigms."""
    print("\n" + "="*50)
    print("Testing Advanced Computing Paradigms")
    print("="*50)
    
    results = {}
    
    try:
        # Test Quantum-Inspired RL
        print("Testing Quantum-Inspired RL...")
        quantum_rl = QuantumInspiredRL(
            state_dim=4,
            action_dim=2,
            num_qubits=4,
            num_layers=2,
        )
        
        state = torch.randn(1, 4)
        action_logits = quantum_rl(state)
        
        print(f"✓ Quantum RL output shape: {action_logits.shape}")
        
        results["quantum_inspired_rl"] = {
            "model_parameters": sum(p.numel() for p in quantum_rl.parameters()),
            "output_shape": list(action_logits.shape),
        }
        
    except Exception as e:
        print(f"✗ Quantum-Inspired RL test failed: {e}")
        results["quantum_inspired_rl"] = {"error": str(e)}
    
    try:
        # Test Neuromorphic Network
        print("Testing Neuromorphic Network...")
        neuromorphic_net = NeuromorphicNetwork(
            input_dim=4,
            hidden_dim=16,
            output_dim=2,
            num_layers=2,
            time_steps=5,
        )
        
        input_spikes = torch.randn(1, 4)
        output_rates = neuromorphic_net(input_spikes)
        
        print(f"✓ Neuromorphic network output shape: {output_rates.shape}")
        
        results["neuromorphic_network"] = {
            "model_parameters": sum(p.numel() for p in neuromorphic_net.parameters()),
            "output_shape": list(output_rates.shape),
        }
        
    except Exception as e:
        print(f"✗ Neuromorphic Network test failed: {e}")
        results["neuromorphic_network"] = {"error": str(e)}
    
    return results


def test_deployment_ethics():
    """Test deployment and ethics components."""
    print("\n" + "="*50)
    print("Testing Deployment and Ethics")
    print("="*50)
    
    results = {}
    
    try:
        # Test Production RL System
        print("Testing Production RL System...")
        model = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        
        production_system = ProductionRLSystem(model)
        
        # Deploy system
        config = {
            "model_path": "test_model.pth",
            "environment_config": {"state_dim": 4, "action_dim": 2},
            "safety_config": {"threshold": 0.8},
        }
        
        deployed = production_system.deploy(config)
        print(f"✓ System deployed: {deployed}")
        
        # Test inference
        if deployed:
            state = torch.randn(1, 4)
            action, info = production_system.inference(state)
            print(f"✓ Inference completed: {action.shape}, info: {info}")
        
        results["production_rl_system"] = {
            "system_deployed": deployed,
            "system_status": production_system.get_system_status(),
        }
        
    except Exception as e:
        print(f"✗ Production RL System test failed: {e}")
        results["production_rl_system"] = {"error": str(e)}
    
    try:
        # Test Safety Monitor
        print("Testing Safety Monitor...")
        safety_monitor = SafetyMonitor(
            safety_thresholds={"inference_time": 0.1, "memory_usage": 0.8},
        )
        
        safety_monitor.start_monitoring()
        safety_monitor.update_metrics({"inference_time": 0.05, "memory_usage": 0.6})
        
        safety_report = safety_monitor.get_safety_report()
        print(f"✓ Safety report generated: {safety_report}")
        
        results["safety_monitor"] = {
            "safety_violations": safety_report["total_violations"],
            "monitoring_status": safety_report["is_monitoring"],
        }
        
    except Exception as e:
        print(f"✗ Safety Monitor test failed: {e}")
        results["safety_monitor"] = {"error": str(e)}
    
    try:
        # Test Ethics Checker
        print("Testing Ethics Checker...")
        ethics_checker = EthicsChecker(
            ethical_guidelines={"bias_threshold": 0.1, "fairness_threshold": 0.8},
        )
        
        predictions = torch.randn(100, 2)
        protected_attributes = torch.randint(0, 2, (100,))
        
        bias_result = ethics_checker.check_bias(predictions, protected_attributes)
        print(f"✓ Bias check completed: {bias_result}")
        
        results["ethics_checker"] = {
            "bias_score": bias_result["bias_score"],
            "biased": bias_result["biased"],
        }
        
    except Exception as e:
        print(f"✗ Ethics Checker test failed: {e}")
        results["ethics_checker"] = {"error": str(e)}
    
    return results


def test_visualizations():
    """Test visualization functions."""
    print("\n" + "="*50)
    print("Testing Visualizations")
    print("="*50)
    
    results = {}
    
    try:
        # Test attention heatmap
        print("Testing attention heatmap...")
        attention_weights = torch.randn(4, 8)  # 4 heads, 8 sequence length
        fig = create_attention_heatmap(attention_weights, "Test Attention")
        print("✓ Attention heatmap created")
        
        results["attention_heatmap"] = {"created": True}
        
    except Exception as e:
        print(f"✗ Attention heatmap test failed: {e}")
        results["attention_heatmap"] = {"error": str(e)}
    
    try:
        # Test training dynamics plot
        print("Testing training dynamics plot...")
        training_data = {
            "losses": [1.0, 0.8, 0.6, 0.4, 0.2],
            "rewards": [0.1, 0.3, 0.5, 0.7, 0.9],
            "episodes": list(range(5)),
        }
        fig = plot_training_dynamics(training_data, "Test Training")
        print("✓ Training dynamics plot created")
        
        results["training_dynamics"] = {"created": True}
        
    except Exception as e:
        print(f"✗ Training dynamics plot test failed: {e}")
        results["training_dynamics"] = {"error": str(e)}
    
    try:
        # Test performance comparison
        print("Testing performance comparison...")
        comparison_data = {
            "Method A": 0.8,
            "Method B": 0.7,
            "Method C": 0.9,
        }
        fig = create_performance_comparison(comparison_data, "Test Comparison")
        print("✓ Performance comparison plot created")
        
        results["performance_comparison"] = {"created": True}
        
    except Exception as e:
        print(f"✗ Performance comparison plot test failed: {e}")
        results["performance_comparison"] = {"error": str(e)}
    
    return results


def run_comprehensive_test():
    """Run comprehensive test of all CA16 components."""
    print("CA16 Comprehensive Test")
    print("=" * 60)
    print("Testing all CA16 components...")
    
    # Create results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Run all tests
    all_results = {}
    
    # Test each module
    test_functions = [
        ("foundation_models", test_foundation_models),
        ("neurosymbolic_rl", test_neurosymbolic_rl),
        ("continual_learning", test_continual_learning),
        ("human_ai_collaboration", test_human_ai_collaboration),
        ("environments", test_environments),
        ("advanced_computing", test_advanced_computing),
        ("deployment_ethics", test_deployment_ethics),
        ("visualizations", test_visualizations),
    ]
    
    for module_name, test_func in test_functions:
        print(f"\nTesting {module_name}...")
        try:
            results = test_func()
            all_results[module_name] = results
            print(f"✓ {module_name} tests completed")
        except Exception as e:
            print(f"✗ {module_name} tests failed: {e}")
            all_results[module_name] = {"error": str(e)}
    
    # Generate summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    successful_tests = 0
    failed_tests = 0
    
    for module_name, results in all_results.items():
        print(f"\n{module_name.upper()}:")
        if "error" in results:
            print(f"  ✗ Module failed: {results['error']}")
            failed_tests += 1
        else:
            print(f"  ✓ Module tests passed")
            successful_tests += 1
        
        # Count individual tests
        for test_name, test_result in results.items():
            total_tests += 1
            if "error" in test_result:
                print(f"    ✗ {test_name}: {test_result['error']}")
            else:
                print(f"    ✓ {test_name}: passed")
    
    print(f"\nOverall Results:")
    print(f"  Total modules: {len(test_functions)}")
    print(f"  Successful modules: {successful_tests}")
    print(f"  Failed modules: {failed_tests}")
    print(f"  Total individual tests: {total_tests}")
    
    # Save results
    results_file = os.path.join("test_results", "comprehensive_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    # Run comprehensive test
    results = run_comprehensive_test()
    
    print("\n" + "=" * 60)
    print("CA16 Comprehensive Test Complete!")
    print("=" * 60)
    print("Check the 'test_results' directory for detailed results.")
    print("All components have been tested and results saved.")
