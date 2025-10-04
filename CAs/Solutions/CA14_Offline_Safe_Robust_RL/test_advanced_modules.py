"""
Advanced Test Suite for CA14 Project
تست‌های پیشرفته برای پروژه CA14

This module tests all advanced components including:
- Advanced Algorithms
- Complex Environments
- Advanced Visualizations
- Advanced Concepts
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("🚀 CA14 Advanced Test Suite")
print("==================================================")

def test_advanced_algorithms():
    """Test advanced RL algorithms."""
    print("🧪 Testing Advanced RL Algorithms...")
    
    try:
        from advanced_algorithms import (
            HierarchicalRLAgent, MetaLearningAgent, CausalRLAgent,
            QuantumInspiredRLAgent, NeurosymbolicRLAgent, FederatedRLAgent
        )
        
        # Test Hierarchical RL
        hierarchical_agent = HierarchicalRLAgent(state_dim=4, action_dim=4, num_options=3)
        print("  ✅ Hierarchical RL Agent created")
        
        # Test Meta Learning
        meta_agent = MetaLearningAgent(state_dim=4, action_dim=4)
        print("  ✅ Meta Learning Agent created")
        
        # Test Causal RL
        causal_agent = CausalRLAgent(state_dim=4, action_dim=4)
        print("  ✅ Causal RL Agent created")
        
        # Test Quantum RL
        quantum_agent = QuantumInspiredRLAgent(state_dim=4, action_dim=4)
        print("  ✅ Quantum RL Agent created")
        
        # Test Neuro-Symbolic RL
        neurosymbolic_agent = NeurosymbolicRLAgent(state_dim=4, action_dim=4)
        print("  ✅ Neuro-Symbolic RL Agent created")
        
        # Test Federated RL
        federated_agent = FederatedRLAgent(state_dim=4, action_dim=4, num_clients=3)
        print("  ✅ Federated RL Agent created")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced algorithms test failed: {e}")
        return False

def test_complex_environments():
    """Test complex environments."""
    print("\n🌍 Testing Complex Environments...")
    
    try:
        from complex_environments import (
            DynamicMultiObjectiveEnvironment, PartiallyObservableEnvironment,
            ContinuousControlEnvironment, AdversarialEnvironment, EnvironmentConfig
        )
        
        config = EnvironmentConfig(size=6, num_agents=2, num_objectives=2)
        
        # Test Dynamic Multi-Objective Environment
        multi_obj_env = DynamicMultiObjectiveEnvironment(config)
        state = multi_obj_env.reset()
        action = np.random.randint(4)
        next_state, reward, done, info = multi_obj_env.step(action)
        print("  ✅ Dynamic Multi-Objective Environment tested")
        
        # Test Partially Observable Environment
        po_env = PartiallyObservableEnvironment(config)
        state = po_env.reset()
        action = np.random.randint(5)
        next_state, reward, done, info = po_env.step(action)
        print("  ✅ Partially Observable Environment tested")
        
        # Test Continuous Control Environment
        cc_env = ContinuousControlEnvironment(config)
        state = cc_env.reset()
        action = np.random.random(3) * 2 - 1  # Continuous action
        next_state, reward, done, info = cc_env.step(action)
        print("  ✅ Continuous Control Environment tested")
        
        # Test Adversarial Environment
        adv_env = AdversarialEnvironment(config)
        state = adv_env.reset()
        action = np.random.randint(4)
        next_state, reward, done, info = adv_env.step(action)
        print("  ✅ Adversarial Environment tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Complex environments test failed: {e}")
        return False

def test_advanced_visualizations():
    """Test advanced visualizations."""
    print("\n🎨 Testing Advanced Visualizations...")
    
    try:
        from advanced_visualizations import (
            Interactive3DVisualizer, RealTimePerformanceMonitor, MultiDimensionalAnalyzer,
            CausalGraphVisualizer, QuantumStateVisualizer, FederatedLearningDashboard,
            AdvancedMetricsAnalyzer, VisualizationConfig
        )
        
        config = VisualizationConfig(figure_size=(10, 8), dpi=150)
        
        # Test 3D Visualizer
        viz_3d = Interactive3DVisualizer(config)
        env_data = {
            'agent_positions': [(i, i, i*0.1) for i in range(10)],
            'target_positions': [(5, 5, 2)],
            'obstacle_positions': [(3, 3, 0)],
            'reward_history': np.random.random(10) * 5
        }
        fig_3d = viz_3d.create_3d_environment_plot(env_data)
        plt.close(fig_3d)
        print("  ✅ 3D Visualizer tested")
        
        # Test Performance Monitor
        monitor = RealTimePerformanceMonitor(config)
        monitor.metrics_history['rewards'] = deque(np.random.random(20) * 5)
        monitor.metrics_history['losses'] = deque(np.random.random(20) * 1)
        fig_dashboard = monitor.create_performance_dashboard()
        plt.close(fig_dashboard)
        print("  ✅ Performance Monitor tested")
        
        # Test Multi-dimensional Analyzer
        analyzer = MultiDimensionalAnalyzer(config)
        data = np.random.random((20, 4))
        labels = ['A', 'B', 'C', 'D']
        fig_parallel = analyzer.create_parallel_coordinates_plot(data, labels)
        plt.close(fig_parallel)
        print("  ✅ Multi-dimensional Analyzer tested")
        
        # Test Causal Graph Visualizer
        causal_viz = CausalGraphVisualizer(config)
        causal_graph = {'A': ['B'], 'B': ['C'], 'C': []}
        fig_causal = causal_viz.create_causal_graph(causal_graph)
        plt.close(fig_causal)
        print("  ✅ Causal Graph Visualizer tested")
        
        # Test Quantum State Visualizer
        quantum_viz = QuantumStateVisualizer(config)
        quantum_state = np.array([0.7 + 0.3j, 0.5 - 0.2j])
        fig_quantum = quantum_viz.create_bloch_sphere(quantum_state)
        print("  ✅ Quantum State Visualizer tested")
        
        # Test Federated Learning Dashboard
        federated_viz = FederatedLearningDashboard(config)
        federated_data = {
            'client_performance': {'Client 1': np.random.random(10) * 5},
            'global_loss': np.random.random(10) * 1,
            'communication_rounds': {'Round 1': 5},
            'data_distribution': {'Client 1': 50}
        }
        fig_federated = federated_viz.create_federated_dashboard(federated_data)
        print("  ✅ Federated Learning Dashboard tested")
        
        # Test Advanced Metrics Analyzer
        advanced_analyzer = AdvancedMetricsAnalyzer(config)
        all_results = {
            'Method A': {
                'sample_efficiency': 0.8, 'asymptotic_performance': 0.9,
                'robustness': 0.7, 'safety': 0.8, 'coordination': 0.6,
                'learning_curve': np.random.random(20) * 5,
                'computational_cost': {'training_time': 100, 'memory_usage': 500}
            }
        }
        fig_comprehensive = advanced_analyzer.create_comprehensive_analysis(all_results)
        plt.close(fig_comprehensive)
        print("  ✅ Advanced Metrics Analyzer tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced visualizations test failed: {e}")
        return False

def test_advanced_concepts():
    """Test advanced RL concepts."""
    print("\n🧠 Testing Advanced RL Concepts...")
    
    try:
        from advanced_concepts import (
            TransferLearningAgent, CurriculumLearningAgent, MultiTaskLearningAgent,
            ContinualLearningAgent, ExplainableRLAgent, AdaptiveMetaLearningAgent,
            AdvancedRLExperimentManager
        )
        
        # Test Transfer Learning Agent
        transfer_agent = TransferLearningAgent(source_state_dim=4, target_state_dim=6, action_dim=4)
        print("  ✅ Transfer Learning Agent created")
        
        # Test Curriculum Learning Agent
        curriculum_agent = CurriculumLearningAgent(state_dim=4, action_dim=4)
        print("  ✅ Curriculum Learning Agent created")
        
        # Test Multi-Task Learning Agent
        multi_task_agent = MultiTaskLearningAgent(state_dim=4, action_dim=4, num_tasks=2)
        print("  ✅ Multi-Task Learning Agent created")
        
        # Test Continual Learning Agent
        continual_agent = ContinualLearningAgent(state_dim=4, action_dim=4)
        print("  ✅ Continual Learning Agent created")
        
        # Test Explainable RL Agent
        explainable_agent = ExplainableRLAgent(state_dim=4, action_dim=4)
        print("  ✅ Explainable RL Agent created")
        
        # Test Adaptive Meta-Learning Agent
        adaptive_meta_agent = AdaptiveMetaLearningAgent(state_dim=4, action_dim=4)
        print("  ✅ Adaptive Meta-Learning Agent created")
        
        # Test Experiment Manager
        experiment_manager = AdvancedRLExperimentManager()
        experiment_manager.create_experiment("test_exp", {"test": True})
        print("  ✅ Experiment Manager created")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced concepts test failed: {e}")
        return False

def test_integration():
    """Test integration of all components."""
    print("\n🔗 Testing Integration...")
    
    try:
        # Test basic integration
        from advanced_algorithms import HierarchicalRLAgent
        from complex_environments import DynamicMultiObjectiveEnvironment, EnvironmentConfig
        from advanced_visualizations import VisualizationConfig, Interactive3DVisualizer
        
        # Create components
        config = EnvironmentConfig(size=4, num_agents=1, num_objectives=1)
        env = DynamicMultiObjectiveEnvironment(config)
        agent = HierarchicalRLAgent(state_dim=8, action_dim=4, num_options=2)
        viz_config = VisualizationConfig(figure_size=(8, 6), dpi=100)
        visualizer = Interactive3DVisualizer(viz_config)
        
        # Run simple interaction
        state = env.reset()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        
        print("  ✅ Basic integration test passed")
        
        # Test visualization integration
        env_data = {
            'agent_positions': [(i, i, i*0.1) for i in range(5)],
            'target_positions': [(2, 2, 1)],
            'obstacle_positions': [(1, 1, 0)],
            'reward_history': np.random.random(5) * 3
        }
        fig = visualizer.create_3d_environment_plot(env_data)
        plt.close(fig)
        
        print("  ✅ Visualization integration test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_performance():
    """Test performance of advanced components."""
    print("\n⚡ Testing Performance...")
    
    try:
        import time
        
        # Test algorithm performance
        from advanced_algorithms import HierarchicalRLAgent
        
        start_time = time.time()
        agent = HierarchicalRLAgent(state_dim=8, action_dim=4, num_options=3)
        
        # Test action selection speed
        state = np.random.random(8)
        action_start = time.time()
        for _ in range(100):
            action = agent.get_action(state)
        action_time = time.time() - action_start
        
        print(f"  ✅ Agent creation: {time.time() - start_time:.3f}s")
        print(f"  ✅ Action selection (100x): {action_time:.3f}s")
        
        # Test environment performance
        from complex_environments import DynamicMultiObjectiveEnvironment, EnvironmentConfig
        
        config = EnvironmentConfig(size=6, num_agents=2, num_objectives=2)
        env = DynamicMultiObjectiveEnvironment(config)
        
        env_start = time.time()
        for _ in range(50):
            state = env.reset()
            for _ in range(10):
                action = np.random.randint(4)
                next_state, reward, done, info = env.step(action)
                if done:
                    break
        env_time = time.time() - env_start
        
        print(f"  ✅ Environment simulation (50 episodes): {env_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage of advanced components."""
    print("\n💾 Testing Memory Usage...")
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple agents
        from advanced_algorithms import (
            HierarchicalRLAgent, MetaLearningAgent, CausalRLAgent,
            QuantumInspiredRLAgent, NeurosymbolicRLAgent, FederatedRLAgent
        )
        
        agents = []
        for i in range(5):
            agents.append(HierarchicalRLAgent(state_dim=8, action_dim=4, num_options=3))
            agents.append(MetaLearningAgent(state_dim=8, action_dim=4))
            agents.append(CausalRLAgent(state_dim=8, action_dim=4))
            agents.append(QuantumInspiredRLAgent(state_dim=8, action_dim=4))
            agents.append(NeurosymbolicRLAgent(state_dim=8, action_dim=4))
            agents.append(FederatedRLAgent(state_dim=8, action_dim=4, num_clients=3))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"  ✅ Initial memory: {initial_memory:.1f} MB")
        print(f"  ✅ Final memory: {final_memory:.1f} MB")
        print(f"  ✅ Memory increase: {memory_increase:.1f} MB")
        print(f"  ✅ Agents created: {len(agents)}")
        
        if memory_increase < 1000:  # Less than 1GB
            print("  ✅ Memory usage is reasonable")
            return True
        else:
            print("  ⚠️ Memory usage is high")
            return False
        
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("\n🎯 Running Comprehensive Test Suite...")
    
    test_results = []
    
    # Run all tests
    test_results.append(("Advanced Algorithms", test_advanced_algorithms()))
    test_results.append(("Complex Environments", test_complex_environments()))
    test_results.append(("Advanced Visualizations", test_advanced_visualizations()))
    test_results.append(("Advanced Concepts", test_advanced_concepts()))
    test_results.append(("Integration", test_integration()))
    test_results.append(("Performance", test_performance()))
    test_results.append(("Memory Usage", test_memory_usage()))
    
    # Print results
    print("\n==================================================")
    print("📊 Test Results Summary:")
    print("==================================================")
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n📈 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! CA14 Advanced Project is ready!")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
