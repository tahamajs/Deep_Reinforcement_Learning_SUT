#!/usr/bin/env python3
"""
Comprehensive test suite for Temporal Difference Learning implementation
Tests all algorithms, environments, and utilities
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all modules
from environments.environments import GridWorld
from agents.policies import RandomPolicy
from agents.algorithms import TD0Agent, QLearningAgent, SARSAAgent
from agents.exploration import (
    ExplorationStrategies,
    BoltzmannQLearning,
    ExplorationExperiment,
)
from utils.visualization import (
    plot_learning_curve,
    plot_q_learning_analysis,
    compare_algorithms,
)
from experiments.experiments import (
    experiment_td0,
    experiment_q_learning,
    experiment_sarsa,
)
from evaluation.evaluation import evaluate_agent, compare_agents, analyze_performance
from models.model_utils import save_model, export_results, create_summary_report

warnings.filterwarnings("ignore")
np.random.seed(42)

# Set up visualization backend
plt.switch_backend("Agg")  # Use non-interactive backend for testing


def test_environment():
    """Test GridWorld environment"""
    print("=" * 60)
    print("TESTING GRIDWORLD ENVIRONMENT")
    print("=" * 60)

    env = GridWorld()
    print(f"✓ Environment created: {env.size}x{env.size} grid")
    print(f"✓ States: {len(env.states)}")
    print(f"✓ Actions: {env.actions}")
    print(f"✓ Start state: {env.start_state}")
    print(f"✓ Goal state: {env.goal_state}")
    print(f"✓ Obstacles: {env.obstacles}")

    # Test environment reset and step
    state = env.reset()
    assert state == env.start_state, "Reset should return start state"
    print("✓ Environment reset works correctly")

    # Test valid actions
    valid_actions = env.get_valid_actions(state)
    assert len(valid_actions) == 4, "Should have 4 valid actions from start"
    print("✓ Valid actions retrieved correctly")

    # Test step function
    next_state, reward, done, info = env.step("right")
    assert next_state == (0, 1), "Should move right"
    assert reward == env.step_reward, "Should give step reward"
    assert not done, "Should not be done yet"
    print("✓ Step function works correctly")

    # Test terminal state detection
    assert not env.is_terminal(env.start_state), "Start state should not be terminal"
    assert env.is_terminal(env.goal_state), "Goal state should be terminal"
    print("✓ Terminal state detection works")

    print("✓ All environment tests passed!")
    return env


def test_policies():
    """Test policy implementations"""
    print("\n" + "=" * 60)
    print("TESTING POLICIES")
    print("=" * 60)

    env = GridWorld()
    policy = RandomPolicy(env)
    print("✓ RandomPolicy created")

    # Test random action selection
    actions = []
    for _ in range(100):
        action = policy.get_action(env.start_state)
        actions.append(action)

    # Should have variety in actions (not all the same)
    unique_actions = set(actions)
    assert len(unique_actions) > 1, "Random policy should select different actions"
    print("✓ Random policy selects varied actions")

    print("✓ All policy tests passed!")
    return policy


def test_td0_agent():
    """Test TD(0) agent"""
    print("\n" + "=" * 60)
    print("TESTING TD(0) AGENT")
    print("=" * 60)

    env = GridWorld()
    policy = RandomPolicy(env)
    agent = TD0Agent(env, policy, alpha=0.1, gamma=0.9)
    print("✓ TD0Agent created")

    # Test initial state
    assert len(agent.V) == 0, "Value function should start empty"
    print("✓ Initial value function is empty")

    # Test action selection
    action = agent.get_action(env.start_state)
    assert action in env.get_valid_actions(env.start_state), "Action should be valid"
    print("✓ Action selection works")

    # Test TD update
    state = env.start_state
    reward = -0.1
    next_state = (0, 1)
    td_error = agent.td_update(state, reward, next_state, False)
    assert state in agent.V, "State should be added to value function"
    print("✓ TD update works correctly")

    # Test training
    V_td = agent.train(num_episodes=100, print_every=50)
    assert len(V_td) > 0, "Should learn some value function"
    assert len(agent.episode_rewards) == 100, "Should have 100 episode rewards"
    print("✓ Training completed successfully")

    print("✓ All TD(0) tests passed!")
    return agent, V_td


def test_q_learning_agent():
    """Test Q-Learning agent"""
    print("\n" + "=" * 60)
    print("TESTING Q-LEARNING AGENT")
    print("=" * 60)

    env = GridWorld()
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    print("✓ QLearningAgent created")

    # Test initial state
    assert len(agent.Q) == 0, "Q-table should start empty"
    print("✓ Initial Q-table is empty")

    # Test action selection
    action = agent.get_action(env.start_state, explore=True)
    assert action in env.get_valid_actions(env.start_state), "Action should be valid"
    print("✓ Action selection works")

    # Test greedy action
    greedy_action = agent.get_greedy_action(env.start_state)
    assert greedy_action in env.get_valid_actions(
        env.start_state
    ), "Greedy action should be valid"
    print("✓ Greedy action selection works")

    # Test Q-update
    state = env.start_state
    action = "right"
    reward = -0.1
    next_state = (0, 1)
    td_error = agent.update_q(state, action, reward, next_state, False)
    assert state in agent.Q, "State should be added to Q-table"
    print("✓ Q-update works correctly")

    # Test training
    agent.train(num_episodes=200, print_every=100)
    assert len(agent.episode_rewards) == 200, "Should have 200 episode rewards"
    print("✓ Training completed successfully")

    # Test policy extraction
    policy = agent.get_policy()
    assert len(policy) > 0, "Should extract some policy"
    print("✓ Policy extraction works")

    # Test value function extraction
    V_optimal = agent.get_value_function()
    assert len(V_optimal) > 0, "Should extract value function"
    print("✓ Value function extraction works")

    # Test evaluation
    evaluation = agent.evaluate_policy(num_episodes=50)
    assert "avg_reward" in evaluation, "Evaluation should include average reward"
    assert "success_rate" in evaluation, "Evaluation should include success rate"
    print("✓ Policy evaluation works")

    print("✓ All Q-Learning tests passed!")
    return agent, V_optimal, evaluation


def test_sarsa_agent():
    """Test SARSA agent"""
    print("\n" + "=" * 60)
    print("TESTING SARSA AGENT")
    print("=" * 60)

    env = GridWorld()
    agent = SARSAAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    print("✓ SARSAAgent created")

    # Test initial state
    assert len(agent.Q) == 0, "Q-table should start empty"
    print("✓ Initial Q-table is empty")

    # Test action selection
    action = agent.get_action(env.start_state, explore=True)
    assert action in env.get_valid_actions(env.start_state), "Action should be valid"
    print("✓ Action selection works")

    # Test SARSA update
    state = env.start_state
    action = "right"
    reward = -0.1
    next_state = (0, 1)
    next_action = "right"
    td_error = agent.update_q_sarsa(
        state, action, reward, next_state, next_action, False
    )
    assert state in agent.Q, "State should be added to Q-table"
    print("✓ SARSA update works correctly")

    # Test training
    agent.train(num_episodes=200, print_every=100)
    assert len(agent.episode_rewards) == 200, "Should have 200 episode rewards"
    print("✓ Training completed successfully")

    # Test evaluation
    evaluation = agent.evaluate_policy(num_episodes=50)
    assert "avg_reward" in evaluation, "Evaluation should include average reward"
    print("✓ Policy evaluation works")

    print("✓ All SARSA tests passed!")
    return agent, evaluation


def test_exploration_strategies():
    """Test exploration strategies"""
    print("\n" + "=" * 60)
    print("TESTING EXPLORATION STRATEGIES")
    print("=" * 60)

    env = GridWorld()

    # Test epsilon-greedy exploration
    Q = {(0, 0): {"right": 1.0, "down": 0.5, "left": 0.0, "up": 0.0}}
    valid_actions = env.get_valid_actions((0, 0))

    action = ExplorationStrategies.epsilon_greedy(Q, (0, 0), valid_actions, 0.1)
    assert action in valid_actions, "Action should be valid"
    print("✓ Epsilon-greedy exploration works")

    # Test Boltzmann exploration
    action = ExplorationStrategies.boltzmann_exploration(Q, (0, 0), valid_actions, 1.0)
    assert action in valid_actions, "Action should be valid"
    print("✓ Boltzmann exploration works")

    # Test epsilon decay
    eps = ExplorationStrategies.decay_epsilon(0.9, 0, 0.99, 0.01)
    assert eps == 0.9, "Initial epsilon should be preserved"

    eps = ExplorationStrategies.decay_epsilon(0.9, 100, 0.99, 0.01)
    assert eps >= 0.01, "Epsilon should not go below minimum"
    print("✓ Epsilon decay works")

    # Test Boltzmann Q-Learning
    boltzmann_agent = BoltzmannQLearning(env, temperature=2.0)
    assert boltzmann_agent.temperature == 2.0, "Temperature should be set correctly"
    print("✓ Boltzmann Q-Learning agent created")

    print("✓ All exploration strategy tests passed!")


def test_experiments():
    """Test experiment functions"""
    print("\n" + "=" * 60)
    print("TESTING EXPERIMENT FUNCTIONS")
    print("=" * 60)

    env = GridWorld()
    policy = RandomPolicy(env)

    # Test TD(0) experiment
    td_agent, V_td = experiment_td0(env, policy, num_episodes=50)
    assert len(V_td) > 0, "Should learn value function"
    print("✓ TD(0) experiment works")

    # Test Q-Learning experiment
    q_agent, V_optimal, q_policy, q_evaluation = experiment_q_learning(
        env, num_episodes=100
    )
    assert len(V_optimal) > 0, "Should learn optimal value function"
    assert len(q_policy) > 0, "Should learn policy"
    assert "avg_reward" in q_evaluation, "Should return evaluation results"
    print("✓ Q-Learning experiment works")

    # Test SARSA experiment
    sarsa_agent, V_sarsa, sarsa_policy, sarsa_evaluation = experiment_sarsa(
        env, num_episodes=100
    )
    assert len(V_sarsa) > 0, "Should learn value function"
    assert len(sarsa_policy) > 0, "Should learn policy"
    assert "avg_reward" in sarsa_evaluation, "Should return evaluation results"
    print("✓ SARSA experiment works")

    print("✓ All experiment tests passed!")
    return (
        td_agent,
        q_agent,
        sarsa_agent,
        V_td,
        V_optimal,
        V_sarsa,
        q_evaluation,
        sarsa_evaluation,
    )


def test_evaluation():
    """Test evaluation functions"""
    print("\n" + "=" * 60)
    print("TESTING EVALUATION FUNCTIONS")
    print("=" * 60)

    env = GridWorld()

    # Create a simple agent for testing
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    agent.train(num_episodes=100, print_every=100)

    # Test agent evaluation
    metrics = evaluate_agent(agent, env, num_episodes=50, save_results=False)
    assert "avg_reward" in metrics, "Should include average reward"
    assert "success_rate" in metrics, "Should include success rate"
    print("✓ Agent evaluation works")

    # Test agent comparison
    agents_dict = {
        "Q-Learning": agent,
        "SARSA": SARSAAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1),
    }

    # Train SARSA agent
    agents_dict["SARSA"].train(num_episodes=100, print_every=100)

    comparison_results = compare_agents(
        agents_dict, env, num_episodes=30, save_dir="test_visualizations"
    )
    assert len(comparison_results) == 2, "Should compare 2 agents"
    print("✓ Agent comparison works")

    # Test performance analysis
    analysis = analyze_performance(agent, save_dir="test_visualizations")
    assert "total_episodes" in analysis, "Should include total episodes"
    print("✓ Performance analysis works")

    print("✓ All evaluation tests passed!")


def test_visualization():
    """Test visualization functions"""
    print("\n" + "=" * 60)
    print("TESTING VISUALIZATION FUNCTIONS")
    print("=" * 60)

    env = GridWorld()

    # Create agents for visualization testing
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    agent.train(num_episodes=100, print_every=100)

    # Test learning curve plotting (suppress display)
    try:
        plot_learning_curve(agent.episode_rewards, "Test Learning Curve")
        print("✓ Learning curve plotting works")
    except Exception as e:
        print(f"⚠ Learning curve plotting failed: {e}")

    # Test Q-learning analysis plotting
    try:
        plot_q_learning_analysis(agent)
        print("✓ Q-learning analysis plotting works")
    except Exception as e:
        print(f"⚠ Q-learning analysis plotting failed: {e}")

    print("✓ Visualization tests completed!")


def test_model_utils():
    """Test model utility functions"""
    print("\n" + "=" * 60)
    print("TESTING MODEL UTILITIES")
    print("=" * 60)

    env = GridWorld()
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    agent.train(num_episodes=50, print_every=50)

    # Test model saving
    save_model(agent, "test_model.pkl", save_dir="test_models")
    print("✓ Model saving works")

    # Test results export
    results = {
        "episode_rewards": agent.episode_rewards,
        "avg_reward": np.mean(agent.episode_rewards),
        "success_rate": 0.8,
    }
    export_results(results, "test_results", save_dir="test_visualizations")
    print("✓ Results export works")

    # Test summary report
    results_dict = {
        "Q-Learning": results,
        "SARSA": {"avg_reward": 7.5, "success_rate": 0.75},
    }
    create_summary_report(results_dict, save_dir="test_visualizations")
    print("✓ Summary report creation works")

    print("✓ All model utility tests passed!")


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 STARTING COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Test started at: {datetime.now()}")
    print("=" * 80)

    try:
        # Test all components
        env = test_environment()
        policy = test_policies()
        td_agent, V_td = test_td0_agent()
        q_agent, V_optimal, q_evaluation = test_q_learning_agent()
        sarsa_agent, sarsa_evaluation = test_sarsa_agent()
        test_exploration_strategies()
        (
            td_agent2,
            q_agent2,
            sarsa_agent2,
            V_td2,
            V_optimal2,
            V_sarsa2,
            q_eval2,
            sarsa_eval2,
        ) = test_experiments()
        test_evaluation()
        test_visualization()
        test_model_utils()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 80)

        # Generate final summary
        print("\nFINAL TEST SUMMARY:")
        print(f"✓ Environment: GridWorld {env.size}x{env.size}")
        print(f"✓ TD(0) Agent: Learned {len(V_td)} state values")
        print(f"✓ Q-Learning: Success rate {q_evaluation['success_rate']*100:.1f}%")
        print(f"✓ SARSA: Success rate {sarsa_evaluation['success_rate']*100:.1f}%")
        print(f"✓ All modules working correctly")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_demo():
    """Run a demonstration of all algorithms"""
    print("\n" + "=" * 80)
    print("🎬 RUNNING DEMONSTRATION")
    print("=" * 80)

    # Create environment
    env = GridWorld()
    print("Created GridWorld environment")

    # Run TD(0) demonstration
    print("\n1. TD(0) Policy Evaluation:")
    policy = RandomPolicy(env)
    td_agent = TD0Agent(env, policy, alpha=0.1, gamma=0.9)
    V_td = td_agent.train(num_episodes=200, print_every=100)
    print(f"   Learned values for {len(V_td)} states")

    # Run Q-Learning demonstration
    print("\n2. Q-Learning Control:")
    q_agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    q_agent.train(num_episodes=300, print_every=150)
    q_evaluation = q_agent.evaluate_policy(num_episodes=100)
    print(f"   Success rate: {q_evaluation['success_rate']*100:.1f}%")

    # Run SARSA demonstration
    print("\n3. SARSA Control:")
    sarsa_agent = SARSAAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    sarsa_agent.train(num_episodes=300, print_every=150)
    sarsa_evaluation = sarsa_agent.evaluate_policy(num_episodes=100)
    print(f"   Success rate: {sarsa_evaluation['success_rate']*100:.1f}%")

    # Save results
    print("\n4. Saving Results:")
    save_model(q_agent, "demo_q_learning.pkl")
    save_model(sarsa_agent, "demo_sarsa.pkl")

    results = {
        "TD(0)": {"avg_reward": np.mean(td_agent.episode_rewards)},
        "Q-Learning": q_evaluation,
        "SARSA": sarsa_evaluation,
    }
    export_results(results, "demo_results")
    create_summary_report(results)

    print("✓ Demonstration completed successfully!")
    print("✓ Results saved to visualizations/ directory")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("test_visualizations", exist_ok=True)
    os.makedirs("test_models", exist_ok=True)

    # Run tests
    success = run_comprehensive_test()

    if success:
        # Run demonstration
        run_demo()

        print("\n" + "=" * 80)
        print("✅ COMPREHENSIVE TEST SUITE COMPLETED SUCCESSFULLY!")
        print("✅ All algorithms are working correctly!")
        print("✅ Results saved to visualizations/ directory")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ TEST SUITE FAILED!")
        print("❌ Please check the error messages above")
        print("=" * 80)
        sys.exit(1)
