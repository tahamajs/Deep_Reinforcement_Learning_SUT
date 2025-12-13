"""
Advanced Policy Gradient Methods - Training Examples
====================================================

This module provides comprehensive implementations and training examples for
Advanced Policy Gradient Methods (CA9).

Key Components:
- REINFORCE algorithm with variance reduction
- Actor-Critic methods (A2C, A3C)
- Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)
- Continuous control with policy gradients
- Advanced analysis and visualization tools

Author: DRL Course Team
"""

import sys
import os

# Ensure src directory is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.experiments.train_utils import (
    set_seed,
    train_reinforce_agent,
    train_ppo_agent,
    train_continuous_ppo_agent,
    compare_policy_gradient_methods,
    plot_policy_gradient_convergence_analysis,
    comprehensive_policy_gradient_comparison,
    policy_gradient_curriculum_learning,
    entropy_regularization_study,
    trust_region_policy_optimization_comparison,
    create_comprehensive_visualization_suite
)
from src.config import Config

if __name__ == "__main__":
    set_seed(Config.SEED)
    print("Advanced Policy Gradient Methods - Training Examples")
    print("=" * 60)
    print("Available training and visualization functions:")
    print("1. train_reinforce_agent() - Trains REINFORCE with/without baseline")
    print("2. train_ppo_agent() - Trains PPO agent")
    print("3. train_continuous_ppo_agent() - Trains Continuous PPO")
    print("4. compare_policy_gradient_methods() - Compares all methods")
    print("5. plot_policy_gradient_convergence_analysis() - Analyzes convergence")
    print("6. comprehensive_policy_gradient_comparison() - Comprehensive comparison")
    print("7. policy_gradient_curriculum_learning() - Curriculum Learning analysis")
    print("8. entropy_regularization_study() - Entropy Regularization study")
    print("9. trust_region_policy_optimization_comparison() - Trust Region comparison")
    print("10. create_comprehensive_visualization_suite() - Creates all visualizations")
    print("\nExample Usage:")
    print("results_ppo = train_ppo_agent(num_episodes=100)")
    print("comparison_results = compare_policy_gradient_methods()")
    print("create_comprehensive_visualization_suite(save_dir=Config.SAVE_DIR)")
