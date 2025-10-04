#!/bin/bash

# CA15: Advanced Deep Reinforcement Learning - Complete Setup and Execution Guide
# This script provides comprehensive instructions for running all CA15 experiments

echo "🚀 CA15: Advanced Deep Reinforcement Learning - Complete Setup Guide"
echo "=================================================================="

# Function to print section headers
print_section() {
    echo ""
    echo "📋 $1"
    echo "$(printf '=%.0s' {1..60})"
}

# Function to print step
print_step() {
    echo "🔸 $1"
}

# Function to print success
print_success() {
    echo "✅ $1"
}

# Function to print warning
print_warning() {
    echo "⚠️  $1"
}

# Function to print info
print_info() {
    echo "ℹ️  $1"
}

print_section "Project Overview"
echo "CA15 implements advanced deep reinforcement learning algorithms including:"
echo "• Model-Based RL: Dynamics models, MPC, Dyna-Q"
echo "• Hierarchical RL: Options, HAC, Goal-conditioned learning, Feudal networks"
echo "• Planning: MCTS, Model-based value expansion, World models"
echo "• Complete experiment framework with visualization and analysis"

print_section "Quick Start (Recommended)"
print_step "1. Install Dependencies"
echo "   pip install -r requirements.txt"
echo ""
print_step "2. Run All Experiments"
echo "   ./run.sh"
echo ""
print_step "3. View Results"
echo "   • Check visualizations/ folder for plots"
echo "   • Read results/ folder for detailed reports"
echo "   • Review logs/ for training details"

print_section "Alternative Execution Methods"
print_step "Method 1: Bash Script (Recommended)"
echo "   ./run.sh                    # Run all experiments"
echo "   ./run.sh model-based        # Run only model-based experiments"
echo "   ./run.sh hierarchical       # Run only hierarchical experiments"
echo "   ./run.sh planning           # Run only planning experiments"
echo "   ./run.sh notebook           # Run Jupyter notebook only"
echo ""

print_step "Method 2: Python Script"
echo "   python3 run_all_experiments.py --all           # Run all experiments"
echo "   python3 run_all_experiments.py --model-based   # Model-based only"
echo "   python3 run_all_experiments.py --hierarchical  # Hierarchical only"
echo "   python3 run_all_experiments.py --planning      # Planning only"
echo ""

print_step "Method 3: Jupyter Notebook"
echo "   jupyter notebook CA15.ipynb"
echo ""

print_step "Method 4: Individual Modules"
echo "   python3 training_examples.py                   # Training examples"
echo "   python3 experiments/runner.py                 # Experiment runner"
echo "   python3 experiments/hierarchical.py           # Hierarchical experiments"
echo "   python3 experiments/planning.py               # Planning experiments"

print_section "Project Structure"
echo "CA15_Model_Based_Hierarchical_RL/"
echo "├── run.sh                           # Main execution script"
echo "├── run_all_experiments.py           # Complete Python runner"
echo "├── CA15.ipynb                       # Main Jupyter notebook"
echo "├── training_examples.py             # Training examples"
echo "├── requirements.txt                 # Dependencies"
echo "├── README.md                        # Documentation"
echo "├── test_structure.py                # Structure test script"
echo "│"
echo "├── model_based_rl/                  # Model-Based RL algorithms"
echo "│   ├── algorithms.py               # DynamicsModel, ModelEnsemble, MPC, DynaQ"
echo "│   └── __init__.py"
echo "│"
echo "├── hierarchical_rl/                 # Hierarchical RL algorithms"
echo "│   ├── algorithms.py               # Options, HAC, Goal-Conditioned, Feudal"
echo "│   ├── environments.py             # Hierarchical environment wrappers"
echo "│   └── __init__.py"
echo "│"
echo "├── planning/                        # Planning algorithms"
echo "│   ├── algorithms.py               # MCTS, MVE, LatentSpacePlanner, WorldModel"
echo "│   └── __init__.py"
echo "│"
echo "├── experiments/                     # Experiment runners"
echo "│   ├── runner.py                   # Unified experiment runner"
echo "│   ├── hierarchical.py             # Hierarchical RL experiments"
echo "│   ├── planning.py                 # Planning algorithm experiments"
echo "│   └── __init__.py"
echo "│"
echo "├── environments/                    # Custom environments"
echo "│   ├── grid_world.py               # Simple grid world"
echo "│   └── __init__.py"
echo "│"
echo "├── utils/                           # Utility functions"
echo "│   └── __init__.py                  # ReplayBuffer, Logger, VisualizationUtils"
echo "│"
echo "├── visualizations/                  # Generated plots (created after running)"
echo "├── results/                         # Experiment results (created after running)"
echo "├── logs/                            # Training logs (created after running)"
echo "└── data/                            # Collected data (created after running)"

print_section "Available Algorithms"

print_step "Model-Based RL"
echo "• DynamicsModel: Neural network for environment dynamics learning"
echo "• ModelEnsemble: Ensemble methods for uncertainty quantification"
echo "• ModelPredictiveController: MPC using learned dynamics"
echo "• DynaQAgent: Combining model-free and model-based learning"

print_step "Hierarchical RL"
echo "• Option: Options framework implementation"
echo "• HierarchicalActorCritic: Multi-level policies with different time scales"
echo "• GoalConditionedAgent: Goal-conditioned RL with Hindsight Experience Replay"
echo "• FeudalNetwork: Manager-worker architecture for goal-directed behavior"

print_step "Planning Algorithms"
echo "• MonteCarloTreeSearch: MCTS with neural network guidance"
echo "• ModelBasedValueExpansion: Recursive value expansion using learned models"
echo "• LatentSpacePlanner: Planning in learned compact representations"
echo "• WorldModel: End-to-end models for environment simulation and control"

print_section "Expected Results"
print_step "Generated Files"
echo "• visualizations/ca15_complete_analysis_*.png: Comprehensive analysis plots"
echo "• results/ca15_experiment_report_*.md: Detailed experiment reports"
echo "• logs/: Training logs and metrics"
echo "• data/: Collected training data"

print_step "Key Metrics"
echo "• Sample Efficiency: Episodes needed to reach performance threshold"
echo "• Final Performance: Average reward in final episodes"
echo "• Computational Overhead: Planning time per episode"
echo "• Success Rate: Goal achievement rate for hierarchical methods"

print_section "Troubleshooting"

print_step "Common Issues"
echo "1. CUDA Out of Memory:"
echo "   export CUDA_VISIBLE_DEVICES=\"\"  # Use CPU only"
echo ""
echo "2. Import Errors:"
echo "   export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)\""
echo ""
echo "3. Permission Denied:"
echo "   chmod +x run.sh run_all_experiments.py"
echo ""
echo "4. Missing Dependencies:"
echo "   pip install -r requirements.txt"

print_step "Performance Issues"
echo "• Slow Training: Reduce number of episodes or use smaller networks"
echo "• High Memory Usage: Reduce batch sizes or use gradient accumulation"
echo "• Convergence Issues: Adjust learning rates or add learning rate scheduling"

print_section "Customization"

print_step "Adding New Algorithms"
echo "1. Create your algorithm class in the appropriate module"
echo "2. Add it to the __init__.py file"
echo "3. Include it in the experiment runner"

print_step "Custom Environments"
echo "1. Create your environment class in environments/"
echo "2. Implement the standard RL interface (reset(), step())"
echo "3. Update the experiment runners to use your environment"

print_step "Hyperparameter Tuning"
echo "Modify hyperparameters in run_all_experiments.py:"
echo "• Learning rates, discount factors, exploration rates"
echo "• Network architectures and training parameters"
echo "• Environment-specific settings"

print_section "Advanced Usage"

print_step "Integration with Other Projects"
echo "from model_based_rl.algorithms import DynaQAgent"
echo "from hierarchical_rl.algorithms import GoalConditionedAgent"
echo "from planning.algorithms import MonteCarloTreeSearch"
echo ""
echo "# Use in your own experiments"
echo "agent = DynaQAgent(state_dim=4, action_dim=2)"
echo "# ... your training loop"

print_step "Research Extensions"
echo "• Meta-Learning: Learning to learn world models across tasks"
echo "• Multi-Agent Hierarchical RL: Coordinating multiple hierarchical agents"
echo "• Safe Model-Based RL: Incorporating safety constraints in planning"
echo "• Continual Learning: Updating world models with new experiences"

print_section "Performance Expectations"

print_step "Sample Efficiency"
echo "• Model-based methods: 5-10x better than model-free"
echo "• Hierarchical RL: Enables complex task decomposition"
echo "• Planning algorithms: Better asymptotic performance"

print_step "Computational Trade-offs"
echo "• MCTS: Highest computational cost, best performance"
echo "• MPC: Moderate cost, good performance"
echo "• Dyna-Q: Low cost, good sample efficiency"
echo "• Goal-Conditioned: Moderate cost, excellent for multi-goal tasks"

print_section "Final Notes"
print_success "CA15 is a complete implementation of advanced deep RL algorithms"
print_success "All algorithms are fully implemented and ready for execution"
print_success "Comprehensive experiment framework with automatic visualization"
print_success "Modular design allows easy extension and customization"

print_info "For questions or issues, check the troubleshooting section"
print_info "All experiments use simple test environments for demonstration"
print_info "Results may vary significantly on more complex environments"

echo ""
echo "🎉 Ready to explore advanced deep reinforcement learning!"
echo "🚀 Start with: ./run.sh"
echo ""
echo "Happy Learning! 🎓"
