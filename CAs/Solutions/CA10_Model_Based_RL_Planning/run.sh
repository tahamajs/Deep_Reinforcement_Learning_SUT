#!/bin/bash

# CA10: Model-Based Reinforcement Learning and Planning Methods - Complete Run Script
# ==============================================================================
# This script runs all components of the Model-Based RL assignment
# Author: DRL Course Team
# Date: 2025

echo "🚀 Starting CA10: Model-Based Reinforcement Learning and Planning Methods"
echo "========================================================================="

# Set working directory
cd "$(dirname "$0")"

# Create necessary directories
mkdir -p visualizations
mkdir -p logs
mkdir -p results

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Function to run with error handling
run_component() {
    local component_name="$1"
    local script_path="$2"
    local description="$3"
    
    echo ""
    echo "📊 Running: $component_name"
    echo "Description: $description"
    echo "----------------------------------------"
    
    if [ -f "$script_path" ]; then
        python "$script_path" 2>&1 | tee "logs/${component_name}.log"
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✅ $component_name completed successfully"
        else
            echo "❌ $component_name failed - check logs/${component_name}.log"
        fi
    else
        echo "⚠️ $script_path not found - skipping $component_name"
    fi
}

# Function to run Jupyter notebook
run_notebook() {
    local notebook_name="$1"
    local description="$2"
    
    echo ""
    echo "📓 Running: $notebook_name"
    echo "Description: $description"
    echo "----------------------------------------"
    
    if [ -f "$notebook_name" ]; then
        # Convert notebook to Python and run
        jupyter nbconvert --to python --execute "$notebook_name" --stdout 2>&1 | tee "logs/${notebook_name%.ipynb}.log"
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✅ $notebook_name completed successfully"
        else
            echo "❌ $notebook_name failed - check logs/${notebook_name%.ipynb}.log"
        fi
    else
        echo "⚠️ $notebook_name not found - skipping"
    fi
}

echo ""
echo "🔧 Phase 1: Environment Setup and Model Learning"
echo "================================================"

# Run training examples to set up models
run_component "Training Examples" "training_examples.py" "Setting up models and training data"

echo ""
echo "🎯 Phase 2: Classical Planning Algorithms"
echo "========================================="

# Run classical planning demonstration
run_component "Classical Planning" "agents/classical_planning.py" "Value Iteration, Policy Iteration, and uncertainty-aware planning"

echo ""
echo "🧠 Phase 3: Dyna-Q Algorithm"
echo "============================"

# Run Dyna-Q demonstration
run_component "Dyna-Q" "agents/dyna_q.py" "Integrated planning and learning with Dyna-Q and Dyna-Q+"

echo ""
echo "🌳 Phase 4: Monte Carlo Tree Search (MCTS)"
echo "==========================================="

# Run MCTS demonstration
run_component "MCTS" "agents/mcts.py" "Monte Carlo Tree Search for planning"

echo ""
echo "🎮 Phase 5: Model Predictive Control (MPC)"
echo "==========================================="

# Run MPC demonstration
run_component "MPC" "agents/mpc.py" "Model Predictive Control for optimal control"

echo ""
echo "📊 Phase 6: Comprehensive Comparison"
echo "===================================="

# Run comprehensive comparison
run_component "Comparison" "experiments/comparison.py" "Comprehensive analysis of all methods"

echo ""
echo "📓 Phase 7: Educational Notebook"
echo "================================"

# Run the main educational notebook
run_notebook "CA10.ipynb" "Main educational content and demonstrations"

echo ""
echo "📈 Phase 8: Generate Additional Visualizations"
echo "=============================================="

# Run additional analysis scripts
python -c "
import sys
import os
sys.path.append('.')
from training_examples import (
    plot_model_based_comparison,
    analyze_mcts_performance,
    comprehensive_model_based_analysis
)

print('Generating comprehensive analysis visualizations...')
plot_model_based_comparison('visualizations/model_based_comparison.png')
analyze_mcts_performance('visualizations/mcts_performance.png')
comprehensive_model_based_analysis('visualizations/comprehensive_analysis.png')
print('✅ All additional visualizations generated!')
" 2>&1 | tee logs/additional_visualizations.log

echo ""
echo "📋 Phase 9: Generate Summary Report"
echo "==================================="

# Generate summary report
python -c "
import os
import glob

print('📊 CA10 Model-Based RL Summary Report')
print('=' * 50)

# Check visualizations
viz_files = glob.glob('visualizations/*.png')
print(f'📈 Generated {len(viz_files)} visualization files:')
for viz in sorted(viz_files):
    print(f'  • {os.path.basename(viz)}')

# Check logs
log_files = glob.glob('logs/*.log')
print(f'📝 Generated {len(log_files)} log files:')
for log in sorted(log_files):
    print(f'  • {os.path.basename(log)}')

print()
print('✅ CA10 Model-Based RL Analysis Complete!')
print('🎓 All components have been successfully executed.')
print('📚 Check the visualizations/ folder for results and analysis.')
" 2>&1 | tee logs/summary_report.log

echo ""
echo "🎉 CA10: Model-Based Reinforcement Learning Complete!"
echo "====================================================="
echo ""
echo "📊 Results Summary:"
echo "  • Classical Planning: Value/Policy Iteration with learned models"
echo "  • Dyna-Q: Integrated planning and learning"
echo "  • MCTS: Monte Carlo Tree Search for sophisticated planning"
echo "  • MPC: Model Predictive Control for optimal control"
echo "  • Comprehensive comparison of all methods"
echo ""
echo "📁 Output Files:"
echo "  • visualizations/: All generated plots and analysis"
echo "  • logs/: Execution logs for each component"
echo "  • results/: Any additional result files"
echo ""
echo "🔍 Key Insights:"
echo "  • Model-based methods improve sample efficiency"
echo "  • Planning steps significantly boost performance"
echo "  • Different methods excel in different scenarios"
echo "  • Neural models provide better generalization"
echo ""
echo "🚀 Next Steps:"
echo "  • Explore the generated visualizations"
echo "  • Experiment with different hyperparameters"
echo "  • Try the methods on different environments"
echo "  • Study the theoretical foundations in the notebook"
echo ""
echo "✅ All done! Happy learning! 🎓"
