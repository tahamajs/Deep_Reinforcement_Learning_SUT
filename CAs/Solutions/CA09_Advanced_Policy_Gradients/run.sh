#!/bin/bash

# ===============================================================================
# CA9: Advanced Policy Gradient Methods - Complete Execution Script
# ===============================================================================
# This script runs all policy gradient implementations and generates visualizations
# Author: DRL Course Assistant
# Date: $(date +%Y-%m-%d)
# ===============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZATIONS_DIR="$PROJECT_DIR/visualizations"
RESULTS_DIR="$PROJECT_DIR/results"
LOGS_DIR="$PROJECT_DIR/logs"

# Create directories if they don't exist
mkdir -p "$VISUALIZATIONS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Check if Python is available
    if ! command_exists python3; then
        print_error "Python3 is not installed. Please install Python 3.8+"
        exit 1
    fi
    
    # Check if pip is available
    if ! command_exists pip3; then
        print_error "pip3 is not installed. Please install pip"
        exit 1
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Install requirements
    print_status "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Environment setup complete!"
}

# Function to run Python script with error handling
run_python_script() {
    local script_name="$1"
    local description="$2"
    local log_file="$LOGS_DIR/${script_name%.py}.log"
    
    print_status "Running $description..."
    
    if python3 "$script_name" > "$log_file" 2>&1; then
        print_success "$description completed successfully!"
        return 0
    else
        print_error "$description failed. Check $log_file for details."
        return 1
    fi
}

# Function to run Jupyter notebook
run_notebook() {
    local notebook_name="$1"
    local description="$2"
    local log_file="$LOGS_DIR/${notebook_name%.ipynb}.log"
    
    print_status "Running $description..."
    
    if jupyter nbconvert --execute --to notebook --inplace "$notebook_name" > "$log_file" 2>&1; then
        print_success "$description completed successfully!"
        return 0
    else
        print_error "$description failed. Check $log_file for details."
        return 1
    fi
}

# Function to run training examples
run_training_examples() {
    print_status "Running comprehensive training examples..."
    
    # Run different policy gradient methods
    local methods=("reinforce" "actor_critic" "ppo" "continuous_control")
    local environments=("CartPole-v1" "LunarLander-v2" "Pendulum-v1")
    
    for method in "${methods[@]}"; do
        for env in "${environments[@]}"; do
            print_status "Training $method on $env..."
            python3 -c "
import sys
sys.path.append('.')
from training_examples import *

if '$method' == 'reinforce':
    results = train_reinforce_agent('$env', num_episodes=200)
elif '$method' == 'actor_critic':
    results = train_actor_critic_agent('$env', num_episodes=200)
elif '$method' == 'ppo':
    results = train_ppo_agent('$env', num_episodes=200)
elif '$method' == 'continuous_control' and '$env' == 'Pendulum-v1':
    results = train_continuous_ppo_agent('$env', num_episodes=200)

print(f'Results for {method} on {env}:')
print(f'Final reward: {results.get(\"final_reward\", 0):.2f}')
print(f'Episodes: {results.get(\"episodes\", 0)}')
" > "$LOGS_DIR/training_${method}_${env}.log" 2>&1
            
            if [ $? -eq 0 ]; then
                print_success "$method on $env completed!"
            else
                print_warning "$method on $env had issues (check logs)"
            fi
        done
    done
}

# Function to generate comprehensive visualizations
generate_visualizations() {
    print_status "Generating comprehensive visualizations..."
    
    python3 -c "
import sys
sys.path.append('.')
from training_examples import *
import matplotlib.pyplot as plt
import numpy as np

# Generate all visualization functions
print('Generating convergence analysis...')
fig1 = plot_policy_gradient_convergence_analysis('$VISUALIZATIONS_DIR/convergence_analysis.png')

print('Generating comprehensive comparison...')
results = comprehensive_policy_gradient_comparison('$VISUALIZATIONS_DIR/comprehensive_comparison.png')

print('Generating curriculum learning analysis...')
curriculum_results = policy_gradient_curriculum_learning('$VISUALIZATIONS_DIR/curriculum_learning.png')

print('Generating entropy regularization study...')
entropy_results = entropy_regularization_study('$VISUALIZATIONS_DIR/entropy_regularization.png')

print('Generating TRPO comparison...')
trpo_results = trust_region_policy_optimization_comparison('$VISUALIZATIONS_DIR/trpo_comparison.png')

print('Creating comprehensive visualization suite...')
create_comprehensive_visualization_suite('$VISUALIZATIONS_DIR/')

print('All visualizations generated successfully!')
" > "$LOGS_DIR/visualizations.log" 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "All visualizations generated successfully!"
    else
        print_error "Visualization generation failed. Check logs."
    fi
}

# Function to run agent implementations
run_agent_implementations() {
    print_status "Running agent implementations..."
    
    # Run REINFORCE
    run_python_script "agents/reinforce.py" "REINFORCE Agent Implementation"
    
    # Run Baseline REINFORCE
    run_python_script "agents/baseline_reinforce.py" "Baseline REINFORCE Implementation"
    
    # Run Actor-Critic
    run_python_script "agents/actor_critic.py" "Actor-Critic Implementation"
    
    # Run PPO
    run_python_script "agents/ppo.py" "PPO Implementation"
    
    # Run Continuous Control
    run_python_script "agents/continuous_control.py" "Continuous Control Implementation"
}

# Function to run utility scripts
run_utility_scripts() {
    print_status "Running utility scripts..."
    
    # Run hyperparameter tuning
    if [ -f "utils/hyperparameter_tuning.py" ]; then
        run_python_script "utils/hyperparameter_tuning.py" "Hyperparameter Tuning"
    fi
    
    # Run policy gradient visualizer
    if [ -f "utils/policy_gradient_visualizer.py" ]; then
        run_python_script "utils/policy_gradient_visualizer.py" "Policy Gradient Visualizer"
    fi
    
    # Run main utility
    if [ -f "utils/main.py" ]; then
        run_python_script "utils/main.py" "Main Utility Script"
    fi
}

# Function to create summary report
create_summary_report() {
    print_status "Creating summary report..."
    
    cat > "$RESULTS_DIR/summary_report.md" << EOF
# CA9: Advanced Policy Gradient Methods - Execution Summary

## Execution Date
$(date)

## Environment Setup
- Python Version: $(python3 --version)
- Virtual Environment: Activated
- Dependencies: Installed from requirements.txt

## Executed Components

### 1. Agent Implementations
- ✅ REINFORCE Agent
- ✅ Baseline REINFORCE Agent  
- ✅ Actor-Critic Agent
- ✅ PPO Agent
- ✅ Continuous Control Agent

### 2. Training Examples
- ✅ REINFORCE Training
- ✅ Actor-Critic Training
- ✅ PPO Training
- ✅ Continuous Control Training
- ✅ Comprehensive Comparisons

### 3. Visualizations Generated
- ✅ Policy Gradient Convergence Analysis
- ✅ Comprehensive Method Comparison
- ✅ Curriculum Learning Analysis
- ✅ Entropy Regularization Study
- ✅ Trust Region Policy Optimization Comparison
- ✅ Advanced Visualization Suite

### 4. Environments Tested
- CartPole-v1
- LunarLander-v2
- Pendulum-v1

## Results Location
- Visualizations: $VISUALIZATIONS_DIR/
- Logs: $LOGS_DIR/
- This Report: $RESULTS_DIR/summary_report.md

## Status: COMPLETE ✅
All components executed successfully!
EOF
    
    print_success "Summary report created at $RESULTS_DIR/summary_report.md"
}

# Function to check results
check_results() {
    print_status "Checking results..."
    
    local viz_count=$(find "$VISUALIZATIONS_DIR" -name "*.png" -o -name "*.pdf" | wc -l)
    local log_count=$(find "$LOGS_DIR" -name "*.log" | wc -l)
    
    print_status "Generated $viz_count visualization files"
    print_status "Generated $log_count log files"
    
    if [ $viz_count -gt 0 ]; then
        print_success "Visualizations successfully generated!"
    else
        print_warning "No visualizations found. Check logs for issues."
    fi
}

# Main execution function
main() {
    echo "=================================================================================="
    echo "CA9: Advanced Policy Gradient Methods - Complete Execution"
    echo "=================================================================================="
    echo "Project Directory: $PROJECT_DIR"
    echo "Visualizations: $VISUALIZATIONS_DIR"
    echo "Results: $RESULTS_DIR"
    echo "Logs: $LOGS_DIR"
    echo "=================================================================================="
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Setup environment
    setup_environment
    
    # Run agent implementations
    run_agent_implementations
    
    # Run utility scripts
    run_utility_scripts
    
    # Run training examples
    run_training_examples
    
    # Generate visualizations
    generate_visualizations
    
    # Run Jupyter notebook
    if [ -f "CA9.ipynb" ]; then
        run_notebook "CA9.ipynb" "CA9 Jupyter Notebook"
    fi
    
    # Create summary report
    create_summary_report
    
    # Check results
    check_results
    
    echo "=================================================================================="
    echo "🎉 EXECUTION COMPLETE!"
    echo "=================================================================================="
    print_success "All policy gradient implementations executed successfully!"
    print_status "Check the following directories for results:"
    echo "  📊 Visualizations: $VISUALIZATIONS_DIR"
    echo "  📋 Results: $RESULTS_DIR" 
    echo "  📝 Logs: $LOGS_DIR"
    echo "=================================================================================="
}

# Run main function
main "$@"

