#!/bin/bash

# CA1 Deep RL Fundamentals - Comprehensive Execution Script
# This script runs all components of the CA1 project and generates results

echo "=========================================="
echo "CA1 Deep RL Fundamentals - Execution Script"
echo "=========================================="

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p visualization
mkdir -p experiments/results
mkdir -p experiments/plots
mkdir -p logs

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a logs/execution.log
}

# Function to check if a command succeeded
check_success() {
    if [ $? -eq 0 ]; then
        log "✓ $1 completed successfully"
    else
        log "✗ $1 failed"
        exit 1
    fi
}

# Function to install dependencies if needed
install_dependencies() {
    log "Installing Python dependencies..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        log "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        check_success "Dependencies installation"
    else
        log "Warning: requirements.txt not found, installing basic dependencies..."
        pip install torch torchvision torchaudio gymnasium numpy matplotlib seaborn pandas jupyter ipykernel tqdm scikit-learn pillow
        check_success "Basic dependencies installation"
    fi
}

# Function to run notebook execution
run_notebook() {
    log "Executing Jupyter notebook..."
    
    # Convert notebook to Python script and execute
    if command -v jupyter &> /dev/null; then
        jupyter nbconvert --to python --execute CA1.ipynb --ExecutePreprocessor.timeout=3600
        check_success "Notebook execution"
        
        # Run the converted Python script
        python CA1.py
        check_success "Python script execution"
    else
        log "Jupyter not found, running Python modules directly..."
        python -c "
import sys
sys.path.append('.')
from agents.ca1_agents import *
from models.ca1_models import *
from utils.ca1_utils import *
from environments.custom_envs import *
from evaluation.evaluators import *
from experiments.experiment_runner import *
print('All modules imported successfully!')
"
        check_success "Module imports"
    fi
}

# Function to run individual experiments
run_experiments() {
    log "Running individual experiments..."
    
    # Run DQN experiment
    python -c "
import sys
sys.path.append('.')
from agents.ca1_agents import DQNAgent, train_dqn_agent
from environments.custom_envs import create_cartpole_env
from utils.ca1_utils import set_seed
import numpy as np

set_seed(42)
env = create_cartpole_env()
agent = DQNAgent(state_size=4, action_size=2, use_dueling=True, use_double_dqn=True)
scores = train_dqn_agent(agent, env, n_episodes=200, max_t=200)
print(f'DQN training completed. Final average score: {np.mean(scores[-50:]):.2f}')
env.close()
" > logs/dqn_experiment.log 2>&1
    check_success "DQN experiment"
    
    # Run REINFORCE experiment
    python -c "
import sys
sys.path.append('.')
from agents.ca1_agents import REINFORCEAgent, train_reinforce_agent
from environments.custom_envs import create_cartpole_env
from utils.ca1_utils import set_seed
import numpy as np

set_seed(42)
env = create_cartpole_env()
agent = REINFORCEAgent(state_size=4, action_size=2)
scores = train_reinforce_agent(agent, env, n_episodes=200, max_t=200)
print(f'REINFORCE training completed. Final average score: {np.mean(scores[-50:]):.2f}')
env.close()
" > logs/reinforce_experiment.log 2>&1
    check_success "REINFORCE experiment"
    
    # Run Actor-Critic experiment
    python -c "
import sys
sys.path.append('.')
from agents.ca1_agents import ActorCriticAgent, train_actor_critic_agent
from environments.custom_envs import create_cartpole_env
from utils.ca1_utils import set_seed
import numpy as np

set_seed(42)
env = create_cartpole_env()
agent = ActorCriticAgent(state_size=4, action_size=2)
scores = train_actor_critic_agent(agent, env, n_episodes=200, max_t=200)
print(f'Actor-Critic training completed. Final average score: {np.mean(scores[-50:]):.2f}')
env.close()
" > logs/actor_critic_experiment.log 2>&1
    check_success "Actor-Critic experiment"
}

# Function to run comprehensive comparison
run_comprehensive_comparison() {
    log "Running comprehensive algorithm comparison..."
    
    python -c "
import sys
sys.path.append('.')
from experiments.experiment_runner import run_quick_demo
import os

# Run quick demo for comprehensive comparison
results = run_quick_demo()
print('Comprehensive comparison completed!')
print('Results saved to demo_experiments/ directory')
" > logs/comprehensive_comparison.log 2>&1
    check_success "Comprehensive comparison"
}

# Function to generate visualizations
generate_visualizations() {
    log "Generating visualizations..."
    
    python -c "
import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# Load existing results if available
if os.path.exists('results.json'):
    with open('results.json', 'r') as f:
        results = json.load(f)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Learning curves
    ax1 = axes[0, 0]
    for algorithm, scores in results.items():
        if len(scores) > 0:
            episodes = range(len(scores))
            ax1.plot(episodes, scores, label=algorithm, linewidth=2, alpha=0.8)
            
            # Add moving average
            if len(scores) >= 10:
                window = min(10, len(scores))
                ma = np.convolve(scores, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(scores)), ma, '--', alpha=0.6)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Learning Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=195, color='red', linestyle=':', alpha=0.7, label='CartPole Threshold')
    
    # Plot 2: Final performance comparison
    ax2 = axes[0, 1]
    final_scores = [np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores) if len(scores) > 0 else 0 for scores in results.values()]
    algorithms = list(results.keys())
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax2.bar(algorithms, final_scores, color=colors[:len(algorithms)], alpha=0.8)
    ax2.set_ylabel('Average Score (Last 10 Episodes)')
    ax2.set_title('Final Performance Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, final_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Score distributions
    ax3 = axes[1, 0]
    for i, (algorithm, scores) in enumerate(results.items()):
        if len(scores) > 0:
            ax3.hist(scores, bins=15, alpha=0.6, label=algorithm, 
                    color=colors[i % len(colors)], density=True)
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Score Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary
    ax4 = axes[1, 1]
    stats_data = []
    for algorithm, scores in results.items():
        if len(scores) > 0:
            stats_data.append({
                'Algorithm': algorithm,
                'Mean': float(np.mean(scores)),
                'Std': float(np.std(scores)),
                'Max': float(np.max(scores)),
                'Min': float(np.min(scores))
            })
    
    # Create summary text
    summary_text = 'Performance Summary:\n\n'
    for stats in stats_data:
        summary_text += f'{stats[\"Algorithm\"]}:\n'
        summary_text += f'  Mean: {stats[\"Mean\"]:.2f} ± {stats[\"Std\"]:.2f}\n'
        summary_text += f'  Range: {stats[\"Min\"]:.2f} - {stats[\"Max\"]:.2f}\n\n'
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='lightblue', alpha=0.7))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Statistical Summary')
    
    plt.tight_layout()
    
    # Save to visualization directory
    os.makedirs('visualization', exist_ok=True)
    plt.savefig('visualization/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('Visualizations generated and saved to visualization/ directory')
else:
    print('No results.json found. Please run experiments first.')
" > logs/visualization.log 2>&1
    check_success "Visualization generation"
}

# Function to run tests
run_tests() {
    log "Running basic tests..."
    
    python -c "
import sys
sys.path.append('.')

# Test 1: Import all modules
try:
    from agents.ca1_agents import DQNAgent, REINFORCEAgent, ActorCriticAgent
    from models.ca1_models import DQN, DuelingDQN, NoisyDQN
    from utils.ca1_utils import set_seed, moving_average
    from environments.custom_envs import SimpleGridWorld, MultiArmedBandit
    from evaluation.evaluators import AgentEvaluator
    from experiments.experiment_runner import ExperimentRunner
    print('✓ All modules imported successfully')
except Exception as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)

# Test 2: Create agents
try:
    set_seed(42)
    dqn_agent = DQNAgent(state_size=4, action_size=2)
    reinforce_agent = REINFORCEAgent(state_size=4, action_size=2)
    ac_agent = ActorCriticAgent(state_size=4, action_size=2)
    print('✓ All agents created successfully')
except Exception as e:
    print(f'✗ Agent creation failed: {e}')
    sys.exit(1)

# Test 3: Create environments
try:
    from environments.custom_envs import create_cartpole_env
    env = create_cartpole_env()
    print('✓ Environment created successfully')
    env.close()
except Exception as e:
    print(f'✗ Environment creation failed: {e}')
    sys.exit(1)

print('✓ All tests passed!')
" > logs/tests.log 2>&1
    check_success "Basic tests"
}

# Function to generate final report
generate_report() {
    log "Generating final report..."
    
    python -c "
import os
import json
from datetime import datetime

# Generate report
report = {
    'timestamp': datetime.now().isoformat(),
    'project': 'CA1 Deep RL Fundamentals',
    'status': 'Completed',
    'files_created': [],
    'experiments_run': [],
    'results_summary': {}
}

# Check created files
files_to_check = [
    'visualization/comprehensive_analysis.png',
    'visualization/learning_curves.png',
    'logs/execution.log',
    'logs/dqn_experiment.log',
    'logs/reinforce_experiment.log',
    'logs/actor_critic_experiment.log',
    'logs/comprehensive_comparison.log',
    'logs/visualization.log',
    'logs/tests.log'
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        report['files_created'].append(file_path)

# Check experiment results
if os.path.exists('results.json'):
    with open('results.json', 'r') as f:
        results = json.load(f)
    report['results_summary'] = {
        algorithm: {
            'mean_score': float(np.mean(scores)) if len(scores) > 0 else 0,
            'max_score': float(np.max(scores)) if len(scores) > 0 else 0,
            'episodes': len(scores)
        } for algorithm, scores in results.items()
    }

# Save report
with open('execution_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('Final report generated: execution_report.json')
" > logs/report.log 2>&1
    check_success "Report generation"
}

# Main execution flow
main() {
    log "Starting CA1 Deep RL Fundamentals execution..."
    
    # Step 1: Install dependencies
    install_dependencies
    
    # Step 2: Run basic tests
    run_tests
    
    # Step 3: Execute notebook
    run_notebook
    
    # Step 4: Run individual experiments
    run_experiments
    
    # Step 5: Run comprehensive comparison
    run_comprehensive_comparison
    
    # Step 6: Generate visualizations
    generate_visualizations
    
    # Step 7: Generate final report
    generate_report
    
    log "CA1 Deep RL Fundamentals execution completed successfully!"
    log "Check the following directories for results:"
    log "  - visualization/: Generated plots and charts"
    log "  - experiments/: Comprehensive experiment results"
    log "  - logs/: Execution logs and individual experiment results"
    log "  - execution_report.json: Final execution summary"
    
    echo ""
    echo "=========================================="
    echo "EXECUTION COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo "Results are available in:"
    echo "  📊 Visualization: ./visualization/"
    echo "  🧪 Experiments: ./experiments/"
    echo "  📝 Logs: ./logs/"
    echo "  📋 Report: ./execution_report.json"
    echo "=========================================="
}

# Handle command line arguments
case "${1:-}" in
    "deps"|"dependencies")
        install_dependencies
        ;;
    "test"|"tests")
        run_tests
        ;;
    "notebook"|"nb")
        run_notebook
        ;;
    "experiments"|"exp")
        run_experiments
        ;;
    "comparison"|"comp")
        run_comprehensive_comparison
        ;;
    "visualize"|"viz")
        generate_visualizations
        ;;
    "report")
        generate_report
        ;;
    "quick")
        log "Running quick execution (tests + experiments + visualization)..."
        install_dependencies
        run_tests
        run_experiments
        generate_visualizations
        generate_report
        log "Quick execution completed!"
        ;;
    "full"|"")
        main
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deps, dependencies  - Install Python dependencies"
        echo "  test, tests         - Run basic tests"
        echo "  notebook, nb        - Execute Jupyter notebook"
        echo "  experiments, exp    - Run individual experiments"
        echo "  comparison, comp    - Run comprehensive comparison"
        echo "  visualize, viz      - Generate visualizations"
        echo "  report              - Generate final report"
        echo "  quick               - Run quick execution (tests + experiments + viz)"
        echo "  full                - Run full execution (default)"
        echo "  help                - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

