#!/bin/bash

# Comprehensive Temporal Difference Learning Experiment Runner
# This script runs all experiments and saves results to visualizations folder

echo "🚀 Starting Temporal Difference Learning Experiments"
echo "=================================================="
echo "Timestamp: $(date)"
echo "=================================================="

# Set up environment
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
VIS_DIR="$PROJECT_DIR/visualizations"
MODELS_DIR="$PROJECT_DIR/models"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p "$VIS_DIR"
mkdir -p "$MODELS_DIR"
mkdir -p "$PROJECT_DIR/results"
mkdir -p "$PROJECT_DIR/logs"

echo "✓ Directories created successfully"

# Function to run Python script with error handling
run_python_script() {
    local script_name="$1"
    local description="$2"
    
    echo ""
    echo "🔧 Running $description..."
    echo "Script: $script_name"
    echo "----------------------------------------"
    
    if python3 "$script_name" > "$PROJECT_DIR/logs/${script_name%.py}.log" 2>&1; then
        echo "✅ $description completed successfully"
        return 0
    else
        echo "❌ $description failed"
        echo "Check log: $PROJECT_DIR/logs/${script_name%.py}.log"
        return 1
    fi
}

# Function to run Jupyter notebook conversion
run_notebook() {
    local notebook_name="$1"
    local description="$2"
    
    echo ""
    echo "📓 Converting $description..."
    echo "Notebook: $notebook_name"
    echo "----------------------------------------"
    
    if jupyter nbconvert --to python --execute "$notebook_name" --output-dir="$PROJECT_DIR/results" > "$PROJECT_DIR/logs/notebook_${notebook_name%.ipynb}.log" 2>&1; then
        echo "✅ $description converted successfully"
        return 0
    else
        echo "❌ $description conversion failed"
        echo "Check log: $PROJECT_DIR/logs/notebook_${notebook_name%.ipynb}.log"
        return 1
    fi
}

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking Python packages..."
python3 -c "
import sys
required_packages = ['numpy', 'matplotlib', 'seaborn', 'pandas']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        missing_packages.append(package)
        print(f'❌ {package} - MISSING')

if missing_packages:
    print(f'\\nMissing packages: {missing_packages}')
    print('Please install them using: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('✓ All required packages are installed')
"

if [ $? -ne 0 ]; then
    echo "❌ Missing required packages. Please install them first."
    exit 1
fi

# Install requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing requirements..."
    pip3 install -r requirements.txt --quiet
    echo "✓ Requirements installed"
fi

# Track success/failure
SUCCESS_COUNT=0
TOTAL_COUNT=0

# 1. Run comprehensive test suite
TOTAL_COUNT=$((TOTAL_COUNT + 1))
if run_python_script "test_ca3.py" "Comprehensive Test Suite"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi

# 2. Run individual algorithm experiments
echo ""
echo "🧪 Running Individual Algorithm Experiments"
echo "==========================================="

# TD(0) Experiment
TOTAL_COUNT=$((TOTAL_COUNT + 1))
cat > run_td0.py << 'EOF'
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.environments import GridWorld
from agents.policies import RandomPolicy
from experiments.experiments import experiment_td0
from utils.visualization import plot_learning_curve
from models.model_utils import save_model, export_results

print("Running TD(0) Policy Evaluation Experiment...")
env = GridWorld()
policy = RandomPolicy(env)
agent, V_td = experiment_td0(env, policy, num_episodes=500)

# Save results
save_model(agent, "td0_model.pkl")
results = {
    'algorithm': 'TD(0)',
    'episode_rewards': agent.episode_rewards,
    'avg_reward': sum(agent.episode_rewards) / len(agent.episode_rewards),
    'learned_states': len(V_td)
}
export_results(results, "td0_results")

print("TD(0) experiment completed successfully!")
EOF

if run_python_script "run_td0.py" "TD(0) Policy Evaluation"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi

# Q-Learning Experiment
TOTAL_COUNT=$((TOTAL_COUNT + 1))
cat > run_qlearning.py << 'EOF'
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.environments import GridWorld
from experiments.experiments import experiment_q_learning
from utils.visualization import plot_q_learning_analysis, show_q_values
from models.model_utils import save_model, export_results

print("Running Q-Learning Experiment...")
env = GridWorld()
agent, V_optimal, policy, evaluation = experiment_q_learning(env, num_episodes=800)

# Save results
save_model(agent, "qlearning_model.pkl")
results = {
    'algorithm': 'Q-Learning',
    'episode_rewards': agent.episode_rewards,
    'episode_steps': agent.episode_steps,
    'evaluation': evaluation,
    'learned_states': len(V_optimal)
}
export_results(results, "qlearning_results")

print("Q-Learning experiment completed successfully!")
print(f"Success rate: {evaluation['success_rate']*100:.1f}%")
EOF

if run_python_script "run_qlearning.py" "Q-Learning Control"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi

# SARSA Experiment
TOTAL_COUNT=$((TOTAL_COUNT + 1))
cat > run_sarsa.py << 'EOF'
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.environments import GridWorld
from experiments.experiments import experiment_sarsa
from utils.visualization import plot_learning_curve
from models.model_utils import save_model, export_results

print("Running SARSA Experiment...")
env = GridWorld()
agent, V_sarsa, policy, evaluation = experiment_sarsa(env, num_episodes=800)

# Save results
save_model(agent, "sarsa_model.pkl")
results = {
    'algorithm': 'SARSA',
    'episode_rewards': agent.episode_rewards,
    'episode_steps': agent.episode_steps,
    'evaluation': evaluation,
    'learned_states': len(V_sarsa)
}
export_results(results, "sarsa_results")

print("SARSA experiment completed successfully!")
print(f"Success rate: {evaluation['success_rate']*100:.1f}%")
EOF

if run_python_script "run_sarsa.py" "SARSA Control"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi

# 3. Run comparison experiment
TOTAL_COUNT=$((TOTAL_COUNT + 1))
cat > run_comparison.py << 'EOF'
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.environments import GridWorld
from agents.policies import RandomPolicy
from experiments.experiments import experiment_td0, experiment_q_learning, experiment_sarsa
from utils.visualization import compare_algorithms
from evaluation.evaluation import compare_agents
from models.model_utils import export_results, create_summary_report

print("Running Algorithm Comparison Experiment...")
env = GridWorld()

# Run all experiments
print("1. Training TD(0)...")
policy = RandomPolicy(env)
td_agent, V_td = experiment_td0(env, policy, num_episodes=300)

print("2. Training Q-Learning...")
q_agent, V_optimal, q_policy, q_evaluation = experiment_q_learning(env, num_episodes=500)

print("3. Training SARSA...")
sarsa_agent, V_sarsa, sarsa_policy, sarsa_evaluation = experiment_sarsa(env, num_episodes=500)

# Compare algorithms
print("4. Comparing algorithms...")
agents_dict = {
    'Q-Learning': q_agent,
    'SARSA': sarsa_agent
}

comparison_results = compare_agents(agents_dict, env, num_episodes=100)

# Create summary report
all_results = {
    'TD(0)': {
        'avg_reward': sum(td_agent.episode_rewards) / len(td_agent.episode_rewards),
        'total_episodes': len(td_agent.episode_rewards),
        'learned_states': len(V_td)
    },
    'Q-Learning': q_evaluation,
    'SARSA': sarsa_evaluation
}

export_results(all_results, "algorithm_comparison")
create_summary_report(all_results)

print("Algorithm comparison completed successfully!")
EOF

if run_python_script "run_comparison.py" "Algorithm Comparison"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi

# 4. Run exploration strategies experiment
TOTAL_COUNT=$((TOTAL_COUNT + 1))
cat > run_exploration.py << 'EOF'
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.environments import GridWorld
from agents.exploration import ExplorationExperiment, BoltzmannQLearning
from agents.algorithms import QLearningAgent
from models.model_utils import export_results

print("Running Exploration Strategies Experiment...")
env = GridWorld()

# Define exploration strategies
strategies = {
    'epsilon_0.1': {'epsilon': 0.1, 'decay': 1.0},
    'epsilon_0.3': {'epsilon': 0.3, 'decay': 1.0},
    'epsilon_decay_fast': {'epsilon': 0.9, 'decay': 0.99},
    'epsilon_decay_slow': {'epsilon': 0.5, 'decay': 0.995},
    'boltzmann': {'temperature': 2.0}
}

print("Testing exploration strategies:")
for name, params in strategies.items():
    print(f"  • {name}: {params}")

# Run exploration experiment
experiment = ExplorationExperiment(env)
results = experiment.run_exploration_experiment(strategies, num_episodes=300, num_runs=2)

# Save results
exploration_results = {}
for strategy, runs in results.items():
    avg_rewards = [run["evaluation"]["avg_reward"] for run in runs]
    success_rates = [run["evaluation"]["success_rate"] for run in runs]
    
    exploration_results[strategy] = {
        'avg_reward': sum(avg_rewards) / len(avg_rewards),
        'success_rate': sum(success_rates) / len(success_rates),
        'std_reward': (sum([(r - sum(avg_rewards)/len(avg_rewards))**2 for r in avg_rewards]) / len(avg_rewards))**0.5
    }

export_results(exploration_results, "exploration_strategies")

print("Exploration strategies experiment completed successfully!")
EOF

if run_python_script "run_exploration.py" "Exploration Strategies Comparison"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi

# 5. Convert Jupyter notebook to Python (if jupyter is available)
if command -v jupyter &> /dev/null; then
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    if run_notebook "CA3.ipynb" "Main Jupyter Notebook"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
else
    echo "⚠ Jupyter not found, skipping notebook conversion"
fi

# 6. Generate final summary
TOTAL_COUNT=$((TOTAL_COUNT + 1))
cat > generate_summary.py << 'EOF'
import sys
import os
import json
import glob
from datetime import datetime

print("Generating Final Summary Report...")

# Collect all results
results_files = glob.glob("visualizations/*.json")
summary_data = {
    'timestamp': datetime.now().isoformat(),
    'total_experiments': len(results_files),
    'experiments': {}
}

for file_path in results_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        filename = os.path.basename(file_path).replace('.json', '')
        summary_data['experiments'][filename] = data

# Save summary
with open('visualizations/experiment_summary.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

# Create readable report
with open('visualizations/EXPERIMENT_REPORT.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("TEMPORAL DIFFERENCE LEARNING - EXPERIMENT REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Generated on: {datetime.now()}\n")
    f.write(f"Total experiments: {len(results_files)}\n\n")
    
    for filename, data in summary_data['experiments'].items():
        f.write(f"EXPERIMENT: {filename.upper()}\n")
        f.write("-" * 50 + "\n")
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value}\n")
                elif isinstance(value, dict) and 'avg_reward' in value:
                    f.write(f"Average Reward: {value['avg_reward']:.3f}\n")
                    f.write(f"Success Rate: {value.get('success_rate', 0)*100:.1f}%\n")
        
        f.write("\n")
    
    f.write("=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print("Final summary report generated successfully!")
EOF

if run_python_script "generate_summary.py" "Final Summary Generation"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi

# Clean up temporary files
echo ""
echo "🧹 Cleaning up temporary files..."
rm -f run_*.py generate_summary.py

echo "✓ Temporary files cleaned up"

# Final report
echo ""
echo "=" * 80
echo "📊 FINAL EXPERIMENT REPORT"
echo "=" * 80
echo "Completed experiments: $SUCCESS_COUNT / $TOTAL_COUNT"
echo "Success rate: $(( SUCCESS_COUNT * 100 / TOTAL_COUNT ))%"
echo ""
echo "📁 Results saved to:"
echo "   • Visualizations: $VIS_DIR"
echo "   • Models: $MODELS_DIR"
echo "   • Logs: $PROJECT_DIR/logs"
echo ""

if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    echo "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
    echo "✅ Temporal Difference Learning implementation is working perfectly!"
    echo ""
    echo "📋 Generated files:"
    ls -la "$VIS_DIR"
    echo ""
    echo "🔍 Check the visualizations folder for all results and plots"
    exit 0
else
    echo "⚠️  SOME EXPERIMENTS FAILED"
    echo "❌ Please check the log files in the logs directory"
    echo ""
    echo "📋 Available results:"
    if [ -d "$VIS_DIR" ]; then
        ls -la "$VIS_DIR"
    fi
    exit 1
fi
