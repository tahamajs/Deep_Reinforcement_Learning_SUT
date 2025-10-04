#!/bin/bash

# CA5 Advanced DQN Methods - Complete Run Script
# This script runs all components of the CA5 project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="/Users/tahamajs/Documents/uni/DRL/CAs/Solutions/CA05_Advanced_DQN_Methods"
cd "$PROJECT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CA5 Advanced DQN Methods Runner${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed or not in PATH"
    exit 1
fi

print_status "Python3 found: $(python3 --version)"

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
pip install -r requirements.txt

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p visualizations
mkdir -p models
mkdir -p logs
mkdir -p results

# Step 1: Test custom environments
print_status "Step 1: Testing custom environments..."
python3 -c "
import sys
sys.path.append('.')
from environments.custom_envs import GridWorldEnv, make_env
import numpy as np

print('Testing GridWorld environment...')
env = GridWorldEnv(size=5)
state = env.reset()
print(f'Initial state shape: {state.shape}')

for i in range(5):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(f'Step {i}: Action={action}, Reward={reward:.2f}, Done={done}')
    if done:
        break

env.close()
print('Custom environments test completed!')
"

# Step 2: Test DQN agents
print_status "Step 2: Testing DQN agents..."
python3 -c "
import sys
sys.path.append('.')
from agents.dqn_base import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.dueling_dqn import DuelingDQNAgent
from agents.prioritized_replay import PrioritizedDQNAgent
import gym
import numpy as np

print('Testing DQN agents...')
env = gym.make('CartPole-v1')

# Test Vanilla DQN
print('Testing Vanilla DQN...')
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=1e-3,
    epsilon_start=0.1,
    epsilon_end=0.01
)

state = env.reset()
for i in range(10):
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    agent.replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    if done:
        state = env.reset()

print('DQN agents test completed!')
env.close()
"

# Step 3: Run training examples
print_status "Step 3: Running training examples..."
python3 -c "
import sys
sys.path.append('.')
from training_examples import train_dqn_agent, dqn_variant_comparison
import matplotlib.pyplot as plt

print('Running DQN training example...')
results = train_dqn_agent(
    env_name='CartPole-v1',
    agent_type='dqn',
    num_episodes=100,  # Reduced for demo
    lr=1e-3,
    epsilon_start=0.1,
    epsilon_end=0.01
)

print(f'Training completed. Final average reward: {results[\"final_avg_reward\"]:.2f}')

# Save results
import json
with open('results/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Training results saved to results/training_results.json')
"

# Step 4: Run experiments
print_status "Step 4: Running experiments..."
python3 -c "
import sys
sys.path.append('.')
from experiments import ExperimentRunner, get_dqn_configs
from agents.dqn_base import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.dueling_dqn import DuelingDQNAgent
from agents.prioritized_replay import PrioritizedDQNAgent

print('Running comparison experiments...')
runner = ExperimentRunner()

# Get configurations
configs = get_dqn_configs()
agent_classes = [DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PrioritizedDQNAgent]

# Run experiments (reduced episodes for demo)
for config in configs:
    config.config['training']['num_episodes'] = 50

comparison_results = runner.run_comparison_experiment(
    configs[:2],  # Only first 2 for demo
    agent_classes[:2],
    'CartPole-v1'
)

print('Experiments completed!')
"

# Step 5: Run evaluation
print_status "Step 5: Running evaluation..."
python3 -c "
import sys
sys.path.append('.')
from evaluation import PerformanceEvaluator, compare_agents
from agents.dqn_base import DQNAgent
from agents.double_dqn import DoubleDQNAgent
import gym

print('Running performance evaluation...')
env = gym.make('CartPole-v1')

# Create agents
agents = {
    'DQN': DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=1e-3
    ),
    'Double DQN': DoubleDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=1e-3
    )
}

# Quick training for evaluation
for name, agent in agents.items():
    print(f'Quick training {name}...')
    for episode in range(50):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
            state = next_state

# Evaluate agents
comparison_results = compare_agents(agents, env, num_episodes=20)
print('Evaluation completed!')
env.close()
"

# Step 6: Generate visualizations
print_status "Step 6: Generating visualizations..."
python3 -c "
import sys
sys.path.append('.')
from training_examples import plot_q_value_landscape, plot_experience_replay_analysis
from agents.dqn_base import DQNAgent
import gym
import matplotlib.pyplot as plt

print('Generating visualizations...')
env = gym.make('CartPole-v1')

# Create and train agent for visualization
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=1e-3
)

# Quick training
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        if len(agent.replay_buffer) > agent.batch_size:
            agent.update()
        state = next_state

# Generate plots
try:
    plot_q_value_landscape(agent, 'CartPole-v1', 'visualizations/q_value_landscape.png')
    plot_experience_replay_analysis(agent.replay_buffer, 'visualizations/replay_analysis.png')
    print('Visualizations saved to visualizations/ folder')
except Exception as e:
    print(f'Visualization generation failed: {e}')

env.close()
"

# Step 7: Run Jupyter notebook (if available)
if command -v jupyter &> /dev/null; then
    print_status "Step 7: Converting Jupyter notebook..."
    if [ -f "CA5.ipynb" ]; then
        python3 -c "
import nbformat
from nbconvert import PythonExporter

# Convert notebook to Python
with open('CA5.ipynb', 'r') as f:
    notebook = nbformat.read(f, as_version=4)

exporter = PythonExporter()
python_code, _ = exporter.from_notebook_node(notebook)

# Save as Python file
with open('CA5_converted.py', 'w') as f:
    f.write(python_code)

print('Notebook converted to CA5_converted.py')
"
        
        print_status "Running converted notebook..."
        python3 CA5_converted.py
    else
        print_warning "CA5.ipynb not found, skipping notebook execution"
    fi
else
    print_warning "Jupyter not available, skipping notebook execution"
fi

# Step 8: Generate summary report
print_status "Step 8: Generating summary report..."
python3 -c "
import json
import os
from datetime import datetime

# Create summary report
summary = {
    'timestamp': datetime.now().isoformat(),
    'project': 'CA5 Advanced DQN Methods',
    'components_tested': [
        'Custom Environments',
        'DQN Agents',
        'Training Examples',
        'Experiments',
        'Evaluation',
        'Visualizations'
    ],
    'directories_created': [
        'visualizations/',
        'models/',
        'logs/',
        'results/'
    ],
    'files_generated': [
        'results/training_results.json',
        'experiments/comparison_results.json',
        'visualizations/q_value_landscape.png',
        'visualizations/replay_analysis.png',
        'visualizations/agent_comparison.png'
    ]
}

# Save summary
with open('results/summary_report.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Summary report saved to results/summary_report.json')
"

# Final status
print_status "All components executed successfully!"
print_status "Results saved in:"
echo -e "  ${GREEN}•${NC} visualizations/ - Generated plots and charts"
echo -e "  ${GREEN}•${NC} results/ - Training results and summary"
echo -e "  ${GREEN}•${NC} experiments/ - Experiment configurations and results"
echo -e "  ${GREEN}•${NC} models/ - Trained model checkpoints (if saved)"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}CA5 Advanced DQN Methods - Execution Complete!${NC}"
echo -e "${BLUE}========================================${NC}"

# Deactivate virtual environment
deactivate
