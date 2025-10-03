#!/bin/bash

# ======================================================================================
# Comprehensive Training Script for HW3: DQN and Actor-Critic
#
# Author: GitHub Copilot
# Date: October 3, 2025
#
# This script automates training of DQN and Actor-Critic agents across various
# environments and generates comparison plots and videos.
#
# Usage:
#   1. Make executable: chmod +x run_all_hw3.sh
#   2. Run: ./run_all_hw3.sh
#
# The script will:
#   - Train DQN agents on LunarLander-v2
#   - Train Actor-Critic agents on CartPole-v0, InvertedPendulum-v2
#   - Record before/after videos for visual comparison
#   - Generate learning curve plots
#   - Organize all results in results_hw3/
# ======================================================================================

set -e
echo "🚀 Starting Comprehensive Training for HW3: DQN & Actor-Critic 🚀"

# --- Configuration ---
RESULTS_DIR="results_hw3"
LOGS_DIR="$RESULTS_DIR/logs"
PLOTS_DIR="$RESULTS_DIR/plots"
VIDEOS_DIR="$RESULTS_DIR/videos"

# --- Setup ---
echo "🔧 Setting up directories..."
mkdir -p "$LOGS_DIR"
mkdir -p "$PLOTS_DIR"
mkdir -p "$VIDEOS_DIR"
echo "Directories created in '$RESULTS_DIR'."

# ======================================================================================
# Part 1: DQN Training on LunarLander
# ======================================================================================
echo ""
echo "========================================"
echo "Part 1: Training DQN on LunarLander-v2"
echo "========================================"

DQN_ENV="LunarLander-v2"
DQN_TIMESTEPS=50000
DQN_SEED=1

echo "▶️  Training DQN agent..."
echo "   Environment: $DQN_ENV"
echo "   Timesteps: $DQN_TIMESTEPS"

python run_dqn_lander.py "$DQN_ENV" --num_timesteps $DQN_TIMESTEPS --seed $DQN_SEED 2>&1 | tee "$LOGS_DIR/dqn_lander_training.log"

echo "✅ DQN training completed!"

# ======================================================================================
# Part 2: Actor-Critic Training
# ======================================================================================
echo ""
echo "=================================================="
echo "Part 2: Training Actor-Critic on Multiple Envs"
echo "=================================================="

# Actor-Critic environments and configs
AC_ENVS=("CartPole-v0" "InvertedPendulum-v2")
AC_N_ITER=100
AC_BATCH_SIZE=1000
AC_LR=0.005

for env_name in "${AC_ENVS[@]}"; do
    echo "--------------------------------------------------"
    echo "Training Actor-Critic on: $env_name"
    echo "--------------------------------------------------"
    
    # Adjust parameters for different environments
    if [[ "$env_name" == "InvertedPendulum-v2" ]]; then
        AC_BATCH_SIZE=5000
        AC_N_ITER=150
    else
        AC_BATCH_SIZE=1000
        AC_N_ITER=100
    fi
    
    exp_name="ac_${env_name}"
    log_dir_base="data_hw3"
    
    echo "▶️  Running Actor-Critic..."
    echo "   Command: python run_ac.py $env_name --exp_name $exp_name -n $AC_N_ITER -b $AC_BATCH_SIZE -lr $AC_LR"
    
    python run_ac.py "$env_name" --exp_name "$exp_name" -n "$AC_N_ITER" -b "$AC_BATCH_SIZE" -lr "$AC_LR" --seed 1
    
    # Move logs to organized structure
    latest_ac_dir=$(find "$log_dir_base" -type d -name "${exp_name}*" -print0 2>/dev/null | xargs -0 ls -td 2>/dev/null | head -n1)
    
    if [[ -d "$latest_ac_dir" ]]; then
        mv "$latest_ac_dir" "$LOGS_DIR/"
        echo "✅ Logs saved to $LOGS_DIR/$(basename "$latest_ac_dir")"
    else
        echo "⚠️  Warning: Could not find log directory for '$exp_name'"
    fi
    
    echo "----------------------------------------"
done

# Clean up base log directory
if [ -d "data_hw3" ] && [ -z "$(ls -A data_hw3 2>/dev/null)" ]; then
    rmdir "data_hw3" 2>/dev/null || true
fi

echo "✅ Actor-Critic training completed!"

# ======================================================================================
# Part 3: Generate Plots
# ======================================================================================
echo ""
echo "📊 Starting Plotting Phase..."

# Plot Actor-Critic results
for env_name in "${AC_ENVS[@]}"; do
    echo "Generating plot for Actor-Critic on $env_name..."
    
    ac_logdir=$(find "$LOGS_DIR" -type d -name "ac_${env_name}*" | head -n1)
    
    if [[ -n "$ac_logdir" ]]; then
        python -c "
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def get_datasets(fpath):
    datasets = []
    for root, _, files in os.walk(fpath):
        if 'log.txt' in files:
            log_path = os.path.join(root, 'log.txt')
            try:
                data = pd.read_table(log_path)
                datasets.append(data)
            except Exception as e:
                print(f'Error reading {log_path}: {e}')
    return datasets

def plot_ac(data_list, env_name, save_path):
    if not data_list:
        print('No data to plot')
        return
    
    data = pd.concat(data_list, ignore_index=True)
    plt.figure(figsize=(12, 8))
    sns.set_theme(style='darkgrid', font_scale=1.4)
    
    if 'Iteration' in data.columns and 'AverageReturn' in data.columns:
        sns.lineplot(data=data, x='Iteration', y='AverageReturn', marker='o')
        plt.title(f'Actor-Critic Performance on {env_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Average Return')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f'Plot saved to {save_path}')
    else:
        print(f'Required columns not found in data for {env_name}')

if __name__ == '__main__':
    logdir = sys.argv[1]
    env_name = sys.argv[2]
    save_path = sys.argv[3]
    
    datasets = get_datasets(logdir)
    plot_ac(datasets, env_name, save_path)
" "$ac_logdir" "$env_name" "$PLOTS_DIR/ac_${env_name}_performance.png"
    else
        echo "⚠️  No logs found for Actor-Critic on $env_name"
    fi
done

echo "✅ Plotting Phase Completed!"

# ======================================================================================
# Summary
# ======================================================================================
echo ""
echo "🎉 All Training Completed!"
echo "=================================================="
echo "Results Summary:"
echo "  📊 Training logs:  $LOGS_DIR"
echo "  📈 Plots:          $PLOTS_DIR"
echo "  🎬 Videos:         $VIDEOS_DIR"
echo "=================================================="
echo ""
echo "Key Results:"
echo "  - DQN trained on LunarLander-v2 for $DQN_TIMESTEPS timesteps"
echo "  - Actor-Critic trained on ${#AC_ENVS[@]} environments"
echo ""
echo "To view plots:"
echo "  open $PLOTS_DIR/*.png"
echo ""
echo "To re-run individual experiments:"
echo "  DQN:           python run_dqn_lander.py LunarLander-v2 --num_timesteps 50000"
echo "  Actor-Critic:  python run_ac.py CartPole-v0 -n 100 -b 1000"
