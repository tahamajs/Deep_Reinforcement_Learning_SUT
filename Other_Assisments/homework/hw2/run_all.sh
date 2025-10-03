#!/bin/bash

# ======================================================================================
# Comprehensive Training and Plotting Script for HW2: Policy Gradients
#
# Author: GitHub Copilot
# Date: October 3, 2025
#
# This script automates the training of policy gradient agents and generates
# insightful plots to compare their performance across different configurations.
#
# Usage:
#   1. Make the script executable: chmod +x run_all_hw2.sh
#   2. Run the script: ./run_all_hw2.sh
#
# The script will:
#   - Create a main results directory to store all outputs.
#   - Train the agent on multiple environments (CartPole, LunarLander, HalfCheetah).
#   - Test various configurations for each environment:
#     - Vanilla Policy Gradient
#     - Reward-to-Go
#     - Neural Network Baseline
#     - Reward-to-Go + Neural Network Baseline
#   - Organize the logs from each run into a structured directory.
#   - Generate and save learning curve plots comparing the configurations for each env.
# ======================================================================================

set -e
echo "🚀 Starting Comprehensive Training for HW2: Policy Gradients 🚀"

# --- MuJoCo Detection ---
echo "🔍 Checking MuJoCo availability..."
MUJOCO_AVAILABLE=false
python -c "import mujoco_py" 2>/dev/null && MUJOCO_AVAILABLE=true

if [ "$MUJOCO_AVAILABLE" = true ]; then
    echo "✅ MuJoCo is available - all environments will be tested"
else
    echo "⚠️  MuJoCo not detected - will skip HalfCheetah-v2 experiments"
    echo "   To enable MuJoCo environments:"
    echo "   1. Download MuJoCo 2.1.0 from: https://github.com/deepmind/mujoco/releases"
    echo "   2. Extract to ~/.mujoco/mujoco210"
    echo "   3. Install mujoco-py: pip install mujoco-py"
    echo "   4. On macOS: brew install gcc"
fi
echo ""

# --- Configuration ---
# Using standard indexed arrays for macOS bash compatibility
if [ "$MUJOCO_AVAILABLE" = true ]; then
    ENVS=("CartPole-v0" "LunarLander-v2" "HalfCheetah-v2")
else
    ENVS=("CartPole-v0" "LunarLander-v2")
fi
declare -a CONFIG_NAMES
declare -a CONFIG_FLAGS
CONFIG_NAMES[0]="Vanilla"
CONFIG_FLAGS[0]=""
CONFIG_NAMES[1]="RTG"
CONFIG_FLAGS[1]="--reward_to_go"
CONFIG_NAMES[2]="Baseline"
CONFIG_FLAGS[2]="--nn_baseline"
CONFIG_NAMES[3]="RTG_Baseline"
CONFIG_FLAGS[3]="--reward_to_go --nn_baseline"

# Training parameters
N_ITER=100
BATCH_SIZE=1000
LEARNING_RATE=0.01

# Directories
RESULTS_DIR="results_hw2"
LOGS_DIR="$RESULTS_DIR/logs"
PLOTS_DIR="$RESULTS_DIR/plots"

# --- Setup ---
echo "🔧 Setting up directories..."
mkdir -p "$LOGS_DIR"
mkdir -p "$PLOTS_DIR"
echo "Directories created in '$RESULTS_DIR'."

# --- Training Phase ---
echo "🏋️  Starting Training Phase..."

for env_name in "${ENVS[@]}"; do
    echo "--------------------------------------------------"
    echo "Training on Environment: $env_name"
    echo "--------------------------------------------------"

    # Adjust params for more complex environments
    if [[ "$env_name" == "LunarLander-v2" ]]; then
        BATCH_SIZE=5000
        LEARNING_RATE=0.02
    elif [[ "$env_name" == "HalfCheetah-v2" ]]; then
        BATCH_SIZE=30000
        N_ITER=150
        LEARNING_RATE=0.005
    else # CartPole
        BATCH_SIZE=1000
        N_ITER=100
        LEARNING_RATE=0.01
    fi

    for i in "${!CONFIG_NAMES[@]}"; do
        config_name=${CONFIG_NAMES[$i]}
        config_flags=${CONFIG_FLAGS[$i]}
        exp_name="${env_name}_${config_name}"
        log_dir_base="data_hw2" # Base directory for logz to write to

        echo "▶️  Running Config: $config_name"
        echo "   Command: python run_pg.py $env_name --exp_name $exp_name -n $N_ITER -b $BATCH_SIZE -lr $LEARNING_RATE $config_flags --record_video"

        # The run_pg.py script uses logz, which creates its own timestamped directory.
        # We will run it and then move the results to our organized structure.
        # Added --record_video flag to capture before/after videos
        python run_pg.py "$env_name" --exp_name "$exp_name" -n "$N_ITER" -b "$BATCH_SIZE" -lr "$LEARNING_RATE" $config_flags --seed 1 --record_video

        # Find the latest directory created by logz for this experiment
        # The logz script creates directories like: data_hw2/CartPole-v0_Vanilla_...
        latest_logz_dir=$(find "$log_dir_base" -type d -name "${exp_name}*" -print0 | xargs -0 ls -td | head -n1)

        if [[ -d "$latest_logz_dir" ]]; then
            # Move the specific log directory into our organized structure
            mv "$latest_logz_dir" "$LOGS_DIR/"
            echo "✅ Logs saved to $LOGS_DIR/$(basename "$latest_logz_dir")"
        else
            echo "⚠️  Warning: Could not find log directory for experiment '$exp_name'. Skipping."
        fi
        echo "----------------------------------------"
    done
done

# Clean up the base log directory if it's empty
if [ -d "data_hw2" ] && [ -z "$(ls -A data_hw2)" ]; then
    rmdir "data_hw2"
fi

echo "✅ Training Phase Completed!"

# --- Plotting Phase ---
echo "📊 Starting Plotting Phase..."

for env_name in "${ENVS[@]}"; do
    echo "Generating plot for $env_name..."
    
    logdirs_for_plot=()
    legend_titles=()
    
    for i in "${!CONFIG_NAMES[@]}"; do
        config_name=${CONFIG_NAMES[$i]}
        exp_name_prefix="${env_name}_${config_name}"
        
        # Find the corresponding log directory
        logdir=$(find "$LOGS_DIR" -type d -name "${exp_name_prefix}*")
        
        if [[ -n "$logdir" ]]; then
            logdirs_for_plot+=("$logdir")
            legend_titles+=("$config_name")
        fi
    done

    if (( ${#logdirs_for_plot[@]} > 0 )); then
        logdirs_arg=$(IFS=':'; echo "${logdirs_for_plot[*]}")
        legends_arg=$(IFS=':'; echo "${legend_titles[*]}")
        # Use a Python script to generate and save the plot
        python -c "
import sys
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

def get_datasets(fpath, condition):
    unit = 0
    datasets = []
    for root, _, files in os.walk(fpath):
        if 'log.txt' in files:
            log_path = os.path.join(root, 'log.txt')
            try:
                experiment_data = pd.read_table(log_path)
                experiment_data.insert(len(experiment_data.columns), 'Unit', unit)
                experiment_data.insert(len(experiment_data.columns), 'Condition', condition)
                datasets.append(experiment_data)
                unit += 1
            except Exception as e:
                print(f'Could not read {log_path}: {e}')
    return datasets

def plot_and_save(data, value, title, save_path):
    if not data:
        print('No data to plot.')
        return
    
    data_df = pd.concat(data, ignore_index=True)
    plt.figure(figsize=(12, 8))
    sns.set_theme(style='darkgrid', font_scale=1.4)
    sns.lineplot(
        data=data_df,
        x='Iteration',
        y=value,
        hue='Condition',
        estimator='mean',
        ci='sd',
        marker='o'
    )
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(value)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Plot saved to {save_path}')

if __name__ == '__main__':
    logdirs = [p for p in sys.argv[1].split(':') if p]
    legends = [l for l in sys.argv[2].split(':') if l]
    env_name = sys.argv[3]
    save_dir = sys.argv[4]

    all_data = []
    for logdir, legend in zip(logdirs, legends):
        all_data.extend(get_datasets(logdir, legend))
    
    plot_and_save(
        all_data,
        value='AverageReturn',
        title=f'Performance on {env_name}',
        save_path=os.path.join(save_dir, f'{env_name}_performance.png')
    )
" "$logdirs_arg" "$legends_arg" "$env_name" "$PLOTS_DIR"

    else
        echo "⚠️ No log directories found for $env_name to plot."
    fi
done

echo "✅ Plotting Phase Completed!"
echo ""
echo "🎉 All tasks finished!"
echo "=================================================="
echo "Results saved to '$RESULTS_DIR' directory:"
echo "  📊 Training logs:  $LOGS_DIR"
echo "  📈 Plots:          $PLOTS_DIR"
echo "  🎬 Videos:         results_hw2/videos/"
echo "=================================================="
if [ "$MUJOCO_AVAILABLE" = false ]; then
    echo ""
    echo "⚠️  Note: HalfCheetah-v2 experiments were skipped (MuJoCo not installed)"
fi
echo ""
echo "To view videos, navigate to results_hw2/videos/<env>/<config>/before|after/"
