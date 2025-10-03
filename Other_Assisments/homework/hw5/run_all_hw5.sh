#!/bin/bash

# ======================================================================================
# Comprehensive Training Script for HW5: Exploration, SAC, and Meta-Learning
#
# Author: GitHub Copilot
# Date: October 3, 2025
#
# This script automates training of advanced RL algorithms including:
# - Soft Actor-Critic (SAC)
# - Exploration methods with density models
# - Meta-learning algorithms
#
# Usage:
#   1. Make executable: chmod +x run_all_hw5.sh
#   2. Run: ./run_all_hw5.sh
#
# The script will:
#   - Run SAC experiments on continuous control tasks
#   - Run exploration experiments with bonus rewards
#   - Run meta-learning experiments for few-shot adaptation
#   - Generate comparison plots
#   - Organize all results in results_hw5/
#
# Note: Some environments require MuJoCo. Script will skip if unavailable.
# ======================================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "🚀 Starting Comprehensive Training for HW5: Exploration, SAC, Meta-Learning 🚀"

# --- Configuration ---
RESULTS_DIR="results_hw5"
LOGS_DIR="$RESULTS_DIR/logs"
PLOTS_DIR="$RESULTS_DIR/plots"
DATA_DIR="$SCRIPT_DIR/data"

# --- Setup ---
echo "🔧 Setting up directories..."
mkdir -p "$LOGS_DIR"
mkdir -p "$PLOTS_DIR"
mkdir -p "$DATA_DIR"
echo "Directories created in '$RESULTS_DIR'."

# --- Check MuJoCo availability ---
echo ""
echo "🔍 Checking MuJoCo availability..."
MUJOCO_AVAILABLE=false

if python3 -c "import mujoco_py" 2>/dev/null; then
    echo "✅ MuJoCo (mujoco-py) is installed and available"
    MUJOCO_AVAILABLE=true
else
    echo "⚠️  MuJoCo (mujoco-py) not found."
    echo "   To enable MuJoCo environments:"
    echo "   1. Install MuJoCo binaries (mujoco.org)"
    echo "   2. Install mujoco-py: pip install mujoco-py"
    echo "   3. On macOS: brew install gcc --without-multilib"
    echo ""
    echo "   Proceeding with non-MuJoCo environments only."
fi

# ======================================================================================
# Part 1: Soft Actor-Critic (SAC)
# ======================================================================================
echo ""
echo "========================================"
echo "Part 1: Soft Actor-Critic (SAC)"
echo "========================================"

# Non-MuJoCo environment: Pendulum
SAC_ENV_BASIC="Pendulum-v0"
SAC_STEPS_BASIC=50000

echo ""
echo "--- SAC on $SAC_ENV_BASIC (Basic Environment) ---"
EXP_NAME_SAC_BASIC="sac_${SAC_ENV_BASIC}_$(date +%s)"

echo "▶️  Running SAC on $SAC_ENV_BASIC..."
echo "   Command: python run_hw5.py sac --env_name $SAC_ENV_BASIC --total_steps $SAC_STEPS_BASIC"

python run_hw5.py sac \
    --env_name "$SAC_ENV_BASIC" \
    --exp_name "$EXP_NAME_SAC_BASIC" \
    --total_steps "$SAC_STEPS_BASIC" \
    --batch_size 256 \
    --learning_rate 3e-3 \
    2>&1 | tee "$LOGS_DIR/sac_${SAC_ENV_BASIC}.log"

# Move results
if [[ -d "$DATA_DIR/$EXP_NAME_SAC_BASIC" ]]; then
    cp -r "$DATA_DIR/$EXP_NAME_SAC_BASIC" "$LOGS_DIR/"
    echo "✅ SAC on $SAC_ENV_BASIC logs saved"
fi

# MuJoCo environments (if available)
if [ "$MUJOCO_AVAILABLE" = true ]; then
    echo ""
    echo "--- SAC on HalfCheetah-v2 (MuJoCo Environment) ---"
    SAC_ENV_MUJOCO="HalfCheetah-v2"
    SAC_STEPS_MUJOCO=100000
    EXP_NAME_SAC_MUJOCO="sac_${SAC_ENV_MUJOCO}_$(date +%s)"
    
    echo "▶️  Running SAC on $SAC_ENV_MUJOCO..."
    python run_hw5.py sac \
        --env_name "$SAC_ENV_MUJOCO" \
        --exp_name "$EXP_NAME_SAC_MUJOCO" \
        --total_steps "$SAC_STEPS_MUJOCO" \
        --batch_size 256 \
        2>&1 | tee "$LOGS_DIR/sac_${SAC_ENV_MUJOCO}.log"
    
    if [[ -d "$DATA_DIR/$EXP_NAME_SAC_MUJOCO" ]]; then
        cp -r "$DATA_DIR/$EXP_NAME_SAC_MUJOCO" "$LOGS_DIR/"
        echo "✅ SAC on $SAC_ENV_MUJOCO logs saved"
    fi
else
    echo "⚠️  Skipping MuJoCo SAC experiments (MuJoCo not available)"
fi

echo "✅ SAC experiments completed!"

# ======================================================================================
# Part 2: Exploration with Density Models
# ======================================================================================
echo ""
echo "========================================"
echo "Part 2: Exploration Methods"
echo "========================================"

# Non-MuJoCo environment: MountainCar
EXPLORE_ENV_BASIC="MountainCar-v0"
EXPLORE_STEPS_BASIC=50000

echo ""
echo "--- Exploration on $EXPLORE_ENV_BASIC ---"
EXP_NAME_EXPLORE_BASIC="exploration_${EXPLORE_ENV_BASIC}_$(date +%s)"

echo "▶️  Running exploration on $EXPLORE_ENV_BASIC..."
echo "   Command: python run_hw5.py exploration --env_name $EXPLORE_ENV_BASIC --total_steps $EXPLORE_STEPS_BASIC"

python run_hw5.py exploration \
    --env_name "$EXPLORE_ENV_BASIC" \
    --exp_name "$EXP_NAME_EXPLORE_BASIC" \
    --total_steps "$EXPLORE_STEPS_BASIC" \
    --bonus_coeff 0.1 \
    --initial_rollouts 10 \
    2>&1 | tee "$LOGS_DIR/exploration_${EXPLORE_ENV_BASIC}.log"

# Move results
if [[ -d "$DATA_DIR/$EXP_NAME_EXPLORE_BASIC" ]]; then
    cp -r "$DATA_DIR/$EXP_NAME_EXPLORE_BASIC" "$LOGS_DIR/"
    echo "✅ Exploration on $EXPLORE_ENV_BASIC logs saved"
fi

# MuJoCo sparse environment (if available)
if [ "$MUJOCO_AVAILABLE" = true ]; then
    echo ""
    echo "--- Exploration on SparseHalfCheetah (MuJoCo) ---"
    EXPLORE_ENV_MUJOCO="SparseHalfCheetah-v0"
    EXPLORE_STEPS_MUJOCO=100000
    EXP_NAME_EXPLORE_MUJOCO="exploration_${EXPLORE_ENV_MUJOCO}_$(date +%s)"
    
    echo "▶️  Running exploration on $EXPLORE_ENV_MUJOCO..."
    python run_hw5.py exploration \
        --env_name "$EXPLORE_ENV_MUJOCO" \
        --exp_name "$EXP_NAME_EXPLORE_MUJOCO" \
        --total_steps "$EXPLORE_STEPS_MUJOCO" \
        --bonus_coeff 1.0 \
        2>&1 | tee "$LOGS_DIR/exploration_${EXPLORE_ENV_MUJOCO}.log" || echo "⚠️  Sparse environment may not be registered"
    
    if [[ -d "$DATA_DIR/$EXP_NAME_EXPLORE_MUJOCO" ]]; then
        cp -r "$DATA_DIR/$EXP_NAME_EXPLORE_MUJOCO" "$LOGS_DIR/"
        echo "✅ Exploration on $EXPLORE_ENV_MUJOCO logs saved"
    fi
else
    echo "⚠️  Skipping MuJoCo exploration experiments (MuJoCo not available)"
fi

echo "✅ Exploration experiments completed!"

# ======================================================================================
# Part 3: Meta-Learning
# ======================================================================================
echo ""
echo "========================================"
echo "Part 3: Meta-Learning"
echo "========================================"

# Non-MuJoCo environment: CartPole
META_ENV_BASIC="CartPole-v0"
META_TASKS=20
META_STEPS=100

echo ""
echo "--- Meta-Learning on $META_ENV_BASIC ---"
EXP_NAME_META_BASIC="meta_${META_ENV_BASIC}_$(date +%s)"

echo "▶️  Running meta-learning on $META_ENV_BASIC..."
echo "   Command: python run_hw5.py meta --env_name $META_ENV_BASIC --num_tasks $META_TASKS --meta_steps $META_STEPS"

python run_hw5.py meta \
    --env_name "$META_ENV_BASIC" \
    --exp_name "$EXP_NAME_META_BASIC" \
    --num_tasks "$META_TASKS" \
    --meta_steps "$META_STEPS" \
    --meta_batch_size 4 \
    --adaptation_steps 5 \
    2>&1 | tee "$LOGS_DIR/meta_${META_ENV_BASIC}.log"

# Move results
if [[ -d "$DATA_DIR/$EXP_NAME_META_BASIC" ]]; then
    cp -r "$DATA_DIR/$EXP_NAME_META_BASIC" "$LOGS_DIR/"
    echo "✅ Meta-learning on $META_ENV_BASIC logs saved"
fi

echo "✅ Meta-learning experiments completed!"

# ======================================================================================
# Part 4: Generate Plots
# ======================================================================================
echo ""
echo "📊 Starting Plotting Phase..."

# Plot SAC results
echo "Generating plots for SAC experiments..."
for exp_dir in "$LOGS_DIR"/sac_*; do
    if [[ -d "$exp_dir" ]] && [[ -f "$exp_dir/episode_rewards.npy" ]]; then
        exp_name=$(basename "$exp_dir")
        echo "  Plotting $exp_name..."
        
        python -c "
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

exp_dir = sys.argv[1]
plot_path = sys.argv[2]

try:
    rewards = np.load(f'{exp_dir}/episode_rewards.npy')
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.6, label='Episode Reward')
    
    # Smoothed curve
    window = min(100, len(rewards) // 10)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed, linewidth=2, label='Smoothed')
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('SAC Training Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f'Saved plot to {plot_path}')
except Exception as e:
    print(f'Error plotting: {e}')
" "$exp_dir" "$PLOTS_DIR/${exp_name}_performance.png"
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
echo "  💾 Raw data:       $DATA_DIR"
echo "=================================================="
echo ""
echo "Key Results:"
if [ "$MUJOCO_AVAILABLE" = true ]; then
    echo "  - SAC trained on Pendulum-v0 and HalfCheetah-v2"
    echo "  - Exploration tested on MountainCar-v0 and sparse environments"
else
    echo "  - SAC trained on Pendulum-v0"
    echo "  - Exploration tested on MountainCar-v0"
    echo "  - (MuJoCo experiments skipped - install MuJoCo to enable)"
fi
echo "  - Meta-learning trained on CartPole-v0"
echo ""
echo "To view plots:"
echo "  open $PLOTS_DIR/*.png"
echo ""
echo "To re-run individual experiments:"
echo "  SAC:         python run_hw5.py sac --env_name Pendulum-v0 --total_steps 50000"
echo "  Exploration: python run_hw5.py exploration --env_name MountainCar-v0"
echo "  Meta:        python run_hw5.py meta --env_name CartPole-v0 --num_tasks 20"
