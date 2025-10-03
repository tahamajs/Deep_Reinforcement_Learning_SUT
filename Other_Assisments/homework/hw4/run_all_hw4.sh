#!/bin/bash

# ======================================================================================
# Comprehensive Training Script for HW4: Model-Based Reinforcement Learning
#
# Author: GitHub Copilot
# Date: October 3, 2025
#
# This script automates training of model-based RL algorithms across various
# configurations and generates comparison plots.
#
# Usage:
#   1. Make executable: chmod +x run_all_hw4.sh
#   2. Run: ./run_all_hw4.sh
#
# The script will:
#   - Run Q1: Dynamics model training and prediction evaluation
#   - Run Q2: MPC with random shooting
#   - Run Q3: On-policy MBRL with various hyperparameters
#   - Generate comparison plots
#   - Organize all results in results_hw4/
#
# Note: Requires MuJoCo and mujoco-py. If not installed, the script will skip runs.
# ======================================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "🚀 Starting Comprehensive Training for HW4: Model-Based RL 🚀"

# --- Configuration ---
RESULTS_DIR="results_hw4"
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
    echo "⚠️  MuJoCo not available — running with Pendulum-v0 instead."
    echo "   (Results may differ from HalfCheetah. Install MuJoCo for full experiments.)"
    exit 0
fi

# ======================================================================================
# Part 1: Q1 - Dynamics Model Training and Evaluation
# ======================================================================================
echo ""
echo "========================================"
echo "Part 1: Q1 - Dynamics Model Training"
echo "========================================"

EXP_NAME_Q1="q1_dynamics_$(date +%s)"

echo "▶️  Running Q1..."
echo "   Command: python run_mbrl.py q1 --exp_name $EXP_NAME_Q1"

python run_mbrl.py q1 --exp_name "$EXP_NAME_Q1" 2>&1 | tee "$LOGS_DIR/q1_training.log"

# Move results
if [[ -d "$DATA_DIR/$EXP_NAME_Q1" ]]; then
    cp -r "$DATA_DIR/$EXP_NAME_Q1" "$LOGS_DIR/"
    echo "✅ Q1 logs and plots saved to $LOGS_DIR/$EXP_NAME_Q1"
fi

echo "✅ Q1 completed!"

# ======================================================================================
# Part 2: Q2 - MPC with Random Shooting
# ======================================================================================
echo ""
echo "========================================"
echo "Part 2: Q2 - MPC with Random Shooting"
echo "========================================"

EXP_NAME_Q2="q2_mpc_$(date +%s)"

echo "▶️  Running Q2..."
echo "   Command: python run_mbrl.py q2 --exp_name $EXP_NAME_Q2"

python run_mbrl.py q2 --exp_name "$EXP_NAME_Q2" 2>&1 | tee "$LOGS_DIR/q2_training.log"

# Move results
if [[ -d "$DATA_DIR/$EXP_NAME_Q2" ]]; then
    cp -r "$DATA_DIR/$EXP_NAME_Q2" "$LOGS_DIR/"
    echo "✅ Q2 logs saved to $LOGS_DIR/$EXP_NAME_Q2"
fi

echo "✅ Q2 completed!"

# ======================================================================================
# Part 3: Q3 - On-policy MBRL
# ======================================================================================
echo ""
echo "=================================================="
echo "Part 3: Q3 - On-policy Model-Based RL"
echo "=================================================="

# Q3a: Default configuration
echo ""
echo "--- Q3a: Default Configuration ---"
EXP_NAME_Q3A="HalfCheetah_q3_default"

echo "▶️  Running Q3 with default settings..."
python main.py q3 --exp_name default 2>&1 | tee "$LOGS_DIR/q3a_default.log"

if [[ -d "$DATA_DIR/$EXP_NAME_Q3A" ]]; then
    cp -r "$DATA_DIR/$EXP_NAME_Q3A" "$LOGS_DIR/"
    echo "✅ Q3a logs saved"
fi

# Generate plot
if [[ -f "$DATA_DIR/$EXP_NAME_Q3A/log.csv" ]]; then
    python plot.py --exps "$EXP_NAME_Q3A" --save HalfCheetah_q3_default
    if [[ -f "plots/HalfCheetah_q3_default.jpg" ]]; then
        mv plots/HalfCheetah_q3_default.jpg "$PLOTS_DIR/"
        echo "✅ Q3a plot saved to $PLOTS_DIR/HalfCheetah_q3_default.jpg"
    fi
fi

# Q3b: Vary number of random action selections
echo ""
echo "--- Q3b: Varying Random Action Selections ---"
for num_actions in 128 4096 16384; do
    exp_name="HalfCheetah_q3_action${num_actions}"
    echo "▶️  Running with $num_actions random actions..."
    python main.py q3 --exp_name "action${num_actions}" --num_random_action_selection "$num_actions" 2>&1 | tee "$LOGS_DIR/q3b_action${num_actions}.log"
    
    if [[ -d "$DATA_DIR/$exp_name" ]]; then
        cp -r "$DATA_DIR/$exp_name" "$LOGS_DIR/"
    fi
done

# Plot action comparison
python plot.py --exps HalfCheetah_q3_action128 HalfCheetah_q3_action4096 HalfCheetah_q3_action16384 --save HalfCheetah_q3_actions 2>/dev/null || true
if [[ -f "plots/HalfCheetah_q3_actions.jpg" ]]; then
    mv plots/HalfCheetah_q3_actions.jpg "$PLOTS_DIR/"
    echo "✅ Action comparison plot saved"
fi

# Q3c: Vary MPC horizon
echo ""
echo "--- Q3c: Varying MPC Horizon ---"
for horizon in 10 15 20; do
    exp_name="HalfCheetah_q3_horizon${horizon}"
    echo "▶️  Running with horizon $horizon..."
    python main.py q3 --exp_name "horizon${horizon}" --mpc_horizon "$horizon" 2>&1 | tee "$LOGS_DIR/q3c_horizon${horizon}.log"
    
    if [[ -d "$DATA_DIR/$exp_name" ]]; then
        cp -r "$DATA_DIR/$exp_name" "$LOGS_DIR/"
    fi
done

# Plot horizon comparison
python plot.py --exps HalfCheetah_q3_horizon10 HalfCheetah_q3_horizon15 HalfCheetah_q3_horizon20 --save HalfCheetah_q3_mpc_horizon 2>/dev/null || true
if [[ -f "plots/HalfCheetah_q3_mpc_horizon.jpg" ]]; then
    mv plots/HalfCheetah_q3_mpc_horizon.jpg "$PLOTS_DIR/"
    echo "✅ Horizon comparison plot saved"
fi

# Q3d: Vary number of NN layers
echo ""
echo "--- Q3d: Varying NN Layers ---"
for layers in 1 2 3; do
    exp_name="HalfCheetah_q3_layers${layers}"
    echo "▶️  Running with $layers layers..."
    python main.py q3 --exp_name "layers${layers}" --nn_layers "$layers" 2>&1 | tee "$LOGS_DIR/q3d_layers${layers}.log"
    
    if [[ -d "$DATA_DIR/$exp_name" ]]; then
        cp -r "$DATA_DIR/$exp_name" "$LOGS_DIR/"
    fi
done

# Plot layers comparison
python plot.py --exps HalfCheetah_q3_layers1 HalfCheetah_q3_layers2 HalfCheetah_q3_layers3 --save HalfCheetah_q3_nn_layers 2>/dev/null || true
if [[ -f "plots/HalfCheetah_q3_nn_layers.jpg" ]]; then
    mv plots/HalfCheetah_q3_nn_layers.jpg "$PLOTS_DIR/"
    echo "✅ Layers comparison plot saved"
fi

# Clean up plots directory
if [ -d "plots" ] && [ -z "$(ls -A plots 2>/dev/null)" ]; then
    rmdir "plots" 2>/dev/null || true
fi

echo "✅ Q3 experiments completed!"

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
echo "  - Q1: Dynamics model training and prediction evaluation"
echo "  - Q2: MPC policy evaluation"
echo "  - Q3: On-policy MBRL with hyperparameter sweeps"
echo ""
echo "To view plots:"
echo "  open $PLOTS_DIR/*.jpg"
echo ""
echo "To re-run individual experiments:"
echo "  Q1: python run_mbrl.py q1 --exp_name my_q1_run"
echo "  Q2: python run_mbrl.py q2 --exp_name my_q2_run"
echo "  Q3: python main.py q3 --exp_name my_q3_run"
