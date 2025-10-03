#!/bin/bash

################################################################################
# Behavioral Cloning - Complete Pipeline Runner
# Runs data collection, training, and evaluation for all environments
#
# Author: Saeed Reza Zouashkiani
# Student ID: 400206262
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
NUM_ROLLOUTS=20
EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=0.001

################################################################################
# Functions
################################################################################

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
}

print_section() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${MAGENTA}→ $1${NC}"
}

################################################################################
# Main Script
################################################################################

clear

print_header "BEHAVIORAL CLONING - COMPLETE PIPELINE RUNNER"

echo ""
echo -e "${CYAN}Author:${NC} Saeed Reza Zouashkiani"
echo -e "${CYAN}Student ID:${NC} 400206262"
echo -e "${CYAN}Course:${NC} CS294-112 Deep Reinforcement Learning"
echo ""

print_section "STEP 0: Pre-flight Checks"

# Check if we're in the right directory
if [ ! -f "run_full_pipeline.py" ]; then
    print_error "run_full_pipeline.py not found!"
    echo "Please run this script from the hw1 directory."
    exit 1
fi

print_success "Found run_full_pipeline.py"

# Check if expert policies exist
if [ ! -d "experts" ]; then
    print_error "experts/ directory not found!"
    exit 1
fi

print_success "Found experts/ directory"

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found!"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_success "Python ${PYTHON_VERSION} detected"

# Check if MuJoCo works
print_info "Checking MuJoCo installation..."
if python -c "import mujoco_py" 2>/dev/null; then
    print_success "MuJoCo is working"
    MUJOCO_AVAILABLE=true
else
    print_warning "MuJoCo is NOT working"
    MUJOCO_AVAILABLE=false
    echo ""
    print_error "MuJoCo is required for the selected environments!"
    echo ""
    echo "You have the following options:"
    echo ""
    echo "  1. Run test without MuJoCo (CartPole, etc.)"
    echo "  2. Fix MuJoCo installation (run: chmod +x fix_mujoco.sh && ./fix_mujoco.sh)"
    echo "  3. Use Google Colab (easiest for MuJoCo)"
    echo "  4. Continue anyway (will fail on MuJoCo environments)"
    echo ""
    read -p "Choose option (1-4) [default: 1]: " MUJOCO_CHOICE
    MUJOCO_CHOICE=${MUJOCO_CHOICE:-1}
    
    case $MUJOCO_CHOICE in
        1)
            print_info "Running test without MuJoCo..."
            python test_without_mujoco.py
            exit 0
            ;;
        2)
            print_info "Running MuJoCo fixer..."
            chmod +x fix_mujoco.sh
            ./fix_mujoco.sh
            exit 0
            ;;
        3)
            print_info "Please upload this folder to Google Colab"
            echo ""
            echo "See MUJOCO_SETUP.md for instructions"
            exit 0
            ;;
        4)
            print_warning "Continuing anyway. MuJoCo environments will fail."
            sleep 2
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
fi

# Create output directories
mkdir -p expert_data
mkdir -p models
mkdir -p logs
mkdir -p results
mkdir -p visualizations
mkdir -p visualizations/training_curves
mkdir -p visualizations/comparisons
mkdir -p visualizations/evaluations

print_success "Output directories created"

################################################################################
# Environment Selection
################################################################################

print_section "STEP 1: Environment Selection"

echo "Available environments:"
echo ""
echo "  1) Hopper-v2         (Recommended - Fast, 5-10 min)"
echo "  2) Reacher-v2        (Fast, 5-10 min)"
echo "  3) HalfCheetah-v2    (Medium, 10-15 min)"
echo "  4) Walker2d-v2       (Medium, 10-15 min)"
echo "  5) Ant-v2            (Slow, 15-20 min)"
echo "  6) Humanoid-v2       (Very Slow, 30+ min)"
echo "  7) ALL Environments  (Run all sequentially)"
echo "  8) Quick Test        (CartPole without MuJoCo)"
echo ""

read -p "Select option (1-8) [default: 1]: " CHOICE
CHOICE=${CHOICE:-1}

ENVIRONMENTS=()

case $CHOICE in
    1)
        ENVIRONMENTS=("Hopper-v2")
        ;;
    2)
        ENVIRONMENTS=("Reacher-v2")
        ;;
    3)
        ENVIRONMENTS=("HalfCheetah-v2")
        ;;
    4)
        ENVIRONMENTS=("Walker2d-v2")
        ;;
    5)
        ENVIRONMENTS=("Ant-v2")
        ;;
    6)
        ENVIRONMENTS=("Humanoid-v2")
        ;;
    7)
        ENVIRONMENTS=("Hopper-v2" "Reacher-v2" "HalfCheetah-v2" "Walker2d-v2" "Ant-v2" "Humanoid-v2")
        ;;
    8)
        print_info "Running quick test without MuJoCo..."
        python test_without_mujoco.py
        exit 0
        ;;
    *)
        print_error "Invalid choice. Defaulting to Hopper-v2"
        ENVIRONMENTS=("Hopper-v2")
        ;;
esac

################################################################################
# Configuration
################################################################################

print_section "STEP 2: Configuration"

read -p "Number of expert rollouts [default: ${NUM_ROLLOUTS}]: " INPUT_ROLLOUTS
NUM_ROLLOUTS=${INPUT_ROLLOUTS:-$NUM_ROLLOUTS}

read -p "Training epochs [default: ${EPOCHS}]: " INPUT_EPOCHS
EPOCHS=${INPUT_EPOCHS:-$EPOCHS}

read -p "Batch size [default: ${BATCH_SIZE}]: " INPUT_BATCH
BATCH_SIZE=${INPUT_BATCH:-$BATCH_SIZE}

echo ""
print_info "Configuration:"
echo "  - Environments: ${ENVIRONMENTS[@]}"
echo "  - Rollouts: ${NUM_ROLLOUTS}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo ""

read -p "Continue? (Y/n): " CONFIRM
CONFIRM=${CONFIRM:-Y}

if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    print_warning "Aborted by user"
    exit 0
fi

################################################################################
# Main Pipeline Loop
################################################################################

# Summary arrays (using simple arrays instead of associative for bash 3.x compatibility)
RESULTS_EXPERT=()
RESULTS_BC=()
RESULTS_RATIO=()
RESULTS_STATUS=()
RESULTS_ENVS=()

TOTAL_ENVS=${#ENVIRONMENTS[@]}
CURRENT_ENV=0
START_TIME=$(date +%s)

for ENV in "${ENVIRONMENTS[@]}"; do
    CURRENT_ENV=$((CURRENT_ENV + 1))
    
    print_section "Processing ${ENV} (${CURRENT_ENV}/${TOTAL_ENVS})"
    
    EXPERT_POLICY="experts/${ENV}.pkl"
    
    # Check if expert policy exists
    if [ ! -f "$EXPERT_POLICY" ]; then
        print_warning "Expert policy not found: ${EXPERT_POLICY}"
        print_warning "Skipping ${ENV}"
        RESULTS_ENVS+=("$ENV")
        RESULTS_STATUS+=("SKIPPED")
        RESULTS_EXPERT+=("N/A")
        RESULTS_BC+=("N/A")
        RESULTS_RATIO+=("N/A")
        continue
    fi
    
    print_success "Found expert policy: ${EXPERT_POLICY}"
    
    # Create log file
    LOG_FILE="logs/${ENV}_$(date +%Y%m%d_%H%M%S).log"
    
    print_info "Logging to: ${LOG_FILE}"
    echo ""
    
    # Run the pipeline
    ENV_START=$(date +%s)
    
    if python run_full_pipeline.py \
        --env "$ENV" \
        --expert_policy "$EXPERT_POLICY" \
        --num_rollouts "$NUM_ROLLOUTS" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee "$LOG_FILE"; then
        
        ENV_END=$(date +%s)
        ENV_DURATION=$((ENV_END - ENV_START))
        
        print_success "Completed ${ENV} in ${ENV_DURATION}s"
        
        RESULTS_ENVS+=("$ENV")
        RESULTS_STATUS+=("SUCCESS")
        
        # Extract results from log
        EXPERT_RETURN=$(grep "Expert.*Mean return:" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        BC_RETURN=$(grep "BC Policy.*Mean return:" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        EXPERT_STD=$(grep "Expert.*Std return:" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        BC_STD=$(grep "BC Policy.*Std return:" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        
        if [ ! -z "$EXPERT_RETURN" ] && [ ! -z "$BC_RETURN" ]; then
            RESULTS_EXPERT+=("$EXPERT_RETURN")
            RESULTS_BC+=("$BC_RETURN")
            RATIO=$(echo "scale=1; $BC_RETURN / $EXPERT_RETURN * 100" | bc 2>/dev/null || echo "N/A")
            RESULTS_RATIO+=("$RATIO")
            
            # Generate visualizations
            print_info "Generating visualizations..."
            
            # 1. Training curve
            if [ -f "models/training_history_${ENV}.pkl" ]; then
                python -c "
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    with open('models/training_history_${ENV}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    losses = data.get('losses', [])
    
    if losses:
        plt.figure(figsize=(10, 6))
        plt.plot(losses, linewidth=2, color='#2E86AB')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss - ${ENV}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/training_curves/${ENV}_training_loss.png', dpi=150)
        plt.close()
        print('✓ Training curve saved')
except Exception as e:
    print(f'⚠ Could not generate training curve: {e}')
" 2>/dev/null || true
            fi
            
            # 2. Performance comparison
            python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

expert_mean = ${EXPERT_RETURN}
expert_std = ${EXPERT_STD:-0}
bc_mean = ${BC_RETURN}
bc_std = ${BC_STD:-0}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
categories = ['Expert', 'BC Policy']
means = [expert_mean, bc_mean]
stds = [expert_std, bc_std]
colors = ['#06A77D', '#F77E21']

x = np.arange(len(categories))
bars = ax1.bar(x, means, yerr=stds, capsize=10, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Average Return', fontsize=12)
ax1.set_title('Performance Comparison - ${ENV}', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.2f}\\n±{std:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Ratio visualization
ratio = (bc_mean / expert_mean * 100) if expert_mean > 0 else 0
ax2.barh(['Performance'], [ratio], color='#D62246', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.barh(['Expert'], [100], color='#06A77D', alpha=0.3)
ax2.set_xlabel('Performance Ratio (%)', fontsize=12)
ax2.set_title('BC vs Expert Performance', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 110)
ax2.grid(True, alpha=0.3, axis='x')
ax2.text(ratio/2, 0, f'{ratio:.1f}%', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('visualizations/comparisons/${ENV}_comparison.png', dpi=150)
plt.close()
print('✓ Comparison chart saved')
" 2>/dev/null || print_warning "Could not generate comparison chart"
            
            print_success "Visualizations saved to visualizations/"
        else
            RESULTS_EXPERT+=("N/A")
            RESULTS_BC+=("N/A")
            RESULTS_RATIO+=("N/A")
        fi
        
    else
        print_error "Failed to process ${ENV}"
        RESULTS_ENVS+=("$ENV")
        RESULTS_STATUS+=("FAILED")
        RESULTS_EXPERT+=("N/A")
        RESULTS_BC+=("N/A")
        RESULTS_RATIO+=("N/A")
    fi
    
    echo ""
done

################################################################################
# Final Summary
################################################################################

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

print_header "FINAL SUMMARY"

echo ""
printf "${CYAN}%-20s %-15s %-15s %-15s %-10s${NC}\n" "Environment" "Expert Return" "BC Return" "Ratio" "Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Iterate through results using index
for i in "${!RESULTS_ENVS[@]}"; do
    ENV="${RESULTS_ENVS[$i]}"
    STATUS="${RESULTS_STATUS[$i]}"
    EXPERT="${RESULTS_EXPERT[$i]}"
    BC="${RESULTS_BC[$i]}"
    RATIO="${RESULTS_RATIO[$i]}"
    
    if [ "$RATIO" != "N/A" ]; then
        RATIO="${RATIO}%"
    fi
    
    if [ "$STATUS" == "SUCCESS" ]; then
        printf "${GREEN}%-20s %-15s %-15s %-15s %-10s${NC}\n" "$ENV" "$EXPERT" "$BC" "$RATIO" "$STATUS"
    elif [ "$STATUS" == "SKIPPED" ]; then
        printf "${YELLOW}%-20s %-15s %-15s %-15s %-10s${NC}\n" "$ENV" "$EXPERT" "$BC" "$RATIO" "$STATUS"
    else
        printf "${RED}%-20s %-15s %-15s %-15s %-10s${NC}\n" "$ENV" "$EXPERT" "$BC" "$RATIO" "$STATUS"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "Total time: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

print_section "Output Locations"

echo "📊 Data & Models:"
echo "  • Expert Data:        expert_data/"
echo "  • Trained Models:     models/"
echo "  • Training History:   models/*_training_history_*.pkl"
echo ""
echo "📈 Visualizations:"
echo "  • Training Curves:    visualizations/training_curves/"
echo "  • Comparisons:        visualizations/comparisons/"
echo "  • Complete Summary:   visualizations/complete_summary.png"
echo "  • Detailed Analysis:  visualizations/detailed_analysis.png"
echo ""
echo "📝 Logs & Reports:"
echo "  • Execution Logs:     logs/"
echo "  • Summary Report:     ${SUMMARY_FILE}"
echo ""

print_section "Next Steps"

echo "🎮 View visualizations:"
echo "  open visualizations/complete_summary.png"
echo "  open visualizations/detailed_analysis.png"
echo ""
echo "🎯 Evaluate a trained policy with rendering:"
echo "  python run_bc.py evaluate --env <ENV> --model_file models/bc_policy_<ENV> --episodes 10 --render"
echo ""
echo "📊 View training curves:"
echo "  open visualizations/training_curves/"
echo ""
echo "📈 View performance comparisons:"
echo "  open visualizations/comparisons/"
echo ""
echo "📝 Read detailed summary:"
echo "  cat ${SUMMARY_FILE}"
echo ""

print_success "Pipeline completed!"
echo ""

# Generate summary file
SUMMARY_FILE="results/summary_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "BEHAVIORAL CLONING RESULTS SUMMARY"
    echo "Generated: $(date)"
    echo "========================================"
    echo ""
    echo "Configuration:"
    echo "  - Rollouts: ${NUM_ROLLOUTS}"
    echo "  - Epochs: ${EPOCHS}"
    echo "  - Batch Size: ${BATCH_SIZE}"
    echo ""
    echo "Results:"
    echo "----------------------------------------"
    printf "%-20s %-15s %-15s %-10s\n" "Environment" "Expert" "BC Policy" "Ratio"
    echo "----------------------------------------"
    for i in "${!RESULTS_ENVS[@]}"; do
        if [ "${RESULTS_STATUS[$i]}" == "SUCCESS" ]; then
            RATIO_VAL="${RESULTS_RATIO[$i]}"
            if [ "$RATIO_VAL" != "N/A" ]; then
                RATIO_VAL="${RATIO_VAL}%"
            fi
            printf "%-20s %-15s %-15s %-10s\n" \
                "${RESULTS_ENVS[$i]}" \
                "${RESULTS_EXPERT[$i]}" \
                "${RESULTS_BC[$i]}" \
                "$RATIO_VAL"
        fi
    done
    echo "----------------------------------------"
    echo ""
    echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
} > "$SUMMARY_FILE"

print_success "Summary saved to: ${SUMMARY_FILE}"

# Generate comprehensive summary visualization
if [ ${#RESULTS_ENVS[@]} -gt 0 ]; then
    print_section "Generating Summary Visualizations"
    
    python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Collect data
envs = []
expert_returns = []
bc_returns = []
ratios = []

results_envs = '${RESULTS_ENVS[@]}'.split()
results_expert = '${RESULTS_EXPERT[@]}'.split()
results_bc = '${RESULTS_BC[@]}'.split()
results_ratio = '${RESULTS_RATIO[@]}'.split()
results_status = '${RESULTS_STATUS[@]}'.split()

for i, env in enumerate(results_envs):
    if results_status[i] == 'SUCCESS' and results_expert[i] != 'N/A':
        envs.append(env.replace('-v2', ''))
        expert_returns.append(float(results_expert[i]))
        bc_returns.append(float(results_bc[i]))
        ratios.append(float(results_ratio[i]))

if not envs:
    print('No successful results to visualize')
    exit(0)

# Create comprehensive dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Bar chart comparison
ax1 = fig.add_subplot(gs[0, :])
x = np.arange(len(envs))
width = 0.35
bars1 = ax1.bar(x - width/2, expert_returns, width, label='Expert', color='#06A77D', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, bc_returns, width, label='BC Policy', color='#F77E21', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Environment', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Return', fontsize=12, fontweight='bold')
ax1.set_title('Expert vs BC Policy Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(envs, rotation=45, ha='right')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9)

# 2. Performance ratio chart
ax2 = fig.add_subplot(gs[1, 0])
colors_ratio = ['#06A77D' if r >= 70 else '#F77E21' if r >= 50 else '#D62246' for r in ratios]
bars = ax2.barh(envs, ratios, color=colors_ratio, alpha=0.8, edgecolor='black')
ax2.axvline(x=100, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Expert Level')
ax2.set_xlabel('Performance Ratio (%)', fontsize=12, fontweight='bold')
ax2.set_title('BC Policy Performance Ratio', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 110)
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, ratio) in enumerate(zip(bars, ratios)):
    width = bar.get_width()
    ax2.text(width + 2, bar.get_y() + bar.get_height()/2.,
            f'{ratio:.1f}%',
            ha='left', va='center', fontsize=10, fontweight='bold')

# 3. Performance gap chart
ax3 = fig.add_subplot(gs[1, 1])
gaps = [e - b for e, b in zip(expert_returns, bc_returns)]
colors_gap = ['#D62246' if g > 0 else '#06A77D' for g in gaps]
ax3.barh(envs, gaps, color=colors_gap, alpha=0.8, edgecolor='black')
ax3.axvline(x=0, color='black', linewidth=2)
ax3.set_xlabel('Performance Gap (Expert - BC)', fontsize=12, fontweight='bold')
ax3.set_title('Performance Gap Analysis', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 4. Statistics summary
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

stats_text = f'''
SUMMARY STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Environments Tested: {len(envs)}

Average Expert Return: {np.mean(expert_returns):.2f} ± {np.std(expert_returns):.2f}
Average BC Return: {np.mean(bc_returns):.2f} ± {np.std(bc_returns):.2f}

Mean Performance Ratio: {np.mean(ratios):.1f}%
Best Performance: {max(ratios):.1f}% ({envs[ratios.index(max(ratios))]})
Worst Performance: {min(ratios):.1f}% ({envs[ratios.index(min(ratios))]})

Environments with >80% Performance: {sum(1 for r in ratios if r >= 80)}
Environments with >70% Performance: {sum(1 for r in ratios if r >= 70)}
Environments with >50% Performance: {sum(1 for r in ratios if r >= 50)}

Configuration:
  • Rollouts: ${NUM_ROLLOUTS}
  • Epochs: ${EPOCHS}
  • Batch Size: ${BATCH_SIZE}
'''

ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Behavioral Cloning - Complete Results Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('visualizations/complete_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print('✓ Complete summary dashboard saved')

# Generate individual metric plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot
ax = axes[0, 0]
ax.scatter(expert_returns, bc_returns, s=100, alpha=0.6, c=ratios, cmap='RdYlGn', edgecolors='black', linewidth=1.5)
max_val = max(max(expert_returns), max(bc_returns))
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect Match')
ax.set_xlabel('Expert Return', fontsize=12, fontweight='bold')
ax.set_ylabel('BC Policy Return', fontsize=12, fontweight='bold')
ax.set_title('Expert vs BC Performance Scatter', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add labels
for i, env in enumerate(envs):
    ax.annotate(env, (expert_returns[i], bc_returns[i]), fontsize=8, alpha=0.7)

# Box plot
ax = axes[0, 1]
bp = ax.boxplot([expert_returns, bc_returns], labels=['Expert', 'BC Policy'],
                patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('#06A77D')
bp['boxes'][1].set_facecolor('#F77E21')
ax.set_ylabel('Return', fontsize=12, fontweight='bold')
ax.set_title('Return Distribution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Ratio distribution
ax = axes[1, 0]
ax.hist(ratios, bins=10, color='#2E86AB', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ratios):.1f}%')
ax.set_xlabel('Performance Ratio (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Performance Ratio Distribution', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Sorted ratios
ax = axes[1, 1]
sorted_idx = np.argsort(ratios)
sorted_envs = [envs[i] for i in sorted_idx]
sorted_ratios = [ratios[i] for i in sorted_idx]
colors_sorted = ['#06A77D' if r >= 70 else '#F77E21' if r >= 50 else '#D62246' for r in sorted_ratios]
ax.barh(sorted_envs, sorted_ratios, color=colors_sorted, alpha=0.8, edgecolor='black')
ax.axvline(100, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Performance Ratio (%)', fontsize=12, fontweight='bold')
ax.set_title('Sorted Performance Rankings', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('visualizations/detailed_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print('✓ Detailed analysis plots saved')
print('✓ All visualizations saved to visualizations/')

" 2>/dev/null && print_success "Summary visualizations generated" || print_warning "Could not generate summary visualizations (matplotlib may not be installed)"
fi

echo ""
