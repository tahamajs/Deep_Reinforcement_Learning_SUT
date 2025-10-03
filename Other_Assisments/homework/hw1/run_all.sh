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
        
        if [ ! -z "$EXPERT_RETURN" ] && [ ! -z "$BC_RETURN" ]; then
            RESULTS_EXPERT+=("$EXPERT_RETURN")
            RESULTS_BC+=("$BC_RETURN")
            RATIO=$(echo "scale=1; $BC_RETURN / $EXPERT_RETURN * 100" | bc 2>/dev/null || echo "N/A")
            RESULTS_RATIO+=("$RATIO")
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

echo "Expert Data:      expert_data/"
echo "Trained Models:   models/"
echo "Logs:            logs/"
echo "Training History: models/*_training_history_*.pkl"
echo ""

print_section "Next Steps"

echo "To visualize a trained policy:"
echo "  python run_bc.py evaluate --env <ENV> --model_file models/bc_policy_<ENV> --episodes 10 --render"
echo ""
echo "To plot training history:"
echo "  python -c \"import pickle; import matplotlib.pyplot as plt; data=pickle.load(open('models/training_history_Hopper-v2.pkl','rb')); plt.plot(data['losses']); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.savefig('training_loss.png'); print('Saved to training_loss.png')\""
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
echo ""
