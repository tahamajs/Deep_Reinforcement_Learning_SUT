#!/bin/bash

# Complete Behavioral Cloning Pipeline Runner
# Author: Saeed Reza Zouashkiani
# Student ID: 400206262

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================================================"
echo "BEHAVIORAL CLONING - COMPLETE PIPELINE"
echo -e "========================================================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "run_full_pipeline.py" ]; then
    echo -e "${RED}Error: run_full_pipeline.py not found!${NC}"
    echo "Please run this script from the hw1 directory."
    exit 1
fi

# Step 1: Check environment setup
echo -e "${YELLOW}Step 1: Checking environment setup...${NC}"
python check_setup.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Environment check failed. Please install required packages.${NC}"
    echo "Run: pip install -r requirements.txt"
    exit 1
fi
echo -e "${GREEN}✓ Environment setup OK${NC}"
echo ""

# Step 2: Choose environment
echo -e "${YELLOW}Step 2: Selecting environment...${NC}"
echo "Available environments:"
echo "  1) Hopper-v2 (Recommended for quick test)"
echo "  2) Reacher-v2"
echo "  3) HalfCheetah-v2"
echo "  4) Walker2d-v2"
echo "  5) Ant-v2"
echo "  6) Humanoid-v2 (Slowest, most complex)"
echo "  7) All environments"
echo ""

# Default to Hopper for quick test
ENV="Hopper-v2"
echo -e "${GREEN}Using default: ${ENV}${NC}"
echo ""

# Step 3: Set parameters
echo -e "${YELLOW}Step 3: Setting parameters...${NC}"
NUM_ROLLOUTS=20
EPOCHS=100
BATCH_SIZE=64

echo "Parameters:"
echo "  - Number of rollouts: $NUM_ROLLOUTS"
echo "  - Training epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo ""

# Step 4: Run the pipeline
echo -e "${YELLOW}Step 4: Running behavioral cloning pipeline...${NC}"
echo -e "${BLUE}========================================================================"
echo "STARTING PIPELINE FOR $ENV"
echo -e "========================================================================${NC}"
echo ""

python run_full_pipeline.py \
    --env "$ENV" \
    --expert_policy "experts/${ENV}.pkl" \
    --num_rollouts $NUM_ROLLOUTS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================================================"
    echo "✓ PIPELINE COMPLETED SUCCESSFULLY!"
    echo -e "========================================================================${NC}"
    echo ""
    echo "Results saved to:"
    echo "  - Expert data: expert_data/${ENV}.pkl"
    echo "  - Trained model: models/bc_policy_${ENV}.*"
    echo "  - Training history: models/training_history_${ENV}.pkl"
    echo ""
    echo "To visualize the trained policy, run:"
    echo "  python run_bc.py evaluate --env $ENV --model_file models/bc_policy_${ENV} --episodes 5 --render"
    echo ""
else
    echo -e "${RED}Pipeline failed! Check the error messages above.${NC}"
    exit 1
fi
