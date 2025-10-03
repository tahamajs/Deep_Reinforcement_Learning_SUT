#!/bin/bash

# Run behavioral cloning pipeline for all available environments
# Author: Saeed Reza Zouashkiani
# Student ID: 400206262

set -e  # Exit on error

echo "========================================================================"
echo "RUNNING BEHAVIORAL CLONING ON ALL ENVIRONMENTS"
echo "========================================================================"

# List of environments
ENVIRONMENTS=(
    "Hopper-v2"
    "Ant-v2"
    "HalfCheetah-v2"
    "Walker2d-v2"
    "Reacher-v2"
    "Humanoid-v2"
)

# Parameters
NUM_ROLLOUTS=20
EPOCHS=100
BATCH_SIZE=64

# Run pipeline for each environment
for ENV in "${ENVIRONMENTS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Processing environment: $ENV"
    echo "========================================================================"
    
    # Check if expert policy exists
    EXPERT_POLICY="experts/${ENV}.pkl"
    if [ ! -f "$EXPERT_POLICY" ]; then
        echo "Warning: Expert policy not found for $ENV, skipping..."
        continue
    fi
    
    # Run the pipeline
    python run_full_pipeline.py \
        --env "$ENV" \
        --expert_policy "$EXPERT_POLICY" \
        --num_rollouts $NUM_ROLLOUTS \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE
    
    echo "✓ Completed $ENV"
    echo ""
done

echo ""
echo "========================================================================"
echo "ALL ENVIRONMENTS PROCESSED SUCCESSFULLY!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  - Expert data saved in: expert_data/"
echo "  - Trained models saved in: models/"
echo ""
echo "To evaluate a specific model, run:"
echo "  python run_bc.py evaluate --env <ENV_NAME> --model_file models/bc_policy_<ENV_NAME> --episodes 10 --render"
