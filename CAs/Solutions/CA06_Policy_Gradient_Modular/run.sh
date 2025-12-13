#!/bin/bash

# CA06 Policy Gradient Methods - Quick Execution Script
# This script runs the main execution script with default quick settings.

set -e  # Exit on any error

echo "=========================================="
echo "CA06: Policy Gradient Methods - Quick Run"
echo "=========================================="

# Navigate to the correct directory if not already there
current_dir=$(basename "$(pwd)")
if [ "$current_dir" != "CA06_Policy_Gradient_Modular" ]; then
    echo "Warning: Not in the CA06_Policy_Gradient_Modular directory. Attempting to navigate."
    cd "CAs/Solutions/CA06_Policy_Gradient_Modular" || {
        echo "Error: Could not navigate to CAs/Solutions/CA06_Policy_Gradient_Modular. Please run this script from the project root or the assignment directory."
        exit 1
    }
fi

# Create necessary directories (main.py also ensures this)
mkdir -p visualizations results logs

# Run the main Python script in quick mode
echo "Starting main.py in quick mode..."
python main.py --quick

echo "\n=========================================="
echo "CA06 Quick Execution Completed"
echo "=========================================="
echo "Timestamp: $(date)"
echo "\nCheck the 'visualizations/' and 'results/' folders for outputs."
echo "For full runs, use 'python main.py' or 'python main.py --episodes <num_episodes>'."
echo "=========================================="

# Optional: Open visualizations folder (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Opening visualizations folder..."
    open visualizations/
fi


