#!/bin/bash

# CA07: Deep Q-Networks (DQN) and Value-Based Methods - Complete Run Script
# ========================================================================
# This script runs all experiments and generates comprehensive results

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Define paths based on config (assuming config values can be accessed or hardcoded for script)
# For a more robust solution, these might be passed as arguments or read from a config file
VISUALIZATIONS_DIR="visualizations"
RESULTS_DIR="results"
LOGS_DIR="logs"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p "$VISUALIZATIONS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$LOGS_DIR/experiments"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists, if not create one
if [ ! -d ".venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Install requirements
print_status "Installing requirements..."
pip install -r requirements.txt

# Set environment variables for Python path to include src
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU if available

# Function to run python script with error handling and logging
run_python_script() {
    local script_path=$1
    local description=$2
    local log_file_name=$(basename "$script_path" .py)
    local log_path="$LOGS_DIR/experiments/${log_file_name}.log"
    
    print_status "Running: $description"
    if python3 "$script_path" > "$log_path" 2>&1; then
        print_success "Completed: $description. Log: $log_path"
    else
        print_error "Failed: $description. Check log: $log_path"
        return 1
    fi
}

# Main execution
print_status "Starting CA07 DQN Experiments..."
echo "========================================"

# 1. Run all training examples (includes variant comparison, hyperparam, robustness, advanced demo)
run_python_script "training_examples.py" "Comprehensive Training Examples and Analysis"

# 2. Run unit tests
print_status "Running unit tests..."
if python3 -m pytest test_implementation.py; then
    print_success "Unit tests passed!"
else
    print_error "Unit tests failed!"
    # Optionally, exit or continue based on severity
    exit 1 
fi

# 3. (Optional) Convert Jupyter notebook to script for static analysis
# This step is removed as the notebook will be for interactive use and visualization
# and we don't want to execute it from the run.sh script directly.

# 4. Generate summary report (Placeholder for a more detailed report generation, if needed)
print_status "Creating results summary in $RESULTS_DIR/summary.txt..."
cat > "$RESULTS_DIR/summary.txt" << EOF
CA07: Deep Q-Networks (DQN) and Value-Based Methods - Experiment Summary
=======================================================================

All primary training and analysis scripts have been executed. 
Detailed logs for each run can be found in the '$LOGS_DIR/experiments/' directory.
Visualizations are saved in the '$VISUALIZATIONS_DIR/' directory.

Summary of Executed Components:
- Comprehensive Training Examples and Analysis (training_examples.py)
- Unit Tests (test_implementation.py)

Review the generated plots in '$VISUALIZATIONS_DIR/' and logs in '$LOGS_DIR/' for detailed results.
EOF

# 5. Clean up and finalize
print_status "Cleaning up temporary files..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

print_success "All CA07 DQN experiments completed successfully!"
print_status "Results saved in:"
echo "  - $VISUALIZATIONS_DIR/ (all plots and charts)"
echo "  - $RESULTS_DIR/ (summary and data)"
echo "  - $LOGS_DIR/ (execution logs)"

echo ""
echo "========================================"
print_success "CA07 DQN Project Execution Complete!"
echo "========================================"

