#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 CA15: Advanced Deep Reinforcement Learning - Complex Experiments"
echo "=================================================================="

# Function to print section headers
print_section() {
    echo ""
    echo "📋 $1"
    echo "$(printf '=%.0s' {1..60})"
}

# Function to print step
print_step() {
    echo "🔸 $1"
}

# Function to print success
print_success() {
    echo "✅ $1"
}

# Function to print warning
print_warning() {
    echo "⚠️  $1"
}

# Function to print info
print_info() {
    echo "ℹ️  $1"
}

print_section "Advanced CA15 Setup and Execution"

# Parse command line arguments
EXPERIMENT_TYPE="all"
if [ $# -gt 0 ]; then
    EXPERIMENT_TYPE="$1"
fi

print_step "1. Environment Setup"
# 1. Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# 3. Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
print_success "Dependencies installed"

# 4. Create necessary directories
echo "Ensuring directories exist..."
mkdir -p visualizations
mkdir -p results
mkdir -p logs
mkdir -p data
print_success "Directories created"

print_step "2. Advanced Experiment Execution"

case $EXPERIMENT_TYPE in
    "all")
        print_info "Running ALL Advanced Experiments..."
        echo "This includes:"
        echo "• Multi-Agent Cooperation Experiments"
        echo "• Hierarchical Curriculum Learning"
        echo "• Model-Based Uncertainty Quantification"
        echo "• Advanced Planning Algorithms"
        echo "• Comprehensive Benchmark"
        echo ""
        python3 run_complex_experiments.py --experiment-type all
        ;;
    "multi-agent")
        print_info "Running Multi-Agent Cooperation Experiments..."
        python3 run_complex_experiments.py --experiment-type multi-agent
        ;;
    "hierarchical")
        print_info "Running Hierarchical Curriculum Learning Experiments..."
        python3 run_complex_experiments.py --experiment-type hierarchical
        ;;
    "model-based")
        print_info "Running Model-Based Uncertainty Quantification Experiments..."
        python3 run_complex_experiments.py --experiment-type model-based
        ;;
    "planning")
        print_info "Running Advanced Planning Algorithms Experiments..."
        python3 run_complex_experiments.py --experiment-type planning
        ;;
    "benchmark")
        print_info "Running Comprehensive Benchmark..."
        python3 run_complex_experiments.py --experiment-type benchmark
        ;;
    "basic")
        print_info "Running Basic CA15 Experiments..."
        python3 run_all_experiments.py
        ;;
    "notebook")
        print_info "Starting Jupyter Notebook..."
        jupyter notebook CA15.ipynb
        ;;
    "test")
        print_info "Running Structure Test..."
        python3 test_structure.py
        ;;
    "visualize")
        print_info "Creating Advanced Visualizations..."
        python3 -c "
from utils.advanced_visualization import AdvancedVisualizer
visualizer = AdvancedVisualizer()
print('Advanced visualization tools loaded successfully!')
"
        ;;
    "help")
        print_section "Available Experiment Types"
        echo "Usage: ./run.sh [experiment_type]"
        echo ""
        echo "Available experiment types:"
        echo "• all          - Run all advanced experiments (default)"
        echo "• multi-agent - Multi-agent cooperation experiments"
        echo "• hierarchical- Hierarchical curriculum learning"
        echo "• model-based - Model-based uncertainty quantification"
        echo "• planning    - Advanced planning algorithms"
        echo "• benchmark   - Comprehensive benchmark"
        echo "• basic       - Basic CA15 experiments"
        echo "• notebook    - Start Jupyter notebook"
        echo "• test        - Run structure test"
        echo "• visualize   - Test visualization tools"
        echo "• help        - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run.sh                    # Run all experiments"
        echo "  ./run.sh multi-agent        # Run multi-agent experiments only"
        echo "  ./run.sh notebook           # Start Jupyter notebook"
        echo "  ./run.sh help               # Show help"
        exit 0
        ;;
    *)
        print_warning "Unknown experiment type: $EXPERIMENT_TYPE"
        print_info "Use './run.sh help' to see available options"
        exit 1
        ;;
esac

print_step "3. Results and Visualizations"
print_success "Advanced CA15 experiments completed!"

echo ""
print_section "Generated Outputs"
echo "📊 Results:"
echo "  • results/benchmark_results.json - Detailed experiment results"
echo "  • results/benchmark_summary.md - Summary report"
echo ""
echo "📈 Visualizations:"
echo "  • visualizations/multi_agent_cooperation_results.png"
echo "  • visualizations/hierarchical_curriculum_results.png"
echo "  • visualizations/model_uncertainty_results.png"
echo "  • visualizations/planning_comparison_results.png"
echo "  • visualizations/comprehensive_dashboard.png"
echo ""
echo "📋 Logs:"
echo "  • logs/ - Training logs and metrics"
echo ""
echo "💾 Data:"
echo "  • data/ - Collected training data"

print_section "Next Steps"
echo "🔍 Explore Results:"
echo "  • Check visualizations/ for plots and analysis"
echo "  • Read results/benchmark_summary.md for detailed report"
echo "  • Review logs/ for training details"
echo ""
echo "🚀 Advanced Usage:"
echo "  • Modify experiments/advanced_experiments.py for custom experiments"
echo "  • Use utils/advanced_visualization.py for custom visualizations"
echo "  • Extend environments/advanced_environments.py for new tasks"
echo ""
echo "📚 Documentation:"
echo "  • README.md - Project overview and setup"
echo "  • SETUP_GUIDE.sh - Comprehensive usage guide"

echo ""
print_success "🎉 CA15 Advanced Deep RL Experiments Completed Successfully!"
echo "🚀 Ready for advanced reinforcement learning research!"

# Deactivate virtual environment (optional, but good practice)
# deactivate