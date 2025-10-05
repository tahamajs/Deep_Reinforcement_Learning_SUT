#!/bin/bash

# CA16: Cutting-Edge Deep Reinforcement Learning
# Foundation Models, Neurosymbolic RL, and Future Paradigms
# 
# This script provides convenient commands to run and test the CA16 implementation

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

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "../../../venv" ]; then
        print_error "Virtual environment not found at ../../../venv"
        print_status "Please create a virtual environment first:"
        echo "cd ../../.. && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
}

# Function to activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source ../../../venv/bin/activate
    print_success "Virtual environment activated"
}

# Function to test imports
test_imports() {
    print_status "Testing module imports..."
    python test_imports.py
    if [ $? -eq 0 ]; then
        print_success "All imports working correctly!"
    else
        print_error "Import tests failed!"
        exit 1
    fi
}

# Function to run the notebook
run_notebook() {
    print_status "Starting Jupyter notebook..."
    print_warning "Make sure to run the cells in order!"
    jupyter notebook CA16.ipynb
}

# Function to run a quick demo
run_demo() {
    print_status "Running quick demo..."
    python -c "
import sys
sys.path.insert(0, '.')
from foundation_models import DecisionTransformer, FoundationModelTrainer
from neurosymbolic import NeurosymbolicAgent, SymbolicKnowledgeBase
from human_ai_collaboration import CollaborativeAgent
from environments import SymbolicGridWorld, CollaborativeGridWorld
import torch

print('🧠 Testing Foundation Models...')
dt = DecisionTransformer(state_dim=8, action_dim=4, model_dim=64, num_heads=4, num_layers=2)
trainer = FoundationModelTrainer(dt, lr=3e-4)
print('✅ DecisionTransformer created successfully')

print('🔮 Testing Neurosymbolic RL...')
kb = SymbolicKnowledgeBase()
ns_agent = NeurosymbolicAgent(state_dim=8, action_dim=4, knowledge_base=kb)
print('✅ NeurosymbolicAgent created successfully')

print('🤝 Testing Human-AI Collaboration...')
collab = CollaborativeAgent(state_dim=8, action_dim=4)
print('✅ CollaborativeAgent created successfully')

print('🌍 Testing Environments...')
env1 = SymbolicGridWorld(size=6)
env2 = CollaborativeGridWorld(size=6)
print('✅ Environments created successfully')

print('🎉 All components working correctly!')
"
}

# Function to install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install torch torchvision numpy matplotlib seaborn jupyter notebook
    print_success "Dependencies installed"
}

# Function to show help
show_help() {
    echo "CA16: Cutting-Edge Deep Reinforcement Learning"
    echo "=============================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  test        Test all module imports"
    echo "  demo        Run a quick demonstration"
    echo "  notebook    Start Jupyter notebook"
    echo "  install     Install required dependencies"
    echo "  clean       Clean up __pycache__ directories"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 test     # Test imports"
    echo "  $0 demo     # Run quick demo"
    echo "  $0 notebook # Start Jupyter notebook"
    echo ""
}

# Function to clean up
clean_up() {
    print_status "Cleaning up __pycache__ directories..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    print_success "Cleanup completed"
}

# Main script logic
main() {
    # Check if we're in the right directory
    if [ ! -f "CA16.ipynb" ]; then
        print_error "CA16.ipynb not found. Please run this script from the CA16 directory."
        exit 1
    fi

    # Check virtual environment
    check_venv
    activate_venv

    # Parse command line arguments
    case "${1:-help}" in
        "test")
            test_imports
            ;;
        "demo")
            run_demo
            ;;
        "notebook")
            run_notebook
            ;;
        "install")
            install_deps
            ;;
        "clean")
            clean_up
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
