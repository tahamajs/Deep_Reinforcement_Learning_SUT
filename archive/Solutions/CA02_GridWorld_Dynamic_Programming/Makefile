# Makefile for Reinforcement Learning GridWorld Dynamic Programming Project

.PHONY: help install test run clean setup check

# Default target
help:
	@echo "Reinforcement Learning GridWorld Dynamic Programming Project"
	@echo "=========================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  help     - Show this help message"
	@echo "  install  - Install required dependencies"
	@echo "  test     - Run installation and functionality tests"
	@echo "  run      - Execute all experiments and generate results"
	@echo "  setup    - Set up the project (install + test)"
	@echo "  check    - Check code style and quality"
	@echo "  clean    - Clean generated files and directories"
	@echo "  jupyter  - Start Jupyter notebook server"
	@echo ""
	@echo "Quick start: make setup && make run"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

# Run tests
test:
	@echo "🧪 Running tests..."
	python3 test_installation.py
	@echo "✅ Tests completed"

# Run all experiments
run:
	@echo "🚀 Running all experiments..."
	./run.sh
	@echo "✅ Experiments completed"

# Set up the project
setup: install test
	@echo "🎯 Project setup completed successfully"
	@echo "You can now run 'make run' to execute all experiments"

# Check code style (if tools are available)
check:
	@echo "🔍 Checking code style..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 --max-line-length=100 --ignore=E203,W503 *.py agents/ environments/ utils/ experiments/ evaluation/ models/; \
	else \
		echo "⚠️  flake8 not installed, skipping style check"; \
	fi
	@if command -v black >/dev/null 2>&1; then \
		black --check --line-length=100 *.py agents/ environments/ utils/ experiments/ evaluation/ models/; \
	else \
		echo "⚠️  black not installed, skipping formatting check"; \
	fi

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf visualizations/*.png
	rm -rf models/policy/*.json
	rm -rf models/value_function/*.json
	rm -rf models/q_table/*.json
	rm -rf evaluation/results/*
	rm -rf __pycache__
	rm -rf agents/__pycache__
	rm -rf environments/__pycache__
	rm -rf utils/__pycache__
	rm -rf experiments/__pycache__
	rm -rf evaluation/__pycache__
	rm -rf models/__pycache__
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup completed"

# Start Jupyter notebook
jupyter:
	@echo "📓 Starting Jupyter notebook server..."
	jupyter notebook CA2.ipynb

# Create directories
dirs:
	@echo "📁 Creating necessary directories..."
	mkdir -p visualizations
	mkdir -p models/policy
	mkdir -p models/value_function
	mkdir -p models/q_table
	mkdir -p evaluation/results
	@echo "✅ Directories created"

# Development setup with additional tools
dev-setup: install
	@echo "🛠️  Setting up development environment..."
	pip install pytest black flake8 mypy
	@echo "✅ Development environment ready"

# Run specific experiment
experiment-policy:
	@echo "🎯 Running policy comparison experiment..."
	python3 -c "from experiments.experiments import experiment_policy_comparison; from environments.environments import GridWorld; experiment_policy_comparison(GridWorld())"

experiment-value:
	@echo "🎯 Running value iteration experiment..."
	python3 -c "from experiments.experiments import experiment_value_iteration; from environments.environments import GridWorld; experiment_value_iteration(GridWorld())"

experiment-qlearning:
	@echo "🎯 Running Q-learning experiment..."
	python3 -c "from experiments.experiments import experiment_q_learning; from environments.environments import GridWorld; experiment_q_learning(GridWorld())"

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	@if command -v sphinx-build >/dev/null 2>&1; then \
		cd docs && make html; \
	else \
		echo "⚠️  Sphinx not installed, skipping documentation generation"; \
	fi

# Package the project
package:
	@echo "📦 Creating project package..."
	python3 setup.py sdist bdist_wheel
	@echo "✅ Package created in dist/"

# Show project status
status:
	@echo "📊 Project Status"
	@echo "================"
	@echo "Python version: $$(python3 --version)"
	@echo "Project files: $$(find . -name "*.py" | wc -l) Python files"
	@echo "Visualizations: $$(find visualizations -name "*.png" 2>/dev/null | wc -l) plots"
	@echo "Saved models: $$(find models -name "*.json" 2>/dev/null | wc -l) models"
	@echo ""
	@echo "Directories:"
	@ls -la | grep "^d"


