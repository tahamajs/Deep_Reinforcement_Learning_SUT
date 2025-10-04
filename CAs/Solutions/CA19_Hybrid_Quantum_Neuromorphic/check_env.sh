#!/bin/bash

# Script to check Python environments and installations
# Run with: ./check_env.sh

echo "🔍 Python Environment Information"
echo "=================================="
echo ""

# Check Python versions
echo "📌 Python Installations:"
echo "------------------------"
which python 2>/dev/null && python --version 2>/dev/null || echo "  ❌ python: not found"
which python3 2>/dev/null && python3 --version 2>/dev/null || echo "  ❌ python3: not found"
which python3.10 2>/dev/null && python3.10 --version 2>/dev/null || echo "  ⚠️  python3.10: not found"
which python3.11 2>/dev/null && python3.11 --version 2>/dev/null || echo "  ⚠️  python3.11: not found"
echo ""

# Check pip versions
echo "📦 Pip Installations:"
echo "---------------------"
which pip 2>/dev/null && pip --version 2>/dev/null || echo "  ❌ pip: not found"
which pip3 2>/dev/null && pip3 --version 2>/dev/null || echo "  ❌ pip3: not found"
echo ""

# Check current Python executable
echo "🎯 Current Default Python:"
echo "--------------------------"
python3 -c "import sys; print(f'  Executable: {sys.executable}')"
python3 -c "import sys; print(f'  Version: {sys.version}')"
python3 -c "import sys; print(f'  Prefix: {sys.prefix}')"
echo ""

# Check if in a virtual environment
echo "🌐 Virtual Environment Status:"
echo "------------------------------"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "  ✅ Inside virtual environment"
    echo "  Location: $VIRTUAL_ENV"
else
    echo "  ℹ️  Not in a virtual environment (using system Python)"
fi
echo ""

# Check for conda
echo "🐍 Conda Environments:"
echo "----------------------"
if command -v conda &> /dev/null; then
    echo "  ✅ Conda is installed"
    conda --version
    echo ""
    echo "  Available conda environments:"
    conda env list
else
    echo "  ℹ️  Conda not found (not installed or not in PATH)"
fi
echo ""

# Check for venv directories in current and parent directories
echo "📁 Virtual Environment Directories:"
echo "-----------------------------------"
SEARCH_DIRS=(
    "."
    ".."
    "$HOME"
)

found_venvs=0
for dir in "${SEARCH_DIRS[@]}"; do
    if [ -d "$dir/venv" ]; then
        echo "  ✅ Found venv in: $(cd "$dir" && pwd)/venv"
        found_venvs=$((found_venvs + 1))
    fi
    if [ -d "$dir/.venv" ]; then
        echo "  ✅ Found .venv in: $(cd "$dir" && pwd)/.venv"
        found_venvs=$((found_venvs + 1))
    fi
    if [ -d "$dir/env" ]; then
        echo "  ✅ Found env in: $(cd "$dir" && pwd)/env"
        found_venvs=$((found_venvs + 1))
    fi
done

if [ $found_venvs -eq 0 ]; then
    echo "  ℹ️  No virtual environment directories found in common locations"
fi
echo ""

# Check Python path
echo "📚 Python Module Search Path:"
echo "-----------------------------"
python3 -c "import sys; [print(f'  {i+1}. {p}') for i, p in enumerate(sys.path[:5])]"
echo "  ... (showing first 5 paths)"
echo ""

# Check installed key packages
echo "📦 Key Installed Packages:"
echo "--------------------------"
packages=("numpy" "torch" "gymnasium" "matplotlib" "qiskit")
for pkg in "${packages[@]}"; do
    version=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  ✅ $pkg: $version"
    else
        echo "  ❌ $pkg: not installed"
    fi
done
echo ""

# Summary
echo "📊 Summary:"
echo "-----------"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "  • You ARE using a virtual environment"
    echo "  • Location: $VIRTUAL_ENV"
else
    echo "  • You are NOT using a virtual environment"
    echo "  • Using system Python: $(python3 -c 'import sys; print(sys.executable)')"
fi
echo ""
echo "💡 Tips:"
echo "--------"
if [ -z "$VIRTUAL_ENV" ]; then
    echo "  • To create a virtual environment: python3 -m venv venv"
    echo "  • To activate it: source venv/bin/activate"
else
    echo "  • To deactivate virtual environment: deactivate"
fi
echo "  • To see all pip packages: pip3 list"
echo "  • To see conda envs: conda env list"
echo ""
