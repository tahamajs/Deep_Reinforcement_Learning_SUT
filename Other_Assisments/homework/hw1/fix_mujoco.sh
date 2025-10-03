#!/bin/bash

################################################################################
# MuJoCo Environment Checker and Fixer
# This script helps diagnose and fix MuJoCo issues
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           MuJoCo Environment Checker & Fixer                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
    echo -e "${RED}⚠ WARNING: Python 3.$PYTHON_MINOR is too new for old MuJoCo (1.50)${NC}"
    echo -e "${YELLOW}Recommendation: Use Python 3.9 or 3.10 for MuJoCo 1.50${NC}"
    echo ""
fi

# Check if MuJoCo works
echo -e "${YELLOW}Testing MuJoCo installation...${NC}"
if python -c "import mujoco_py; print('MuJoCo version:', mujoco_py.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓ MuJoCo is working!${NC}"
    exit 0
else
    echo -e "${RED}✗ MuJoCo is NOT working${NC}"
    echo ""
fi

# Check macOS version
echo -e "${YELLOW}Checking macOS version...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    MACOS_VERSION=$(sw_vers -productVersion)
    echo "macOS version: $MACOS_VERSION"
    
    # Check for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo -e "${RED}⚠ Apple Silicon (M1/M2/M3) detected${NC}"
        echo -e "${YELLOW}Old MuJoCo (1.50) has limited support on Apple Silicon${NC}"
    fi
fi
echo ""

# Provide solutions
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    RECOMMENDED SOLUTIONS                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Option 1: Run Test Without MuJoCo (Easiest)${NC}"
echo "  This tests your BC implementation without MuJoCo:"
echo -e "  ${CYAN}python test_without_mujoco.py${NC}"
echo ""

echo -e "${GREEN}Option 2: Use Google Colab (Best for MuJoCo)${NC}"
echo "  1. Go to https://colab.research.google.com"
echo "  2. Upload your hw1 folder"
echo "  3. Run the setup cells (install MuJoCo in Linux)"
echo "  4. Execute your pipeline"
echo ""

echo -e "${GREEN}Option 3: Use Docker (Linux Container)${NC}"
echo "  Run MuJoCo in a Linux container:"
echo -e "  ${CYAN}docker run -v \$(pwd):/workspace -it tensorflow/tensorflow:latest-py3 bash${NC}"
echo "  Then install dependencies inside container"
echo ""

echo -e "${GREEN}Option 4: Upgrade to Modern Gymnasium (Recommended Long-term)${NC}"
echo "  Modern gymnasium works on Apple Silicon:"
echo ""
echo "  Step 1: Create new environment"
echo -e "  ${CYAN}deactivate${NC}"
echo -e "  ${CYAN}python3.10 -m venv venv_modern${NC}"
echo -e "  ${CYAN}source venv_modern/bin/activate${NC}"
echo ""
echo "  Step 2: Install modern packages"
echo -e "  ${CYAN}pip install gymnasium[mujoco] tensorflow numpy matplotlib seaborn${NC}"
echo ""
echo "  Step 3: Minor code updates needed (see MUJOCO_SETUP.md)"
echo ""

echo -e "${GREEN}Option 5: Install GCC and Try Old MuJoCo${NC}"
echo "  (This may not work on Apple Silicon)"
echo ""
echo "  Step 1: Install older GCC"
echo -e "  ${CYAN}brew install gcc@7${NC}"
echo ""
echo "  Step 2: Use Python 3.9 or 3.10"
echo -e "  ${CYAN}pyenv install 3.9.13${NC}"
echo -e "  ${CYAN}pyenv local 3.9.13${NC}"
echo ""
echo "  Step 3: Reinstall mujoco-py"
echo -e "  ${CYAN}pip install mujoco-py==1.50.1.56${NC}"
echo ""

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    QUICK ACTION                                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

read -p "Do you want to run the test WITHOUT MuJoCo now? (Y/n): " RUN_TEST
RUN_TEST=${RUN_TEST:-Y}

if [[ $RUN_TEST =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${GREEN}Running test without MuJoCo...${NC}"
    echo ""
    python test_without_mujoco.py
else
    echo ""
    echo -e "${YELLOW}Please choose one of the options above and try again.${NC}"
    echo ""
    echo "For detailed help, see:"
    echo "  • MUJOCO_SETUP.md"
    echo "  • FINAL_GUIDE.md"
    echo ""
fi
