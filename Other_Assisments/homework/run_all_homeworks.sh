#!/bin/bash

# ======================================================================================
# Master Automatio    cd "$SCRIPT_DIR"
    echo ""
    echo "✅ HW4 Phase Complete"
    echo ""
fi

# ======================================================================================
# HW5: Exploration, SAC, and Meta-Learning
# ======================================================================================
if [ "$RUN_HW5" = true ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║  HW5: Exploration, SAC, and Meta-Learning                                  ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    cd "$SCRIPT_DIR/hw5"
    
    if [ -f "run_all_hw5.sh" ]; then
        chmod +x run_all_hw5.sh
        ./run_all_hw5.sh || echo "⚠️  HW5 completed with warnings"
    else
        echo "❌ run_all_hw5.sh not found in hw5/"
    fi
    
    cd "$SCRIPT_DIR"
    echo ""
    echo "✅ HW5 Phase Complete"
    echo ""
fiipt for Deep RL Homework Assignments
#
# Author: GitHub Copilot
# Date: October 3, 2025
#
# This script runs all homework automation scripts sequentially.
#
# Usage:
#   chmod +x run_all_homeworks.sh
#   ./run_all_homeworks.sh
#
# Options:
#   --skip-mujoco    Skip all MuJoCo-dependent experiments
#   --hw2-only       Run only HW2
#   --hw3-only       Run only HW3
#   --hw4-only       Run only HW4
# ======================================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_MUJOCO=false
RUN_HW2=true
RUN_HW3=true
RUN_HW4=true
RUN_HW5=true

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-mujoco)
            SKIP_MUJOCO=true
            ;;
        --hw2-only)
            RUN_HW2=true
            RUN_HW3=false
            RUN_HW4=false
            ;;
        --hw3-only)
            RUN_HW2=false
            RUN_HW3=true
            RUN_HW4=false
            ;;
        --hw4-only)
            RUN_HW2=false
            RUN_HW3=false
            RUN_HW4=true
            RUN_HW5=false
            ;;
        --hw5-only)
            RUN_HW2=false
            RUN_HW3=false
            RUN_HW4=false
            RUN_HW5=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--skip-mujoco] [--hw2-only] [--hw3-only] [--hw4-only] [--hw5-only]"
            exit 1
            ;;
    esac
done

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║        🚀 Deep RL Homework - Master Automation Script 🚀                  ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Run HW2: $RUN_HW2"
echo "  Run HW3: $RUN_HW3"
echo "  Run HW4: $RUN_HW4"
echo "  Run HW5: $RUN_HW5"
echo "  Skip MuJoCo: $SKIP_MUJOCO"
echo ""

# Check MuJoCo availability
MUJOCO_AVAILABLE=false
if python3 -c "import mujoco_py" 2>/dev/null; then
    MUJOCO_AVAILABLE=true
    echo "✅ MuJoCo (mujoco-py) detected and available"
else
    echo "⚠️  MuJoCo (mujoco-py) not found - some experiments will be skipped"
    if [ "$SKIP_MUJOCO" = false ]; then
        echo "   To install: See AUTOMATION_GUIDE.md for MuJoCo setup instructions"
    fi
fi

if [ "$SKIP_MUJOCO" = true ] && [ "$MUJOCO_AVAILABLE" = false ]; then
    echo "   Proceeding with non-MuJoCo experiments only"
fi

echo ""
read -p "Press Enter to start automation (or Ctrl+C to cancel)..." 

START_TIME=$(date +%s)

# ======================================================================================
# HW2: Policy Gradients
# ======================================================================================
if [ "$RUN_HW2" = true ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║  HW2: Policy Gradients                                                     ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    cd "$SCRIPT_DIR/hw2"
    
    if [ -f "run_all_hw2.sh" ]; then
        chmod +x run_all_hw2.sh
        ./run_all_hw2.sh || echo "⚠️  HW2 completed with warnings"
    else
        echo "❌ run_all_hw2.sh not found in hw2/"
    fi
    
    cd "$SCRIPT_DIR"
    echo ""
    echo "✅ HW2 Phase Complete"
    echo ""
fi

# ======================================================================================
# HW3: DQN and Actor-Critic
# ======================================================================================
if [ "$RUN_HW3" = true ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║  HW3: DQN and Actor-Critic                                                 ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    cd "$SCRIPT_DIR/hw3"
    
    if [ -f "run_all_hw3.sh" ]; then
        chmod +x run_all_hw3.sh
        ./run_all_hw3.sh || echo "⚠️  HW3 completed with warnings"
    else
        echo "❌ run_all_hw3.sh not found in hw3/"
    fi
    
    cd "$SCRIPT_DIR"
    echo ""
    echo "✅ HW3 Phase Complete"
    echo ""
fi

# ======================================================================================
# HW4: Model-Based RL
# ======================================================================================
if [ "$RUN_HW4" = true ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║  HW4: Model-Based RL                                                       ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    if [ "$MUJOCO_AVAILABLE" = false ]; then
        echo "⚠️  Skipping HW4 - Requires MuJoCo"
        echo "   Install MuJoCo and mujoco-py to enable HW4 experiments"
    else
        cd "$SCRIPT_DIR/hw4"
        
        if [ -f "run_all_hw4.sh" ]; then
            chmod +x run_all_hw4.sh
            ./run_all_hw4.sh || echo "⚠️  HW4 completed with warnings"
        else
            echo "❌ run_all_hw4.sh not found in hw4/"
        fi
        
        cd "$SCRIPT_DIR"
        echo ""
        echo "✅ HW4 Phase Complete"
        echo ""
    fi
fi

# ======================================================================================
# Summary
# ======================================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║                   🎉 All Homework Automation Complete! 🎉                  ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Total execution time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "Results Summary"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

if [ "$RUN_HW2" = true ]; then
    echo "📂 HW2: Policy Gradients"
    if [ -d "$SCRIPT_DIR/hw2/results_hw2" ]; then
        echo "   ├─ Logs:   hw2/results_hw2/logs/"
        echo "   ├─ Plots:  hw2/results_hw2/plots/"
        echo "   └─ Videos: hw2/results_hw2/videos/"
    else
        echo "   └─ No results directory found"
    fi
    echo ""
fi

if [ "$RUN_HW3" = true ]; then
    echo "📂 HW3: DQN and Actor-Critic"
    if [ -d "$SCRIPT_DIR/hw3/results_hw3" ]; then
        echo "   ├─ Logs:   hw3/results_hw3/logs/"
        echo "   ├─ Plots:  hw3/results_hw3/plots/"
        echo "   └─ Videos: hw3/results_hw3/videos/"
    else
        echo "   └─ No results directory found"
    fi
    echo ""
fi

if [ "$RUN_HW4" = true ] && [ "$MUJOCO_AVAILABLE" = true ]; then
    echo "📂 HW4: Model-Based RL"
    if [ -d "$SCRIPT_DIR/hw4/results_hw4" ]; then
        echo "   ├─ Logs:   hw4/results_hw4/logs/"
        echo "   └─ Plots:  hw4/results_hw4/plots/"
    else
        echo "   └─ No results directory found"
    fi
    echo ""
fi

if [ "$RUN_HW5" = true ]; then
    echo "📂 HW5: Exploration, SAC, Meta-Learning"
    if [ -d "$SCRIPT_DIR/hw5/results_hw5" ]; then
        echo "   ├─ Logs:   hw5/results_hw5/logs/"
        echo "   └─ Plots:  hw5/results_hw5/plots/"
    else
        echo "   └─ No results directory found"
    fi
    echo ""
fi

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "Quick View Commands"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
if [ "$RUN_HW2" = true ]; then
    echo "View HW2 plots:"
    echo "  open hw2/results_hw2/plots/*.png"
    echo "  open hw2/results_hw2/videos/*.mp4"
    echo ""
fi
if [ "$RUN_HW3" = true ]; then
    echo "View HW3 plots:"
    echo "  open hw3/results_hw3/plots/*.png"
    echo "  open hw3/results_hw3/videos/*.mp4"
    echo ""
fi
if [ "$RUN_HW4" = true ] && [ "$MUJOCO_AVAILABLE" = true ]; then
    echo "View HW4 plots:"
    echo "  open hw4/results_hw4/plots/*.jpg"
    echo ""
fi
if [ "$RUN_HW5" = true ]; then
    echo "View HW5 plots:"
    echo "  open hw5/results_hw5/plots/*.png"
    echo ""
fi

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "Documentation"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📖 Full automation guide: cat AUTOMATION_GUIDE.md"
echo "📖 Individual READMEs:    cat hw{2,3,4}/README.md"
echo ""
echo "To re-run individual homeworks:"
echo "  cd hw2 && ./run_all_hw2.sh"
echo "  cd hw3 && ./run_all_hw3.sh"
echo "  cd hw4 && ./run_all_hw4.sh"
echo "  cd hw5 && ./run_all_hw5.sh"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
