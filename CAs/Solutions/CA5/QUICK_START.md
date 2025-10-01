# Quick Start Guide - CA5

## TL;DR
Run the notebook `CA5.ipynb` cell by cell. All code is properly organized in `.py` files.

---

## Setup (One-Time)

```bash
# Navigate to CA5 directory
cd /Users/tahamajs/Documents/uni/DRL/CAs/Solutions/CA5

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, gymnasium, numpy, matplotlib; print('✅ All dependencies installed')"
```

---

## Running the Notebook

### Option 1: VS Code (Recommended)
1. Open `CA5.ipynb` in VS Code
2. Click "Run All" or run cells sequentially
3. Watch the agents train and compare

### Option 2: Jupyter Lab
```bash
jupyter lab CA5.ipynb
```

### Option 3: Jupyter Notebook
```bash
jupyter notebook CA5.ipynb
```

---

## What Happens When You Run

### Cells 1-2: Setup (30 seconds)
- Imports all modules from `agents/` and `utils/`
- Configures plotting and random seeds
- Creates test environment (CartPole-v1)

### Cell 3-5: Basic DQN Training (5-10 minutes)
- Trains standard DQN agent for 500 episodes
- Plots learning curves
- Shows final performance

### Cell 6-8: Double DQN (5-10 minutes)
- Trains Double DQN agent
- Compares with standard DQN
- Analyzes overestimation bias

### Cell 9-11: Dueling DQN (5-10 minutes)
- Trains Dueling DQN agent
- Visualizes value-advantage decomposition
- Compares performance

### Cell 12-14: Prioritized DQN (5-10 minutes)
- Trains Prioritized DQN agent
- Analyzes priority distribution
- Shows sample efficiency improvements

### Cell 15-17: Rainbow DQN (5-10 minutes)
- Trains Rainbow DQN (combining all improvements)
- Comprehensive comparison
- Final performance analysis

### Cell 18-20: Comprehensive Analysis (Instant)
- Statistical comparisons
- Convergence analysis
- Performance heatmaps
- Key findings summary

**Total Runtime**: ~30-60 minutes (depends on CPU/GPU)

---

## Expected Output

You will see:

### ✅ Training Progress
```
Training Basic DQN Agent...
==================================================
Episode  100 | Avg Score:   45.23 | Avg Loss:   0.1234 | Epsilon: 0.605 | Buffer Size: 3245
Episode  200 | Avg Score:   89.45 | Avg Loss:   0.0987 | Epsilon: 0.366 | Buffer Size: 6789
...
```

### ✅ Visualizations
- Learning curves for each agent
- Box plots comparing final performance
- Overestimation bias analysis
- Priority distribution heatmaps
- Comprehensive comparison plots

### ✅ Performance Tables
```
COMPREHENSIVE PERFORMANCE SUMMARY
============================================================
Agent                Mean       Std        Max        Improvement
------------------------------------------------------------
Standard DQN        165.23     25.45      198.00     —
Double DQN          178.45     22.34      205.00     +13.22
Dueling DQN         189.67     19.87      218.00     +24.44
Prioritized DQN     201.34     18.23      225.00     +36.11
Rainbow DQN         234.56     15.67      248.00     +69.33
============================================================
```

### ✅ Analysis Insights
- Convergence speed comparisons
- Sample efficiency metrics
- Stability analysis
- Key findings and recommendations

---

## Quick Test (2 minutes)

Test that everything works before full run:

```python
# Run this in a cell or Python terminal
from agents.dqn_base import DQNAgent
from utils.ca5_helpers import create_test_environment

env, state_size, action_size = create_test_environment()
agent = DQNAgent(state_size, action_size)

# Quick 10-episode test
scores, _ = agent.train(env, num_episodes=10, print_every=5)
print(f"✅ Test passed! Final score: {scores[-1]:.2f}")
```

---

## Troubleshooting

### Issue: Import Errors
```python
# Solution: Check you're in the right directory
import os
print(os.getcwd())  # Should end with /CA5

# If not, change directory
os.chdir('/Users/tahamajs/Documents/uni/DRL/CAs/Solutions/CA5')
```

### Issue: Module Not Found
```bash
# Solution: Install missing packages
pip install torch gymnasium numpy matplotlib seaborn
```

### Issue: Slow Training
```python
# Solution: Use GPU if available
import torch
print(f"GPU Available: {torch.cuda.is_available()}")

# Or reduce episodes for quick demo
agent.train(env, num_episodes=100)  # Instead of 500
```

### Issue: Out of Memory
```python
# Solution: Reduce batch size or buffer size
agent = DQNAgent(
    state_size, 
    action_size,
    batch_size=16,      # Default: 32
    buffer_size=5000    # Default: 10000
)
```

---

## Skipping to Specific Sections

### Just Want to See Comparisons?
Run cells: 1-2 (setup), then jump to cells 18-20 (analysis)
*Note: You'll need to train agents first or load pre-trained models*

### Just Want One Algorithm?
1. Run cells 1-2 (setup)
2. Run cells for your chosen algorithm:
   - Basic DQN: Cells 3-5
   - Double DQN: Cells 6-8
   - Dueling DQN: Cells 9-11
   - Prioritized DQN: Cells 12-14
   - Rainbow DQN: Cells 15-17

---

## Customization

### Change Environment
```python
# In cell after imports
import gymnasium as gym

env = gym.make('LunarLander-v2')  # Instead of CartPole
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
```

### Adjust Training Duration
```python
# Shorter training for quick demo
num_episodes = 100  # Default: 500

# Longer training for better results
num_episodes = 1000
```

### Modify Hyperparameters
```python
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    lr=0.0005,              # Learning rate
    gamma=0.99,             # Discount factor
    epsilon=1.0,            # Initial exploration
    epsilon_decay=0.995,    # Exploration decay
    buffer_size=10000,      # Replay buffer size
    batch_size=32,          # Batch size
    target_update_freq=1000 # Target network update
)
```

---

## Save and Load Models

### Save Trained Agent
```python
# After training
torch.save(agent.q_network.state_dict(), 'trained_dqn.pth')
print("✅ Model saved to trained_dqn.pth")
```

### Load Trained Agent
```python
# Create agent and load weights
agent = DQNAgent(state_size, action_size)
agent.q_network.load_state_dict(torch.load('trained_dqn.pth'))
print("✅ Model loaded from trained_dqn.pth")
```

---

## Performance Tips

### Speed Up Training
1. **Use GPU**: Agents automatically use CUDA if available
2. **Reduce Print Frequency**: `print_every=200` instead of `100`
3. **Smaller Buffer**: `buffer_size=5000` instead of `10000`
4. **Larger Batch**: `batch_size=64` instead of `32` (if memory allows)

### Improve Results
1. **More Episodes**: Train for 1000+ episodes
2. **Tune Learning Rate**: Try 0.0001 to 0.001
3. **Adjust Epsilon Decay**: Slower decay = more exploration
4. **Increase Buffer Size**: More diverse experiences

---

## Next Steps After Running

1. **Experiment**: Modify hyperparameters and see effects
2. **New Environments**: Try LunarLander, Acrobot, MountainCar
3. **Extend**: Add new features or algorithms
4. **Analyze**: Deep dive into specific aspects
5. **Apply**: Use these agents for your own problems

---

## Getting Help

### Check Documentation
- See `README.md` for project overview
- See `COMPLETION_SUMMARY.md` for detailed info
- See `CA5.md` for theoretical background
- Check docstrings in `.py` files

### Debug Mode
```python
# Enable detailed output
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Verify Imports
```python
# Test all modules load correctly
from agents.dqn_base import DQNAgent
from agents.double_dqn import DoubleDQNAgent  
from agents.dueling_dqn import DuelingDQNAgent
from agents.prioritized_replay import PrioritizedDQNAgent
from agents.rainbow_dqn import RainbowDQNAgent
from utils.ca5_helpers import create_test_environment
from utils.analysis_tools import DQNComparator

print("✅ All modules imported successfully!")
```

---

## Summary

1. **Install**: `pip install -r requirements.txt`
2. **Run**: Open `CA5.ipynb` and run all cells
3. **Wait**: ~30-60 minutes for complete training
4. **Enjoy**: Comprehensive DQN analysis and comparisons!

**That's it!** 🎉

---

*For detailed information, see COMPLETION_SUMMARY.md and README.md*
