# CA7 Quick Start Guide

## What Changed?

Your notebook has been completely restructured:
- ✅ **Old notebook**: 978KB with mixed code and markdown
- ✅ **New notebook**: 23KB with IEEE format and clean imports
- ✅ **All code**: Now in separate Python files in `agents/` directory

## How to Run the Notebook

### Option 1: Jupyter Notebook
```bash
cd /Users/tahamajs/Documents/uni/DRL/CAs/Solutions/CA7
jupyter notebook CA7.ipynb
```

### Option 2: VS Code
1. Open `CA7.ipynb` in VS Code
2. Select Python kernel
3. Run cells sequentially

### Option 3: JupyterLab
```bash
jupyter lab CA7.ipynb
```

## Notebook Structure

The notebook is organized like an IEEE paper:

1. **Abstract** - Overview and keywords
2. **I. INTRODUCTION** - Motivation and contributions
3. **II. THEORETICAL FOUNDATIONS** - Math and theory
4. **III. SETUP** - Import all modules
5. **IV. VISUALIZATIONS** - Core concepts
6. **V. BASIC DQN** - Implementation and training
7. **VI. DOUBLE DQN** - Bias reduction
8. **VII. DUELING DQN** - Value decomposition
9. **VIII. COMPREHENSIVE COMPARISON** - Final results
10. **IX. CONCLUSIONS** - Summary and references

## Key Imports

```python
from agents.core import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.dueling_dqn import DuelingDQNAgent
from agents.utils import QNetworkVisualization, PerformanceAnalyzer
```

## Quick Example

```python
import gymnasium as gym
from agents.core import DQNAgent

# Create environment
env = gym.make('CartPole-v1')

# Create and train agent
agent = DQNAgent(
    state_dim=4,
    action_dim=2,
    lr=1e-3,
    gamma=0.99
)

# Train for 100 episodes
for episode in range(100):
    reward, steps = agent.train_episode(env)
    
# Evaluate
results = agent.evaluate(env, num_episodes=10)
print(f"Mean Reward: {results['mean_reward']:.2f}")
```

## Running Experiments

### Basic DQN Experiment
```bash
python experiments/basic_dqn_experiment.py
```

### Comprehensive Analysis
```bash
python experiments/comprehensive_dqn_analysis.py
```

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`:
```python
import sys
sys.path.append('.')  # Add current directory to path
```

### Visualization Issues
If plots don't show:
```python
%matplotlib inline  # In Jupyter
# or
plt.show()  # At end of plotting code
```

### Dependencies
Install all requirements:
```bash
pip install -r requirements.txt
```

## What's Where?

- **Notebook**: `CA7.ipynb` - Clean IEEE-formatted notebook
- **Core DQN**: `agents/core.py` - Basic DQN implementation
- **Double DQN**: `agents/double_dqn.py` - Overestimation bias reduction
- **Dueling DQN**: `agents/dueling_dqn.py` - Value-advantage decomposition
- **Utilities**: `agents/utils.py` - Visualization and analysis
- **Examples**: `training_examples.py` - Standalone training scripts
- **Experiments**: `experiments/` - Full experiment scripts

## Tips

1. **Run cells in order** - The notebook builds progressively
2. **Check imports first** - Run the setup cell before others
3. **Adjust episodes** - Reduce for faster testing, increase for better results
4. **Save checkpoints** - Use `agent.save('checkpoint.pt')` to save progress
5. **Compare variants** - The final section compares all algorithms

## Need Help?

- Check `README.md` for detailed documentation
- Check `CHANGES.md` for what changed
- Review the code in `agents/` directory
- Look at `training_examples.py` for standalone examples

## Original Notebook

Your original notebook is backed up as `CA7_old.ipynb`
