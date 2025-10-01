# CA10 Usage Guide

## Getting Started

### Prerequisites
Make sure you have all required packages installed:
```bash
pip install -r requirements.txt
```

### Testing the Installation
Run the test script to verify everything is set up correctly:
```bash
python test_imports.py
```

## Running the Notebook

### Option 1: Run All Cells
Open `CA10.ipynb` in Jupyter and run all cells sequentially:
```bash
jupyter notebook CA10.ipynb
```

The notebook is organized into 8 main sections:
1. **Setup** - Imports and configuration
2. **Section 1** - Theoretical foundations
3. **Section 2** - Environment model learning
4. **Section 3** - Classical planning algorithms
5. **Section 4** - Dyna-Q integrated learning
6. **Section 5** - Monte Carlo Tree Search
7. **Section 6** - Model Predictive Control
8. **Section 7-8** - Comprehensive comparison and summary

### Option 2: Run Specific Demonstrations
You can import and run individual demonstrations:

```python
# In Python or Jupyter
from agents.classical_planning import demonstrate_classical_planning
from agents.dyna_q import demonstrate_dyna_q
from agents.mcts import demonstrate_mcts
from agents.mpc import demonstrate_mpc
from experiments.comparison import demonstrate_comparison

# Run individual demonstrations
demonstrate_classical_planning()  # ~2 minutes
demonstrate_dyna_q()              # ~3 minutes
demonstrate_mcts()                # ~2 minutes
demonstrate_mpc()                 # ~2 minutes
demonstrate_comparison()          # ~5 minutes
```

## Understanding the Output

### Visualizations
All visualizations are saved to the `visualizations/` folder:
- `classical_planning.png` - Value/policy iteration results
- `dyna_q_comparison.png` - Dyna-Q vs baselines
- `mcts_analysis.png` - MCTS performance analysis
- `mpc_analysis.png` - MPC experiments
- `comprehensive_comparison.png` - Full method comparison

### Console Output
Each demonstration provides:
- Training progress updates
- Performance metrics
- Statistical analysis
- Key insights and recommendations

## Common Use Cases

### 1. Learning About Model-Based RL
Start with Section 1 and 2 to understand:
- Differences between model-free and model-based RL
- How to learn environment models
- Model validation and accuracy

### 2. Implementing Planning Algorithms
Study Section 3 and 4 for:
- Value iteration with learned models
- Policy iteration
- Dyna-Q algorithm
- Handling model uncertainty

### 3. Advanced Planning Methods
Explore Section 5 and 6 for:
- Monte Carlo Tree Search (MCTS)
- Model Predictive Control (MPC)
- Planning horizon effects
- Optimization methods

### 4. Comparative Analysis
Run Section 7-8 to:
- Compare all methods
- Understand trade-offs
- Get practical recommendations
- Identify best practices

## Customization

### Modifying Environments
Edit `environments/environments.py` to:
- Change grid world size
- Add obstacles or rewards
- Create new environments

### Adjusting Algorithms
Edit files in `agents/` to:
- Tune hyperparameters
- Modify planning strategies
- Add new planning methods

### Custom Experiments
Edit `experiments/comparison.py` to:
- Add new methods to compare
- Change evaluation metrics
- Run different environments

## Performance Optimization

### Reducing Runtime
- Decrease number of episodes in demonstrations
- Reduce planning steps or simulations
- Use smaller environments

### Improving Accuracy
- Increase model training epochs
- Use larger ensemble sizes
- Collect more training data

## Troubleshooting

### Import Errors
If you get import errors:
```python
import sys
import os
sys.path.insert(0, os.getcwd())
```

### CUDA/GPU Issues
If you have GPU issues, the code will automatically fall back to CPU:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Memory Issues
For large experiments:
- Reduce batch sizes
- Use smaller neural networks
- Decrease number of planning simulations

## Advanced Topics

### Adding New Algorithms
1. Create new file in `agents/`
2. Implement agent class with `train_episode()` method
3. Add `demonstrate_*()` function
4. Import in notebook

### Custom Environments
1. Inherit from base environment
2. Implement `reset()` and `step()` methods
3. Define `num_states` and `num_actions`
4. Test with existing agents

### Extending Analysis
1. Add methods to `comparison.py`
2. Implement new metrics
3. Create custom visualizations
4. Run statistical tests

## Best Practices

1. **Always run test_imports.py first** to verify setup
2. **Start with small experiments** before full runs
3. **Save visualizations** for later reference
4. **Document your experiments** in markdown cells
5. **Use version control** to track changes

## Getting Help

- Check the README.md for overview
- Review code comments for details
- Examine demonstration functions for examples
- Look at existing visualizations for expected output

## Next Steps

After completing CA10, you should:
1. Understand model-based RL fundamentals
2. Be able to implement planning algorithms
3. Know when to use model-based methods
4. Understand trade-offs and best practices

Continue to advanced topics:
- World models and latent planning
- Hybrid model-free/model-based methods
- Meta-learning for models
- Real-world applications

---

**Happy Learning! 🚀**
