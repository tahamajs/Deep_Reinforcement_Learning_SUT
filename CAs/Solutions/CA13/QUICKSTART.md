# CA13 Quick Start Guide

Welcome to Computer Assignment 13: Advanced Deep Reinforcement Learning!

## Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify Installation**:
```python
import CA13
print(CA13.get_version())
```

## Running the Notebook

### Option 1: Jupyter Notebook
```bash
jupyter notebook CA13.ipynb
```

### Option 2: VS Code
Open `CA13.ipynb` in VS Code with Jupyter extension installed.

### Option 3: Run Demo Script
```bash
cd experiments
python demo_comprehensive.py
```

## Notebook Structure

The notebook is organized into sections:

1. **Setup & Imports** - Configure environment
2. **Model-Free vs Model-Based** - Compare approaches
3. **World Models** - VAE-based dynamics learning
4. **Imagination Planning** - Plan in latent space
5. **Sample Efficiency** - Prioritized replay, augmentation
6. **Transfer Learning** - Knowledge reuse
7. **Hierarchical RL** - Options-Critic, Feudal Networks
8. **Comprehensive Evaluation** - Compare all methods
9. **Integration** - Combined advanced techniques
10. **Conclusions** - Summary and future directions

## Running Individual Sections

Each section can be run independently. Simply:

1. Run the setup cells (Sections 1)
2. Jump to any section of interest
3. Execute cells sequentially within that section

## Expected Runtime

- **Full Notebook**: ~30-45 minutes
- **Per Section**: ~5-10 minutes
- **Quick Demo**: ~10 minutes (fewer episodes)

## Common Issues

### Issue: Import Errors
**Solution**: Ensure you're in the correct directory
```bash
cd /path/to/CA13
python -c "import CA13; print('Success!')"
```

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or use CPU
```python
device = torch.device("cpu")
```

### Issue: Environment Not Found
**Solution**: Install gymnasium
```bash
pip install gymnasium
```

## Key Features

✅ **Modular Design**: Clean separation of agents, environments, utilities  
✅ **Type Hints**: Full type annotations for clarity  
✅ **Documentation**: Comprehensive docstrings  
✅ **Visualization**: Built-in plotting utilities  
✅ **Reproducibility**: Fixed random seeds  
✅ **Error Handling**: Robust error messages  

## Next Steps

After completing this assignment:

1. **Experiment**: Try different hyperparameters
2. **Extend**: Add new environments or agents
3. **Compare**: Run your own ablation studies
4. **Apply**: Use techniques in your own projects

## Getting Help

- Check docstrings: `help(CA13.DQNAgent)`
- Read CA13.md for detailed theory
- Review code in `agents/`, `models/`, `environments/`
- Run tests: `python -m pytest tests/` (if available)

## Acknowledgments

Based on state-of-the-art research in:
- World Models (Ha & Schmidhuber, 2018)
- Dreamer (Hafner et al., 2020)
- Options-Critic (Bacon et al., 2017)
- Feudal Networks (Vezhnevets et al., 2017)

---

**Good luck and happy learning!** 🚀
