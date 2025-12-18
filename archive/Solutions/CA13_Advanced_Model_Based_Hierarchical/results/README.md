# CA13 Experiment Results

This directory contains results from running various advanced RL experiments.

## Files

- `comprehensive_demo.png`: Visualization from comprehensive demonstration
- `experiment_logs/`: Detailed training logs
- `plots/`: Generated comparison plots

## Quick Summary

### Methods Compared:
1. Model-Free DQN
2. Model-Based Agent
3. Sample-Efficient Agent
4. Hierarchical Agents (Options-Critic, Feudal)
5. Integrated Advanced Agent

### Key Findings:
- Model-based methods show better sample efficiency
- Sample-efficient techniques reduce training time by 2-3x
- Hierarchical methods excel in long-horizon tasks
- Integrated approaches balance complexity and performance

## Reproducing Results

Run the comprehensive demonstration:
```bash
python experiments/demo_comprehensive.py
```

Or run the full notebook:
```bash
jupyter notebook CA13.ipynb
```
