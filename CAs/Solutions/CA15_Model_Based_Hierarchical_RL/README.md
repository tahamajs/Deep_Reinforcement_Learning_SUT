# CA15: Advanced Deep Reinforcement Learning - Model-Based RL and Hierarchical RL

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project directory
cd CA15_Model_Based_Hierarchical_RL

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x run.sh
chmod +x run_all_experiments.py
```

### Run All Experiments

```bash
# Option 1: Using bash script
./run.sh

# Option 2: Using Python script
python3 run_all_experiments.py --all

# Option 3: Run specific experiments
python3 run_all_experiments.py --model-based
python3 run_all_experiments.py --hierarchical
python3 run_all_experiments.py --planning
```

### Run Jupyter Notebook

```bash
jupyter notebook CA15.ipynb
```

## 📁 Project Structure

```
CA15_Model_Based_Hierarchical_RL/
├── run.sh                           # Main bash script for all experiments
├── run_all_experiments.py           # Complete Python experiment runner
├── CA15.ipynb                       # Main Jupyter notebook
├── training_examples.py             # Training examples and utilities
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── model_based_rl/                  # Model-Based RL implementations
│   ├── __init__.py
│   └── algorithms.py                # DynamicsModel, ModelEnsemble, MPC, DynaQ
│
├── hierarchical_rl/                 # Hierarchical RL implementations
│   ├── __init__.py
│   ├── algorithms.py                # Options, HAC, Goal-Conditioned, Feudal
│   └── environments.py              # Hierarchical environment wrappers
│
├── planning/                        # Advanced planning algorithms
│   ├── __init__.py
│   └── algorithms.py                # MCTS, MVE, LatentSpacePlanner, WorldModel
│
├── experiments/                     # Experiment runners and evaluation
│   ├── __init__.py
│   ├── runner.py                    # Unified experiment runner
│   ├── hierarchical.py              # Hierarchical RL experiments
│   └── planning.py                  # Planning algorithm experiments
│
├── environments/                    # Custom test environments
│   ├── __init__.py
│   └── grid_world.py                # Simple grid world environment
│
├── utils/                           # Utility functions and classes
│   └── __init__.py                  # ReplayBuffer, Logger, VisualizationUtils, etc.
│
├── visualizations/                  # Generated plots and analysis (created after running)
├── results/                         # Experiment results and reports (created after running)
├── logs/                            # Training logs (created after running)
└── data/                            # Collected training data (created after running)
```

## 🧪 Available Experiments

### 1. Model-Based RL Algorithms

- **DynamicsModel**: Neural network for learning environment dynamics
- **ModelEnsemble**: Ensemble methods for uncertainty quantification
- **ModelPredictiveController**: MPC using learned dynamics
- **DynaQAgent**: Combining model-free and model-based learning

### 2. Hierarchical RL Algorithms

- **Option**: Options framework implementation
- **HierarchicalActorCritic**: Multi-level policies with different time scales
- **GoalConditionedAgent**: Goal-conditioned RL with Hindsight Experience Replay
- **FeudalNetwork**: Manager-worker architecture for goal-directed behavior

### 3. Planning Algorithms

- **MonteCarloTreeSearch**: MCTS with neural network guidance
- **ModelBasedValueExpansion**: Recursive value expansion using learned models
- **LatentSpacePlanner**: Planning in learned compact representations
- **WorldModel**: End-to-end models for environment simulation and control

## 📊 Expected Results

After running the experiments, you'll find:

### Generated Files

- `visualizations/ca15_complete_analysis_*.png`: Comprehensive analysis plots
- `results/ca15_experiment_report_*.md`: Detailed experiment report
- `results/`: All experiment outputs and data
- `logs/`: Training logs and metrics
- `data/`: Collected training data

### Key Metrics

- **Sample Efficiency**: Episodes needed to reach performance threshold
- **Final Performance**: Average reward in final episodes
- **Computational Overhead**: Planning time per episode
- **Success Rate**: Goal achievement rate for hierarchical methods

## 🔧 Customization

### Adding New Algorithms

1. Create your algorithm class in the appropriate module (`model_based_rl/`, `hierarchical_rl/`, or `planning/`)
2. Add it to the `__init__.py` file
3. Include it in the experiment runner (`run_all_experiments.py`)

### Custom Environments

1. Create your environment class in `environments/`
2. Implement the standard RL interface (`reset()`, `step()`)
3. Update the experiment runners to use your environment

### Hyperparameter Tuning

Modify the hyperparameters in `run_all_experiments.py`:

```python
# Example: Modify Dyna-Q parameters
dyna_agent = DynaQAgent(
    env.state_dim,
    env.action_dim,
    lr=1e-3,        # Learning rate
    gamma=0.99,      # Discount factor
    epsilon=0.1      # Exploration rate
)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Reduce batch sizes or use CPU
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Import Errors**

   ```bash
   # Make sure you're in the project directory
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Permission Denied**

   ```bash
   # Make scripts executable
   chmod +x run.sh run_all_experiments.py
   ```

4. **Missing Dependencies**
   ```bash
   # Install all requirements
   pip install -r requirements.txt
   ```

### Performance Issues

- **Slow Training**: Reduce number of episodes or use smaller networks
- **High Memory Usage**: Reduce batch sizes or use gradient accumulation
- **Convergence Issues**: Adjust learning rates or add learning rate scheduling

## 📈 Understanding Results

### Learning Curves

- **Steep Initial Rise**: Good sample efficiency
- **Plateau**: Algorithm has converged
- **High Variance**: Unstable training (try reducing learning rate)

### Performance Comparison

- **Model-Based RL**: Best for sample efficiency
- **Hierarchical RL**: Best for complex multi-goal tasks
- **Planning**: Best for final performance (with computational cost)

### Computational Trade-offs

- **MCTS**: Highest computational cost, best performance
- **MPC**: Moderate cost, good performance
- **Dyna-Q**: Low cost, good sample efficiency

## 🔬 Advanced Usage

### Running Specific Experiments

```bash
# Only model-based experiments
python3 run_all_experiments.py --model-based

# Only hierarchical experiments
python3 run_all_experiments.py --hierarchical

# Only planning experiments
python3 run_all_experiments.py --planning
```

### Custom Experiment Configuration

Modify `run_all_experiments.py` to:

- Change number of episodes
- Modify environment parameters
- Adjust algorithm hyperparameters
- Add new evaluation metrics

### Integration with Other Projects

```python
# Import specific algorithms
from model_based_rl.algorithms import DynaQAgent
from hierarchical_rl.algorithms import GoalConditionedAgent
from planning.algorithms import MonteCarloTreeSearch

# Use in your own experiments
agent = DynaQAgent(state_dim=4, action_dim=2)
# ... your training loop
```

## 📚 References

1. **Model-Based RL**: Deisenroth et al. (2011) - PILCO
2. **Dyna-Q**: Sutton (1990) - Integrated Architectures
3. **Options Framework**: Sutton et al. (1999) - Between MDPs and Semi-MDPs
4. **HAC**: Levy et al. (2019) - Hierarchical Actor-Critic
5. **HER**: Andrychowicz et al. (2017) - Hindsight Experience Replay
6. **Feudal Networks**: Vezhnevets et al. (2017) - Feudal Networks
7. **MCTS**: Coulom (2006) - Efficient Selectivity
8. **World Models**: Ha & Schmidhuber (2018) - World Models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Deep Reinforcement Learning course at Sharif University of Technology.

---

**Happy Learning! 🎓**

For questions or issues, please check the troubleshooting section or create an issue in the repository.
