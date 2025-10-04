# CA18 Advanced RL Paradigms - Setup Guide

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Option 1: Automated Setup (Recommended)

```bash
# Make the script executable and run it
chmod +x run.sh
./run.sh
```

This will:

- Create a virtual environment
- Install all dependencies
- Run all demos and tests
- Generate visualizations and reports

### Option 2: Manual Setup

#### Step 1: Create Virtual Environment

```bash
python3 -m venv ca18_env
source ca18_env/bin/activate  # On Windows: ca18_env\Scripts\activate
```

#### Step 2: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# OR install core dependencies manually
pip install torch numpy matplotlib seaborn pandas networkx scikit-learn scipy tqdm
```

#### Step 3: Test Installation

```bash
python test_modules.py
```

#### Step 4: Run Demos

```bash
# Run comprehensive demo
python comprehensive_demo.py

# Or run individual demos
python quantum_rl/quantum_rl_demo.py
python world_models/world_models_demo.py
python multi_agent_rl/multi_agent_rl_demo.py
python causal_rl/causal_rl_demo.py
python federated_rl/federated_rl_demo.py
```

## 📁 Project Structure

```
CA18_Advanced_RL_Paradigms/
├── run.sh                          # Main execution script
├── test_modules.py                 # Module testing script
├── comprehensive_demo.py           # Comprehensive demo (auto-generated)
├── integration_demo.py             # Integration demonstration
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # Main documentation
├── SETUP_GUIDE.md                  # This setup guide
├── quantum_rl/                     # Quantum RL algorithms
│   ├── quantum_rl.py
│   └── quantum_rl_demo.py
├── world_models/                   # World model-based RL
│   ├── world_models.py
│   └── world_models_demo.py
├── multi_agent_rl/                 # Multi-agent systems
│   ├── multi_agent_rl.py
│   └── multi_agent_rl_demo.py
├── causal_rl/                      # Causal RL algorithms
│   ├── causal_rl.py
│   └── causal_rl_demo.py
├── federated_rl/                   # Federated learning
│   ├── federated_rl.py
│   └── federated_rl_demo.py
├── advanced_safety/                # Safety mechanisms
│   └── advanced_safety.py
├── utils/                          # Utility functions
│   └── utils.py
├── environments/                   # Test environments
│   └── environments.py
├── experiments/                    # Experiment frameworks
│   └── experiments.py
├── visualizations/                 # Generated plots (after running)
├── results/                        # Results and reports
└── logs/                          # Execution logs
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "No module named 'torch'" Error

**Solution**: Install PyTorch in your virtual environment

```bash
source ca18_env/bin/activate
pip install torch torchvision
```

#### 2. Virtual Environment Issues

**Solution**: Recreate the virtual environment

```bash
rm -rf ca18_env
python3 -m venv ca18_env
source ca18_env/bin/activate
pip install -r requirements.txt
```

#### 3. Permission Errors on macOS/Linux

**Solution**: Make scripts executable

```bash
chmod +x run.sh
chmod +x test_modules.py
```

#### 4. Import Errors

**Solution**: Ensure you're in the correct directory and virtual environment is activated

```bash
cd /path/to/CA18_Advanced_RL_Paradigms
source ca18_env/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 🎯 Usage Examples

### Basic Usage

```python
from quantum_rl.quantum_rl import QuantumQLearning
from world_models.world_models import WorldModel
from multi_agent_rl.multi_agent_rl import MADDPGAgent

# Create a quantum RL agent
agent = QuantumQLearning(n_qubits=4, n_actions=4)

# Create a world model
world_model = WorldModel(obs_dim=8, action_dim=2)

# Create a multi-agent system
agent = MADDPGAgent(agent_idx=0, obs_dim=6, action_dim=2, n_agents=3)
```

### Running Experiments

```python
from experiments.experiments import QuantumRLExperiment
from quantum_rl.quantum_rl import QuantumQLearning
from environments.environments import QuantumEnvironment

# Create experiment
experiment = QuantumRLExperiment(
    agent_class=QuantumQLearning,
    environment_class=QuantumEnvironment,
    experiment_name="quantum_rl_test"
)

# Run experiment
results = experiment.run_experiment(
    agent_kwargs={'n_qubits': 4},
    env_kwargs={'n_qubits': 4}
)
```

## 📊 Generated Outputs

After running the demos, you'll find:

### Visualizations (`visualizations/`)

- `quantum_rl_results.png` - Quantum learning curves
- `world_models_training.png` - Model training losses
- `multi_agent_rl_results.png` - Multi-agent coordination
- `causal_rl_training.png` - Causal model training
- `federated_rl_results.png` - Federated learning performance
- `safety_results.png` - Safety constraint analysis
- `environment_testing.png` - Environment test results
- `utility_functions.png` - Utility function demonstrations
- `final_summary.png` - Overall completion status

### Results (`results/`)

- `final_report.md` - Complete execution summary
- `execution_summary.md` - Technical details
- Various JSON files with detailed metrics

### Logs (`logs/`)

- Detailed execution logs for each module
- Error logs if any issues occurred
- Performance metrics and timing information

## 🔬 Advanced Features

### Custom Configuration

Edit `config.py` to modify hyperparameters:

```python
from config import get_config, update_config

# Get default config
quantum_config = get_config("quantum")

# Update specific parameters
updated_config = update_config("quantum", {
    "n_episodes_q": 200,
    "learning_rate_q": 0.05
})
```

### Custom Environments

Create your own environments by extending the base classes:

```python
from environments.environments import QuantumEnvironment

class CustomQuantumEnv(QuantumEnvironment):
    def __init__(self):
        super().__init__(n_qubits=6, max_steps=100)
        # Add custom initialization

    def step(self, action):
        # Add custom dynamics
        return super().step(action)
```

### Custom Agents

Extend existing agent classes:

```python
from quantum_rl.quantum_rl import QuantumQLearning

class CustomQuantumAgent(QuantumQLearning):
    def __init__(self, custom_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param

    def custom_method(self):
        # Add custom functionality
        pass
```

## 🚀 Next Steps

1. **Experiment with Parameters**: Try different hyperparameters in `config.py`
2. **Create Custom Environments**: Build domain-specific environments
3. **Extend Algorithms**: Add new features to existing algorithms
4. **Real Quantum Hardware**: Integrate with actual quantum computers
5. **Applications**: Apply to real-world problems in your domain

## 📚 Documentation

- **README.md**: Main project documentation
- **Individual module files**: Detailed code documentation
- **Demo files**: Usage examples and demonstrations
- **Generated reports**: Execution summaries and results

## 🤝 Contributing

To contribute to CA18:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:

1. Check this setup guide
2. Review the logs in `logs/` directory
3. Ensure all dependencies are installed
4. Verify you're using the correct Python version
5. Check that the virtual environment is activated

---

_CA18 Advanced RL Paradigms - Comprehensive Implementation_
_Ready for cutting-edge reinforcement learning research_
