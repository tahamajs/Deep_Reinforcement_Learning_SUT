# AI Coding Guidelines for DRL Course Repository

## Project Overview

This is an educational repository for Deep Reinforcement Learning (DRL) course assignments at Sharif University of Technology. Each Computer Assignment (CA) implements RL algorithms from basic concepts to advanced topics like quantum RL and neuromorphic computing.

## Architecture & Structure

- **Modular Design**: Each CA has `agents/`, `environments/`, `experiments/`, `models/`, `utils/` subdirectories
- **Progressive Complexity**: CAs build from fundamentals (CA1: MDP basics) to cutting-edge (CA19: quantum RL)
- **Framework Stack**: PyTorch for neural networks, Gymnasium for environments, NumPy/Matplotlib for computation/visualization

## Key Patterns & Conventions

### Agent Implementations

- **Class Naming**: `{Algorithm}Agent` (e.g., `DQNAgent`, `REINFORCEAgent`)
- **Core Methods**: `select_action()`, `update()`, `train()`
- **Training Functions**: `train_{algorithm}_agent()` returning score history
- **Example**: `agents/dqn_agent.py` in CA4 shows DQN with experience replay and target networks

### Neural Network Models

- **Base Class**: Inherit from `torch.nn.Module`
- **Naming**: `{Algorithm}Network` or `{Component}Network` (e.g., `PolicyNetwork`, `ValueNetwork`)
- **Forward Pass**: Return action distributions or Q-values
- **Example**: `models/policy_network.py` uses MLP with softmax for discrete policies

### Training Loops

- **Progress Tracking**: Use `tqdm` for episode progress bars
- **Metrics**: Track episode rewards, losses, and convergence metrics
- **Evaluation**: Run evaluation episodes every N training episodes
- **Example**: Training functions in `experiments/` return dict with `'scores'`, `'losses'`, `'eval_scores'`

### Environment Wrappers

- **Standardization**: Wrap Gymnasium envs for consistent interfaces
- **Preprocessing**: Normalize states, handle action spaces
- **Example**: `environments/wrappers.py` provides `NormalizedEnv` and `FrameStack`

## Critical Workflows

### Running Experiments

```bash
# Install dependencies (run once)
pip install -r requirements.txt

# Train agent (example from CA4)
python -m experiments.train_dqn --env CartPole-v1 --episodes 1000

# Run notebook for interactive exploration
jupyter notebook CA4.ipynb
```

### Debugging Training

- **Loss Monitoring**: Plot policy/value losses over time
- **Reward Curves**: Use moving averages to detect convergence
- **Gradient Checks**: Verify gradients flow through networks
- **Environment Reset**: Ensure proper env resets between episodes

### Hyperparameter Tuning

- **Grid Search**: Use `experiments/hyperparameter_sweep.py` patterns
- **Logging**: Save configs and results to JSON/CSV
- **Reproducibility**: Set seeds with `utils.set_seed(seed)`

## Integration Points

### External Dependencies

- **PyTorch**: Use `torch.nn.functional` for activations, `torch.optim` for optimizers
- **Gymnasium**: Register custom envs, handle both discrete/continuous spaces
- **Visualization**: Seaborn for statistical plots, Matplotlib for curves

### Cross-CA Communication

- **Shared Utils**: Common functions in `utils/` (e.g., `plot_learning_curve()`)
- **Environment Interfaces**: Consistent `reset()`, `step()` across CAs
- **Model Loading**: Save/load PyTorch models with `torch.save/load`

## Common Pitfalls & Solutions

### Training Instability

- **Gradient Clipping**: Clip gradients at 1.0 in policy gradients
- **Target Networks**: Use in DQN/Actor-Critic for stability
- **Entropy Bonus**: Add to continuous action policies for exploration

### Sample Efficiency

- **Experience Replay**: Buffer size 10k-100k for DQN variants
- **Prioritized Replay**: Use for important transitions
- **Model-Based**: Implement Dyna-Q for planning (see CA10)

### Environment Handling

- **Action Spaces**: Check discrete vs continuous, handle properly
- **Observation Spaces**: Flatten/box handling for neural nets
- **Termination**: Respect `done` flags, don't update after episode end

## File Organization Examples

- `CA4/agents/dqn_agent.py`: DQN implementation with Double/Dueling variants
- `CA6/experiments/policy_gradient_comparison.py`: Compare REINFORCE vs Actor-Critic
- `CA10/models/neural_dynamics.py`: Neural network for environment modeling

Focus on implementing RL algorithms correctly, following PyTorch best practices, and ensuring reproducible results through proper seeding and logging.
