# Author: Taha Majlesi - 810101504, University of Tehran
# Homework 1: Imitation Learning - Behavior Cloning and DAgger

**Author:** Taha Majlesi - 810101504, University of Tehran

## Overview

This assignment introduces imitation learning, a paradigm in reinforcement learning where an agent learns to mimic expert behavior from demonstrations. You will implement Behavior Cloning (BC) and Dataset Aggregation (DAgger), comparing their performance and understanding the limitations of supervised learning approaches in sequential decision-making tasks.

## Learning Objectives

- Understand the fundamentals of imitation learning
- Implement Behavior Cloning using supervised learning on expert trajectories
- Implement DAgger to address compounding errors in BC
- Analyze the differences between offline and online imitation learning
- Evaluate policies in continuous control environments

## Key Concepts

- **Behavior Cloning**: Treats imitation as a supervised learning problem, training a policy to predict expert actions from states
- **DAgger**: Iteratively aggregates new data by querying the expert on states visited by the current policy, reducing distribution shift
- **Compounding Errors**: How small mistakes accumulate in sequential prediction tasks
- **Expert Demonstrations**: Using pre-collected trajectories from an optimal policy

## Structure

- `cs285/`: Core RL codebase (agents, policies, infrastructure)
- `data/`: Datasets for offline RL (if any)
- `README.md`: This file
- `*.ipynb`: Jupyter notebooks for experiments and analysis

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Set up a virtual environment for isolation.

## Running Experiments

- Use the provided scripts or Jupyter notebooks to run experiments:
  ```bash
  python cs285/scripts/run_hw1.py --env_name CartPole-v1 --exp_name test_pg
  ```
- Results and logs will be saved in the `results/` directory.

## One-click runner script

To automate environment setup and execute the default Behavior Cloning and DAgger experiments on MuJoCo tasks, use the helper script:

```bash
chmod +x run_hw1.sh
./run_hw1.sh
```

The script will create (or reuse) a virtual environment in `.venv/`, install dependencies from `requirements.txt`, register the `cs285` package in editable mode, and then launch both experiments for `HalfCheetah-v4`. Set the following environment variables to customize the run:

- `ENV_NAME` – choose among `Ant-v4`, `HalfCheetah-v4`, `Hopper-v4`, `Walker2d-v4`, or `Humanoid-v4` (requires you to supply the missing expert dataset file).
- `RUN_BEHAVIOR_CLONING` / `RUN_DAGGER` – set to `0` to skip either phase.
- `PYTHON_BIN` – specify a different Python executable (defaults to `python3`).
- `SKIP_INSTALL` – set to `1` to reuse existing dependencies.

Example: run only DAgger for Hopper with an existing environment.

```bash
SKIP_INSTALL=1 RUN_BEHAVIOR_CLONING=0 ENV_NAME=Hopper-v4 ./run_hw1.sh
```

## Key Files

- `cs285/agents/`: RL agent implementations
- `cs285/policies/`: Policy networks
- `cs285/critics/`: Value function approximators
- `cs285/infrastructure/`: Utilities and trainers

## Submission

- Submit your code, results, and a report (if required) as specified in the assignment PDF.

## References

- Ross, S., & Bagnell, J. A. (2010). Efficient reductions for imitation learning. AISTATS.
- Pomerleau, D. A. (1988). ALVINN: An autonomous land vehicle in a neural network. NIPS.
- Sutton & Barto, "Reinforcement Learning: An Introduction"
- OpenAI Gymnasium documentation
