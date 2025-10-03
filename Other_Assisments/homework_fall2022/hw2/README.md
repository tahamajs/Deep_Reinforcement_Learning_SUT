# Author: Taha Majlesi - 810101504, University of Tehran
# Homework 2: Policy Gradient Methods

**Author:** Taha Majlesi - 810101504, University of Tehran

## Overview

This assignment explores policy gradient methods, a class of reinforcement learning algorithms that directly optimize the policy using gradient ascent on expected returns. You will implement REINFORCE (Monte Carlo policy gradients), analyze variance reduction techniques like reward-to-go and baselines, and experiment with generalized advantage estimation (GAE).

## Learning Objectives

- Understand the theory behind policy gradient methods
- Implement REINFORCE algorithm with variance reduction techniques
- Analyze the impact of different advantage estimators
- Compare policy gradients with value-based methods
- Gain experience with continuous control tasks

## Key Concepts

- **Policy Gradients**: Optimizing policies directly using gradient information from trajectories
- **REINFORCE**: Monte Carlo policy gradient algorithm
- **Reward-to-Go**: Using future rewards instead of total returns to reduce variance
- **Baselines**: Subtracting a baseline from returns to reduce variance without bias
- **Generalized Advantage Estimation (GAE)**: Balancing bias and variance in advantage estimation
- **Advantage Normalization**: Standardizing advantages for stable training

## Structure

- `cs285/`: Core RL codebase
- `data/`: Datasets (if any)
- `README.md`: This file
- `*.ipynb`: Experiment notebooks

## Setup

You can run this code on your own machine.

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions. If you completed this installation for homework 1, you do not need to repeat it.

- Make sure to run `pip install -e .` in hw2 folder.
- Also run `pip install gym[box2d]`

## Complete the code

The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with "TODO: get this from hw1".

- [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
- [infrastructure/utils.py](cs285/infrastructure/utils.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy.py)

You will then need to complete the following new files for homework 2. The relevant sections are marked with "TODO".

- [agents/pg_agent.py](cs285/agents/pg_agent.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy)

You will also want to look through [scripts/run_hw2.py](cs285/scripts/run_hw2.py), though you will not need to edit this files beyond changing runtime arguments.

You will be running your policy gradients implementation in five experiments total, investigating the effects of design decisions like reward-to-go estimators, neural network baselines and generalized advantage estimation for variance reduction, and advantage normalization. See the [assignment PDF](cs285_hw2.pdf) for more details.

## Running Experiments

- Example command:
  ```bash
  python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --exp_name pg_test
  ```
- Logs are stored inside `cs285/data/` by default.

## One-click runner script

You can automate setup and launch a sequence of policy-gradient experiments using the helper script:

```bash
chmod +x run_hw2.sh
./run_hw2.sh
```

The script will create (or reuse) a virtual environment in `.venv/`, install dependencies from `requirements.txt`, register the package in editable mode, and then execute a suite of presets (vanilla PG, reward-to-go, baseline, and reward-to-go with baseline) on `LunarLander-v2`.

Customize runs by overriding environment variables when invoking the script:

- `ENV_NAME` – target Gym environment (e.g. `CartPole-v1`, `LunarLander-v2`, `Ant-v4`).
- `PRESETS` – comma-separated list choosing among `vanilla`, `rtg`, `baseline`, `rtg_baseline`, `gae`, `no_std_adv`, or `custom`.
- `GAE_LAMBDA` – lambda for the `gae` preset (defaults to `0.95`).
- `N_ITER`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_AGENT_TRAIN_STEPS`, etc. – tune training hyperparameters.
- `SKIP_INSTALL=1` – reuse an existing environment without reinstalling requirements.
- `CUSTOM_FLAGS` – extra CLI flags appended to every run; required when using the `custom` preset.

Examples:

Run only reward-to-go with baseline for Hopper:

```bash
PRESETS=rtg_baseline ENV_NAME=Hopper-v4 ./run_hw2.sh
```

Evaluate a GAE sweep without reinstalling dependencies:

```bash
SKIP_INSTALL=1 PRESETS=gae ENV_NAME=HalfCheetah-v4 GAE_LAMBDA=0.97 ./run_hw2.sh
```

## Key Files

- `cs285/agents/`: Policy gradient agents
- `cs285/policies/`: Policy networks
- `cs285/critics/`: Value networks for baselines
- `cs285/infrastructure/`: Utilities and trainers

## Submission

- Submit code, results, and report as required.

## References

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. ICLR.
- Sutton & Barto, "Reinforcement Learning: An Introduction"
- OpenAI Gymnasium documentation
