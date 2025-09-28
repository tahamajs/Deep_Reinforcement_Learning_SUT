# Homework 2: Policy Gradient Methods

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

You can run this code on your own machine or on Google Colab.

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions. If you completed this installation for homework 1, you do not need to repeat it.
- Make sure to run `pip install -e .` in hw2 folder.
- Also run `pip install gym[box2d]`
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/master/hw2/cs285/scripts/run_hw2.ipynb)

## Complete the code

The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with "TODO: get this from hw1".

- [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
- [infrastructure/utils.py](cs285/infrastructure/utils.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy.py)

You will then need to complete the following new files for homework 2. The relevant sections are marked with "TODO".
- [agents/pg_agent.py](cs285/agents/pg_agent.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy)

You will also want to look through [scripts/run_hw2.py](cs285/scripts/run_hw2.py) (if running locally) or [scripts/run_hw2.ipynb](cs285/scripts/run_hw2.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

You will be running your policy gradients implementation in five experiments total, investigating the effects of design decisions like reward-to-go estimators, neural network baselines and generalized advantage estimation for variance reduction, and advantage normalization. See the [assignment PDF](cs285_hw2.pdf) for more details.

## Running Experiments
- Example command:
  ```bash
  python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --exp_name pg_test
  ```
- Results are saved in `results/`.

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
