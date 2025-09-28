# Assignment 1: Imitation Learning

Author: Taha Majlesi - 810101504, University of Tehran

This assignment implements various imitation learning algorithms: Behavioral Cloning (BC), Dataset Aggregation (DAgger), Covariance Matrix Adaptation Evolution Strategy (CMA-ES), and Generative Adversarial Imitation Learning (GAIL).

## Structure

- `src/models.py`: Neural network model definitions
- `src/utils.py`: Utility functions for episode generation and evaluation
- `src/bc_dagger.py`: Behavioral Cloning and DAgger implementation
- `src/cmaes.py`: CMA-ES implementation
- `src/gail.py`: GAIL implementation
- `run_bc_dagger.py`: Script to run BC and DAgger
- `run_cmaes.py`: Script to run CMA-ES
- `run_gail.py`: Script to run GAIL
- `expert.h5`: Pre-trained expert policy
- `requirements.txt`: Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Behavioral Cloning and DAgger
```bash
python run_bc_dagger.py
```

### CMA-ES
```bash
python run_cmaes.py
```

### GAIL
```bash
python run_gail.py
```

## Algorithms

1. **Behavioral Cloning (BC)**: Learns policy by supervised learning on expert demonstrations.

2. **DAgger**: Iteratively aggregates data from both expert and learned policy to improve performance.

3. **CMA-ES**: Evolutionary algorithm that optimizes policy parameters using covariance matrix adaptation.

4. **GAIL**: Uses adversarial training to learn policy that matches expert behavior distribution.

## Environment

- CartPole-v1 from Gymnasium

## Dependencies

- gymnasium
- tensorflow
- numpy
- matplotlib
- scikit-learn
- cma