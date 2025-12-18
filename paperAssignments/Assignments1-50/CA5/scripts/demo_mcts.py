\"\"\"Demo script: wire PUCT MCTS to a toy discrete environment.

This script is notebook-friendly: it can be imported into a notebook or run as a script.
It demonstrates:
 - a tiny deterministic toy environment (1D position, actions: -1, 0, +1)
 - a model exposing `policy(state)` and `value(state)` used by PUCT
 - running PUCT search and printing visit counts and selected action
\"\"\"
from typing import List, Tuple
import math
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.mcts.puct import PUCT, Node


class ToyModel:
    \"\"\"Simple 1D toy model where state is integer position.

    Goal is at position `goal`. Episode horizon is limited externally by search depth.
    """

    def __init__(self, goal: int = 3):
        self.goal = goal
        # discrete actions: left, stay, right
        self.actions = [-1, 0, 1]

    def policy(self, state: int) -> List[float]:
        \"\"\"Return prior probabilities over actions for a given state.
        Simple heuristic: prefer actions that move toward goal.
        \"\"\"
        d = self.goal - state
        priors = []
        for a in self.actions:
            score = -abs(d - a)
            priors.append(math.exp(score))
        s = sum(priors)
        return [p / s for p in priors]

    def value(self, state: int) -> float:
        \"\"\"Heuristic value: negative absolute distance to goal (higher is better).\"\"\"
        return -abs(self.goal - state)

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        \"\"\"Deterministic transition.\"\"\"
        next_state = state + action
        reward = 1.0 if next_state == self.goal else 0.0
        done = next_state == self.goal
        return next_state, reward, done


def run_demo(num_simulations: int = 50, verbose: bool = True):
    model = ToyModel(goal=3)
    action_space = [0, 1, 2]  # indices corresponding to [-1,0,1]
    puct = PUCT(model, action_space=action_space, c_puct=1.0, dirichlet_alpha=0.3)

    root_state = 0
    root = puct.search(root_state, num_simulations=num_simulations)

    # Collect visit counts and priors
    visits = {a: child.visits for a, child in root.children.items()}
    priors = {a: child.prior for a, child in root.children.items()}

    if verbose:
        print(f\"Root state: {root_state}\")
        print(\"Action index -> prior, visits, value:\")
        for a in sorted(root.children.keys()):
            child = root.children[a]
            print(f\"  {a} -> prior={child.prior:.3f}, visits={child.visits}, value={child.value():.3f}\")

    # choose action with highest visits
    best_action_idx = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
    best_action = [-1, 0, 1][best_action_idx]
    if verbose:
        print(f\"Selected action index {best_action_idx} => action {best_action}\")

    # Simple plot of visit counts
    try:
        labels = [str(a) for a in sorted(visits.keys())]
        counts = [visits[a] for a in sorted(visits.keys())]
        plt.bar(labels, counts)
        plt.xlabel(\"Action index\")
        plt.ylabel(\"Visit counts\")
        plt.title(\"PUCT Visit Counts (root)\")
        plt.tight_layout()
        plt.show()
    except Exception:
        # matplotlib may not be available in some headless environments; ignore plotting errors
        pass

    return root, best_action_idx


if __name__ == \"__main__\":
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--sims\", type=int, default=50)
    args = parser.parse_args()
    run_demo(num_simulations=args.sims)










