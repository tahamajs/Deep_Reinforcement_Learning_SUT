import random
import numpy as np
import torch
from src.mcts.puct import PUCT
from src.mcts.nn_model import SmallDiscreteModel


def run_search_with_seed(seed: int, sims: int = 50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SmallDiscreteModel(n_actions=3, hidden=32)
    puct = PUCT(model, action_space=[0, 1, 2], c_puct=1.0, dirichlet_alpha=0.3)
    root = puct.search(0, num_simulations=sims)
    visits = {a: child.visits for a, child in root.children.items()}
    return visits


def test_deterministic_seed_reproducible():
    v1 = run_search_with_seed(123, sims=30)
    v2 = run_search_with_seed(123, sims=30)
    assert v1 == v2


def test_visit_counts_sum_to_simulations():
    sims = 40
    visits = run_search_with_seed(7, sims=sims)
    total = sum(visits.values())
    assert total == sims






