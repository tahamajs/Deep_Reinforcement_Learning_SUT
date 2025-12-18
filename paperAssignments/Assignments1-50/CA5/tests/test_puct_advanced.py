import threading
from src.mcts.puct import PUCT


class ConstantModel:
    def __init__(self, n_actions=3, value=1.0):
        self.n_actions = n_actions
        self._value = value

    def policy(self, state):
        return [1.0 / self.n_actions] * self.n_actions

    def value(self, state):
        return float(self._value)


def test_puct_invert_backup_behavior():
    model = ConstantModel(n_actions=3, value=2.0)
    p_no_invert = PUCT(model, action_space=[0, 1, 2], c_puct=1.0, invert_value=False)
    root1 = p_no_invert.search(0, num_simulations=10)
    sum_values_pos = sum(child.value_sum for child in root1.children.values())

    p_invert = PUCT(model, action_space=[0, 1, 2], c_puct=1.0, invert_value=True)
    root2 = p_invert.search(0, num_simulations=10)
    sum_values_neg = sum(child.value_sum for child in root2.children.values())

    # With constant positive value, invert should produce negative accumulated sums
    assert sum_values_pos > 0
    assert sum_values_neg < 0


def test_parallel_search_runs():
    model = ConstantModel(n_actions=3, value=1.0)
    p = PUCT(model, action_space=[0, 1, 2], c_puct=1.0)

    results = []

    def worker():
        root = p.search(0, num_simulations=20)
        results.append(sum(child.visits for child in root.children.values()))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 2
    assert all(r == 20 for r in results)

