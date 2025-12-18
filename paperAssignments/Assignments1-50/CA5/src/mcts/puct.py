from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import math
import random


@dataclass
class Node:
    state: Any
    parent: Optional["Node"] = None
    prior: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class PUCT:
    """PUCT MCTS for discrete action spaces.

    Model interface expected:
      - policy(state) -> priors: Sequence[float] over actions
      - value(state) -> scalar value estimate
      - step(state, action) -> next_state, reward, done (optional)
    """

    def __init__(
        self,
        model,
        action_space: Sequence[int],
        c_puct: float = 1.0,
        dirichlet_alpha: Optional[float] = None,
        invert_value: bool = False,
    ):
        self.model = model
        self.action_space = list(action_space)
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.invert_value = invert_value

    def search(self, root_state: Any, num_simulations: int = 100) -> Node:
        root = Node(state=root_state)
        # init priors from model
        priors = list(self.model.policy(root_state))
        assert len(priors) == len(self.action_space)
        for a, p in zip(self.action_space, priors):
            root.children[a] = Node(state=None, parent=root, prior=p)

        # optional exploration noise
        if self.dirichlet_alpha is not None:
            noise = [random.gammavariate(self.dirichlet_alpha, 1.0) for _ in priors]
            s = sum(noise)
            for n, a in zip(noise, self.action_space):
                root.children[a].prior = root.children[a].prior * 0.75 + (n / s) * 0.25

        for _ in range(num_simulations):
            node, path = self._select(root)
            value = self._expand_and_evaluate(node)
            self._backup(path, value)
        return root

    def _select(self, node: Node):
        path = [node]
        while node.expanded():
            # compute UCB for each child
            total_visits = sum(child.visits for child in node.children.values()) + 1
            best_score = -float("inf")
            best_action = None
            best_child = None
            for a, child in node.children.items():
                q = child.value()
                u = (
                    self.c_puct
                    * child.prior
                    * math.sqrt(total_visits)
                    / (1 + child.visits)
                )
                score = q + u
                if score > best_score:
                    best_score = score
                    best_action = a
                    best_child = child
            node = best_child
            path.append(node)
        return node, path

    def _expand_and_evaluate(self, node: Node) -> float:
        # get model priors and value
        priors = list(self.model.policy(node.state))
        value = float(self.model.value(node.state))
        # create children with priors
        for a, p in zip(self.action_space, priors):
            if a not in node.children:
                node.children[a] = Node(state=None, parent=node, prior=p)
        return value

    def _backup(self, path: List[Node], value: float):
        for node in reversed(path):
            node.visits += 1
            if self.invert_value:
                node.value_sum += -value
            else:
                node.value_sum += value













