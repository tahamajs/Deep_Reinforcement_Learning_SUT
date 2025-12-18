from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
import heapq

import torch
import torch.nn.functional as F

from ..mcts.gumbel_topk import topk_joint, topk_factored


class MCTSNode:
    """Node for latent-space MCTS."""

    __slots__ = ("h", "parent", "action_index", "children", "N", "W", "Q", "P")

    def __init__(
        self,
        h: torch.Tensor,
        parent: Optional["MCTSNode"] = None,
        action_index: Optional[int] = None,
    ):
        # latent state (1, latent_dim)
        self.h = h
        self.parent = parent
        self.action_index = action_index
        self.children: Dict[int, MCTSNode] = {}
        self.N = 0  # visit count
        self.W = 0.0  # total value
        self.Q = 0.0  # mean value
        self.P = 0.0  # prior probability (from policy)


class MCTS:
    """Simple PUCT MCTS operating in latent space using the provided network."""

    def __init__(
        self,
        network,
        c_puct: float = 1.5,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_frac: float = 0.0,
    ):
        """
        Args:
            network: MAEZV2Network instance implementing predict_from_latent(h) and dynamics(h,a)
            c_puct: exploration constant
        """
        self.net = network
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

    def run(
        self, h0: torch.Tensor, num_simulations: int = 50, topk: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MCTS from latent h0 and return visit counts and child priors.

        Args:
            h0: (B, latent_dim) batch of roots (we assume batch size 1 for simplicity)
            num_simulations: number of sims
            topk: candidate expansions per node (joint actions)
        Returns:
            visit_counts: (A,) visit counts over joint actions (summing to sims)
            policy_from_visits: softmax of visit counts (A,)
        """
        # For simplicity handle batch size 1
        assert h0.shape[0] == 1, "MCTS.run currently supports batch size 1"
        device = h0.device
        root = MCTSNode(h0[0:1])

        # get root policy and value
        logits_joint, logits_agents, v = self.net.predict_from_latent(root.h)
        probs = F.softmax(logits_joint, dim=-1).squeeze(0)  # (A,)

        A = logits_joint.shape[-1]
        # initialize priors for possible actions; we'll expand topk when needed
        priors = probs.detach().cpu()

        # optionally add dirichlet noise to priors
        if self.dirichlet_alpha is not None and self.dirichlet_frac > 0.0:
            noise = torch.distributions.Dirichlet(
                torch.ones_like(priors) * self.dirichlet_alpha
            ).sample()
            priors = (1 - self.dirichlet_frac) * priors + self.dirichlet_frac * noise

        root.P = 1.0

        # bookkeeping for children created at root; keys are action indices
        for sim in range(num_simulations):
            node = root
            search_path: List[MCTSNode] = [node]

            # selection
            while node.children:
                # pick child with max Q+U
                total_N = sum(child.N for child in node.children.values())
                best_score = -float("inf")
                best_action = None
                best_child = None
                for a_idx, child in node.children.items():
                    U = (
                        self.c_puct
                        * child.P
                        * math.sqrt(total_N + 1e-8)
                        / (1 + child.N)
                    )
                    score = child.Q + U
                    if score > best_score:
                        best_score = score
                        best_action = a_idx
                        best_child = child
                assert best_child is not None
                node = best_child
                search_path.append(node)

            # expand leaf
            # use network policy to propose topk actions from node.h
            logits_joint, logits_agents, value_leaf = self.net.predict_from_latent(
                node.h
            )
            logits_joint = logits_joint.detach()
            # pick topk candidates
            k = min(topk, logits_joint.shape[-1])
            indices, scores = topk_joint(logits_joint, k)
            # indices: (1,k)
            indices = indices.squeeze(0).tolist()
            scores = scores.squeeze(0)
            probs_k = F.softmax(logits_joint, dim=-1).squeeze(0)[indices]

            # create children for each candidate if not existing
            for idx, p in zip(indices, probs_k):
                if idx not in node.children:
                    # apply dynamics with action vector placeholder: use one-hot encoding
                    a_vec = torch.zeros(1, logits_joint.shape[-1], device=device)
                    a_vec[0, idx] = 1.0
                    h_next, r = self.net.dynamics(node.h, a_vec)
                    child = MCTSNode(h_next.detach(), parent=node, action_index=idx)
                    child.P = float(p.detach().cpu().item())
                    node.children[idx] = child

            # evaluate leaf using value from network (value_leaf)
            value = float(value_leaf.detach().cpu().item())

            # backup
            for n in reversed(search_path):
                n.N += 1
                n.W += value
                n.Q = n.W / (n.N + 1e-8)

        # after sims, compute visit counts at root across action space
        visit_counts = torch.zeros(A, dtype=torch.float32)
        for a_idx, child in root.children.items():
            visit_counts[a_idx] = float(child.N)

        # produce policy from visits (temperature 1)
        policy = visit_counts / (visit_counts.sum() + 1e-8)
        return visit_counts, policy


__all__ = ["MCTS", "MCTSNode"]
