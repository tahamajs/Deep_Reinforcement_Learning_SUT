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
    """Simple PUCT MCTS operating in latent space using the provided network.

    Supports batched roots and optional factored beam expansion using per-agent logits.
    """

    def __init__(
        self,
        network,
        c_puct: float = 1.5,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_frac: float = 0.0,
        factored_search: bool = False,
        max_beam: int = 64,
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
        self.factored_search = factored_search
        self.max_beam = max_beam

    def run(
        self,
        h0: torch.Tensor,
        num_simulations: int = 50,
        topk: int = 8,
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
        # Support batch of roots by running independent MCTS per element.
        batch = h0.shape[0]
        device = h0.device
        visit_results = []
        policy_results = []
        joint_visits_all = []

        for b in range(batch):
            root = MCTSNode(h0[b : b + 1])
            # get root policy and value
            logits_joint, logits_agents, v, _ = self.net.predict_from_latent(root.h)
            probs = F.softmax(logits_joint, dim=-1).squeeze(0)  # (A,)
            A = logits_joint.shape[-1]

            priors = probs.detach().cpu()
            # optionally add dirichlet noise to priors
            if self.dirichlet_alpha is not None and self.dirichlet_frac > 0.0:
                noise = torch.distributions.Dirichlet(
                    torch.ones_like(priors) * self.dirichlet_alpha
                ).sample()
                priors = (
                    1 - self.dirichlet_frac
                ) * priors + self.dirichlet_frac * noise

            root.P = 1.0

            joint_counts: Dict[tuple, int] = {}
            for sim in range(num_simulations):
                node = root
                search_path: List[MCTSNode] = [node]

                # selection
                while node.children:
                    total_N = sum(child.N for child in node.children.values())
                    best_score = -float("inf")
                    best_child = None
                    for child in node.children.values():
                        U = (
                            self.c_puct
                            * child.P
                            * math.sqrt(total_N + 1e-8)
                            / (1 + child.N)
                        )
                        score = child.Q + U
                        if score > best_score:
                            best_score = score
                            best_child = child
                    assert best_child is not None
                    node = best_child
                    search_path.append(node)

                # expand leaf
                logits_joint, logits_agents, value_leaf, _ = self.net.predict_from_latent(
                    node.h
                )
                logits_joint = logits_joint.detach()

                if self.factored_search and logits_agents is not None:
                    # logits_agents is a list of tensors (1, A_i)
                    # convert per-agent logits for this node to list
                    per_agent_logits = [la for la in logits_agents]
                    # get joint candidates via factored beam
                    joint_idx, joint_scores = topk_factored(
                        per_agent_logits, topk, self.max_beam
                    )
                    # joint_idx: (1, beam, N_agents)
                    beam = joint_idx.shape[1]
                    candidates = []
                    for j in range(beam):
                        # build concatenated action vector across agents (one-hot per agent slot)
                        # assume sum of per-agent dims equals dynamics action_dim
                        agent_indices = joint_idx[0, j].tolist()
                        # construct action vector as concatenation of one-hots
                        parts = []
                        for ag_i, ai in enumerate(agent_indices):
                            A_i = per_agent_logits[ag_i].shape[-1]
                            one = torch.zeros(
                                1, A_i, device=device, dtype=logits_joint.dtype
                            )
                            one[0, ai] = 1.0
                            parts.append(one)
                        a_vec = torch.cat(parts, dim=-1)
                        candidates.append((tuple(agent_indices), a_vec))
                else:
                    # joint top-k
                    k = min(topk, logits_joint.shape[-1])
                    indices, scores = topk_joint(logits_joint, k)
                    indices = indices.squeeze(0).tolist()
                    candidates = []
                    for idx in indices:
                        a_vec = torch.zeros(1, logits_joint.shape[-1], device=device)
                        a_vec[0, idx] = 1.0
                        candidates.append((idx, a_vec))

                # create children
                for idx_repr, a_vec in candidates:
                    # for factored, idx_repr is tuple; use hashed int key via str
                    key = idx_repr if isinstance(idx_repr, int) else tuple(idx_repr)
                    if key not in node.children:
                        h_next, r = self.net.dynamics(node.h, a_vec)
                        child = MCTSNode(h_next.detach(), parent=node, action_index=key)
                        # set prior from network logits if possible
                        if self.factored_search and logits_agents is not None:
                            # approximate prior by joint_scores
                            # (we won't set precise per-candidate priors here)
                            child.P = 1.0 / max(1, len(candidates))
                        else:
                            # use softmax probability for this index
                            prob = F.softmax(logits_joint, dim=-1).squeeze(0)
                            if isinstance(key, int):
                                child.P = float(prob[key].detach().cpu().item())
                            else:
                                child.P = 1.0 / max(1, len(candidates))
                        node.children[key] = child

                # mark candidate counts on selection when expanded
                # here we track joint candidate counts (initialized on creation or selection)
                # increment joint count for the chosen candidate (leaf)
                # use the final selected child 'node' (which is leaf prior to expansion)
                # after expansion, pick the child with highest Q+U for backup traversal
                # For bookkeeping, increment joint_counts for expanded children (approximate)
                for idx_repr, _ in candidates:
                    jkey = idx_repr if isinstance(idx_repr, tuple) else (idx_repr,)
                    joint_counts[jkey] = (
                        joint_counts.get(jkey, 0) + 0
                    )  # ensure key exists

                # evaluate
                value = float(value_leaf.detach().cpu().item())

                # backup
                for n in reversed(search_path):
                    n.N += 1
                    n.W += value
                    n.Q = n.W / (n.N + 1e-8)

            # collect visit counts (flattening factored keys where possible)
            visit_counts = torch.zeros(A, dtype=torch.float32)
            for a_idx, child in root.children.items():
                if isinstance(a_idx, int) and 0 <= a_idx < A:
                    visit_counts[a_idx] = float(child.N)
                elif isinstance(a_idx, tuple):
                    # for factored joint actions, map to a flattened index if sum of per-agent dims equals A
                    # best-effort: ignore if mapping unknown
                    pass
            visit_results.append(visit_counts)
            policy_results.append(visit_counts / (visit_counts.sum() + 1e-8))
            # also prepare joint visit tensor and keys
            joint_keys = list(joint_counts.keys())
            joint_vals = (
                torch.tensor([joint_counts[k] for k in joint_keys], dtype=torch.float32)
                if joint_keys
                else torch.tensor([], dtype=torch.float32)
            )
            # store as attribute for later inspection
            root._joint_visit_keys = joint_keys
            root._joint_visit_vals = joint_vals
            joint_visits_all.append((joint_keys, joint_vals))

        visit_tensor = torch.stack(visit_results, dim=0)
        policy_tensor = torch.stack(policy_results, dim=0)
        # return (visit_tensor, policy_tensor, joint_visits_all)
        return visit_tensor, policy_tensor, joint_visits_all


__all__ = ["MCTS", "MCTSNode"]














