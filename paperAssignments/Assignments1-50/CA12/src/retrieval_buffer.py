import math
from typing import Optional, Tuple

import torch


class RetrievalBuffer:
    """A simple circular trajectory buffer with retrieval by nearest-neighbor + RTG filtering.

    This implementation is exact (uses brute-force L2) and is intended for
    correctness and ease-of-use. For large buffers replace KD-Tree / FAISS.
    """

    def __init__(
        self,
        max_size: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ) -> None:
        self.max_size = int(max_size)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        # accept either a string or a torch.device and normalize
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

        self.states = torch.zeros(
            (self.max_size, self.state_dim), dtype=torch.float32, device=self.device
        )
        self.actions = torch.zeros(
            (self.max_size, self.action_dim), dtype=torch.float32, device=self.device
        )
        self.returns_to_go = torch.full(
            (self.max_size, 1), -float("inf"), dtype=torch.float32, device=self.device
        )

        # trajectory metadata: list of (start_idx, end_idx, length)
        self.traj_meta = []  # type: ignore
        self.ptr = 0
        self.size = 0

    @staticmethod
    def compute_rtg(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute return-to-go (monte-carlo) per timestep for a trajectory.

        rewards: tensor of shape (T,)
        returns: tensor of shape (T,1)
        """
        T = rewards.shape[0]
        rtg = torch.zeros((T,), dtype=torch.float32, device=rewards.device)
        acc = 0.0
        for t in reversed(range(T)):
            acc = rewards[t].item() + gamma * acc
            rtg[t] = acc
        return rtg.unsqueeze(1)

    def add_trajectory(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        gamma: float = 0.99,
    ) -> None:
        """Add a whole trajectory to the circular buffer.

        states: (T, state_dim)
        actions: (T, action_dim)
        rewards: (T,)
        """
        assert states.ndim == 2 and actions.ndim == 2 and rewards.ndim == 1
        T = states.shape[0]
        rtg = self.compute_rtg(rewards, gamma=gamma)  # (T,1)

        indices = (
            torch.arange(self.ptr, self.ptr + T, device=self.states.device)
            % self.max_size
        ).long()
        self.states[indices] = states.to(self.device)
        self.actions[indices] = actions.to(self.device)
        self.returns_to_go[indices] = rtg.to(self.device)

        self.traj_meta.append((int(indices[0].item()), int(indices[-1].item()), int(T)))
        self.ptr = int((self.ptr + T) % self.max_size)
        self.size = min(self.size + T, self.max_size)

    def retrieve_best_k(
        self, query_state: torch.Tensor, k: int = 10, nn: int = 50
    ) -> torch.Tensor:
        """Retrieve best-k actions near query_state by L2 + RTG ranking.

        Returns tensor of shape (k, action_dim).
        """
        assert query_state.shape == (self.state_dim,)
        if self.size == 0:
            # return empty tensor on correct device
            return torch.zeros((0, self.action_dim), device=self.device)

        # compute L2 distances to all stored states
        stored = self.states[: self.size]  # (N, state_dim)
        dists = torch.norm(stored - query_state.to(self.device), dim=1)  # (N,)

        nn = min(int(nn), int(self.size))
        _, nn_idxs = torch.topk(dists, k=nn, largest=False)

        neighbor_rtgs = self.returns_to_go[nn_idxs].squeeze(1)  # (nn,)
        k2 = min(int(k), nn)
        _, best_sub_idxs = torch.topk(neighbor_rtgs, k=k2, largest=True)
        best_global_idxs = nn_idxs[best_sub_idxs]

        return self.actions[best_global_idxs]

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform sample transitions for critic/actor updates.

        Returns (states, actions, returns_to_go)
        """
        if self.size == 0:
            raise RuntimeError("Buffer is empty")
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.states[idxs], self.actions[idxs], self.returns_to_go[idxs]










