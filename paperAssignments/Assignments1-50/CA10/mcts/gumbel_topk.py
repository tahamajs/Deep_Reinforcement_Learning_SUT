from typing import List, Tuple
import torch


def gumbel_noise(
    shape: Tuple[int, ...], device=None, dtype=torch.float32
) -> torch.Tensor:
    """Sample Gumbel(0,1) noise."""
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u.clamp(min=1e-9)))


def topk_joint(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k joint actions from a joint logits vector.

    Args:
        logits: (B, A) joint action logits
        k: number to select
    Returns:
        indices: (B, k) indices of selected joint actions
        scores: (B, k) scores after adding Gumbel noise
    """
    g = gumbel_noise(logits.shape, device=logits.device, dtype=logits.dtype)
    scores = logits + g
    topk = torch.topk(scores, k, dim=-1)
    return topk.indices, topk.values


def topk_factored(
    per_agent_logits: List[torch.Tensor], k_each: int, max_beam: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factored top-k per agent combined into a beam of joint actions.

    Args:
        per_agent_logits: list of (B, A_i) tensors
        k_each: top-k per agent
        max_beam: cap on total joint combinations
    Returns:
        joint_indices: (B, beam, N) indices per agent forming joint actions
        joint_scores: (B, beam) combined scores (sum of agent scores)
    """
    batch = per_agent_logits[0].shape[0]
    device = per_agent_logits[0].device
    dtype = per_agent_logits[0].dtype
    topk_idx_per_agent = []
    topk_score_per_agent = []
    for logits in per_agent_logits:
        g = gumbel_noise(logits.shape, device=device, dtype=dtype)
        scores = logits + g
        vals, idx = torch.topk(scores, k_each, dim=-1)
        topk_idx_per_agent.append(idx)  # (B, k)
        topk_score_per_agent.append(vals)

    # Combine cartesian product but cap at max_beam via greedy beam search
    # Start with first agent candidates
    joint_indices = topk_idx_per_agent[0].unsqueeze(
        1
    )  # (B, 1, k) treat k as candidates along last dim
    joint_scores = topk_score_per_agent[0].unsqueeze(1)
    N = len(per_agent_logits)
    for i in range(1, N):
        idx_i = topk_idx_per_agent[i]  # (B, k)
        score_i = topk_score_per_agent[i]
        # expand current beams with each candidate of agent i
        # joint_indices: (B, beam, i) -> expand to (B, beam * k, i+1)
        beam = joint_indices.shape[1]
        beam_exp = joint_indices.unsqueeze(2).expand(-1, -1, idx_i.shape[1], -1)
        beam_exp = beam_exp.reshape(batch, beam * idx_i.shape[1], -1)
        # new agent indices to append
        idx_rep = (
            idx_i.unsqueeze(1)
            .expand(-1, beam, -1)
            .reshape(batch, beam * idx_i.shape[1], 1)
        )
        # combine scores
        score_rep = (
            score_i.unsqueeze(1)
            .expand(-1, beam, -1)
            .reshape(batch, beam * idx_i.shape[1])
        )
        joint_indices = torch.cat([beam_exp, idx_rep], dim=-1)  # (B, beam*k, i+1)
        joint_scores = (
            joint_scores.unsqueeze(2)
            .expand(-1, -1, score_i.shape[1])
            .reshape(batch, -1)
            + score_rep
        )
        # prune to max_beam
        if joint_scores.shape[1] > max_beam:
            vals, ord = torch.topk(joint_scores, max_beam, dim=-1)
            joint_scores = vals
            # reorder joint_indices accordingly
            batch_idx = torch.arange(batch, device=device).unsqueeze(-1)
            joint_indices = joint_indices[batch_idx, ord]

    return joint_indices, joint_scores







