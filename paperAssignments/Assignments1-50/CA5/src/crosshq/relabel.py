from typing import Dict
import torch


def off_policy_correction(
    manager_batch: Dict[str, torch.Tensor], worker_policy, k: int = 8
):
    """
    Relabel goals in manager_batch using the current worker_policy.

    manager_batch expected keys:
      - 'obs' : (B, s_dim) starting state
      - 'action_seq' : (B, c, a_dim) sequence of low-level actions observed
      - 'goal' : (B, g_dim) original goal
      - 'next_obs' : (B, s_dim) s_{t+c}

    This implementation follows the HIRO-style candidate generation and likelihood scoring.
    It is a straightforward, vectorized implementation suitable for unit tests and integration.
    """
    s = manager_batch["obs"]
    a_seq = manager_batch["action_seq"]
    original_g = manager_batch["goal"]
    s_next = manager_batch["next_obs"]

    B = s.shape[0]
    device = s.device

    # candidates: original, transition, plus k noisy around transition
    candidates = [original_g, (s_next - s)]
    for _ in range(k):
        candidates.append((s_next - s) + torch.randn_like(original_g) * 0.5)

    # stack -> (C, B, g_dim)
    candidates_t = torch.stack(candidates, dim=0)

    C = candidates_t.shape[0]
    c_len = a_seq.shape[1]

    # compute log probs: shape (C, B)
    log_probs = torch.zeros(C, B, device=device)

    # For each candidate goal, compute sum log prob of observing action sequence under worker_policy
    # worker_policy must implement a method `dist_for_goal(states, goal)` or accept concatenated input.
    for i in range(C):
        g_c = candidates_t[i]  # (B, g_dim)
        total = torch.zeros(B, device=device)
        # iterate through time steps (vectorized across batch)
        for t in range(c_len):
            s_t = manager_batch.get("states_seq", None)
            # If a sequence of states is provided, use it; otherwise assume actions were conditioned only on s and g constant
            if s_t is None:
                # fallback: use s and s_next as proxies (best-effort)
                s_input = s  # best-effort; caller can provide states_seq for accurate relabeling
            else:
                # states_seq expected (B, c, s_dim)
                s_input = s_t[:, t, :]
            a_t = a_seq[:, t, :]
            # worker_policy is expected to expose `dist(obs, goal)` returning a torch.distributions object
            dist = worker_policy.dist(torch.cat([s_input, g_c], dim=-1))
            logp = dist.log_prob(a_t).sum(-1)
            total = total + logp
        log_probs[i] = total

    # choose best candidate per batch element
    best_idx = torch.argmax(log_probs, dim=0)  # (B,)
    # gather
    best = candidates_t[best_idx, torch.arange(B, device=device)]
    return best












