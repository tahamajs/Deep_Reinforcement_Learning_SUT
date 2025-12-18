from typing import Optional, Tuple
import torch


def lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    rhos: Optional[torch.Tensor] = None,
    c_rho: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute lambda-returns (forward-view) for a batch of sequences.

    Args:
        rewards: [B, L] rewards at timesteps t
        values: [B, L+1] value estimates (including bootstrap at t+1)
        dones: [B, L] done flags (1.0 if terminal at step)
        gamma: discount
        lam: lambda parameter
        rhos: optional importance sampling ratios [B, L] used in products
        c_rho: clipping value for IS ratios (optional)

    Returns:
        returns: [B, L] lambda-returns G_t^lambda
    """
    B, L = rewards.shape
    returns = torch.zeros(B, L, device=rewards.device, dtype=rewards.dtype)
    gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    # iterate backwards
    for t in reversed(range(L)):
        next_value = values[:, t + 1]
        value = values[:, t]
        delta = rewards[:, t] + gamma * (1.0 - dones[:, t]) * next_value - value
        if rhos is not None:
            rho_t = rhos[:, t]
            if c_rho is not None:
                rho_t = torch.clamp(rho_t, max=c_rho)
            delta = delta * rho_t
        gae = delta + gamma * lam * (1.0 - dones[:, t]) * gae
        returns[:, t] = gae + value
    return returns


def critic_loss_lambda(
    critic,
    target_critic,
    obs: torch.Tensor,
    acts: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    behavior_logp: Optional[torch.Tensor],
    gamma: float,
    lam: float,
    c_rho: Optional[float] = None,
    recompute_hidden: bool = True,
    policy=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes critic MSE loss using lambda-returns. This is the forward-view approach
    described in the README and avoids manual per-parameter trace accumulation while
    being numerically equivalent under autograd.

    Args:
        critic: current critic module (returns q, h)
        target_critic: target critic (used for bootstrap)
        obs: [B, L, obs_dim]
        acts: [B, L, action_dim]
        rewards: [B, L]
        dones: [B, L]
        behavior_logp: optional stored log-probs of behavior policy [B, L]
        gamma, lam: scalars
        c_rho: max clipping for IS ratios (optional)
        recompute_hidden: if True, forward runs critic to get fresh hidden states

    Returns:
        loss: scalar tensor (MSE)
        returns: [B, L] computed lambda returns
    """
    # compute q values and bootstrap values
    q_vals, _ = critic(obs, acts)  # [B, L, 1]
    q_vals = q_vals.squeeze(-1)

    with torch.no_grad():
        # sample next actions from current critic/actor is outside scope;
        # we use target critic with same actions shifted by one timestep for bootstrap
        next_acts = torch.roll(acts, shifts=-1, dims=1)
        next_q, _ = target_critic(obs, next_acts)
        next_q = next_q.squeeze(-1)

    # values needs to be [B, L+1] with last column being bootstrap next value after last step
    values = torch.cat([q_vals, next_q[:, -1:].detach()], dim=1)

    # compute IS ratios if behavior_logp provided and policy given
    rhos = None
    if behavior_logp is not None and policy is not None:
        # compute current policy logp for the actions under current policy
        # policy.log_prob returns [B, L]
        with torch.no_grad():
            pi_logp = policy.log_prob(obs, acts)
        rhos = torch.exp(pi_logp - behavior_logp)
        if c_rho is not None:
            rhos = torch.clamp(rhos, max=c_rho)
    elif behavior_logp is not None:
        # behavior logp provided but no policy; fall back to ones to avoid crashing
        rhos = torch.ones_like(rewards)

    returns = lambda_returns(rewards, values, dones, gamma, lam, rhos=rhos, c_rho=c_rho)
    # MSE between returns and current Q estimates
    loss = 0.5 * (returns - q_vals).pow(2).mean()
    return loss, returns












