from typing import Tuple, Sequence
import torch
import torch.nn as nn
from src.losses import critic_loss_lambda


def soft_update(src: nn.Module, tgt: nn.Module, tau: float):
    for p, q in zip(src.parameters(), tgt.parameters()):
        q.data.mul_(1 - tau)
        q.data.add_(tau * p.data)


def sac_update(
    critics: Sequence[nn.Module],
    target_critics: Sequence[nn.Module],
    actor: nn.Module,
    optim_critics,
    optim_actor,
    batch: Tuple,
    cfg,
):
    """
    Minimal SAC-style update: critic MSE to lambda-returns handled elsewhere,
    here we implement the actor update using critic Q estimates and entropy.
    """
    obs, acts, rews, dones, beh_logp = batch
    device = obs.device

    # compute current policy actions and log probs
    actions_pi, logp_pi, _ = actor.sample(obs)

    # query all critics and take min
    q_vals = []
    for c in critics:
        qc, _ = c(obs, actions_pi)
        q_vals.append(qc.squeeze(-1))
    q_pi = torch.min(torch.stack(q_vals, dim=0), dim=0).values

    # actor loss: minimize alpha * logp - Q
    alpha = getattr(cfg, "alpha", 0.1)
    actor_loss = (alpha * logp_pi - q_pi).mean()

    optim_actor.zero_grad()
    actor_loss.backward()
    optim_actor.step()

    # soft update targets for each critic pair
    tau = getattr(cfg, "tau", 0.005)
    for c_src, c_tgt in zip(critics, target_critics):
        soft_update(c_src, c_tgt, tau)
    return actor_loss.item()












