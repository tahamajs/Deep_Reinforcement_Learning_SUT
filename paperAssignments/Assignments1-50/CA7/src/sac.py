from typing import Tuple
import torch
import torch.nn as nn


def soft_update(src: nn.Module, tgt: nn.Module, tau: float):
    for p, q in zip(src.parameters(), tgt.parameters()):
        q.data.mul_(1 - tau)
        q.data.add_(tau * p.data)


def sac_update(critic: nn.Module, target_critic: nn.Module, actor: nn.Module, optim_critic, optim_actor, batch: Tuple, cfg):
    """
    Minimal SAC-style update: critic MSE to lambda-returns handled elsewhere,
    here we implement the actor update using critic Q estimates and entropy.
    """
    obs, acts, rews, dones, beh_logp = batch
    device = obs.device

    # Actor update: sample actions and logp from policy, compute Q and actor loss
    with torch.no_grad():
        pass

    # compute current policy actions and log probs
    actions_pi, logp_pi, _ = actor.sample(obs)
    q_pi, _ = critic(obs, actions_pi)
    q_pi = q_pi.squeeze(-1)
    # actor loss: minimize alpha * logp - Q
    alpha = getattr(cfg, "alpha", 0.1)
    actor_loss = (alpha * logp_pi - q_pi).mean()

    optim_actor.zero_grad()
    actor_loss.backward()
    optim_actor.step()

    # soft update targets
    tau = getattr(cfg, "tau", 0.005)
    soft_update(critic, target_critic, tau)
    return actor_loss.item()

