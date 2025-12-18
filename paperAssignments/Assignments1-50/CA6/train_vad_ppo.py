"""
Minimal training script for VAD-PPO (Variance-Adaptive Discount PPO).
This file is import-safe (no training on import). Run as a script to start training.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import gymnasium as gym
except Exception:
    import gym

from advantage import compute_advantages, update_gamma_with_ema
from ppo_core import ActorCritic
from utils import set_seed


def collect_rollout(
    env,
    actor_critic: ActorCritic,
    rollout_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Collect a single rollout of length `rollout_steps` using the current policy.
    Returns a dict with tensors: obs, actions, rewards, dones, logp_old, values
    """
    obs_list, act_list, rew_list, done_list, logp_list, val_list = [], [], [], [], [], []
    obs, _ = env.reset()
    for _ in range(rollout_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, logp, value = actor_critic.get_action(obs_t)
        # to numpy for env
        action_np = action.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = float(terminated or truncated)
        obs_list.append(obs)
        act_list.append(action.cpu().numpy()[0])
        rew_list.append(float(reward))
        done_list.append(done)
        logp_list.append(float(logp.cpu().numpy()))
        val_list.append(float(value.cpu().numpy()))
        obs = next_obs
        if done:
            obs, _ = env.reset()
    # append last value bootstrap
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    _, last_val = actor_critic.get_action(obs_t)
    val_list.append(float(last_val.cpu().numpy()))

    batch = {
        "obs": torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.asarray(act_list), dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(np.asarray(rew_list), dtype=torch.float32, device=device),
        "dones": torch.as_tensor(np.asarray(done_list), dtype=torch.float32, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_list), dtype=torch.float32, device=device),
        "values": torch.as_tensor(np.asarray(val_list), dtype=torch.float32, device=device),
    }
    return batch


def ppo_update(
    actor_critic: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
    vf_coef: float,
    ent_coef: float,
) -> Dict[str, float]:
    """
    Single PPO update over the provided batch (one epoch over full batch).
    For simplicity we do a single large-batch update here; splitting into minibatches is straightforward.
    """
    obs = batch["obs"]
    actions = batch["actions"]
    logp_old = batch["logp_old"]

    logp, ent, values = actor_critic.evaluate_actions(obs, actions)
    ratio = torch.exp(logp - logp_old)
    adv = advantages
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    loss_pi = -torch.min(surr1, surr2).mean()
    loss_v = F.mse_loss(values, returns)
    loss_ent = -ent.mean()
    loss = loss_pi + vf_coef * loss_v + ent_coef * loss_ent
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)
    optimizer.step()
    with torch.no_grad():
        approx_kl = (logp_old - logp).mean().item()
    return {
        "loss_pi": float(loss_pi.item()),
        "loss_v": float(loss_v.item()),
        "loss_ent": float(loss_ent.item()),
        "approx_kl": approx_kl,
    }


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma-init", type=float, default=0.95)
    p.add_argument("--gamma-min", type=float, default=0.90)
    p.add_argument("--gamma-max", type=float, default=0.999)
    p.add_argument("--alpha-gamma", type=float, default=5e-4)
    p.add_argument("--sigma-target", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--total-updates", type=int, default=10)
    return p


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env)

    obs_space = env.observation_space
    act_space = env.action_space
    if hasattr(obs_space, "shape"):
        obs_dim = int(np.prod(obs_space.shape))
    else:
        raise NotImplementedError("Unsupported observation space")
    if hasattr(act_space, "shape"):
        act_dim = int(np.prod(act_space.shape))
        continuous = True
    else:
        act_dim = int(act_space.n)
        continuous = False

    actor_critic = ActorCritic(obs_dim, act_dim, continuous=continuous).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    gamma = float(args.gamma_init)
    ema_varA = None
    beta = 0.9

    for update in range(args.total_updates):
        t0 = time.time()
        batch = collect_rollout(env, actor_critic, args.rollout_steps, device)
        adv, returns = compute_advantages(batch["rewards"], batch["values"], batch["dones"], gamma, args.lam)
        varA = float(adv.var(unbiased=True).item())
        gamma, ema_varA = update_gamma_with_ema(gamma, varA, ema_varA, args.alpha_gamma, beta, args.sigma_target, args.gamma_min, args.gamma_max)
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        # run multiple PPO epochs (simple full-batch updates here)
        stats = {}
        for _ in range(args.ppo_epochs):
            s = ppo_update(actor_critic, optimizer, batch, returns, adv_norm, args.clip_ratio, vf_coef=0.5, ent_coef=0.0)
            stats.update(s)
        t1 = time.time()
        print(f"Update {update+1}/{args.total_updates}  gamma={gamma:.5f} varA={varA:.5f} time={t1-t0:.2f}s  stats={stats}")


if __name__ == "__main__":
    main()

