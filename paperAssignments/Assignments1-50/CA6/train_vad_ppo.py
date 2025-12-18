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
    # Support vectorized envs: env.reset() / step() may return arrays with shape (num_envs, ...).
    obs_list, act_list, rew_list, done_list, logp_list, val_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    obs = env.reset()
    # gymnasium returns (obs, info) for reset
    if isinstance(obs, tuple) or (isinstance(obs, list) and len(obs) == 2):
        obs = obs[0]
    for _ in range(rollout_steps):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
        # get_action supports batched observations
        action, logp, value = actor_critic.get_action(obs_t)
        action_np = action.cpu().numpy()
        step_out = env.step(action_np)
        # step may return (next_obs, rewards, terminations, truncations, infos) or the 4-tuple older API
        if len(step_out) == 5:
            next_obs, rewards, terminated, truncated, infos = step_out
            dones = np.logical_or(terminated, truncated).astype(float)
        else:
            next_obs, rewards, dones, infos = step_out
            dones = np.asarray(dones).astype(float)
        obs_list.append(np.asarray(obs))
        act_list.append(action_np)
        rew_list.append(np.asarray(rewards, dtype=np.float32))
        done_list.append(np.asarray(dones, dtype=np.float32))
        logp_list.append(logp.cpu().numpy())
        val_list.append(value.cpu().numpy())
        obs = next_obs
        # Vectorized envs handle resets internally; no manual reset needed
    # bootstrap last values
    obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    _, last_val = actor_critic.get_action(obs_t)
    last_val_np = last_val.cpu().numpy()
    val_list.append(last_val_np)

    # Convert lists to numpy arrays: shapes -> [T, num_envs, ...]
    obs_arr = np.asarray(obs_list)
    actions_arr = np.asarray(act_list)
    rewards_arr = np.asarray(rew_list)
    dones_arr = np.asarray(done_list)
    logp_arr = np.asarray(logp_list)
    values_arr = np.asarray(val_list)  # shape [T+1, num_envs]

    # Flatten time and env dims for batch usage: [T * N, ...]
    T = obs_arr.shape[0]
    if rewards_arr.ndim == 1:
        # non-vectorized env (single env) -> keep shapes consistent
        batch = {
            "obs": torch.as_tensor(obs_arr, dtype=torch.float32, device=device),
            "actions": torch.as_tensor(actions_arr, dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(rewards_arr, dtype=torch.float32, device=device),
            "dones": torch.as_tensor(dones_arr, dtype=torch.float32, device=device),
            "logp_old": torch.as_tensor(logp_arr, dtype=torch.float32, device=device),
            "values": torch.as_tensor(values_arr, dtype=torch.float32, device=device),
        }
        return batch

    # vectorized: rewards_arr shape [T, N]
    N = rewards_arr.shape[1]
    # flatten with order 'C' (time major then env)
    batch = {
        "obs": torch.as_tensor(
            obs_arr.reshape(T * N, *obs_arr.shape[2:]),
            dtype=torch.float32,
            device=device,
        ),
        "actions": torch.as_tensor(
            actions_arr.reshape(T * N, *actions_arr.shape[2:]),
            dtype=torch.float32,
            device=device,
        ),
        "rewards": torch.as_tensor(
            rewards_arr.reshape(T * N), dtype=torch.float32, device=device
        ),
        "dones": torch.as_tensor(
            dones_arr.reshape(T * N), dtype=torch.float32, device=device
        ),
        "logp_old": torch.as_tensor(
            logp_arr.reshape(T * N), dtype=torch.float32, device=device
        ),
        # keep values as numpy [T+1, N] converted to tensor
        "values": torch.as_tensor(values_arr, dtype=torch.float32, device=device),
        "env_T": T,
        "env_N": N,
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
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short smoke test (small rollouts/updates)",
    )
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
    p.add_argument(
        "--save-ckpt", action="store_true", help="Save checkpoint after each update"
    )
    p.add_argument("--ckpt-path", type=str, default="ckpt_vadppo.pt")
    return p


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create (vectorized) envs if requested
    if args.num_envs > 1:

        def make_one(i):
            def _thunk():
                env = gym.make(args.env)
                env.reset(seed=args.seed + i)
                return env

            return _thunk

        env_fns = [make_one(i) for i in range(args.num_envs)]
        try:
            env = gym.vector.SyncVectorEnv(env_fns)
        except Exception:
            # fallback: create single env
            env = gym.make(args.env)
    else:
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
    if args.smoke:
        args.rollout_steps = min(64, args.rollout_steps)
        args.total_updates = min(3, args.total_updates)

    for update in range(args.total_updates):
        t0 = time.time()
        batch = collect_rollout(env, actor_critic, args.rollout_steps, device)
        # If vectorized, values are shape [T+1, N] and rewards/dones were flattened
        if "env_T" in batch and "env_N" in batch:
            T = int(batch["env_T"])
            N = int(batch["env_N"])
            # compute per-env advantages then flatten
            adv_list = []
            ret_list = []
            rewards_np = batch["rewards"].cpu().numpy().reshape(T, N)
            dones_np = batch["dones"].cpu().numpy().reshape(T, N)
            values_np = batch["values"].cpu().numpy()  # shape [T+1, N]
            for i in range(N):
                rew_i = torch.as_tensor(
                    rewards_np[:, i], dtype=torch.float32, device=device
                )
                vals_i = torch.as_tensor(
                    values_np[:, i], dtype=torch.float32, device=device
                )
                dones_i = torch.as_tensor(
                    dones_np[:, i], dtype=torch.float32, device=device
                )
                adv_i, ret_i = compute_advantages(
                    rew_i, vals_i, dones_i, gamma, args.lam
                )
                adv_list.append(adv_i)
                ret_list.append(ret_i)
            adv = torch.cat(adv_list, dim=0)
            returns = torch.cat(ret_list, dim=0)
        else:
            adv, returns = compute_advantages(
                batch["rewards"], batch["values"], batch["dones"], gamma, args.lam
            )
        varA = float(adv.var(unbiased=True).item())
        gamma, ema_varA = update_gamma_with_ema(
            gamma,
            varA,
            ema_varA,
            args.alpha_gamma,
            beta,
            args.sigma_target,
            args.gamma_min,
            args.gamma_max,
        )
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        # run multiple PPO epochs (simple full-batch updates here)
        stats = {}
        for _ in range(args.ppo_epochs):
            s = ppo_update(
                actor_critic,
                optimizer,
                batch,
                returns,
                adv_norm,
                args.clip_ratio,
                vf_coef=0.5,
                ent_coef=0.0,
            )
            stats.update(s)
        t1 = time.time()
        print(
            f"Update {update+1}/{args.total_updates}  gamma={gamma:.5f} varA={varA:.5f} time={t1-t0:.2f}s  stats={stats}"
        )
        # logging
        try:
            import os, csv

            os.makedirs("logs", exist_ok=True)
            log_path = "logs/vadppo_log.csv"
            header = [
                "update",
                "gamma",
                "varA",
                "varA_ema",
                "return_mean",
                "loss_pi",
                "loss_v",
                "loss_ent",
                "approx_kl",
            ]
            return_mean = (
                float(returns.mean().item())
                if isinstance(returns, torch.Tensor)
                else float(np.mean(returns))
            )
            row = [
                update,
                gamma,
                varA,
                ema_varA if ema_varA is not None else "",
                return_mean,
                stats.get("loss_pi", ""),
                stats.get("loss_v", ""),
                stats.get("loss_ent", ""),
                stats.get("approx_kl", ""),
            ]
            write_header = not os.path.exists(log_path)
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)
        except Exception:
            pass
        if args.save_ckpt:
            ckpt = {
                "model": actor_critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "gamma": gamma,
                "ema_varA": ema_varA,
                "update": update,
            }
            torch.save(ckpt, args.ckpt_path)
            print(f"Saved checkpoint to {args.ckpt_path}")

        # smoke eval: run a short deterministic episode on single env
        if args.smoke:
            try:
                eval_env = gym.make(args.env)
                obs, _ = eval_env.reset()
                done = False
                ep_ret = 0.0
                steps = 0
                while not done and steps < 1000:
                    obs_t = torch.as_tensor(
                        np.asarray(obs), dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    action, _, _ = actor_critic.get_action(obs_t, deterministic=True)
                    action_np = action.cpu().numpy()[0]
                    next_obs, reward, terminated, truncated, info = eval_env.step(
                        action_np
                    )
                    done = bool(terminated or truncated)
                    ep_ret += float(reward)
                    obs = next_obs
                    steps += 1
                print(f"Smoke eval return: {ep_ret:.2f} steps: {steps}")
            except Exception:
                pass


if __name__ == "__main__":
    main()










