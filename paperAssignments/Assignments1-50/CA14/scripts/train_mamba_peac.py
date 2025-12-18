"""Training loop for MAMBA-PEAC (minimal, runnable skeleton).

This script implements a compact training loop that:
- collects episodes from a Gymnasium env using the actor
- stores episodes in a simple replay buffer
- updates world model, morphology encoder, actor and value networks

The implementation is intentionally simple to be readable and easy to extend.
"""

from __future__ import annotations

import argparse
import yaml
import os
import random
from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mamba_core.morph_encoder import MorphEncoder
from mamba_core.world_model import WorldModel
from mamba_core.actor import Actor
from mamba_core.value import ValueNet
from mamba_core.losses import kl_normal, world_model_loss, td_lambda
from mamba_core.replay import ReplayBuffer


def build_models(cfg: Dict, obs_dim: int, act_dim: int):
    morph_dim = cfg.get("morph", {}).get("latent_dim", 16)
    wm_latent = cfg.get("world_model", {}).get("latent_dim", 64)
    wm = WorldModel(
        obs_dim=obs_dim, act_dim=act_dim, stoch_dim=wm_latent, morph_dim=morph_dim
    )
    morph = MorphEncoder(obs_dim=obs_dim, act_dim=act_dim, latent_dim=morph_dim)
    actor = Actor(latent_dim=wm_latent, morph_dim=morph_dim, act_dim=act_dim)
    value = ValueNet(latent_dim=wm_latent, morph_dim=morph_dim)
    return wm, morph, actor, value


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/mamba_peac.yaml")
    p.add_argument("--env", type=str, default="Walker2d-v4")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument(
        "--train_morphs",
        nargs="+",
        default=["walker2d-v4", "hopper-v4", "halfcheetah-v4"],
    )
    p.add_argument("--heldout_morph", type=str, default="ant-v4")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument(
        "--smoke", action="store_true", help="run a very short smoke training loop"
    )
    return p.parse_args()


def infer_morph_from_history(
    morph_encoder: MorphEncoder,
    obs_seq: np.ndarray,
    act_seq: np.ndarray,
    rew_seq: np.ndarray,
    done_seq: np.ndarray,
    device: torch.device,
):
    # obs_seq: (B, L, D)
    obs_t = torch.from_numpy(obs_seq).float().to(device)
    act_t = torch.from_numpy(act_seq).float().to(device)
    rew_t = torch.from_numpy(rew_seq).float().to(device)
    done_t = torch.from_numpy(done_seq).float().to(device)
    z_m, mu_m, logvar_m = morph_encoder(obs_t, act_t, rew_t, done_t)
    return z_m, mu_m, logvar_m


def collect_episode(
    env,
    actor: Actor,
    morph_enc: MorphEncoder,
    device: torch.device,
    max_steps: int = 1000,
):
    obs_list, act_list, rew_list, done_list = [], [], [], []
    obs, _ = env.reset()
    for _ in range(max_steps):
        obs_arr = np.array(obs, dtype=np.float32)
        # simple zero morph conditioning for collection; actor expects torch input
        z = torch.zeros(
            1, actor.fc1.in_features - actor.film.scale.in_features, device=device
        )
        z = torch.randn(1, actor.film.scale.in_features, device=device) * 0.0  # zeros
        obs_t = torch.from_numpy(obs_arr).float().unsqueeze(0).to(device)
        # sample random action initially (exploration)
        with torch.no_grad():
            a = actor.act(
                torch.randn(
                    1,
                    actor.fc1.in_features - actor.film.scale.in_features,
                    device=device,
                ),
                z,
            )
            a = a.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(a)
        done = bool(terminated or truncated)
        obs_list.append(obs_arr)
        act_list.append(a)
        rew_list.append(float(reward))
        done_list.append(done)
        obs = next_obs
        if done:
            break
    return obs_list, act_list, rew_list, done_list


def train_loop(cfg: Dict, env_name: str, steps: int = 20000, smoke: bool = False):
    device = torch.device("cpu")
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    wm, morph, actor, value = build_models(cfg, obs_dim, act_dim)
    wm.to(device)
    morph.to(device)
    actor.to(device)
    value.to(device)

    # optimizers
    wm_opt = optim.Adam(wm.parameters(), lr=cfg.get("world_model", {}).get("lr", 3e-4))
    morph_opt = optim.Adam(
        morph.parameters(),
        lr=(
            cfg.get("morph", {}).get("lr", 3e-4)
            if cfg.get("morph", {}).get("lr")
            else 3e-4
        ),
    )
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.get("actor", {}).get("lr", 3e-4))
    value_opt = optim.Adam(value.parameters(), lr=cfg.get("value", {}).get("lr", 3e-4))

    replay = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        capacity=10000,
        seq_len=cfg.get("morph", {}).get("history_len", 50),
    )

    # warmup: collect a few random episodes
    warmup_eps = 5 if not smoke else 1
    for _ in range(warmup_eps):
        ep = collect_episode(env, actor, morph, device, max_steps=200)
        replay.add_episode(*ep)
    print(f"Replay size after warmup: {len(replay)} episodes")

    total_steps = 0
    update_every = cfg.get("train", {}).get("updates_per_env", 2)
    batch_size = cfg.get("train", {}).get("batch_size", 16)
    horizon = cfg.get("train", {}).get("horizon", 15)
    beta_z = cfg.get("world_model", {}).get("beta_z", 1.0)
    beta_m = cfg.get("morph", {}).get("beta_m", 0.5)
    free_bits_z = cfg.get("world_model", {}).get("free_bits", 1.0)
    free_bits_m = cfg.get("morph", {}).get("free_bits", 1.0)

    while total_steps < steps:
        # collect one episode with current actor (simple behavior)
        ep = collect_episode(env, actor, morph, device, max_steps=500)
        replay.add_episode(*ep)
        total_steps += len(ep[0])

        # training updates
        for _ in range(update_every):
            if len(replay) == 0:
                break
            batch = replay.sample_batch(batch_size)
            # batch arrays: obs (B,L,D), acts (B,L,A), rews (B,L), dones (B,L)
            obs_b = torch.from_numpy(batch["obs"]).float().to(device)
            acts_b = torch.from_numpy(batch["acts"]).float().to(device)
            rews_b = torch.from_numpy(batch["rews"]).float().to(device)
            dones_b = torch.from_numpy(batch["dones"]).float().to(device)

            # infer morphology latent from full history
            z_m, mu_m, logvar_m = morph(obs_b, acts_b, rews_b, dones_b)
            # compute KL for morph
            kl_m = kl_normal(mu_m, logvar_m).mean()

            # World model supervised updates on sequence (simple per-timestep ELBO)
            B, L, D = obs_b.shape
            h, z = wm.init_state(B, device)
            recon_losses = []
            reward_losses = []
            kl_z_terms = []
            for t in range(L):
                a_t = acts_b[:, t]
                obs_t = obs_b[:, t]
                h, z, mu_post, logvar_post = wm.observe(h, a_t, z_m)
                # prior on current h
                mu_prior, logvar_prior = wm.rssm.prior(h)
                recon = wm.decode_obs(h, z, z_m)
                r_pred = wm.predict_reward(h, z, z_m)
                recon_losses.append(((recon - obs_t) ** 2).mean())
                reward_losses.append(((r_pred - rews_b[:, t]) ** 2).mean())
                kl_z_terms.append(
                    kl_normal(mu_post, logvar_post, mu_prior, logvar_prior)
                )
            recon_loss = torch.stack(recon_losses).mean()
            reward_loss = torch.stack(reward_losses).mean()
            kl_z = torch.stack(kl_z_terms).mean()

            wm_loss = (
                recon_loss
                + reward_loss
                + beta_z * torch.clamp(kl_z - free_bits_z, min=0.0).mean()
                + beta_m * torch.clamp(kl_m - free_bits_m, min=0.0).mean()
            )

            wm_opt.zero_grad()
            morph_opt.zero_grad()
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                wm.parameters(), cfg.get("train", {}).get("grad_clip", 40)
            )
            wm_opt.step()
            morph_opt.step()

            # Imagined rollouts for actor/value updates
            # start from zero state for each batch element
            h0, z0 = wm.init_state(B, device)
            # for simplicity use mean of inferred z_m across batch as conditioning for imagination
            z_m_imagine = z_m.detach()
            # run imagined rollout
            imag_rewards = []
            imag_gammas = []
            imag_values = []
            zs = []
            hs = []
            z_curr = z0
            h_curr = h0
            for t in range(horizon):
                # actor samples action from current stochastic latent z_curr
                # flatten z_curr to match actor input size
                z_input = z_curr
                with torch.no_grad():
                    a_mu, a_std = actor(z_input, z_m_imagine)
                    a = a_mu + a_std * torch.randn_like(a_std)
                # imagine one step
                h_curr, z_curr, mu_prior, logvar_prior = wm.imagine(
                    h_curr, a, z_m_imagine
                )
                r_pred = wm.predict_reward(h_curr, z_curr, z_m_imagine)
                gamma = wm.predict_discount(h_curr, z_curr, z_m_imagine)
                v = value(z_curr, z_m_imagine)
                imag_rewards.append(r_pred)
                imag_gammas.append(gamma)
                imag_values.append(v)
                zs.append(z_curr)
                hs.append(h_curr)

            # compute TD(lambda) targets and update value
            targets = td_lambda(imag_rewards, imag_gammas, imag_values, lambda_=0.95)
            # value loss
            value_loss = 0.0
            for v_pred, target in zip(imag_values, targets):
                value_loss = value_loss + ((v_pred - target.detach()) ** 2).mean()
            value_loss = value_loss / len(imag_values)

            value_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                value.parameters(), cfg.get("train", {}).get("grad_clip", 40)
            )
            value_opt.step()

            # actor loss: negative imagined return (policy gradient through imagined model not implemented here)
            # We use a simple surrogate: maximize sum of predicted rewards minus value baseline
            returns = targets
            actor_loss = 0.0
            for a_step, r_step, v_step in zip(
                range(len(returns)), imag_rewards, imag_values
            ):
                adv = (returns[a_step].detach() - v_step).mean()
                # encourage actions that lead to higher advantage via simple negated advantage times action magnitude
                actor_loss = actor_loss - adv
            actor_loss = actor_loss / len(returns)

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), cfg.get("train", {}).get("grad_clip", 40)
            )
            actor_opt.step()

        if total_steps % 1000 == 0:
            print(
                f"steps={total_steps}, wm_loss={wm_loss.item():.4f}, value_loss={value_loss.item():.4f}, actor_loss={actor_loss.item():.4f}"
            )

        if smoke:
            break

    env.close()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config {cfg_path} not found. Using defaults from README skeleton.")
        cfg = {}
    else:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

    if args.eval_only:
        print("Eval-only mode not implemented in skeleton.")
        return

    train_loop(cfg, args.env, steps=args.steps, smoke=args.smoke)


if __name__ == "__main__":
    main()












