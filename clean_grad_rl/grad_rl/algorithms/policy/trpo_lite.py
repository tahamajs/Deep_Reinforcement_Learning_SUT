from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from grad_rl.algorithms.policy.common import CategoricalActor, compute_gae
from grad_rl.core import RunLogger, evaluate_agent, set_seed
from grad_rl.core.networks import ValueNet
from grad_rl.envs import make_env


@dataclass
class TRPOConfig:
    env: str = "CartPole-v1"
    total_steps: int = 40000
    horizon: int = 1024
    gamma: float = 0.99
    lam: float = 0.95
    max_kl: float = 0.01
    cg_iters: int = 10
    damping: float = 0.1
    vf_lr: float = 1e-3
    vf_iters: int = 5
    hidden_sizes: tuple = (128, 128)


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p))
        idx += n


def flat_grad(y, model, retain_graph=False):
    grads = torch.autograd.grad(y, model.parameters(), retain_graph=retain_graph)
    return torch.cat([g.contiguous().view(-1) for g in grads])


def conjugate_gradient(hvp_fn, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(iters):
        Ap = hvp_fn(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


class TRPOAgent:
    def __init__(self, obs_dim, act_dim, cfg: TRPOConfig, device):
        self.actor = CategoricalActor(obs_dim, act_dim, hidden=cfg.hidden_sizes).to(device)
        self.critic = ValueNet(obs_dim, hidden=cfg.hidden_sizes).to(device)
        self.vf_opt = optim.Adam(self.critic.parameters(), lr=cfg.vf_lr)
        self.cfg = cfg
        self.device = device

    def act(self, obs, deterministic=False):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor(x)
        if deterministic:
            return int(torch.argmax(dist.logits, dim=1).item())
        return int(dist.sample().item())


def train_trpo_lite(config: Dict, out_dir: Path, seed: int) -> Dict:
    cfg = TRPOConfig(**{k: v for k, v in config.items() if k in TRPOConfig.__annotations__})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(cfg.env, seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n
    agent = TRPOAgent(obs_dim, act_dim, cfg, device)

    logger = RunLogger(
        run_id=f"trpo_lite_{cfg.env.replace('/', '_')}_s{seed}",
        chain="policy",
        algo="trpo_lite",
        env=cfg.env,
        seed=seed,
        budget={"steps": cfg.total_steps},
        out_dir=out_dir,
    )

    obs, _ = env.reset()
    step_count = 0

    while step_count < cfg.total_steps:
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        for _ in range(cfg.horizon):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = agent.actor(obs_t)
            action = dist.sample()
            logp = dist.log_prob(action).item()
            value = agent.critic(obs_t).item()
            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            obs_buf.append(obs.copy())
            act_buf.append(int(action.item()))
            logp_buf.append(float(logp))
            rew_buf.append(float(reward))
            done_buf.append(float(done))
            val_buf.append(float(value))

            obs = next_obs
            step_count += 1
            if done:
                logger.log_train(step_count, sum(rew_buf[-20:]))
                obs, _ = env.reset()
            if step_count >= cfg.total_steps:
                break

        returns, adv = compute_gae(rew_buf, val_buf, done_buf, cfg.gamma, cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.int64, device=device)
        logp_old_t = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)

        # value updates
        for _ in range(cfg.vf_iters):
            values = agent.critic(obs_t).squeeze(1)
            v_loss = ((ret_t - values) ** 2).mean()
            agent.vf_opt.zero_grad()
            v_loss.backward()
            agent.vf_opt.step()

        dist = agent.actor(obs_t)
        old_logits = dist.logits.detach()
        logp = dist.log_prob(act_t)
        ratio = torch.exp(logp - logp_old_t)
        surr = (ratio * adv_t).mean()
        old_dist = torch.distributions.Categorical(logits=old_logits)
        kl = torch.distributions.kl_divergence(old_dist, dist).mean()

        g = flat_grad(surr, agent.actor, retain_graph=True).detach()

        def hvp(v):
            new_dist = agent.actor(obs_t)
            kl_local = torch.distributions.kl_divergence(old_dist, new_dist).mean()
            grad_kl = flat_grad(kl_local, agent.actor, retain_graph=True)
            kl_v = (grad_kl * v).sum()
            h = flat_grad(kl_v, agent.actor, retain_graph=True).detach()
            return h + cfg.damping * v

        step_dir = conjugate_gradient(hvp, g, iters=cfg.cg_iters)
        shs = 0.5 * (step_dir * hvp(step_dir)).sum()
        scale = torch.sqrt(torch.tensor(cfg.max_kl, device=device) / (shs + 1e-8))
        full_step = step_dir * scale

        old_params = flat_params(agent.actor)

        def surrogate_and_kl():
            d = agent.actor(obs_t)
            lp = d.log_prob(act_t)
            rat = torch.exp(lp - logp_old_t)
            sur = (rat * adv_t).mean()
            kl_val = torch.distributions.kl_divergence(old_dist, d).mean()
            return sur, kl_val

        expected_improve = (g * full_step).sum()
        step_frac = 1.0
        for _ in range(10):
            new_params = old_params + step_frac * full_step
            set_flat_params(agent.actor, new_params)
            new_surr, new_kl = surrogate_and_kl()
            improve = new_surr - surr
            if improve > 0 and new_kl <= cfg.max_kl:
                break
            step_frac *= 0.5
        else:
            set_flat_params(agent.actor, old_params)

    eval_stats = evaluate_agent(agent, env, episodes=5, deterministic=True)
    payload = logger.finalize(eval_stats)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}, out_dir / "model.pt")
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return payload
