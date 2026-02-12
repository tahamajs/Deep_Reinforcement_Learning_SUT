from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from grad_rl.core import RunLogger, mean_std_ci95, set_seed
from grad_rl.core.networks import CategoricalActor, ValueNet


@dataclass
class IPPOConfig:
    env: str = "simple_spread_v3"
    total_steps: int = 20000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple = (128, 128)
    max_cycles: int = 25


class IPPOAgent:
    def __init__(self, obs_dim, act_dim, cfg: IPPOConfig, device):
        self.actor = CategoricalActor(obs_dim, act_dim, hidden=cfg.hidden_sizes).to(device)
        self.critic = ValueNet(obs_dim, hidden=cfg.hidden_sizes).to(device)
        self.optim = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.learning_rate)
        self.device = device
        self.cfg = cfg

    def act(self, obs, deterministic: bool = False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor(obs_t)
        if deterministic:
            return int(torch.argmax(dist.logits, dim=1).item())
        return int(dist.sample().item())


def _make_env(max_cycles=25):
    from pettingzoo.mpe import simple_spread_v3

    return simple_spread_v3.parallel_env(max_cycles=max_cycles, continuous_actions=False)


def train_ippo(config: Dict, out_dir: Path, seed: int) -> Dict:
    cfg = IPPOConfig(**{k: v for k, v in config.items() if k in IPPOConfig.__annotations__})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = _make_env(max_cycles=cfg.max_cycles)
    obs_dict, _ = env.reset(seed=seed)
    agents = list(env.possible_agents)
    obs_dim = len(obs_dict[agents[0]])
    act_dim = env.action_space(agents[0]).n

    model = IPPOAgent(obs_dim, act_dim, cfg, device)
    logger = RunLogger(
        run_id=f"ippo_{cfg.env}_s{seed}",
        chain="marl",
        algo="ippo",
        env=cfg.env,
        seed=seed,
        budget={"steps": cfg.total_steps},
        out_dir=out_dir,
    )

    step_count = 0
    ep_reward = 0.0
    ep_returns: List[float] = []

    while step_count < cfg.total_steps:
        actions = {}
        cache = {}
        for agent in agents:
            if agent not in obs_dict:
                continue
            obs = obs_dict[agent]
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = model.actor(obs_t)
            value = model.critic(obs_t)
            action_t = dist.sample()
            actions[agent] = int(action_t.item())
            cache[agent] = {
                "obs": obs_t,
                "logp": dist.log_prob(action_t),
                "value": value.squeeze(1),
            }

        next_obs, rewards, terms, truncs, infos = env.step(actions)
        done = all(terms.values()) or all(truncs.values())

        losses = []
        for agent in actions.keys():
            r = float(rewards.get(agent, 0.0))
            obs_next = next_obs.get(agent)
            if obs_next is None:
                target_v = torch.tensor([0.0], device=device)
            else:
                with torch.no_grad():
                    target_v = model.critic(torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(1)
            td_target = torch.tensor([r], dtype=torch.float32, device=device) + cfg.gamma * target_v * (0.0 if done else 1.0)
            adv = td_target - cache[agent]["value"]
            policy_loss = -(cache[agent]["logp"] * adv.detach()).mean()
            value_loss = (adv.pow(2)).mean()
            entropy = model.actor(cache[agent]["obs"]).entropy().mean()
            losses.append(policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy)
            ep_reward += r

        if losses:
            loss = torch.stack(losses).mean()
            model.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.actor.parameters()) + list(model.critic.parameters()), cfg.max_grad_norm)
            model.optim.step()

        step_count += 1
        obs_dict = next_obs

        if done:
            ep_returns.append(ep_reward)
            logger.log_train(step_count, ep_reward)
            obs_dict, _ = env.reset()
            ep_reward = 0.0

    eval_stats = {
        "mean_reward": float(np.mean(ep_returns[-20:])) if ep_returns else 0.0,
        "std_reward": float(np.std(ep_returns[-20:])) if ep_returns else 0.0,
        "ci95": float(1.96 * np.std(ep_returns[-20:]) / np.sqrt(max(len(ep_returns[-20:]), 1))) if ep_returns else 0.0,
        "episodes": min(len(ep_returns), 20),
    }
    payload = logger.finalize(eval_stats)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": model.actor.state_dict(), "critic": model.critic.state_dict()}, out_dir / "model.pt")
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return payload
