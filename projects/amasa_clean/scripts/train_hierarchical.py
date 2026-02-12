"""Train HIRO-style hierarchy with optional config/scenario support."""
from __future__ import annotations

import argparse
import os
import torch
from tqdm import trange

from projects.amasa_clean.scripts.common import add_common_config_flags, resolve_config, make_env
from projects.amasa_clean.amasa.hierarchical.hiro import HIROAgent, HIROConfig, TrajectoryBuffer, MetaBuffer
from projects.amasa_clean.amasa.offline.cql import CQLAgent, CQLConfig


def main(args):
    cfg = resolve_config(args)
    cfg["env"]["max_steps"] = args.max_steps
    env = make_env(cfg, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    hcfg = HIROConfig(obs_dim=obs_dim, act_dim=act_dim, device=args.device)
    agent = HIROAgent(hcfg)

    if args.offline_checkpoint:
        try:
            offline = CQLAgent(CQLConfig(obs_dim=obs_dim, act_dim=act_dim, device=args.device))
            offline.load(args.offline_checkpoint, map_location=args.device)
            src = offline.actor.trunk.net[0]
            dst = agent.lo_actor.trunk.net[0]
            with torch.no_grad():
                padded = torch.zeros_like(dst.weight)
                padded[:, : src.weight.shape[1]] = src.weight
                dst.weight.copy_(padded)
                dst.bias.copy_(src.bias)
                agent.lo_actor.trunk.net[2].weight.copy_(offline.actor.trunk.net[2].weight)
                agent.lo_actor.trunk.net[2].bias.copy_(offline.actor.trunk.net[2].bias)
                agent.lo_actor.trunk.net[4].weight.copy_(offline.actor.trunk.net[4].weight)
                agent.lo_actor.trunk.net[4].bias.copy_(offline.actor.trunk.net[4].bias)
            print(f"Loaded offline checkpoint {args.offline_checkpoint} into low-level actor")
        except Exception as exc:
            print(f"Warn: failed to load offline checkpoint: {exc}")

    traj_buf = TrajectoryBuffer(args.buffer_size, obs_dim, act_dim, hcfg.goal_dim, device=args.device)
    meta_buf = MetaBuffer(max(1, args.buffer_size // hcfg.horizon), obs_dim, hcfg.goal_dim, device=args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    for episode in trange(args.episodes, desc="episodes"):
        obs, _ = env.reset()
        goal = agent.propose_goal(obs)
        goal_start_obs = obs.copy()
        ret = 0.0
        for t in range(env.max_steps):
            action = agent.act(obs, goal)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = float(terminated or truncated)
            traj_buf.add(obs, goal, action, reward, next_obs, done)
            ret += reward

            if (t + 1) % hcfg.horizon == 0 or terminated or truncated:
                meta_buf.add(goal_start_obs, goal, ret, next_obs, done)
                goal_start_obs = next_obs.copy()
                ret = 0.0
                if not (terminated or truncated):
                    goal = agent.propose_goal(next_obs)

            obs = next_obs
            if traj_buf.ptr >= args.batch_size:
                agent.update_low(traj_buf.sample(args.batch_size))
            if meta_buf.ptr >= max(1, args.batch_size // hcfg.horizon):
                agent.update_meta(meta_buf.sample(max(1, args.batch_size // hcfg.horizon)))
            if terminated or truncated:
                break

        if (episode + 1) % args.save_every == 0:
            torch.save(agent, os.path.join(args.out_dir, f"hiro_ep{episode+1}.pt"))

    torch.save(agent, os.path.join(args.out_dir, "hiro_final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_config_flags(parser)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--offline_checkpoint", type=str, default="")
    args = parser.parse_args()
    main(args)
