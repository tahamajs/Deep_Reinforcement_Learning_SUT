"""Train HIRO-style hierarchy on the suturing env."""
import argparse
import os
import numpy as np
import torch
from tqdm import trange

from projects.amasa.amasa.envs.suturing_env import SuturingEnv
from projects.amasa.amasa.hierarchical.hiro import HIROAgent, HIROConfig, TrajectoryBuffer, MetaBuffer
from projects.amasa.amasa.offline.cql import CQLAgent


def main(args):
    env = SuturingEnv(max_steps=args.max_steps, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    cfg = HIROConfig(obs_dim=obs_dim, act_dim=act_dim, device=args.device)
    agent = HIROAgent(cfg)

    # optionally warm-start low-level policy from offline CQL actor
    if args.offline_checkpoint:
        try:
            from projects.amasa.amasa.offline.cql import CQLConfig

            offline_cfg = CQLConfig(obs_dim=obs_dim, act_dim=act_dim, device=args.device)
            offline_agent = CQLAgent(offline_cfg)
            offline_agent.load(args.offline_checkpoint, map_location=args.device)

            # copy weights, padding first layer for goal dims (obs+goal)
            src = offline_agent.actor.trunk.net[0]  # Linear
            dst = agent.lo_actor.trunk.net[0]
            with torch.no_grad():
                padded_w = torch.zeros_like(dst.weight)
                padded_w[:, : src.weight.shape[1]] = src.weight
                dst.weight.copy_(padded_w)
                dst.bias.copy_(src.bias)
                # remaining layers share shape; copy directly
                dst_idx = 2  # after first Linear+ReLU
                src_idx = 2
                agent.lo_actor.trunk.net[dst_idx].weight.copy_(offline_agent.actor.trunk.net[src_idx].weight)
                agent.lo_actor.trunk.net[dst_idx].bias.copy_(offline_agent.actor.trunk.net[src_idx].bias)
                agent.lo_actor.trunk.net[dst_idx + 2].weight.copy_(offline_agent.actor.trunk.net[src_idx + 2].weight)
                agent.lo_actor.trunk.net[dst_idx + 2].bias.copy_(offline_agent.actor.trunk.net[src_idx + 2].bias)
            print(f"Loaded offline checkpoint {args.offline_checkpoint} into low-level actor (padded goals)")
        except Exception as exc:
            print(f"Warn: failed to load offline checkpoint: {exc}")

    traj_buf = TrajectoryBuffer(args.buffer_size, obs_dim, act_dim, cfg.goal_dim, device=args.device)
    meta_buf = MetaBuffer(args.buffer_size // cfg.horizon, obs_dim, cfg.goal_dim, device=args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0
    for episode in trange(args.episodes, desc="episodes"):
        obs, _ = env.reset()
        goal = agent.propose_goal(obs)
        goal_start_obs = obs.copy()
        ret = 0.0
        for t in range(env.max_steps):
            action = agent.act(obs, goal)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = float(terminated)
            traj_buf.add(obs, goal, action, reward, next_obs, done)
            ret += reward
            global_step += 1

            if (t + 1) % cfg.horizon == 0 or terminated or truncated:
                meta_buf.add(goal_start_obs, goal, ret, next_obs, done)
                goal_start_obs = next_obs.copy()
                ret = 0.0
                if not terminated and not truncated:
                    goal = agent.propose_goal(next_obs)

            obs = next_obs
            if traj_buf.ptr >= args.batch_size:
                lo_batch = traj_buf.sample(args.batch_size)
                agent.update_low(lo_batch)
            if meta_buf.ptr >= args.batch_size // cfg.horizon:
                meta_batch = meta_buf.sample(max(1, args.batch_size // cfg.horizon))
                agent.update_meta(meta_batch)
            if terminated or truncated:
                break
        if (episode + 1) % args.save_every == 0:
            torch.save(agent, os.path.join(args.out_dir, f"hiro_ep{episode+1}.pt"))
    torch.save(agent, os.path.join(args.out_dir, "hiro_final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--offline_checkpoint", type=str, default="")
    args = parser.parse_args()
    main(args)
