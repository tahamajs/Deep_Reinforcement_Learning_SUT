"""Safe offline-to-online fine-tuning with PID Lagrangian and optional shield."""
import argparse
import os
import numpy as np
import torch
from tqdm import trange

from projects.amasa.amasa.envs.suturing_env import SuturingEnv
from projects.amasa.amasa.offline.cql import CQLAgent, CQLConfig
from projects.amasa.amasa.safety.pid_lagrangian import PIDLagrangian, PIDConfig
from projects.amasa.amasa.safety.replay_buffer import ReplayBuffer
from projects.amasa.amasa.safety.shield import SafetyShield


def main(args):
    env = SuturingEnv(max_steps=args.max_steps, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    cfg = CQLConfig(obs_dim=obs_dim, act_dim=act_dim, device=args.device, cql_alpha=args.cql_alpha)
    agent = CQLAgent(cfg)
    if args.checkpoint:
        agent.load(args.checkpoint, map_location=args.device)
        print(f"Loaded checkpoint {args.checkpoint}")

    pid = PIDLagrangian(PIDConfig(kp=args.kp, ki=args.ki, kd=args.kd, lambda_max=args.lambda_max))
    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim, device=args.device)
    shield = SafetyShield() if args.use_shield else None
    shield_data = {"states": [], "actions": [], "costs": [], "term": []}

    os.makedirs(args.out_dir, exist_ok=True)

    obs, _ = env.reset()
    ep_cost = 0.0
    ep_reward = 0.0
    metrics = {}
    for step in trange(args.steps, desc="online"):
        if len(buf) < args.random_steps:
            action = env.action_space.sample()
            info_reason = []
        else:
            action = agent.act(obs)
            info_reason = []
        if shield and shield.trained:
            action, info_reason = shield.filter(obs, action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        cost = info.get("cost", 0.0)
        violation = max(0.0, cost - args.cost_limit)
        lam = pid.update(violation)
        shaped_reward = reward - lam * cost

        done_flag = float(terminated or truncated)
        buf.add(obs, action, shaped_reward, next_obs, done_flag, cost)
        shield_data["states"].append(obs)
        shield_data["actions"].append(action)
        shield_data["costs"].append(cost)
        shield_data["term"].append(done_flag)

        obs = next_obs
        ep_cost += cost
        ep_reward += reward

        if len(buf) >= args.batch_size:
            b_obs, b_act, b_rew, b_next, b_done, _ = buf.sample(args.batch_size)
            # reuse CQL update; costs already folded into reward via shaping
            metrics = agent.update((b_obs, b_act, b_rew, b_next, b_done))

        if terminated or truncated:
            obs, _ = env.reset()
            ep_cost = 0.0
            ep_reward = 0.0
            pid.reset()

        if shield and not shield.trained and len(shield_data["states"]) > args.shield_train_after:
            states = np.array(shield_data["states"], dtype=np.float32)
            actions = np.array(shield_data["actions"], dtype=np.float32)
            costs = np.array(shield_data["costs"], dtype=np.float32)
            terminals = np.array(shield_data["term"], dtype=np.float32)
            shield.fit(states, actions, costs, terminals)
            print("Shield trained on", len(states), "samples")

        if (step + 1) % args.log_every == 0 and metrics:
            print({"step": step + 1, "lambda": round(pid.lmbda, 3), **{k: round(v,3) for k,v in metrics.items()}})
        if (step + 1) % args.save_every == 0:
            path = os.path.join(args.out_dir, f"safe_step{step+1}.pt")
            agent.save(path)
            if shield and shield.trained:
                shield.save(os.path.join(args.out_dir, f"shield_step{step+1}.joblib"))
    agent.save(os.path.join(args.out_dir, "safe_final.pt"))
    if shield and shield.trained:
        shield.save(os.path.join(args.out_dir, "shield_final.joblib"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--random_steps", type=int, default=2000)
    parser.add_argument("--cost_limit", type=float, default=0.0)
    parser.add_argument("--kp", type=float, default=0.5)
    parser.add_argument("--ki", type=float, default=0.05)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--lambda_max", type=float, default=10.0)
    parser.add_argument("--cql_alpha", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--use_shield", action="store_true")
    parser.add_argument("--shield_train_after", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10000)
    args = parser.parse_args()
    main(args)
