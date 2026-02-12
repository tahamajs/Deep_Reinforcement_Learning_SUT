"""Generate offline dataset for AMASA suturing env."""

import argparse
import os
import numpy as np
from tqdm import trange

from projects.amasa.amasa.envs.suturing_env import SuturingEnv


def heuristic_policy(obs):
    # obs layout: q(7) dq(7) needle(3) stress progress phase(4)
    needle = obs[14:17]
    # rough guess of current target progression
    progress = obs[18]
    suture_idx = int(progress * 4 + 1e-6)
    # default target list must match env._sample_targets; approximate arc
    base_targets = np.array([
        [0.02, 0.00, -0.01],
        [0.025, 0.005, -0.011],
        [0.03, -0.005, -0.012],
        [0.035, 0.0, -0.013],
    ])
    target = base_targets[min(suture_idx, 3)]
    delta = target - needle
    # PD gains chosen for stability
    action = np.zeros(7, dtype=np.float32)
    action[:3] = 3.0 * delta
    action = np.clip(action, -0.8, 0.8)
    return action


def main(args):
    env = SuturingEnv(max_steps=args.max_steps, seed=args.seed)
    obs_buf, act_buf, rew_buf, next_obs_buf, done_buf, cost_buf = [], [], [], [], [], []
    for _ in trange(args.episodes, desc="rollouts"):
        obs, _ = env.reset()
        done = False
        while not done:
            action = heuristic_policy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            cost = info.get("cost", 0.0)
            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            next_obs_buf.append(next_obs)
            done_buf.append(float(terminated))
            cost_buf.append(float(cost))
            obs = next_obs
            done = terminated or truncated
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    np.savez_compressed(
        args.out,
        obs=np.array(obs_buf, dtype=np.float32),
        actions=np.array(act_buf, dtype=np.float32),
        rewards=np.array(rew_buf, dtype=np.float32),
        next_obs=np.array(next_obs_buf, dtype=np.float32),
        dones=np.array(done_buf, dtype=np.float32),
        costs=np.array(cost_buf, dtype=np.float32),
    )
    print(f"Saved {len(obs_buf)} transitions to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/amasa_offline.npz")
    args = parser.parse_args()
    main(args)
