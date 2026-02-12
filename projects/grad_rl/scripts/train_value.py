import argparse
import json
import os
from pathlib import Path

import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import VecFrameStack
from sb3_contrib import QRDQN

from grad_rl import make_atari_env, make_env, evaluate_sb3, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Value-based chain: DQN/Double/Dueling")
    parser.add_argument("--config", default="projects/grad_rl/configs/value_dqn.yaml")
    parser.add_argument("--env", help="Gymnasium env id", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--double", action="store_true", help="force Double DQN (on by default)")
    parser.add_argument("--dueling", action="store_true", help="use dueling architecture")
    parser.add_argument("--qrdqn", action="store_true", help="use QR-DQN distributional variant")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    env_id = args.env or cfg["env"]
    total_steps = args.total_steps or cfg.get("total_steps", 200_000)
    dueling = args.dueling or bool(cfg.get("dueling", True))
    seed = args.seed

    is_atari = env_id.startswith("ALE/")
    set_seed(seed)
    if is_atari:
        env = make_atari_env(env_id, num_envs=int(cfg.get("num_envs", 4)), seed=seed)
        policy = "CnnPolicy"
    else:
        env = make_env(env_id, seed=seed, num_envs=int(cfg.get("num_envs", 8)))
        policy = cfg.get("policy", "MlpPolicy")

    algo_cls = QRDQN if args.qrdqn else DQN
    model = algo_cls(
        policy,
        env,
        learning_rate=cfg.get("learning_rate", 1e-4),
        buffer_size=cfg.get("buffer_size", 100_000),
        batch_size=cfg.get("batch_size", 32),
        gamma=cfg.get("gamma", 0.99),
        exploration_fraction=cfg.get("exploration_fraction", 0.1),
        exploration_final_eps=cfg.get("exploration_final_eps", 0.01),
        target_update_interval=cfg.get("target_update_interval", 1000),
        train_freq=cfg.get("train_freq", 4),
        learning_starts=cfg.get("learning_starts", 50_000),
        tensorboard_log="projects/grad_rl/outputs/tb/value",
        policy_kwargs={"dueling": dueling},
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_steps, callback=ProgressBarCallback())

    out_dir = Path(cfg.get("save_path", "projects/grad_rl/outputs/value_dqn"))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))

    eval_env = env
    metrics = evaluate_sb3(model, eval_env, n_eval_episodes=5)
    metrics["env"] = env_id
    metrics["steps"] = total_steps
    metrics_path = out_dir.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model to", out_dir)
    print("Eval:", metrics)


if __name__ == "__main__":
    main()
