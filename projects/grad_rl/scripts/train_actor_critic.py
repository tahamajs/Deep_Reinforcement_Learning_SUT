import argparse
import json
from pathlib import Path

import yaml
from sb3_contrib import TRPO
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.callbacks import ProgressBarCallback

from grad_rl import make_env, set_seed, evaluate_sb3


def parse_args():
    p = argparse.ArgumentParser(description="Actor-Critic chain: A3C (A2C stand-in) and SAC")
    p.add_argument("--config", default="projects/grad_rl/configs/actor_critic.yaml")
    p.add_argument("--algo", choices=["a3c", "sac"], default=None)
    p.add_argument("--env", default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def run_a3c(env_id, cfg, total_steps, seed):
    set_seed(seed)
    # Using A2C (synchronous) as lightweight stand-in for A3C
    env = make_env(env_id, seed=seed, num_envs=int(cfg.get("num_envs", 8)))
    model = A2C(
        cfg.get("policy", "MlpPolicy"),
        env,
        learning_rate=cfg.get("learning_rate", 7e-4),
        n_steps=5,
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("lam", 0.95),
        tensorboard_log="projects/grad_rl/outputs/tb/actor_critic",
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_steps, callback=ProgressBarCallback())
    return model, evaluate_sb3(model, env, n_eval_episodes=5)


def run_sac(env_id, cfg, total_steps, seed):
    set_seed(seed)
    env = make_env(env_id, seed=seed, num_envs=1)  # SAC expects continuous single env
    model = SAC(
        cfg.get("policy", "MlpPolicy"),
        env,
        learning_rate=cfg.get("learning_rate", 3e-4),
        buffer_size=cfg.get("buffer_size", 200000),
        batch_size=cfg.get("batch_size", 256),
        tau=cfg.get("tau", 0.005),
        gamma=cfg.get("gamma", 0.99),
        ent_coef=cfg.get("entropy_coef", "auto"),
        tensorboard_log="projects/grad_rl/outputs/tb/actor_critic",
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_steps, callback=ProgressBarCallback())
    return model, evaluate_sb3(model, env, n_eval_episodes=5)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    algo = args.algo or cfg.get("algo", "sac")
    env_id = args.env or cfg.get("env", "Pendulum-v1")
    total_steps = args.total_steps or cfg.get("total_steps", 200_000)

    if algo == "a3c":
        model, metrics = run_a3c(env_id, cfg, total_steps, args.seed)
    else:
        model, metrics = run_sac(env_id, cfg, total_steps, args.seed)

    out = Path(cfg.get("save_path", f"projects/grad_rl/outputs/actor_critic/{algo}.zip"))
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    metrics["algo"] = algo
    metrics_path = out.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved", out)
    print("Eval", metrics)


if __name__ == "__main__":
    main()
