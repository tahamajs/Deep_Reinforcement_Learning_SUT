import argparse
from pathlib import Path
import json
import yaml

import ray
from ray import air, tune
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.algorithms.maddpg import MADDPGConfig
from ray.rllib.env import PettingZooEnv
from pettingzoo.mpe import simple_spread_v3


def make_env(config):
    return PettingZooEnv(simple_spread_v3.parallel_env(max_cycles=25))


def run_qmix(cfg):
    algo_cfg = (
        QMixConfig()
        .environment(make_env)
        .framework("torch")
        .rollouts(num_rollout_workers=cfg.get("num_workers", 2), num_envs_per_worker=cfg.get("num_envs_per_worker", 4))
        .training(gamma=cfg.get("gamma", 0.99), lr=cfg.get("lr", 5e-4))
    )
    return algo_cfg.build()


def run_maddpg(cfg):
    algo_cfg = (
        MADDPGConfig()
        .environment(make_env)
        .framework("torch")
        .rollouts(num_rollout_workers=cfg.get("num_workers", 2), num_envs_per_worker=cfg.get("num_envs_per_worker", 4))
        .training(gamma=cfg.get("gamma", 0.99), lr=cfg.get("lr", 5e-4))
    )
    return algo_cfg.build()


def main():
    parser = argparse.ArgumentParser(description="Multi-agent chain: QMIX and MADDPG on MPE simple_spread")
    parser.add_argument("--config", default="projects/grad_rl/configs/marl.yaml")
    parser.add_argument("--algo", choices=["qmix", "maddpg"], default=None)
    parser.add_argument("--iters", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    algo = args.algo or cfg.get("algo", "qmix")
    iters = args.iters or cfg.get("iters", 50)

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    if algo == "qmix":
        trainer = run_qmix(cfg)
    else:
        trainer = run_maddpg(cfg)

    results = []
    for i in range(iters):
        res = trainer.train()
        results.append({"iter": i, "episode_reward_mean": res["episode_reward_mean"]})
        if (i + 1) % 5 == 0:
            print(f"Iter {i+1}: reward {res['episode_reward_mean']:.2f}")

    out_dir = Path(cfg.get("save_dir", "projects/grad_rl/outputs/marl"))
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out_dir / algo))
    with open(out_dir / f"{algo}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    ray.shutdown()
    print("Saved trainer and metrics to", out_dir)


if __name__ == "__main__":
    main()
