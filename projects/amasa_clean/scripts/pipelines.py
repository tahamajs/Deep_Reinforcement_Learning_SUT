"""Shared train/eval pipelines used by legacy scripts and the unified runner."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Any, List

import numpy as np
import torch
from tqdm import trange

from projects.amasa_clean.amasa.core.metrics import StepRecord, save_records_jsonl, save_summary_csv
from projects.amasa_clean.scripts.common import make_env, make_agent, load_dataset
from projects.amasa_clean.amasa.safety import ReplayBuffer, GuardConfig, SafetyGuard


def _sample_batch(buffer: Dict[str, np.ndarray], batch_size: int, device: str):
    n = buffer["obs"].shape[0]
    idx = np.random.randint(0, n, size=batch_size)
    return (
        torch.as_tensor(buffer["obs"][idx], device=device),
        torch.as_tensor(buffer["actions"][idx], device=device),
        torch.as_tensor(buffer["rewards"][idx], device=device).unsqueeze(-1),
        torch.as_tensor(buffer["next_obs"][idx], device=device),
        torch.as_tensor(buffer["dones"][idx], device=device).unsqueeze(-1),
        torch.as_tensor(buffer.get("costs", np.zeros((n,), dtype=np.float32))[idx], device=device).unsqueeze(-1),
    )


def generate_dataset_pipeline(cfg: Dict[str, Any], out_path: str, episodes: int = 50, max_steps: int | None = None):
    env_cfg = dict(cfg)
    if max_steps is not None:
        env_cfg = {**cfg, "env": {**cfg["env"], "max_steps": max_steps}}
    env = make_env(env_cfg, seed=cfg["experiment"]["seed"])

    obs_buf, act_buf, rew_buf, next_obs_buf, done_buf, cost_buf = [], [], [], [], [], []

    def heuristic_policy(obs):
        needle = obs[14:17]
        progress = obs[18]
        suture_idx = int(progress * 4 + 1e-6)
        base_targets = np.array(
            [[0.02, 0.00, -0.01], [0.025, 0.005, -0.011], [0.03, -0.005, -0.012], [0.035, 0.0, -0.013]],
            dtype=np.float32,
        )
        target = base_targets[min(suture_idx, 3)]
        delta = target - needle
        action = np.zeros(7, dtype=np.float32)
        action[:3] = 3.0 * delta
        return np.clip(action, -0.8, 0.8)

    for _ in trange(episodes, desc="dataset"):
        obs, _ = env.reset()
        done = False
        while not done:
            action = heuristic_policy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            next_obs_buf.append(next_obs)
            done_buf.append(float(terminated or truncated))
            cost_buf.append(float(info.get("cost", 0.0)))
            obs = next_obs
            done = terminated or truncated

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    np.savez_compressed(
        out_path,
        obs=np.array(obs_buf, dtype=np.float32),
        actions=np.array(act_buf, dtype=np.float32),
        rewards=np.array(rew_buf, dtype=np.float32),
        next_obs=np.array(next_obs_buf, dtype=np.float32),
        dones=np.array(done_buf, dtype=np.float32),
        costs=np.array(cost_buf, dtype=np.float32),
    )
    return len(obs_buf)


def train_offline_pipeline(cfg: Dict[str, Any], dataset_path: str, out_dir: str):
    buffer = load_dataset(dataset_path)
    obs_dim = buffer["obs"].shape[1]
    act_dim = buffer["actions"].shape[1]

    algo_name = cfg["algo"]["name"]
    if algo_name not in {"cql", "iql"}:
        raise ValueError("offline pipeline supports only cql/iql")

    agent = make_agent(algo_name, obs_dim, act_dim, cfg)
    os.makedirs(out_dir, exist_ok=True)

    steps = cfg["train"]["steps"]
    batch_size = cfg["train"]["batch_size"]
    log_every = cfg["train"].get("eval_every", 200)
    save_every = cfg["train"].get("save_every", 500)

    last_metrics = {}
    for step in trange(steps, desc=f"offline-{algo_name}"):
        obs, act, rew, next_obs, done, _ = _sample_batch(buffer, batch_size, cfg["experiment"]["device"])
        last_metrics = agent.update((obs, act, rew, next_obs, done))
        if (step + 1) % log_every == 0:
            print({"step": step + 1, **{k: round(v, 4) for k, v in last_metrics.items()}})
        if (step + 1) % save_every == 0:
            agent.save(os.path.join(out_dir, f"{algo_name}_step{step+1}.pt"))

    final_path = os.path.join(out_dir, f"{algo_name}_final.pt")
    agent.save(final_path)

    # Offline quality gates: finite losses + improvement over random baseline.
    env = make_env(cfg, seed=cfg["experiment"]["seed"])
    eval_reward, eval_cost, success = _evaluate_agent(agent, env, cfg["eval"]["episodes"])
    random_rewards = []
    for _ in range(cfg["eval"]["episodes"]):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            done = term or trunc
        random_rewards.append(ep_reward)
    random_reward = float(np.mean(random_rewards))
    reward_ratio = eval_reward / max(1e-6, abs(random_reward))
    q_finite = int(np.isfinite(np.array(list(last_metrics.values()), dtype=np.float32)).all())

    summary = {
        "algo": algo_name,
        "scenario": cfg["scenario"]["type"],
        "seed": cfg["experiment"]["seed"],
        "avg_reward": eval_reward,
        "avg_cost": eval_cost,
        "success_rate": success,
        "reward_ratio_vs_random": reward_ratio,
        "q_finite": q_finite,
    }
    save_summary_csv(os.path.join(out_dir, "summary.csv"), [summary])
    return {"checkpoint": final_path, **last_metrics, **summary}


def _evaluate_agent(agent, env, episodes: int, use_guard: SafetyGuard | None = None):
    rewards, costs, successes = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_r, ep_c, steps = 0.0, 0.0, 0
        while not done:
            if hasattr(agent, "act"):
                a_out = agent.act(obs)
                action = a_out[0] if isinstance(a_out, tuple) else a_out
            else:
                action = env.action_space.sample()
            if use_guard is not None:
                action, _ = use_guard.process_action(obs, action)
            next_obs, reward, term, trunc, info = env.step(action)
            ep_r += reward
            ep_c += info.get("cost", 0.0)
            steps += 1
            obs = next_obs
            done = term or trunc
        rewards.append(ep_r)
        costs.append(ep_c / max(1, steps))
        successes.append(int(info.get("success", False)))
    return float(np.mean(rewards)), float(np.mean(costs)), float(np.mean(successes))


def train_online_pipeline(cfg: Dict[str, Any], out_dir: str, checkpoint_path: str = ""):
    env = make_env(cfg, seed=cfg["experiment"]["seed"])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    algo_name = cfg["algo"]["name"]
    if algo_name not in {"sac_lag", "ppo_lag", "cql"}:
        raise ValueError("online pipeline supports sac_lag/ppo_lag/cql")

    agent = make_agent(algo_name, obs_dim, act_dim, cfg)
    if checkpoint_path:
        try:
            agent.load(checkpoint_path, map_location=cfg["experiment"]["device"])
            print(f"Loaded checkpoint {checkpoint_path}")
        except Exception as exc:
            print(f"Warn: failed to load checkpoint {checkpoint_path}: {exc}")

    guard = SafetyGuard(
        GuardConfig(
            state_dim=obs_dim,
            action_dim=act_dim,
            kp=cfg["safety"]["kp"],
            ki=cfg["safety"]["ki"],
            kd=cfg["safety"]["kd"],
            lambda_max=cfg["safety"]["lambda_max"],
            cost_limit=cfg["safety"]["cost_limit"],
            risk_threshold=cfg["safety"].get("risk_threshold", 0.65),
            shield_enabled=cfg["safety"]["shield"].get("enabled", True),
            risk_enabled=cfg["safety"]["risk_critic"].get("enabled", True),
            device=cfg["experiment"]["device"],
        )
    )

    os.makedirs(out_dir, exist_ok=True)
    steps = cfg["train"]["steps"]
    save_every = cfg["train"]["save_every"]
    log_every = cfg["train"].get("eval_every", 200)

    replay = ReplayBuffer(cfg["train"]["buffer_size"], obs_dim, act_dim, device=cfg["experiment"]["device"])
    ppo_buf = {"obs": [], "act": [], "logp": [], "rew": [], "cost": [], "done": []}
    shield_fit = {"states": [], "actions": [], "costs": [], "term": []}
    safety_records: List[StepRecord] = []

    obs, _ = env.reset()
    metrics = {}
    last_info = {"cost": 0.0, "force": 0.0, "corridor_violation": 0}
    for step in trange(steps, desc=f"online-{algo_name}"):
        if algo_name == "ppo_lag":
            action, logp = agent.act(obs)
        else:
            action = env.action_space.sample() if len(replay) < cfg["train"].get("random_steps", 0) else agent.act(obs)
            logp = 0.0

        action, guard_info = guard.process_action(obs, action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done_flag = float(terminated or truncated)
        cost = float(info.get("cost", 0.0))

        guard.update_cost(cost)
        guard.observe_for_risk(obs, action, cost)
        s_log = guard.safety_log(info, guard_info["risk_score"], guard_info["shield_blocked"])
        safety_records.append(
            StepRecord(
                step=step + 1,
                reward=float(reward),
                cost=float(s_log["cost"]),
                lambda_value=float(s_log["lambda"]),
                risk_score=float(s_log["risk_score"]),
                shield_blocked=int(s_log["shield_blocked"]),
                force=float(s_log["force"]),
                corridor_violation=int(s_log["corridor_violation"]),
            )
        )

        shield_fit["states"].append(obs)
        shield_fit["actions"].append(action)
        shield_fit["costs"].append(cost)
        shield_fit["term"].append(done_flag)
        if len(shield_fit["states"]) > cfg["train"].get("shield_train_after", 200) and guard.shield is not None and not guard.shield.trained:
            guard.maybe_fit_shield(
                np.array(shield_fit["states"], dtype=np.float32),
                np.array(shield_fit["actions"], dtype=np.float32),
                np.array(shield_fit["costs"], dtype=np.float32),
                np.array(shield_fit["term"], dtype=np.float32),
            )

        if algo_name == "ppo_lag":
            ppo_buf["obs"].append(obs)
            ppo_buf["act"].append(action)
            ppo_buf["logp"].append(logp)
            ppo_buf["rew"].append(reward)
            ppo_buf["cost"].append(cost)
            ppo_buf["done"].append(done_flag)
            if len(ppo_buf["obs"]) >= cfg["train"]["batch_size"]:
                traj = agent.build_trajectory(ppo_buf)
                metrics = agent.update(traj, guard.lambda_value)
                for k in ppo_buf:
                    ppo_buf[k].clear()
        else:
            replay.add(obs, action, reward, next_obs, done_flag, cost)
            if len(replay) >= cfg["train"]["batch_size"]:
                batch = replay.sample(cfg["train"]["batch_size"])
                if algo_name == "sac_lag":
                    metrics = agent.update(batch, guard.lambda_value)
                else:
                    b_obs, b_act, b_rew, b_next, b_done, _ = batch
                    metrics = agent.update((b_obs, b_act, b_rew, b_next, b_done))

        obs = next_obs
        last_info = info
        if terminated or truncated:
            obs, _ = env.reset()
            guard.reset_episode()

        if (step + 1) % log_every == 0 and metrics:
            print(
                {
                    "step": step + 1,
                    "lambda": round(guard.lambda_value, 3),
                    "risk": round(guard_info["risk_score"], 3),
                    "blocked": int(guard_info["shield_blocked"]),
                    **{k: round(v, 3) for k, v in metrics.items()},
                }
            )

        if (step + 1) % save_every == 0:
            agent.save(os.path.join(out_dir, f"{algo_name}_step{step+1}.pt"))
            if guard.shield is not None and guard.shield.trained:
                guard.shield.save(os.path.join(out_dir, f"shield_step{step+1}.joblib"))

    final_ckpt = os.path.join(out_dir, f"{algo_name}_final.pt")
    agent.save(final_ckpt)
    if guard.shield is not None and guard.shield.trained:
        guard.shield.save(os.path.join(out_dir, "shield_final.joblib"))

    avg_reward, avg_cost, success = _evaluate_agent(agent, env, cfg["eval"]["episodes"], guard)
    block_rate = float(np.mean([r.shield_blocked for r in safety_records])) if safety_records else 0.0
    avg_lambda = float(np.mean([r.lambda_value for r in safety_records])) if safety_records else 0.0
    avg_risk = float(np.mean([r.risk_score for r in safety_records])) if safety_records else 0.0
    summary = {
        "algo": algo_name,
        "scenario": cfg["scenario"]["type"],
        "seed": cfg["experiment"]["seed"],
        "avg_reward": avg_reward,
        "avg_cost": avg_cost,
        "success_rate": success,
        "kp": cfg["safety"]["kp"],
        "ki": cfg["safety"]["ki"],
        "kd": cfg["safety"]["kd"],
        "avg_lambda": avg_lambda,
        "avg_risk": avg_risk,
        "block_rate": block_rate,
    }
    save_summary_csv(os.path.join(out_dir, "summary.csv"), [summary])
    if safety_records:
        save_records_jsonl(os.path.join(out_dir, "safety_timeline.jsonl"), safety_records)
        save_summary_csv(os.path.join(out_dir, "safety_timeline.csv"), [asdict(r) for r in safety_records])
        _plot_safety_timeline(safety_records, os.path.join(out_dir, "safety_timeline.png"))
    return {"checkpoint": final_ckpt, **summary}


def evaluate_checkpoints_pipeline(cfg: Dict[str, Any], checkpoints_dir: str, out_plot: str = ""):
    env = make_env(cfg, seed=cfg["experiment"]["seed"])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    points = []
    for name in sorted(os.listdir(checkpoints_dir)):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(checkpoints_dir, name)
        try:
            ckpt = torch.load(path, map_location=cfg["experiment"]["device"], weights_only=False)
        except Exception:
            continue
        algo = ckpt.get("algo", "cql") if isinstance(ckpt, dict) else "cql"
        if algo not in {"cql", "iql", "sac_lag", "ppo_lag"}:
            continue
        agent = make_agent(algo, obs_dim, act_dim, cfg)
        try:
            agent.load(path, map_location=cfg["experiment"]["device"])
        except Exception:
            continue
        avg_reward, avg_cost, success = _evaluate_agent(agent, env, cfg["eval"]["episodes"])
        points.append({"checkpoint": name, "avg_reward": avg_reward, "avg_cost": avg_cost, "success_rate": success, "algo": algo, "scenario": cfg["scenario"]["type"], "seed": cfg["experiment"]["seed"]})

    save_summary_csv(os.path.join(checkpoints_dir, "summary.csv"), points)

    if out_plot and points:
        import matplotlib.pyplot as plt

        x = [p["avg_cost"] for p in points]
        y = [p["avg_reward"] for p in points]
        labels = [p["checkpoint"].replace(".pt", "") for p in points]
        os.makedirs(os.path.dirname(out_plot), exist_ok=True) if os.path.dirname(out_plot) else None
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, c="tab:blue")
        for i, lbl in enumerate(labels):
            plt.annotate(lbl, (x[i], y[i]))
        plt.xlabel("Average cost per step")
        plt.ylabel("Episode reward")
        plt.title(f"Reward-Safety Pareto ({cfg['scenario']['type']})")
        plt.tight_layout()
        plt.savefig(out_plot)

    return points


def _plot_safety_timeline(records: List[StepRecord], out_path: str):
    import matplotlib.pyplot as plt

    steps = np.array([r.step for r in records], dtype=np.int32)
    lambdas = np.array([r.lambda_value for r in records], dtype=np.float32)
    risks = np.array([r.risk_score for r in records], dtype=np.float32)
    blocked = np.array([r.shield_blocked for r in records], dtype=np.float32)
    window = min(100, max(10, len(records) // 20))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    block_rate = np.convolve(blocked, kernel, mode="same")

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(steps, lambdas, label="lambda", color="tab:red", linewidth=1.4)
    axes[0].plot(steps, risks, label="risk_score", color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(steps, block_rate, label="block_rate(ma)", color="tab:green", linewidth=1.3)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right")

    fig.suptitle("Safety Timeline")
    fig.tight_layout()
    fig.savefig(out_path)
