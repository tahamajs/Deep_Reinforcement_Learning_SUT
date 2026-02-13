#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import sys
import signal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grad_rl.core import load_yaml
from grad_rl.envs import make_env
from scripts.run_experiment import resolve_experiment_config, run_one


def parse_args():
    p = argparse.ArgumentParser(description="Run benchmark suite matrix")
    p.add_argument("--suite", default="configs/suites/fast_5chain.yaml")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--timeout-sec", type=int, default=1800)
    return p.parse_args()


def apply_budget(cfg: Dict, task: Dict, mode: str):
    if mode == "smoke":
        if "smoke_steps" in task:
            cfg["total_steps"] = int(task["smoke_steps"])
        if "smoke_episodes" in task:
            cfg["episodes"] = int(task["smoke_episodes"])
    else:
        if "full_steps" in task:
            cfg["total_steps"] = int(task["full_steps"])
        if "full_episodes" in task:
            cfg["episodes"] = int(task["full_episodes"])
    return cfg


def preflight_task(task: Dict) -> str | None:
    chain = task.get("chain")
    env_id = task.get("env")
    try:
        if chain == "marl":
            from pettingzoo.mpe import simple_spread_v3

            env = simple_spread_v3.parallel_env(max_cycles=1, continuous_actions=False)
            env.reset(seed=0)
            env.close()
        else:
            env = make_env(env_id, seed=0)
            try:
                env.close()
            except Exception:
                pass
    except Exception as exc:
        return f"preflight failed for env '{env_id}': {exc}"
    return None


def run_one_with_timeout(chain: str, algo: str, cfg: Dict, seed: int, out_dir: Path, timeout_sec: int):
    if timeout_sec is None or timeout_sec <= 0:
        return run_one(chain, algo, cfg, seed, out_dir)
    if not hasattr(signal, "SIGALRM"):
        return run_one(chain, algo, cfg, seed, out_dir)

    def _on_alarm(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"run timed out after {timeout_sec}s")

    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_sec)
    try:
        return run_one(chain, algo, cfg, seed, out_dir)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    args = parse_args()
    suite = load_yaml(args.suite)
    seeds = list(suite.get("seeds", [0, 1, 2]))
    tasks = list(suite.get("tasks", []))

    out_root = Path(args.out_root)
    suite_out = out_root / "suite_reports"
    suite_out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "suite": args.suite,
        "mode": args.mode,
        "timestamp": int(time.time()),
        "seeds": seeds,
        "runs": [],
    }

    count = 0
    for task in tasks:
        chain = task["chain"]
        algo = task["algo"]
        env = task.get("env")
        config_path = task.get("config")
        preflight_error = preflight_task(task)

        for seed in seeds:
            if args.max_runs is not None and count >= args.max_runs:
                break
            count += 1
            run_rec = {
                "chain": chain,
                "algo": algo,
                "env": env,
                "seed": seed,
                "status": "pending",
            }
            if preflight_error is not None:
                run_rec["status"] = "skipped"
                run_rec["error"] = preflight_error
                manifest["runs"].append(run_rec)
                continue
            try:
                cfg = resolve_experiment_config(chain, algo, config_path)
                if env is not None:
                    cfg["env"] = env
                cfg = apply_budget(cfg, task, args.mode)
                out_dir = out_root / "runs" / chain / algo / str(cfg.get("env", "unknown")).replace("/", "_") / f"seed_{seed}"
                payload = run_one_with_timeout(chain, algo, cfg, seed, out_dir, args.timeout_sec)
                run_rec["status"] = "ok"
                run_rec["out_dir"] = str(out_dir)
                run_rec["eval"] = payload.get("eval", {})
            except Exception as exc:
                run_rec["status"] = "failed"
                run_rec["error"] = str(exc)
            manifest["runs"].append(run_rec)

        if args.max_runs is not None and count >= args.max_runs:
            break

    with (suite_out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "total_runs": len(manifest["runs"]),
        "ok": sum(1 for r in manifest["runs"] if r["status"] == "ok"),
        "failed": sum(1 for r in manifest["runs"] if r["status"] == "failed"),
        "skipped": sum(1 for r in manifest["runs"] if r["status"] == "skipped"),
    }
    with (suite_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
