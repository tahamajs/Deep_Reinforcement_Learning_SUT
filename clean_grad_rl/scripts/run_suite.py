#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from grad_rl.core import load_yaml
from scripts.run_experiment import resolve_experiment_config, run_one


def parse_args():
    p = argparse.ArgumentParser(description="Run benchmark suite matrix")
    p.add_argument("--suite", default="configs/suites/fast_5chain.yaml")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--max-runs", type=int, default=None)
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
            try:
                cfg = resolve_experiment_config(chain, algo, config_path)
                if env is not None:
                    cfg["env"] = env
                cfg = apply_budget(cfg, task, args.mode)
                out_dir = out_root / "runs" / chain / algo / str(cfg.get("env", "unknown")).replace("/", "_") / f"seed_{seed}"
                payload = run_one(chain, algo, cfg, seed, out_dir)
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
    }
    with (suite_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
