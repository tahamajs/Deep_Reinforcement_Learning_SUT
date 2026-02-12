"""Unified runner for offline/online training, benchmark matrix, PID sweep, and report bundle."""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import csv
from pathlib import Path

from projects.amasa_clean.scripts.common import add_common_config_flags, resolve_config, project_root
from projects.amasa_clean.scripts.pipelines import train_offline_pipeline, train_online_pipeline, evaluate_checkpoints_pipeline
from projects.amasa_clean.amasa.bench import build_benchmark_jobs, build_pid_sweep_jobs, aggregate_results


def _save_rows(path: str, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _evaluate_thresholds(rows):
    import yaml

    th_path = project_root() / "configs" / "thresholds.yaml"
    thresholds = yaml.safe_load(open(th_path, "r", encoding="utf-8"))["thresholds"]
    checks = []
    for row in rows:
        scenario = row.get("scenario")
        if scenario in {"nominal", "perturbed", "adversarial"} and "avg_reward" in row and "avg_cost" in row:
            r = float(row["avg_reward"])
            c = float(row["avg_cost"])
            tier = "below_min"
            for name in ["minimum", "target", "stretch"]:
                if r > thresholds[scenario][name]["reward"] and c < thresholds[scenario][name]["cost"]:
                    tier = name
            checks.append({"algo": row.get("algo", ""), "scenario": scenario, "seed": row.get("seed", 0), "tier": tier, "avg_reward": r, "avg_cost": c})
        if row.get("algo") in {"cql", "iql"} and "reward_ratio_vs_random" in row:
            ratio_ok = float(row["reward_ratio_vs_random"]) >= float(thresholds["offline"]["random_policy_multiplier"])
            finite_ok = int(row.get("q_finite", 0)) == int(thresholds["offline"]["q_finite"])
            checks.append({"algo": row.get("algo", ""), "scenario": row.get("scenario", ""), "seed": row.get("seed", 0), "tier": "offline_gate_pass" if (ratio_ok and finite_ok) else "offline_gate_fail", "avg_reward": row.get("avg_reward", 0.0), "avg_cost": row.get("avg_cost", 0.0)})
    return checks


def _job_out_dir(base_out: str, name: str):
    out = os.path.join(base_out, name)
    os.makedirs(out, exist_ok=True)
    return out


def _run_single(cfg, mode: str, dataset: str, checkpoint: str, out_root: str):
    name = cfg["experiment"].get("name", f"{cfg['algo']['name']}_{cfg['scenario']['type']}_s{cfg['experiment']['seed']}")
    run_dir = _job_out_dir(out_root, name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if mode == "offline_train":
        return train_offline_pipeline(cfg, dataset_path=dataset, out_dir=ckpt_dir)
    if mode == "online_train":
        return train_online_pipeline(cfg, out_dir=ckpt_dir, checkpoint_path=checkpoint)
    raise ValueError(f"Unknown single-run mode '{mode}'")


def _run_benchmark(cfg, args):
    jobs = build_benchmark_jobs(cfg)
    all_rows = []
    for idx, job in enumerate(jobs, start=1):
        # Keep storage bounded during large matrix runs: save only final checkpoint.
        job["train"]["save_every"] = int(job["train"]["steps"]) + 1
        job_name = f"job{idx}_{job['algo']['name']}_{job['scenario']['type']}_seed{job['experiment']['seed']}"
        job["experiment"]["name"] = job_name
        run_dir = os.path.join(args.out_dir, job_name, "checkpoints")
        summary_path = os.path.join(run_dir, "summary.csv")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    all_rows.append(rows[-1])
                    print(f"skip {job_name} (summary exists)")
                    continue
            except Exception:
                pass
        result = _run_single(
            job,
            mode=job["experiment"]["mode"],
            dataset=args.dataset,
            checkpoint=args.checkpoint,
            out_root=args.out_dir,
        )
        all_rows.append(result)

    _save_rows(os.path.join(args.out_dir, "benchmark_summary.csv"), all_rows)
    checks = _evaluate_thresholds(all_rows)
    _save_rows(os.path.join(args.out_dir, "benchmark_thresholds.csv"), checks)
    aggregate_results(args.out_dir, os.path.join(args.out_dir, "plots"))
    return all_rows


def _run_pid_sweep(cfg, args):
    import yaml

    sweep_file = project_root() / "configs" / "sweep" / "pid_grid.yaml"
    sweep = yaml.safe_load(open(sweep_file, "r", encoding="utf-8"))
    kp_vals = sweep["sweep"]["kp"]
    kd_vals = sweep["sweep"]["kd"]
    ki = sweep["sweep"]["ki"]

    jobs = build_pid_sweep_jobs(cfg, kp_vals, kd_vals, ki=ki)
    rows = []
    for job in jobs:
        # Keep storage bounded during grid sweeps: save only final checkpoint.
        job["train"]["save_every"] = int(job["train"]["steps"]) + 1
        mode = "online_train"
        run_name = job["experiment"]["name"]
        summary_path = os.path.join(args.out_dir, run_name, "checkpoints", "summary.csv")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    prev_rows = list(csv.DictReader(f))
                if prev_rows:
                    rows.append(prev_rows[-1])
                    print(f"skip {run_name} (summary exists)")
                    continue
            except Exception:
                pass
        result = _run_single(job, mode=mode, dataset=args.dataset, checkpoint=args.checkpoint, out_root=args.out_dir)
        rows.append(result)

    _save_rows(os.path.join(args.out_dir, "pid_sweep_summary.csv"), rows)
    checks = _evaluate_thresholds(rows)
    _save_rows(os.path.join(args.out_dir, "pid_sweep_thresholds.csv"), checks)
    aggregate_results(args.out_dir, os.path.join(args.out_dir, "plots"))
    return rows


def _run_report_bundle():
    root = project_root()
    eng = root / "report.tex"
    fa = root / "report_fa.tex"
    # Build English report with pdflatex.
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", str(eng)], cwd=root, check=True)
    except Exception as exc:
        print(f"Warn: failed to build English report: {exc}")
    # Build Persian report with xelatex if available.
    try:
        subprocess.run(["xelatex", "-interaction=nonstopmode", str(fa)], cwd=root, check=True)
    except Exception as exc:
        print(f"Warn: failed to build Persian report: {exc}")


def main(args):
    cfg = resolve_config(args)
    if args.mode == "offline_train":
        res = _run_single(cfg, "offline_train", args.dataset, args.checkpoint, args.out_dir)
        print(res)
    elif args.mode == "online_train":
        res = _run_single(cfg, "online_train", args.dataset, args.checkpoint, args.out_dir)
        print(res)
    elif args.mode == "benchmark":
        rows = _run_benchmark(cfg, args)
        print(f"Completed {len(rows)} benchmark jobs")
    elif args.mode == "pid_sweep":
        rows = _run_pid_sweep(cfg, args)
        print(f"Completed {len(rows)} PID sweep jobs")
    elif args.mode == "report_bundle":
        _run_report_bundle()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_config_flags(parser)
    parser.add_argument("--mode", type=str, required=True, choices=["offline_train", "online_train", "benchmark", "pid_sweep", "report_bundle"])
    parser.add_argument("--dataset", type=str, default="data/amasa_offline.npz")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="projects/amasa_clean/results")
    args = parser.parse_args()
    main(args)
