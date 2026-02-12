#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Generate plots and aggregate CSV from suite outputs")
    p.add_argument("--suite-dir", default="outputs/suite_reports")
    p.add_argument("--runs-root", default="outputs/runs")
    return p.parse_args()


def collect_metrics(runs_root: Path):
    records = []
    for path in runs_root.glob("**/metrics.json"):
        with path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    fig_dir = suite_dir / "figures"
    tab_dir = suite_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    records = collect_metrics(Path(args.runs_root))
    grouped = defaultdict(list)
    for r in records:
        key = (r.get("chain"), r.get("algo"), r.get("env"))
        grouped[key].append(r)

    rows = ["chain,algo,env,seeds,eval_mean,eval_std,eval_ci95"]

    for (chain, algo, env), recs in grouped.items():
        eval_means = [float(x.get("eval", {}).get("mean_reward", 0.0)) for x in recs]
        eval_std = float(np.std(eval_means)) if eval_means else 0.0
        eval_mean = float(np.mean(eval_means)) if eval_means else 0.0
        eval_ci95 = float(1.96 * eval_std / np.sqrt(max(len(eval_means), 1))) if eval_means else 0.0
        rows.append(f"{chain},{algo},{env},{len(recs)},{eval_mean:.4f},{eval_std:.4f},{eval_ci95:.4f}")

        # training curves per seed
        plt.figure(figsize=(7, 4))
        for r in recs:
            curve = r.get("train_curve", [])
            if not curve:
                continue
            xs = [p["x"] for p in curve]
            ys = [p["reward"] for p in curve]
            plt.plot(xs, ys, alpha=0.75, linewidth=1.6, label=f"seed {r.get('seed')}")
        plt.title(f"{chain} | {algo} | {env}")
        plt.xlabel("step/episode")
        plt.ylabel("reward")
        if len(recs) <= 5:
            plt.legend()
        plt.tight_layout()
        out_png = fig_dir / f"{chain}_{algo}_{str(env).replace('/', '_')}.png"
        plt.savefig(out_png, dpi=140)
        plt.close()

    aggregate_csv = tab_dir / "aggregate_metrics.csv"
    aggregate_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    summary = {
        "num_records": len(records),
        "num_groups": len(grouped),
        "figures_dir": str(fig_dir),
        "aggregate_csv": str(aggregate_csv),
    }
    with (tab_dir / "plot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
