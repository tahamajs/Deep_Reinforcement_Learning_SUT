#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser(description="Build LaTeX report table fragments from aggregate CSV")
    p.add_argument("--aggregate-csv", default="outputs/suite_reports/tables/aggregate_metrics.csv")
    p.add_argument("--out-root", default=".")
    return p.parse_args()


def load_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "-"


def best_match(rows: List[Dict], chain: str, algo: str, env_contains: str = ""):
    candidates = [r for r in rows if r.get("chain") == chain and r.get("algo") == algo]
    if env_contains:
        candidates = [r for r in candidates if env_contains in r.get("env", "")]
    return candidates[0] if candidates else None


def write(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def build_tables(rows: List[Dict], out_root: Path):
    def get_val(row, key, default="-"):
        if not row:
            return default
        return row.get(key, default)

    # DQN table
    dqn = best_match(rows, "value", "dqn")
    rainbow = best_match(rows, "value", "rainbow_lite")
    dqn_tex = "\n".join(
        [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Value-based results (auto-filled).}",
            "\\label{tab:dqnresults}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Variant & Seeds & Mean Return & Std Across Seeds \\\\",
            "\\midrule",
            f"DQN & {get_val(dqn, 'seeds')} & {fmt(get_val(dqn, 'eval_mean', 0.0)) if dqn else '-'} & {fmt(get_val(dqn, 'eval_std', 0.0)) if dqn else '-'} \\\\",
            f"Rainbow-lite & {get_val(rainbow, 'seeds')} & {fmt(get_val(rainbow, 'eval_mean', 0.0)) if rainbow else '-'} & {fmt(get_val(rainbow, 'eval_std', 0.0)) if rainbow else '-'} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    write(out_root / "report_table_dqn.tex", dqn_tex)

    ppo_cp = best_match(rows, "policy", "ppo", "CartPole-v1")
    ppo_mc = best_match(rows, "policy", "ppo", "MountainCarContinuous-v0")
    trpo = best_match(rows, "policy", "trpo_lite")
    reinf = best_match(rows, "policy", "reinforce")
    cpo = best_match(rows, "policy", "cpo_lite")
    policy_tex = "\n".join(
        [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Policy-gradient / trust-region results (auto-filled).}",
            "\\label{tab:policyresults}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Algo & Env & Seeds & Mean Return & Std Across Seeds \\\\",
            "\\midrule",
            f"REINFORCE & {get_val(reinf, 'env')} & {get_val(reinf, 'seeds')} & {fmt(get_val(reinf, 'eval_mean', 0.0)) if reinf else '-'} & {fmt(get_val(reinf, 'eval_std', 0.0)) if reinf else '-'} \\\\",
            f"PPO & {get_val(ppo_cp, 'env')} & {get_val(ppo_cp, 'seeds')} & {fmt(get_val(ppo_cp, 'eval_mean', 0.0)) if ppo_cp else '-'} & {fmt(get_val(ppo_cp, 'eval_std', 0.0)) if ppo_cp else '-'} \\\\",
            f"PPO & {get_val(ppo_mc, 'env')} & {get_val(ppo_mc, 'seeds')} & {fmt(get_val(ppo_mc, 'eval_mean', 0.0)) if ppo_mc else '-'} & {fmt(get_val(ppo_mc, 'eval_std', 0.0)) if ppo_mc else '-'} \\\\",
            f"TRPO-lite & {get_val(trpo, 'env')} & {get_val(trpo, 'seeds')} & {fmt(get_val(trpo, 'eval_mean', 0.0)) if trpo else '-'} & {fmt(get_val(trpo, 'eval_std', 0.0)) if trpo else '-'} \\\\",
            f"CPO-lite & {get_val(cpo, 'env')} & {get_val(cpo, 'seeds')} & {fmt(get_val(cpo, 'eval_mean', 0.0)) if cpo else '-'} & {fmt(get_val(cpo, 'eval_std', 0.0)) if cpo else '-'} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    write(out_root / "report_table_policy.tex", policy_tex)

    a2c = best_match(rows, "actor_critic", "a2c")
    sac = best_match(rows, "actor_critic", "sac")
    ac_tex = "\n".join(
        [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Actor-critic / maximum-entropy results (auto-filled).}",
            "\\label{tab:acresults}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Algo & Seeds & Mean Return & Std Across Seeds \\\\",
            "\\midrule",
            f"A2C & {get_val(a2c, 'seeds')} & {fmt(get_val(a2c, 'eval_mean', 0.0)) if a2c else '-'} & {fmt(get_val(a2c, 'eval_std', 0.0)) if a2c else '-'} \\\\",
            f"SAC & {get_val(sac, 'seeds')} & {fmt(get_val(sac, 'eval_mean', 0.0)) if sac else '-'} & {fmt(get_val(sac, 'eval_std', 0.0)) if sac else '-'} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    write(out_root / "report_table_ac.tex", ac_tex)

    dyna = best_match(rows, "model_based", "dyna_q")
    mbpo = best_match(rows, "model_based", "mbpo_lite")
    model_tex = "\n".join(
        [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Model-based results (auto-filled).}",
            "\\label{tab:modelresults}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Algo & Environment & Seeds & Mean Return \\\\",
            "\\midrule",
            f"Dyna-Q & {get_val(dyna, 'env')} & {get_val(dyna, 'seeds')} & {fmt(get_val(dyna, 'eval_mean', 0.0)) if dyna else '-'} \\\\",
            f"MBPO-lite & {get_val(mbpo, 'env')} & {get_val(mbpo, 'seeds')} & {fmt(get_val(mbpo, 'eval_mean', 0.0)) if mbpo else '-'} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    write(out_root / "report_table_model.tex", model_tex)

    ippo = best_match(rows, "marl", "ippo")
    qmix = best_match(rows, "marl", "qmix_lite")
    marl_tex = "\n".join(
        [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Multi-agent results (auto-filled).}",
            "\\label{tab:marlresults}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Algo & Seeds & Episode Reward Mean \\\\",
            "\\midrule",
            f"IPPO & {get_val(ippo, 'seeds')} & {fmt(get_val(ippo, 'eval_mean', 0.0)) if ippo else '-'} \\\\",
            f"QMIX-lite & {get_val(qmix, 'seeds')} & {fmt(get_val(qmix, 'eval_mean', 0.0)) if qmix else '-'} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    write(out_root / "report_table_marl.tex", marl_tex)


def main():
    args = parse_args()
    rows = load_rows(Path(args.aggregate_csv))
    build_tables(rows, Path(args.out_root))
    print(f"Wrote tables in {Path(args.out_root).resolve()}")


if __name__ == "__main__":
    main()
