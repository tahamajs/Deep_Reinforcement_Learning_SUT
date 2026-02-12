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
    # DQN table
    dqn = best_match(rows, "value", "dqn")
    rainbow = best_match(rows, "value", "rainbow_lite")
    dqn_tex = """\\begin{table}[h]
\\centering
\\caption{Value-based results (auto-filled).}
\\label{tab:dqnresults}
\\begin{tabular}{lccc}
\\toprule
Variant & Seeds & Mean Return & Std Across Seeds \\\\
\\midrule
DQN & {dqn_seeds} & {dqn_mean} & {dqn_std} \\\\
Rainbow-lite & {rb_seeds} & {rb_mean} & {rb_std} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""".format(
        dqn_seeds=dqn["seeds"] if dqn else "-",
        dqn_mean=fmt(dqn["eval_mean"]) if dqn else "-",
        dqn_std=fmt(dqn["eval_std"]) if dqn else "-",
        rb_seeds=rainbow["seeds"] if rainbow else "-",
        rb_mean=fmt(rainbow["eval_mean"]) if rainbow else "-",
        rb_std=fmt(rainbow["eval_std"]) if rainbow else "-",
    )
    write(out_root / "report_table_dqn.tex", dqn_tex)

    ppo = best_match(rows, "policy", "ppo")
    trpo = best_match(rows, "policy", "trpo_lite")
    reinf = best_match(rows, "policy", "reinforce")
    cpo = best_match(rows, "policy", "cpo_lite")
    policy_tex = """\\begin{table}[h]
\\centering
\\caption{Policy-gradient / trust-region results (auto-filled).}
\\label{tab:policyresults}
\\begin{tabular}{lcccc}
\\toprule
Algo & Env & Seeds & Mean Return & Std Across Seeds \\\\
\\midrule
REINFORCE & {re_env} & {re_s} & {re_m} & {re_std} \\\\
PPO & {pp_env} & {pp_s} & {pp_m} & {pp_std} \\\\
TRPO-lite & {tr_env} & {tr_s} & {tr_m} & {tr_std} \\\\
CPO-lite & {cp_env} & {cp_s} & {cp_m} & {cp_std} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""".format(
        re_env=reinf["env"] if reinf else "-",
        re_s=reinf["seeds"] if reinf else "-",
        re_m=fmt(reinf["eval_mean"]) if reinf else "-",
        re_std=fmt(reinf["eval_std"]) if reinf else "-",
        pp_env=ppo["env"] if ppo else "-",
        pp_s=ppo["seeds"] if ppo else "-",
        pp_m=fmt(ppo["eval_mean"]) if ppo else "-",
        pp_std=fmt(ppo["eval_std"]) if ppo else "-",
        tr_env=trpo["env"] if trpo else "-",
        tr_s=trpo["seeds"] if trpo else "-",
        tr_m=fmt(trpo["eval_mean"]) if trpo else "-",
        tr_std=fmt(trpo["eval_std"]) if trpo else "-",
        cp_env=cpo["env"] if cpo else "-",
        cp_s=cpo["seeds"] if cpo else "-",
        cp_m=fmt(cpo["eval_mean"]) if cpo else "-",
        cp_std=fmt(cpo["eval_std"]) if cpo else "-",
    )
    write(out_root / "report_table_policy.tex", policy_tex)

    a2c = best_match(rows, "actor_critic", "a2c")
    sac = best_match(rows, "actor_critic", "sac")
    ac_tex = """\\begin{table}[h]
\\centering
\\caption{Actor-critic / maximum-entropy results (auto-filled).}
\\label{tab:acresults}
\\begin{tabular}{lccc}
\\toprule
Algo & Seeds & Mean Return & Std Across Seeds \\\\
\\midrule
A2C & {a2c_s} & {a2c_m} & {a2c_std} \\\\
SAC & {sac_s} & {sac_m} & {sac_std} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""".format(
        a2c_s=a2c["seeds"] if a2c else "-",
        a2c_m=fmt(a2c["eval_mean"]) if a2c else "-",
        a2c_std=fmt(a2c["eval_std"]) if a2c else "-",
        sac_s=sac["seeds"] if sac else "-",
        sac_m=fmt(sac["eval_mean"]) if sac else "-",
        sac_std=fmt(sac["eval_std"]) if sac else "-",
    )
    write(out_root / "report_table_ac.tex", ac_tex)

    dyna = best_match(rows, "model_based", "dyna_q")
    mbpo = best_match(rows, "model_based", "mbpo_lite")
    model_tex = """\\begin{table}[h]
\\centering
\\caption{Model-based results (auto-filled).}
\\label{tab:modelresults}
\\begin{tabular}{lccc}
\\toprule
Algo & Environment & Seeds & Mean Return \\\\
\\midrule
Dyna-Q & {dy_env} & {dy_s} & {dy_m} \\\\
MBPO-lite & {mb_env} & {mb_s} & {mb_m} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""".format(
        dy_env=dyna["env"] if dyna else "-",
        dy_s=dyna["seeds"] if dyna else "-",
        dy_m=fmt(dyna["eval_mean"]) if dyna else "-",
        mb_env=mbpo["env"] if mbpo else "-",
        mb_s=mbpo["seeds"] if mbpo else "-",
        mb_m=fmt(mbpo["eval_mean"]) if mbpo else "-",
    )
    write(out_root / "report_table_model.tex", model_tex)

    ippo = best_match(rows, "marl", "ippo")
    qmix = best_match(rows, "marl", "qmix_lite")
    marl_tex = """\\begin{table}[h]
\\centering
\\caption{Multi-agent results (auto-filled).}
\\label{tab:marlresults}
\\begin{tabular}{lcc}
\\toprule
Algo & Seeds & Episode Reward Mean \\\\
\\midrule
IPPO & {ip_s} & {ip_m} \\\\
QMIX-lite & {qm_s} & {qm_m} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""".format(
        ip_s=ippo["seeds"] if ippo else "-",
        ip_m=fmt(ippo["eval_mean"]) if ippo else "-",
        qm_s=qmix["seeds"] if qmix else "-",
        qm_m=fmt(qmix["eval_mean"]) if qmix else "-",
    )
    write(out_root / "report_table_marl.tex", marl_tex)


def main():
    args = parse_args()
    rows = load_rows(Path(args.aggregate_csv))
    build_tables(rows, Path(args.out_root))
    print(f"Wrote tables in {Path(args.out_root).resolve()}")


if __name__ == "__main__":
    main()
