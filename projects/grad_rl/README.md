# Graduate Deep RL Frameworks (5 Paper Chains)

End-to-end, runnable baselines that mirror the five paper chains described in the course brief:

1) **Value-Based** — DQN + Double + Dueling (Atari, LunarLander)
2) **Policy-Gradient / Trust-Region** — REINFORCE, PPO, TRPO, CPO-lite
3) **Actor–Critic / Max-Entropy** — A3C (async) and SAC (off-policy, entropy regularized)
4) **Model-Based** — Dyna-Q (tabular) and ME-TRPO-style model ensemble rollouts
5) **Multi-Agent** — MADDPG (CTDE, mixed) and QMIX (monotonic value factorization)

The goal is reproducible graduate-level experiments, not leaderboard scores. Each script has sane defaults and light-weight configs so you can run on a laptop or scale to a GPU server.

## Quickstart
```bash
cd /Users/tahamajs/Documents/uni/DRL
python -m venv .venv && source .venv/bin/activate
pip install -r projects/grad_rl/requirements.txt

# Value-based Atari Breakout
python projects/grad_rl/scripts/train_value.py --env ALE/Breakout-v5 --total-steps 200000

# Policy gradients: PPO on MuJoCo HalfCheetah (requires mujoco installed)
python projects/grad_rl/scripts/train_policy.py --algo ppo --env HalfCheetah-v4 --total-steps 300000

# Actor–Critic: SAC on Pendulum
python projects/grad_rl/scripts/train_actor_critic.py --algo sac --env Pendulum-v1 --total-steps 200000

# Model-based: Dyna-Q on CliffWalking
python projects/grad_rl/scripts/train_model_based.py --algo dyna-q --env CliffWalking-v0 --episodes 2000

# Multi-agent: QMIX on MPE simple_spread (Ray will start automatically)
python projects/grad_rl/scripts/train_marl.py --algo qmix --iters 50
```

Artifacts land in `projects/grad_rl/outputs/<algo>/<run_id>/` (checkpoints, TensorBoard logs, metrics.json).

## Folder layout
- `requirements.txt` — pip deps for all chains
- `configs/` — minimal YAMLs with default hyperparameters
- `scripts/` — CLI entrypoints per chain
- `grad_rl/` — shared helpers and lightweight algorithm components
- LaTeX deliverables: `report.tex` (experiment template), `solutions.tex` (theory notes), and `university_project.tex` (full university-style report with BibTeX). Tables for results live in `report_table_*.tex`; compile with `make -C projects/grad_rl` (requires `latexmk`).

## Environment notes
- Atari needs `gymnasium[atari,accept-rom-license]` and will auto-download ROMs via `AutoROM`.
- MuJoCo tasks require the Mujoco 2.3+ runtime and `pip install mujoco` if not already present.
- Multi-agent uses Ray RLlib; expect ~1–2 GB RAM per worker by default. Adjust `--num-workers`.
- Safety/CPO-lite expects envs that emit `info['cost']`; Gymnasium safety tasks or the custom Amasa env work.

## What’s implemented vs. theory
- **Exact papers**: mechanics (double/dueling, PPO clip, TRPO CG+LS, entropy bonus in SAC, monotonic mixer in QMIX, centralized critic in MADDPG) are implemented in minimal but faithful form.
- **Approx / pragmatic**: CPO is implemented as a Lagrangian-penalized PPO update (trust-region style) to keep code short; MuZero/PlaNet are not reproduced here due to size—ME-TRPO-style rollouts cover the model-based chain.

## Reproducibility checklist
- Deterministic seeds: `--seed` in all scripts
- Logging: TensorBoard + JSON metrics in outputs folder
- Evaluation: built-in eval loops with mean/σ over fixed episodes
- Configs: YAML defaults under `configs/`; override any flag via CLI

## Next steps
- Swap environments to match your assignment (ALE suite, MuJoCo humanoid, SMACv2, Safety-Gymnasium)
- Increase `--total-steps` for serious runs; defaults are laptop-friendly smoke tests
- Plug in WandB by setting `WANDB_RUN` env var; scripts will auto-log if present
