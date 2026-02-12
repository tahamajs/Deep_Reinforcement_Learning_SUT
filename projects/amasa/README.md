# AMASA: Autonomous Multi-Agent Surgical Assistant Challenge

Reference implementation for the "Autonomous Surgical Suturing Challenge" described in the course note. The code is organized as a teaching scaffold that walks students through four phases:

1. **Offline causal pre‑training** (pessimistic CQL with causal feature masking)
2. **Hierarchical control** (HIRO‑style goal‑conditioned low level + meta policy)
3. **Safe online fine‑tuning** (PID‑controlled Lagrange multiplier)
4. **Interpretability shield** (three‑level decision‑tree guardrail)

The simulator is a lightweight, differentiable approximation of a 7‑DOF surgical arm interacting with soft tissue. It is Gymnasium‑compatible and tuned for fast iteration on laptops.

## Quickstart

```bash
# create env
python -m venv .venv && source .venv/bin/activate
pip install -r projects/amasa/requirements.txt

# generate a small offline buffer
python -m projects.amasa.scripts.generate_dataset --episodes 50 --out data/amasa_offline.npz

# train pessimistic offline policy
python -m projects.amasa.scripts.train_offline --dataset data/amasa_offline.npz --steps 200000

# train hierarchical controller (uses offline policy to warm start low level)
python -m projects.amasa.scripts.train_hierarchical --steps 150000

# safe online finetune with shield + PID λ
python -m projects.amasa.scripts.train_safe_online --steps 100000 --checkpoint checkpoints/hiro.pt

# evaluate reward/safety Pareto frontier
python -m projects.amasa.scripts.evaluate --checkpoints checkpoints --out plots/pareto.png
```

Generated artifacts land in `checkpoints/`, `logs/`, and `plots/` by default (created on first run).

## Folder layout
- `amasa/envs/suturing_env.py` — Gymnasium surgical task (state, reward, cost)
- `amasa/offline/cql.py` — Conservative Q‑Learning for continuous actions
- `amasa/hierarchical/hiro.py` — Two‑level HIRO‑style controller
- `amasa/safety/pid_lagrangian.py` — PID dual ascent wrapper for safety constraints
- `amasa/safety/shield.py` — Decision‑tree action filter with explanations
- `scripts/` — CLI entrypoints for dataset gen, training, evaluation
- `configs/` — Hyperparameter presets (YAML)

## Key design choices (teaching hooks)
- **Causal masking**: randomizes non‑causal pixels/forces; mask learned via gradient saliency and frozen to stabilize offline CQL.
- **Pessimism**: CQL penalty α defaults to 5.0 to counter OOD overestimation; behavior cloning loss keeps policy near πβ.
- **Hierarchy**: meta‑policy horizon H=20, goal relabeling matches HIRO off‑policy correction.
- **Safety**: cost for force >5N or penetration outside target corridor; λ updated by PID tuned for damping (Kd>0).
- **Shield**: three CART trees (state risk, allowed actions, reason) trained on simulated rollouts with privileged cost labels.

## Tested setup
- Python 3.10+
- CPU‑only or CUDA (optional)
- PyTorch 2.2, Gymnasium 0.29

The implementation favors clarity over raw performance to support classroom use.
