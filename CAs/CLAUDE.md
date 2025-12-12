# CAs — Computer Assignments (Solutions & Templates)

**Technology**: Python (PyTorch, Gymnasium/Gym, NumPy, Matplotlib), notebooks for walkthroughs, lightweight pytest where provided.
**Entry Points**: Each CA folder under `CAs/Solutions/CAxx_*` with paired notebooks (e.g., `CA5.ipynb`) and scripts (`main.py`, `run.sh`).
**Parent Context**: Extends [../CLAUDE.md](../CLAUDE.md) with assignment-focused rules.

## Development Commands

### From this folder (pick the assignment you touch)
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r CAs/Solutions/CA05_Advanced_DQN_Methods/requirements.txt
python -m pytest CAs/Solutions/CA05_Advanced_DQN_Methods/test_project.py
python -m py_compile CAs/Solutions/CA07_DQN_Value_Based_Methods/agents/dqn_agent.py
# Run a scoped script
python CAs/Solutions/CA08_Causal_MultiModal_RL/main.py --help || true
```

### Quick smoke patterns (adjust paths)
```bash
bash CAs/Solutions/CA05_Advanced_DQN_Methods/run.sh           # short driver
python CAs/Solutions/CA09_Advanced_Policy_Gradients/quick_run.py
python CAs/Solutions/CA15_Model_Based_Hierarchical_RL/main.py --config configs/base.yaml  # if config exists
```

### Pre-PR checklist
```bash
python -m py_compile <touched .py files>
python -m pytest <assignment tests>  # e.g., CA07 test_implementation.py
nbstripout <touched .ipynb> || true  # optional
```

## Architecture

### Directory Shape (typical solution folder)
```
CAxx_*/
├── agents/          # algorithm implementations
├── environments/    # gym-style env wrappers
├── experiments/     # configs / experiment runners
├── evaluation/      # metrics and plotting helpers
├── models/          # neural nets
├── utils/           # shared helpers (seeding, buffers)
├── *.ipynb          # narrative notebook for the assignment
├── README.md        # assignment-specific guide
├── run.sh / main.py # entrypoints
└── tests (varies)   # pytest files when provided
```

### Code Organization Patterns

#### Agents & Training
- ✅ **DO** keep seeding and device handling explicit (see `CAs/Solutions/CA05_Advanced_DQN_Methods/agents/dqn_agent.py`).
- ✅ **DO** keep replay buffers and exploration logic isolated (`utils/replay_buffer.py`).
- ❌ **DON'T** add side effects on import—wrap script logic in `if __name__ == "__main__":`.
- ❌ **DON'T** leave unnecessary comments or commented-out code; keep implementations clean and self-explanatory.

#### Environments
- ✅ Use wrappers in `environments/` for compatibility (e.g., `gym_reset`, `gym_step`).
- ❌ Avoid hardcoding environment names—keep them configurable from CLI or constants.

#### Experiments & Results
- ✅ Save outputs under `results/` or `logs/` within the assignment; keep paths relative.
- ✅ Provide lightweight configs for quick verification runs (e.g., `experiments/base_config.py`).
- ❌ Do not commit large videos/checkpoints; add to `.gitignore` if produced.

#### Notebooks
- ✅ Keep cells short; mark heavy cells with comments.
- ✅ Mirror code changes with notebook narrative where relevant.
- ❌ Avoid committing large embedded outputs; strip when unsure.

## Key Files & Touch Points
- `CAs/Solutions/CA05_Advanced_DQN_Methods/main.py` — DQN entry script.
- `CAs/Solutions/CA07_DQN_Value_Based_Methods/test_implementation.py` — pytest target.
- `CAs/Solutions/CA08_Causal_MultiModal_RL/README_EXECUTION.md` — run guidance.
- `CAs/Solutions/CA09_Advanced_Policy_Gradients/complete_run.py` — orchestrated run.
- `CAs/Solutions/CA15_Model_Based_Hierarchical_RL/README.md` — hierarchical RL notes.
- `CAs/No Answer/*.ipynb` — templates; treat as read-mostly unless authoring blanks.

## JIT Search Hints
```bash
# Find agents and buffers
rg -n "class .*Agent" CAs/Solutions
rg -n "ReplayBuffer" CAs/Solutions
# Find env wrappers
rg -n "gym_reset|gym_step" CAs/Solutions
# Locate tests
find CAs/Solutions -name "test_*.py"
# Locate notebooks
find CAs -name "CA*.ipynb"
```

## Common Gotchas
- Gym vs Gymnasium API differences: handle `(obs, info)` returns and `terminated/truncated` flags.
- Some folders carry checked-in `venv/` or `logs/`; do not modify or rely on them.
- `run.sh` scripts may expect relative paths—run from the assignment root.
- Large PNG/PDF visualizations exist; avoid regenerating unless needed.
- Keep epsilon schedules and seed utilities consistent across agents and experiments.

## Math & Theory Alignment
- When adjusting algorithms, update the accompanying notebook/README math (losses, Bellman updates, gradients) to match code changes.
- Keep symbol names consistent between equations and implementation (learning rates, discount factors, entropy terms, KL weights).
- For novel ideas, add short derivations and note any new hyperparameters or constraints; ensure defaults in scripts mirror the formulas.

## Testing Guidelines
- Prefer pytest files shipped with the assignment (CA05 `test_project.py`, CA07 `test_implementation.py`, CA08 `test_basic.py`, CA03 `test_ca3.py`).
- For assignments without tests, run abbreviated training (reduced episodes/steps) and assert shapes/returns.
- Do not run long sweeps by default; provide a small config or `--fast` flag when adding scripts.
- Document any skipped tests with rationale in commit messages.

## Pre-PR Validation
```bash
python -m py_compile CAs/Solutions/<assignment>/**/*.py
python -m pytest CAs/Solutions/<assignment>/test*.py  # if present
nbstripout CAs/Solutions/<assignment>/*.ipynb || true
```

## Policies Specific to CAs
- Keep `No Answer/` pristine unless explicitly filling templates.
- Do not rename `CAx_files/`, `logs/`, or `visualizations/` directories.
- Respect per-assignment README instructions for dependencies and entrypoints.
- Prefer incremental changes within a single CA per commit.
