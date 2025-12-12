# Other_Assisments — External & Archived Assignments

**Technology**: Python (PyTorch, Gym/Gymnasium), Mujoco requirements for CS285 sets, mixed legacy code and data artifacts.
**Entry Points**: `berkeley-deep-RL-pytorch-solutions/hw*/cs285/scripts/run_hw*.py`, legacy assignments under `Deep-RL-Assignments/`, `homework/`, `homework_fall2022/`.
**Parent Context**: Extends [../CLAUDE.md](../CLAUDE.md) with external-assignment safeguards.

## Development Commands (scoped)
```bash
# CS285 homework 1 example
cd Other_Assisments/berkeley-deep-RL-pytorch-solutions/hw1
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .  # installs cs285 package
python cs285/scripts/run_hw1_behavior_cloning.py --help

# Legacy assignment quick check
python -m pip install -r Other_Assisments/Deep-RL-Assignments/Assignment2/requirements.txt
python Other_Assisments/Deep-RL-Assignments/Assignment2/hw2-VI-PI-DQN/Q2-VI-PI/main.py --help || true
```

### Pre-PR checklist
```bash
python -m py_compile Other_Assisments/berkeley-deep-RL-pytorch-solutions/hw1/cs285/**/*.py
nbstripout Other_Assisments/**/*\.ipynb || true
```

## Architecture & Patterns

### CS285 folders (`berkeley-deep-RL-pytorch-solutions`)
```
hw*/
├── cs285/
│   ├── agents/
│   ├── policies/
│   ├── infrastructure/
│   ├── scripts/
│   └── data/ | expert_data/ | results/
├── requirements.txt
├── setup.py
└── cs285_hwX.pdf / README.md
```
- ✅ Keep data under `expert_data/` and `results/` untouched; do not commit new rollouts.
- ✅ Use `scripts/run_hw*_*.py` as entrypoints; keep paths relative.
- ❌ Do not move or rename `cs285/` modules; import paths are fixed.

### Legacy homework collections (`homework`, `homework_fall2022`, `Deep-RL-Assignments`)
- ✅ Respect existing directory names; PDFs reference them.
- ✅ Keep notebooks light; strip outputs before commit when possible.
- ❌ Avoid committing Mujoco downloads or new large binaries.

## Key Files & Touch Points
- `Other_Assisments/berkeley-deep-RL-pytorch-solutions/hw1/cs285/scripts/run_hw1_behavior_cloning.py` — behavior cloning entry.
- `Other_Assisments/berkeley-deep-RL-pytorch-solutions/hw3/cs285/scripts/run_hw3_dqn.py` — DQN driver.
- `Other_Assisments/Deep-RL-Assignments/Assignment2/hw2-VI-PI-DQN/Q2-VI-PI/main.py` — VI/PI assignment code.
- `Other_Assisments/homework_fall2022/hw4/cs285/scripts/run_hw4_mb.py` — model-based homework script.

## JIT Search Hints
```bash
rg -n "class .*Agent" Other_Assisments/berkeley-deep-RL-pytorch-solutions
rg -n "def train" Other_Assisments/Deep-RL-Assignments
find Other_Assisments -name "run_hw*.py"
find Other_Assisments -name "*.ipynb"
```

## Common Gotchas
- Mujoco dependencies may be required; do not attempt installs in automation without approval.
- Large `downloads/` and `results/` directories are present; avoid scanning or modifying them.
- Some data files use `.local` or `.DESKTOP-*` extensions—treat as artifacts, not source.
- Gym versions differ across assignments; align API handling per folder.

## Testing Guidelines
- Few automated tests exist; favor short smoke runs (reduced episodes) for changed modules.
- Use `python -m py_compile` for touched CS285 modules to ensure import safety.
- Avoid long training runs; document any lengthy commands before executing.

## Pre-PR Validation
```bash
python -m py_compile Other_Assisments/berkeley-deep-RL-pytorch-solutions/hw*/cs285/**/*.py
nbstripout Other_Assisments/**/*\.ipynb || true
```

## Policies Specific to Other_Assisments
- Keep `downloads/`, `expert_data/`, and `results/` intact; do not delete or re-run unless requested.
- Do not vendor new checkpoints or datasets into these folders.
- Limit edits to the specific homework you are addressing; avoid repo-wide search/replace across archives.
