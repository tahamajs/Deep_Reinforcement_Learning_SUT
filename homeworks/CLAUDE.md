# homeworks — Templates, Notebooks, and Solutions

**Technology**: Python notebooks (Jupyter), occasional typed packages (e.g., HW10 BootstrapDQN), LaTeX templates for submissions.
**Entry Points**: Per-homework notebooks under `code/` or `base_code/` (e.g., `HW12_Hierarchical_RL/base_code/HW12_Notebook.ipynb`), and package code under `HW10_Multi_Agent/base_code/BootstrapDQN/src/bootstrapdqn/`.
**Parent Context**: Extends [../CLAUDE.md](../CLAUDE.md) with homework-specific practices.

## Development Commands

### Notebook-first workflow
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r homeworks/HW1/code/requirements.txt
jupyter notebook homeworks/HW1/code/HW1_Notebook.ipynb
```

### Typed package example (HW10 BootstrapDQN)
```bash
cd homeworks/HW10_Multi_Agent/base_code/BootstrapDQN
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m py_compile src/bootstrapdqn/*.py
```

### Pre-PR checklist
```bash
python -m py_compile homeworks/HW10_Multi_Agent/base_code/BootstrapDQN/src/bootstrapdqn/*.py
nbstripout homeworks/HW*/**/*.ipynb || true
```

## Architecture & Patterns

### Common layout
```
HWx/
├── base_code/   # starter notebook/code
├── code/        # working copy / student solutions
├── answers/     # published answers (read-only)
├── Homework-x-Template/ (LaTeX)
└── reports/     # PDF with questions
```

### Notebook edits
- ✅ Keep outputs minimal; clear/strip when unsure.
- ✅ Mark long-running cells with comments and prefer reduced episode counts.
- ❌ Do not alter `answers/` unless explicitly preparing official solutions.
- ❌ Avoid embedding large videos/figures; link or store externally.

### Package-style homework (HW10)
- ✅ Keep types intact (`py.typed` present under `src/bootstrapdqn`).
- ✅ Use relative saves for logs/checkpoints; add to `.gitignore` if generated.
- ❌ Do not change package names or move `src/` structure.

### LaTeX templates
- ✅ Keep main files in `Homework-*-Template/main.tex`; images in `figs/`.
- ✅ Avoid committing build artifacts (`*.aux`, `*.log`, `*.pdf`) unless requested.

## Key Files & Examples
- `homeworks/HW10_Multi_Agent/base_code/BootstrapDQN/src/bootstrapdqn/base_agent.py` — agent abstractions.
- `homeworks/HW10_Multi_Agent/base_code/Task 2 Random Network Distillation/README.md` — RND instructions.
- `homeworks/HW3_Policy_Gradients/code/HW3_Notebook.ipynb` — classic policy gradient notebook.
- `homeworks/HW13_Offline_RL/base_codes/HW_13_first_part.ipynb` — offline RL starter.
- `homeworks/HW12_Hierarchical_RL/base_code/HW12_Notebook.ipynb` — hierarchy-focused notebook.

## JIT Search Hints
```bash
find homeworks -maxdepth 3 -name "*Notebook*.ipynb"
rg -n "TODO" homeworks/HW10_Multi_Agent
rg -n "def .*policy" homeworks/HW10_Multi_Agent/base_code
rg -n "\\begin{document}" homeworks/*/Homework-*-Template/main.tex
```

## Common Gotchas
- Mixed Gym APIs across notebooks; adjust `env.reset()` and `env.step()` handling carefully.
- Tensorboard logs and `.mp4` videos exist under `code/`; avoid recomputing unless necessary.
- Some archives contain zipped templates; do not unzip into version control unless needed.
- Homework folders may contain `requirements.txt` duplicates—install from the folder you are editing.

## Testing Guidelines
- Most notebooks lack automated tests; add quick assertions (shape checks, reward thresholds) in new cells if you change core logic.
- For HW10 packages, run `python -m py_compile` across `src/bootstrapdqn` and any task-specific modules.
- Avoid long training; prefer 5–10 episode smoke runs when demonstrating changes.

## Pre-PR Validation
```bash
python -m py_compile homeworks/HW10_Multi_Agent/base_code/BootstrapDQN/src/bootstrapdqn/*.py
nbstripout homeworks/HW*/**/*.ipynb || true
```

## Policies Specific to homeworks
- Keep `answers/` intact unless explicitly tasked with official updates.
- Do not rename homework folders or move templates; links in PDFs rely on stable paths.
- Keep notebook kernel metadata consistent; avoid switching kernels mid-file.
- Store any generated artifacts under ignored paths (`tmp/`, `runs/`) rather than committing.
