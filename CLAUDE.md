# Deep Reinforcement Learning Course Workspace — CLAUDE Constitution

## Project Identity

- **Type**: Educational multi-project repository (Python-heavy) containing computer assignment solutions, homework templates/solutions, and lecture resources.
- **Stack**: Python 3.8–3.11, PyTorch, Gymnasium/Gym, NumPy/SciPy, Matplotlib/Seaborn; Jupyter notebooks for walkthroughs; LaTeX for templates.
- **Architecture**: Independent assignment folders under `CAs/`, `homeworks/`, and `Other_Assisments/`; most code organized by assignment with `agents/`, `environments/`, `experiments/`, and notebooks; supporting PDFs/notes at root.
- **Authority**: This file is the root, immutable rule set. The nearest `CLAUDE.md` to your working directory augments (never weakens) these rules.
- **Execution surface**: Prefer notebooks and scoped scripts; avoid repo-wide training runs without explicit approval.

## Universal Development Rules

### Code Quality (MUST)
- **MUST** keep Python modules import-safe (no training on import; guard main blocks).
- **MUST** preserve type hints, shape checks, and seeding utilities where present (`set_seed`, `gym_reset`, etc.).
- **MUST** keep notebooks light: avoid committing heavyweight outputs; annotate long cells with comments.
- **MUST** run targeted sanity checks on touched `.py` files when feasible (e.g., `python -m py_compile <file>`).
- **MUST** document non-obvious logic with concise docstrings or comments near the code.
- **MUST NOT** commit secrets, tokens, or personal data; `.env` stays local and ignored.
- **MUST NOT** add blanket `# type: ignore`; justify narrow ignores if unavoidable.
- **MUST NOT** delete/move assignment folders, test files, or templates; keep structure stable.
- **MUST** keep random seeding deterministic in scripts and notebooks; thread seeds through configs and helper calls.
- **MUST** respect Gym/Gymnasium API differences (reset/step signatures) when editing older assignments.
- **MUST** keep code clean and self-explanatory; avoid unnecessary comments or commented-out code—prefer clear naming and structure instead.

### Best Practices (SHOULD)
- **SHOULD** keep functions under ~60 lines; extract helpers in `utils/` or `experiments/`.
- **SHOULD** co-locate quick tests/examples with the module you touch (respect assignment boundaries).
- **SHOULD** prefer pathlib/f-strings and avoid hardcoded absolute paths; keep save paths relative.
- **SHOULD** log training metrics minimally (progress bars, moving averages) and avoid noisy prints.
- **SHOULD** mirror terminology between code and notebooks/READMEs for each assignment.
- **SHOULD** gate long training or downloads behind explicit confirmation; offer small demo configs.

### Anti-Patterns (MUST NOT)
- **MUST NOT** force-push, rewrite history, or run destructive commands without confirmation.
- **MUST NOT** run repo-wide recursive jobs that sweep binary-heavy folders (`.zip`, `.mp4`, `downloads/`).
- **MUST NOT** commit large artifacts (videos, checkpoints) unless required and documented.
- **MUST NOT** edit published answers/`answers/` PDFs unless explicitly tasked.
- **MUST NOT** bypass failing tests with skips unless documented with rationale.

## Core Commands

### Environment
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Upgrade pip: `python -m pip install --upgrade pip`
- Install per assignment: `python -m pip install -r <assignment>/requirements.txt`
- Install extras for CS285 tasks: `python -m pip install -e Other_Assisments/berkeley-deep-RL-pytorch-solutions/hw1` (from within that folder)

### Development & Checks (scoped)
- Run a CA smoke test: `python -m pytest CAs/Solutions/CA05_Advanced_DQN_Methods/test_project.py`
- Import-safety check: `python -m py_compile CAs/Solutions/CA07_DQN_Value_Based_Methods/agents/dqn_agent.py`
- Notebook lint (if nbqa available): `nbqa ruff <notebook.ipynb>` (optional, warn-only)
- Strip outputs (optional): `nbstripout <notebook.ipynb>`

### Quality Gate (before commit/PR)
```bash
python -m py_compile <touched .py files>
python -m pytest <targeted tests you touched>  # e.g., CA07, CA08, or HW10 modules
```

## Project Structure

### Assignments
- **`CAs/`** → Computer assignments  
  - `Solutions/` → full Python solutions with tests/notebooks (see `CAs/CLAUDE.md`)
  - `No Answer/` → blank notebooks/templates (read-only unless authoring)

### Coursework
- **`homeworks/`** → Homework templates, code, and answers (see `homeworks/CLAUDE.md`)  
  - `HW10_Multi_Agent/base_code/BootstrapDQN/src/bootstrapdqn/` → modern typed package example  
  - `HW1`–`HW14` → notebooks and LaTeX templates in `Homework-*/`

### Additional Assignments
- **`Other_Assisments/`** → External/archived sets (see `Other_Assisments/CLAUDE.md`)  
  - `berkeley-deep-RL-pytorch-solutions/hw1...hw5` → CS285 PyTorch code  
  - `Deep-RL-Assignments/`, `homework/`, `homework_fall2022/` → legacy/homework collections

### Documentation & Resources
- **`course_notes/`, `Slides/`, `QuestionsAndNotes/`, `summaries/`, `notes_related/`, `guests/`, `quizzes/`, `recitations/`** → PDFs/Markdown reference (treat as read-mostly).
- **`README.md`** → repository overview and syllabus-style description.
- **`LICENSE`** → MIT license.

## Quick Find Commands (JIT Index)

### Code Navigation
```bash
rg -n "train_.*agent" CAs/Solutions
rg -n "ReplayBuffer" CAs/Solutions/CA05_Advanced_DQN_Methods
rg -n "set_seed" CAs/Solutions
rg -n "class .*Policy" Other_Assisments/berkeley-deep-RL-pytorch-solutions
rg -n "envs.py" homeworks/HW10_Multi_Agent
find CAs/Solutions -name "test_*.py"
find homeworks -name "*.ipynb" -maxdepth 3
```

### Dependency/Env Checks
```bash
python -m pip list | grep torch
python -m pip check
```

## Security & Secrets
- **NEVER** commit `.env`, API keys, tokens, or private datasets.
- Keep downloads and checkpoints out of git; prefer local cache paths (`./outputs`, `./tmp`), add to `.gitignore` if needed.
- Confirm before running commands that delete or rewrite data (`rm -rf`, `git clean`, DB drops).
- Redact PII from notebooks, comments, and logs.

## Git Workflow
- Branch from `main`: `feature/<topic>` or `fix/<issue>`.
- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`.
- Keep commits scoped to one assignment/folder; avoid cross-assignment churn.
- Squash on merge is fine; avoid force pushes without approval.
- Include a short note in commit body summarizing scope and tests run/skipped.

## Testing Strategy
- Prefer targeted pytest where present (e.g., `CAs/Solutions/CA07_DQN_Value_Based_Methods/test_implementation.py`).
- For script-only folders, run the associated `run.sh`/`complete_run.py` with reduced steps when possible; avoid long trainings.
- Notebook changes: open-run only the touched cells for sanity; do not commit heavy outputs.
- If no tests exist, add small deterministic checks (shape assertions, quick env rollouts).
- Document skipped checks with rationale in PR/commit description.

## Available Tools
- ✅ Read/write repo files, run Python scripts, use `rg`, `pytest`, `pip`, `nbstripout`.
- ✅ Create/activate venvs per assignment.
- ❌ Edit `answers/` or published PDFs without explicit request.
- ❌ Force push, `rm -rf` at root, or destructive git operations without approval.
- ❌ Download large datasets/checkpoints into version control.

## Specialized CLAUDE.md Files
- `CAs/CLAUDE.md` — rules for computer assignments (solutions/templates).
- `homeworks/CLAUDE.md` — notebook-centric homework guidance.
- `Other_Assisments/CLAUDE.md` — external/archived assignment policies.
- Add more scoped files if new major areas appear; nearest file wins on conflicts.

## Quick Safety Rails
- Review shell commands before execution; block dangerous patterns via hooks.
- Keep automation fail-soft: formatting/testing hooks should warn, not stop work.
- Avoid running notebooks/tests that touch GPU/long rollouts without confirmation.

## Documentation Alignment
- Keep assignment notebooks, README files, and code consistent (hyperparameters, algorithm names).
- When updating diagrams/figures, ensure corresponding code reflects the same defaults.

## Notebook Policy
- Default to unexecuted outputs unless demonstrating specific results.
- Use lightweight configs when adding new cells; mark heavy cells with comments.
- Save plots with relative paths inside the assignment folder; avoid absolute paths.

## Math & LaTeX Alignment
- Keep formulas, derivations, and code consistent: when changing logic, update the matching math in notebooks/READMEs and any LaTeX templates.
- Prefer expressing updates with explicit equations and variable definitions; reference the exact loss/objective implemented in code.
- When proposing novel ideas, add concise derivations (gradient steps, Bellman variants, KL terms, constraints) and ensure hyperparameters in code mirror the written equations.
- Use LaTeX blocks for new math in README/notebook sections; keep symbol names synchronized between text and code.

## Data Handling
- Do not commit `downloads/`, `.zip`, `.mp4`, or `.pkl` artifacts; prefer ignored temp dirs.
- For provided data (e.g., `Other_Assisments/berkeley.../expert_data/*.pkl`), keep paths intact and avoid edits.
- When adding new assets, update `.gitignore` appropriately.

## Interaction Protocol for Claude Code
- Always read the nearest `CLAUDE.md` before editing; hierarchy is authoritative.
- Use `.claude/commands/` for repeatable flows; extend as new workflows emerge.
- Hooks in `.claude/settings.json` enforce safety/formatting; keep them minimal and scoped.
- Ask before expanding scope (new tools, long trainings, dataset downloads).

## Tool Permissions & Escalation
- Default allowed: reading/writing code/docs, scoped tests, venv setup, searches.
- Requires approval: editing binary artifacts, altering `answers/` PDFs, large downloads, git history edits.
- Hard-block: committing secrets, force pushes, destructive deletes at root.
- When unsure, pause and request clarification; document any deviations in commits.

## Review & PR Checklist
- One logical scope per PR/commit (e.g., a single CA or homework).
- Confirm formatting/py_compile on touched `.py` files.
- Run targeted tests or small rollouts; log what you ran/skipped.
- Verify README/notebook instructions still match code changes.
- Scan diffs for secrets, absolute paths, or large artifacts.

## Lineage & Hierarchy Reminder
- This root file is authoritative; subdirectory `CLAUDE.md` files add context.
- On conflicts, the closest `CLAUDE.md` overrides higher-level guidance except for safety/secret prohibitions.
