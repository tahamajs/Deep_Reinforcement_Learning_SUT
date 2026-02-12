# Deep Reinforcement Learning (DRL) Repository – Complete Guide  
*Updated: February 12, 2026 – Sharif University of Technology*

This repository is the single source for the DRL course: slides, notes, homeworks, computer assignments (CAs), paper replications, workshops, and research projects. It is organized so you can (1) study theory, (2) run reference implementations, and (3) extend them for projects or research.

---
## Table of Contents
1. Quick Start
2. Environment & Dependencies
3. Repository Map
4. How to Pick Where to Begin
5. Assignment Workflows (Notebooks vs. Scripts)
6. Data & Assets
7. Validation & Repro Tips
8. Contribution Guidelines
9. License & Attribution

---
## 1) Quick Start
```bash
# 1) Create and activate an isolated env (Python 3.8–3.11)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 2) Install dependencies for the task you want to run
pip install -r homeworks/HW3_Policy_Gradients/requirements.txt

# 3) Launch notebooks or scripts
jupyter lab                     # for .ipynb
python train.py --help          # when a script is provided
```
- **GPU**: Optional for early assignments; recommended for Atari/MuJoCo-heavy tasks. Match CUDA with your PyTorch wheel.
- **Determinism**: Many notebooks set seeds; keep CuDNN deterministic when comparing results.

---
## 2) Environment & Dependencies
- **Per-task `requirements.txt`**: Always install from the specific folder (homeworks, paperAssignments, archive solutions, etc.). Avoid a single monolithic environment.
- **Common stack**: PyTorch, Gymnasium/Classic Control, NumPy, Matplotlib, Jupyter. Some tasks require MuJoCo, Atari ROMs, or image libs—check the local README.
- **Conflicts**: If two assignments need conflicting versions, create separate virtualenvs (e.g., `.venv-hw8`, `.venv-ca28`).

---
## 3) Repository Map (Top Level)
- `archive/` – CAs CA01–CA19 with **Solutions** and answer-free notebooks in **No Answer**.
- `CA_extra/` – Supplemental notebooks s20–s28 (advanced or make-up sessions).
- `CA_extra_versions/` – Cleaned vs. pre-cleaned copies of the extra sessions.
- `course_notes/` – Topic Markdown notes (bandits, exploration, hierarchical, imitation, meta, etc.).
- `guests/` – Bios and summaries for invited lectures.
- `homeworks/` – Core homework track HW1–HW14, weekly drills, special tasks, and term archives.
- `notes_related/` – Numbered PDF lecture notes (1–19) aligned with slides.
- `Other_Assisments/` – External/alternate assignment collections (Berkeley CS285 ports, deep RL class units, Fall 2022 homeworks, DL 2022 sets, etc.).
- `paperAssignments/` – Paper-driven coding assignments (CA1–CA31 style) with prompts and requirements.
- `projects/` – Standalone project codebases (`amasa`, `grad_rl`).
- `QuestionsAndNotes/` – Session Q&A PDFs paired with slide sets.
- `quizzes/` – Quiz solution PDFs.
- `Slides/` – Lecture slide decks (1–19).
- `summaries/` – One-page lecture summaries (10–19).
- `Workshops/` – Six hands-on workshop folders with runnable notebooks.
- `LICENSE` – MIT License.

Every directory listed above now has its own README describing contents and how to run them.

---
## 4) How to Pick Where to Begin
- **Taking the course**: follow `homeworks/` in order; pair each HW with matching slide number and `course_notes/`.
- **Practicing concepts**: start with `archive/No Answer` notebooks, then check solutions under `archive/Solutions`.
- **Research replication**: use `paperAssignments/` (choose the tree `Assignments1-50/` or `Assignments1_50/` based on the path expected by your notebook).
- **Workshop sprint**: open `Workshops/README.md` and run sessions sequentially (good for quick refreshers).
- **External curricula**: browse `Other_Assisments/` and pick the set aligned with your course (e.g., `berkeley-deep-RL-pytorch-solutions/`).
- **Project work**: see `projects/` for starting points; clone a project into a fresh branch/environment.

---
## 5) Assignment Workflows
### Notebooks
1. Install the local `requirements.txt`.
2. Launch `jupyter lab` inside the assignment folder.
3. Run cells top-to-bottom; keep copies of your completed notebook separate from provided solutions.

### Python scripts
1. Install requirements.
2. Check `--help` for CLI flags (e.g., env name, seed, total steps).
3. Save outputs (plots, checkpoints) inside the assignment folder or a `runs/` subdir; avoid committing large binaries.

### Multi-env tasks
Some CAs and paper assignments need MuJoCo or Atari:
- Install MuJoCo and set `MUJOCO_PY_MUJOCO_PATH` (see per-folder README).
- For Atari, install `ale-py`/ROMs as directed.

---
## 6) Data & Assets
- Large assets are rarely bundled. If a README lists downloads (e.g., D4RL datasets, ROMs), follow those steps before running.
- Keep downloaded data under a local `data/` sibling when possible; avoid adding to git.
- Workshop 5 ships small assets in `Workshops/Workshop-5-Material/assets/`; keep paths relative.

---
## 7) Validation & Repro Tips
- **Seeds**: use provided seeds for grading; log them when experimenting.
- **Hardware notes**: if you switch between CPU/GPU, expect minor numeric drift; compare learning curves, not single-step losses.
- **Plots**: many notebooks auto-save figures; verify output directories before running on shared machines.
- **Time**: Atari and MuJoCo runs can be lengthy—consider shorter `--num-steps` for smoke tests.

---
## 8) Contribution Guidelines
- Add new material in its own folder with:
  - A concise README (scope, how to run, dependencies).
  - A `requirements.txt` (pinned if needed for grading).
  - Clear entrypoints (`main.py`, notebooks, or scripts) and sample commands.
- Use relative paths, avoid hard-coding machine-specific directories.
- Keep checkpoints and large datasets out of the repo; prefer download scripts or `.gitignore`.
- When modifying shared utilities, ensure downstream notebooks still run; note breaking changes in the relevant README.

---
## 9) License & Attribution
Licensed under MIT (see `LICENSE`). Cite the course and original paper authors when reusing code or figures. Guest materials remain the property of their presenters.

Happy learning and experimenting!  
