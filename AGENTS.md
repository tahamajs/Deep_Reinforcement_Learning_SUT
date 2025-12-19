Project Snapshot

- Multi-project research workspace (Python/PyTorch) with lecture notes, synthetic safety/flow models, and vendored research repos.
- Execution lives in notebooks; src modules are library-style and typed. External code under `external/` is vendored; hard-assignment briefs live under `assignments-hard/`.
- Each major area has its own AGENTS.md; read the nearest one before editing.

Root Setup Commands

- python -m venv .venv && source .venv/bin/activate
- python -m pip install --upgrade pip
- python -m pip install torch numpy matplotlib # add extras per package AGENT
- (optional) python -m pip install -e external/mamba # only if touching mamba code
- No global build tool; run package-specific commands from sub AGENTS.
- Tests: python -m pytest assignments-hard/31_Actor-Critic_GFlowNets/tests # only if you edit those files

Universal Conventions

- Python 3.10+, strict type hints, dataclasses, and explicit docstrings for algorithms.
- Keep execution in notebooks; modules under `src/` must stay import-safe (no side effects on import).
- Preserve long-form math/theory comments; avoid TODO/placeholder logic.
- Prefer small, isolated edits; do not mix unrelated prototypes in one commit.
- Commit messages should describe intent and scope of the touched files.

Security & Secrets

- Never commit API keys, tokens, or private datasets. No secrets belong in the repo.
- If env vars are needed, document names in code comments and keep values in a local `.env` ignored from git.
- PDFs and slide assets are static; do not embed credentials or PII.

JIT Index (what to open, not what to paste)

- Core models & utilities: `src/` → see `src/AGENTS.md`
- Hard-mode assignment briefs: `assignments-hard/` → see `assignments-hard/AGENTS.md`
- Vendored research deps: `external/` (mamba, Swin-Transformer) → see `external/AGENTS.md`
- Demo and visualization runs: `notebooks/` → see `notebooks/AGENTS.md`
- Papers/notes: `report.tex`, `README.md`, `slides/`, `stochastic_refs/` (reference only; no AGENT)
- Paper assignments: `paperAssignments/` → LaTeX reports for CA1-CA50 assignments
- Homework solutions: `homeworks/` → HW1-HW14 with solutions and archives
- Course notes: `course_notes/` → Markdown files on RL topics (bandits, exploration, etc.)
- Guest lectures: `guests/` → Notes on guest speakers
- Quizzes and summaries: `quizzes/`, `summaries/` → Assessment materials
- Workshops: `Workshops/` → Workshop materials and slides

LaTeX Setup and Conventions

- Use LaTeX for all paper assignments; compile with `pdflatex` or `xelatex` for Unicode support.
- Bibliography: Use BibTeX with `.bib` files; run `bibtex` after initial compile.
- Templates: Many assignments use IEEE or NeurIPS styles; check `report_neurips.tex` for examples.
- Figures: Generate placeholders or use scripts like `gen_placeholder_figures.py`.
- Build command: `cd paperAssignments/Assignments1-50/CAXX && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex`
- Avoid committing build artifacts (*.aux, *.log, *.pdf unless static).

Detailed Directory Structure

- `archive/`: Old solutions and no-answer archives for reference.
- `assignments-hard/`: Advanced RL assignments with tests (e.g., Actor-Critic GFlowNets).
- `course_notes/`: Comprehensive notes on RL subfields (policy-based, model-based, etc.).
- `external/`: Vendored libraries like mamba and Swin-Transformer; install with `pip install -e external/mamba`.
- `guests/`: Biographies and notes from RL experts (Sutton, Watkins, etc.).
- `homeworks/`: Homework assignments HW1-HW14, including archives and special projects.
- `notes_related/`: Additional notes and assessments.
- `Other_Assisments/`: Berkeley Deep RL and other course materials.
- `paperAssignments/`: CA1-CA50 paper assignments in LaTeX, with reports, references, and scripts.
- `QuestionsAndNotes/`: Q&A and additional notes.
- `Slides/`: Presentation slides for lectures.
- `src/`: Core Python modules (library-style, typed, import-safe).
- `stochastic_refs/`: Reference materials on stochastic processes.

Quick Find Commands (Expanded)

- Search a symbol in core code: rg -n "GuardedToolPolicy" src
- Find diffusion safety pieces: rg -n "SafetyConstrainedDiffusion" src
- Locate GFlowNet actor-critic bits: rg -n "ActorCriticGFlowNet" src/model.py src/train.py
- Spot assignment briefs: rg -n "##" assignments-hard | head
- List tests touched: rg -n "def test" assignments-hard external
- Find LaTeX reports: find paperAssignments -name "*.tex" | head -10
- Search for citations: rg -n "@" paperAssignments/Assignments1-50/*/references.bib
- Locate homework solutions: rg -n "HW" homeworks | grep -v archive
- Check for TODOs: rg -n "TODO" src course_notes
- Find Python imports: rg -n "^import|^from" src
- Locate Jupyter notebooks: find . -name "*.ipynb" | grep -v checkpoints
- Search for math equations: rg -n "\\\$" course_notes  # Inline math
- Find figure includes in LaTeX: rg -n "\\includegraphics" paperAssignments

Notebook Execution Guidelines

- All execution should happen in Jupyter notebooks under `notebooks/`.
- Use `configure_notebook` tool before running cells for the first time.
- Install packages via `notebook_install_packages` if needed.
- Avoid running notebooks in production; keep them for experimentation and demos.
- Clean metadata before committing: remove execution counts and outputs unless intentional.

Dependencies and Installation

- Core: Python 3.10+, PyTorch, NumPy, Matplotlib.
- Optional: Install vendored packages like `pip install -e external/mamba` for mamba-related work.
- For assignments: Check individual AGENTS.md in subfolders for specific deps.
- Use virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Update pip first: `python -m pip install --upgrade pip`.

Testing and Validation

- Run tests for assignments-hard: `python -m pytest assignments-hard/XX/tests`
- Validate code: Use type checkers like mypy on src/ modules.
- For LaTeX: Compile reports and check for errors.
- Notebook validation: Run cells and ensure outputs are as expected.
- Pre-commit: Ensure no syntax errors, import issues, or linting failures.

Research Focus and Topics

- Deep Reinforcement Learning (DRL) with emphasis on safety, flow models, and advanced algorithms.
- Key areas: Policy gradients, actor-critic, GFlowNets, model-based RL, exploration, multi-agent, hierarchical RL, offline RL, safe RL.
- Course notes cover bandits, exploration, imitation, inverse RL, meta-learning, etc.
- Guest lectures from RL experts like Richard Sutton, Peter Dayan, etc.

Best Practices

- Follow type hints and docstrings for all algorithms.
- Preserve mathematical comments and derivations.
- Use dataclasses for configuration and state.
- Keep commits small and focused; describe intent clearly.
- Avoid mixing unrelated changes in one commit.
- Document environment variables in comments; never commit secrets.
- For LaTeX: Use consistent citation styles; generate figures programmatically where possible.

Troubleshooting

- Import errors: Check virtual environment activation and package installation.
- LaTeX compilation fails: Ensure BibTeX is run after initial pdflatex; check for missing packages.
- Notebook kernel issues: Reconfigure notebook or restart kernel.
- Git conflicts: Pull latest changes; resolve merges carefully.
- Performance issues: Profile code; consider GPU usage for PyTorch models.

External Resources

- PyTorch docs: https://pytorch.org/docs/
- RL papers: Check stochastic_refs/ and course_notes/ for references.
- GitHub repos: Vendored code in external/ links to original sources.
- Course materials: Slides/, Workshops/ for additional context.

Contribution Guidelines

- Read relevant AGENTS.md before editing.
- Test changes locally before committing.
- Update documentation if adding new features.
- Follow universal conventions for code style.
- For new assignments: Add to appropriate folder with AGENTS.md.

Version and Changelog

- This workspace is for DRL course assignments and research.
- Updates: Regularly sync with course repo; add new assignments as completed.
- Changelog: See git log for recent changes; major updates noted in README.md.

Definition of Done

- New/changed code import-safe with type hints and docstrings aligned to nearby theory.
- Relevant package-specific checks from sub AGENTS executed (tests if you touched them).
- No edits to vendored code unless intentionally scoped and documented; notebooks remain non-executed with clean metadata.
