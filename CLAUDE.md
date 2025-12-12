# Safe Tool Use & Research Lab — CLAUDE Constitution

## Project Identity

- **Type**: Research monorepo (Python/PyTorch) with multiple prototypes (guarded tool use, coastal forecasting hybrid, actor-critic GFlowNets, safety diffusion) plus hard-mode assignments and lecture artifacts.
- **Stack**: Python 3.10+, PyTorch, NumPy, Matplotlib/Seaborn (viz in notebooks), LaTeX for paper drafts.
- **Architecture**: Library-style `src/` modules (import-safe), execution and visualization confined to notebooks, hard assignments in `assignments-hard/`, vendored research deps under `external/`, papers/notes in `README.md` and `report.tex`.
- **Authority**: This CLAUDE.md is the top-level immutable rule set. Subdirectory CLAUDE.md files extend these rules with narrower scope; when scopes conflict, the nearest CLAUDE.md to the working directory wins.

## Universal Development Rules

### Code Quality (MUST)

- **MUST** keep all Python modules import-safe (no side effects, no training on import).
- **MUST** preserve and add type hints, shape assertions, and masks where used.
- **MUST** align code/doc math with `README.md` and `report.tex` when touching theory-bound modules.
- **MUST** document non-trivial logic with concise docstrings; keep comments theory-focused.
- **MUST** run targeted sanity checks (e.g., `python -m py_compile src/<file>.py`) on files you edit when feasible.
- **MUST** avoid executing heavy training loops by default; notebooks are the only execution surface.
- **MUST NOT** commit secrets, API keys, credentials, or private datasets.
- **MUST NOT** bypass errors with blanket `# type: ignore` or `@ignore`; justify narrow ignores if unavoidable.
- **MUST NOT** introduce TODO/pass placeholders in core paths.
- **MUST** keep tensor device/dtype explicit (float32 activations, long ids/masks), avoid silent casts or CPU↔GPU ping-pong.
- **MUST** keep random seeding deterministic in entrypoints; thread seeds via configs.
- **MUST** preserve mask semantics on every reduction; assert shapes and value ranges where ambiguity exists.

### Best Practices (SHOULD)

- **SHOULD** keep functions under ~60 lines; extract helpers for clarity.
- **SHOULD** co-locate small tests/examples near target modules when adding them (respect assignment constraints).
- **SHOULD** keep datasets/configs explicit via dataclasses (`config.py`, per-module configs).
- **SHOULD** gate long-running commands behind explicit user approval.
- **SHOULD** prefer pure functions and deterministic seeds for reproducibility.
- **SHOULD** use pathlib over `os.path`, f-strings over concatenation, and structured logging (dict payloads) over prints.
- **SHOULD** note default device/precision in docstrings when adding new public functions.
- **SHOULD** keep math symbols consistent with papers/notes; mirror naming in code.

### Anti-Patterns (MUST NOT)

- **MUST NOT** run or auto-trigger notebook execution in automated hooks; keep notebooks clean.
- **MUST NOT** edit grading fixtures or assignment prompts outside the scoped assignment.
- **MUST NOT** force-push or rewrite history without explicit approval.
- **MUST NOT** delete/rename assignment folders or vendored external code.
- **MUST NOT** add CLI side effects to library modules (no argparse in `src/*.py`).
- **MUST NOT** catch broad `Exception` without re-raising; be specific.
- **MUST NOT** rely on global mutable state for configs or seeds.

## Core Commands

### Environment

- `python -m venv .venv && source .venv/bin/activate`
- `python -m pip install --upgrade pip`
- `python -m pip install torch numpy matplotlib` (add extras per folder AGENT)

### Lint/Format (use if available; otherwise skip politely)

- `python -m ruff check src` (or `ruff format src` for formatting)
- `python -m black src` (only if project adopts Black)

### Tests (targeted)

- `python -m pytest assignments-hard/<id>*/tests` (only when editing that assignment)
- `python -m py_compile src/<file>.py` (import safety smoke check)
- Optional doctests if you add examples: `python -m pytest --doctest-glob='*.md' <path>`

### Utilities

- `rg -n "<pattern>" <path>` for search
- `python -m pip install -e external/mamba` only if touching SSM kernels in `external/mamba`
- Strip notebook outputs when needed: `nbstripout <notebook.ipynb>` (install if available)

### Quality Gate (before PR / major change)

- Run targeted: `python -m ruff check <touched files>` (if configured)
- Run targeted: `python -m py_compile <touched src files>`
- Run targeted pytest only where you edited tests/modules.
- Note skipped checks with rationale in PR/commit description.

## Project Structure

- `src/` — core research code (multiple prototypes). See `src/CLAUDE.md`.
- `assignments-hard/` — curriculum briefs/tests. See `assignments-hard/CLAUDE.md`.
- `notebooks/` — execution + visualization only. See `notebooks/CLAUDE.md`.
- `external/` — vendored research deps; treat as read-mostly. See `external/CLAUDE.md`.
- `slides/` — lecture slides (PDFs); do not modify without approval.
- `stochastic_refs/` — reference PDFs/notes.
- `README.md` — lecture-style theory aligned with guard prototype.
- `report.tex` — full paper draft; keep consistent with code/math when edited.

## Quick Find Commands (JIT Index)

- Guarded tool policy: `rg -n "GuardedToolPolicy" src/model.py`
- Static specs/dataset: `rg -n "ToolSpec" src/data.py`
- Ocean forecasting hybrid: `rg -n "BoundaryAwareOceanModel" src/model.py`
- Flow-matching losses: `rg -n "sinkhorn_log_domain" src/losses.py`
- Actor-critic GFlowNet: `rg -n "ActorCritic" src/model.py src/train.py`
- Safety diffusion: `rg -n "SafetyConstrainedDiffusion" src/safety_model.py`
- Assignments overview: `rg -n "##" assignments-hard/MASTER_CURRICULUM_*`
- Find tests: `rg -n "def test" assignments-hard`
- Notebook list: `ls notebooks`

## Security & Secrets

- **NEVER** commit `.env`, tokens, API keys, or dataset creds. `.env` stays local and ignored.
- **Confirm** before executing destructive commands: `rm -rf`, `git push --force`, `git reset --hard`, db drops.
- **Do not** edit `report.tex`/`slides/` without explicit intent; they are publication artifacts.
- **PII**: redact in logs, comments, and outputs.
- Secrets handling: document env var names only; never commit values. If needed, add `.env.example` with placeholders.

## Git Workflow

- Branching encouraged for substantive work: `feature/<topic>`.
- Follow Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`.
- Keep commits scoped to related files; avoid mixing assignment edits with core src changes.
- Squash or rebase per team preference; never force-push without approval.
- Before commit: run targeted checks relevant to touched paths.
- For theory or loss changes, add a brief rationale in commit body describing intent/impact.

## Testing Strategy

- Unit-style checks: targeted pytest in the specific assignment folder if you modified it.
- Import-safety: `python -m py_compile src/<file>.py` for edited library modules (skip modules that intentionally reference missing optional deps).
- No blanket repo-wide test runs unless requested; this monorepo mixes unrelated prototypes.
- Prefer fast smoke tests over long training loops.
- For numerical routines, add small deterministic tolerance checks instead of long stochastic tests.
- For mask/shape edits, add assertions on shapes, dtype, and valid ranges (e.g., masks ∈ {0,1}).

## Available Tools & Permissions

- Allowed: read/write repo files, run targeted Python checks, use ripgrep, manage venvs, use `gh` for PR/issue info when configured.
- Restricted (ask first): editing `external/`, `report.tex`, `slides/`, any destructive git ops, dataset downloads/migrations, long-running training or benchmarks.
- Blocked: committing secrets, deleting assignment folders, force-push without approval.
- If unsure about a tool or command (e.g., installs, long jobs), pause and request approval.

## Hook & Automation Guidance

- PreToolUse (recommended): validate shell commands for destructive patterns; block `rm -rf` root or force pushes.
- PostToolUse: optionally auto-format Python (`ruff format` or `black`) on touched `.py` files; optionally run targeted pytest when editing `tests/` files.
- Never auto-run notebooks; strip outputs only when requested.
- Hooks should fail-soft for formatting/tests (warn instead of block) to avoid interrupting critical edits.
- Keep hook logs minimal (single-line echo) to reduce noise.

## Dangerous Patterns to Block

- `rm -rf /` or workspace root; `git push --force` or `git reset --hard` without confirmation.
- Editing `.env`, secrets, `report.tex`, `slides/` unless explicitly approved.
- Running unbounded `find`/`grep` that scan large vendored trees without purpose.
- Executing long training loops in CI/hooks.
- Redirecting output into shared assets/binaries; avoid `>` overwrites without backup.

## Directory-Specific CLAUDE.md Files

- `src/CLAUDE.md` — PyTorch library patterns, configs, masking rules.
- `assignments-hard/CLAUDE.md` — curriculum editing rules, test scope.
- `notebooks/CLAUDE.md` — execution/visualization standards, no auto-run.
- `external/CLAUDE.md` — vendored code is read-mostly; patch only with intent.

## Documentation Alignment

- Keep `README.md` theory, `report.tex` math, and `src/` implementations mutually consistent.
- When adjusting equations or losses, update both code and corresponding text.

## Notebook Policy

- Notebooks are the only place to execute training/inference.
- Do not commit executed outputs unless explicitly requested; prefer clean metadata.
- Save plots via `plt.savefig` with relative paths (e.g., `../pictures/fig_XX.png`).
- Mark heavy cells with comments (e.g., `# long_run`) and keep default params light.
- Keep kernel selection stable; avoid switching kernels mid-notebook.

## Data Handling

- Synthetic data generation lives in `src/data.py`; respect bounds/masks.
- Do not add large binaries to git; store datasets externally and document paths.
- Document dataset provenance/licensing in README/report when adding references.
- Cache small artifacts under ignored paths (e.g., `tmp/`, `.cache/`) rather than committing.

## Interaction Protocol for Claude Code

- Always read nearest CLAUDE.md before edits; rules are hierarchical.
- Use custom slash commands in `.claude/commands/` for repeatable workflows.
- Use hooks in `.claude/settings.json` to enforce safety and formatting.
- Ask for explicit permission before expanding tool scopes or running heavy commands.
- Prefer scoped search/edits to relevant directories; avoid repo-wide churn.

## Memory & Updates

- Use `#` memories during sessions to capture context; refresh this CLAUDE monthly as the project evolves.
- Keep sections modular; add new specialized CLAUDE files instead of bloating root rules.

## References

- Upstream guidance: `AGENTS.md` (root), `src/AGENTS.md`, `assignments-hard/AGENTS.md`.
- Follow their constraints in addition to this constitution.

## Modeling & Math Alignment

- When touching `src/model.py` / `src/losses.py` / `src/train.py`, cross-check with equations in `README.md` and `report.tex`; keep terminology and symbols consistent.
- Preserve mask semantics (`arg_mask`, `node_mask`, coastline masks); every tensor op must honor masks to avoid silent drift.
- Calibration/repair heads (guarded tool policy) must keep monotonicity w.r.t. static scores; do not remove static bias terms.
- For spectral/GraphCast hybrids, keep positional encodings and boundary handling intact; do not drop boundary channels or Sobel-derived cues.
- For flow-matching and OT routines, preserve log-domain stability and detach/stop-grad choices noted in code comments.

## Logging & Checkpointing

- Keep logging utilities lightweight and import-safe (no file writes on import).
- If adding checkpoints, default paths should be user-provided; never hardcode absolute paths.
- Prefer structured logging (dicts) over print statements; avoid noisy logging in library code.

## Data & Dataloaders

- Synthetic datasets live in `src/data.py` and `src/safety_data.py`; keep sampling reproducible via seeds passed through configs.
- Respect shapes: args/context typically float32; masks float/bool; tool ids long.
- Do not embed large datasets; reference external locations in docs if needed.
- Keep splits reproducible; store split seeds in configs. If adding augmentations, document effects on masks/labels.

## Notebooks (Extended)

- Add rich plotting (loss curves, heatmaps, repair histograms) but keep code import-clean; no side effects at import time.
- Save figures to `../pictures/` with DPI>=300 and descriptive filenames.
- If adding cells that would take long to run, annotate with comments and provide smaller default configs for demos.
- Prefer tight demo loops (few batches/epochs); summarize metrics rather than printing large tensors.

## Assignments Policy

- Each `assignments-hard/<id>_*` folder is self-contained; never rename or move folders/files.
- When editing a brief, append clarifications instead of rewriting prompts; keep headings intact.
- If tests exist in an assignment, run only those tests when you touched that folder.
- Do not mix changes across multiple assignments in one commit.
- Cite any new external references and match the brief’s citation style.

## External Dependencies

- `external/` holds vendored code; treat as read-only unless explicitly working on that dependency.
- If modifying vendored code, document upstream differences and keep patches minimal; never run destructive commands in `external/`.
- Do not vendor new dependencies without approval; prefer documented pip installs.

## Hook Recommendations (Claude Code)

- **PreToolUse**: block destructive shell (`rm -rf`, `git push --force`, `git reset --hard`); echo target files for Edit/Write; guard against editing `.env`, `report.tex`, `slides/` unless confirmed.
- **PostToolUse**: on `.py` edits run formatter (`ruff format` or `black`) if available; run targeted pytest when `tests/` files are touched; avoid notebook execution.
- Keep hooks fail-soft (warn, not break) for formatting/test steps to preserve developer control.
- Consider adding spell-check for markdown/papers (warn-only) when editing briefs or `report.tex`.

## Custom Slash Commands (to place under `.claude/commands/`)

- `/review` — run structured code review per repository standards.
- `/fix-issue <id>` — guide issue triage/fix with `gh issue view` + targeted search + tests.
- `/migrate-db <name>` — only if DB tooling is added; ensure migrations are safe and tested.
- Add more commands for repeatable workflows (doc sync, notebook clean) as the repo evolves.
- Future ideas: `/sync-docs` to align README/report with code diffs; `/lint-scope <path>` to run scoped ruff/py_compile.

## Tool Permissions & Escalation

- Default allowed: Read/Write files under scope, run `rg`, run import-safety checks, small pytest targets.
- Requires explicit approval: edits to `external/`, `report.tex`, `slides/`, secrets, dataset downloads, long trainings, system package installs.
- Hard-block: force pushes, repository-wide destructive deletes, committing secrets.
- Escalate to humans for ambiguous operations or when touching publication artifacts.
- If a command could mutate publication artifacts (report/slides), pause and confirm before proceeding.

## Review & PR Checklist

- Scope: one logical change set; avoid cross-prototype edits in a single PR.
- Consistency: code <-> README <-> report.tex alignment verified.
- Safety: no side effects on import; masks and shapes asserted.
- Tests: targeted pytest/py_compile executed where applicable; note if skipped with justification.
- Formatting: Python formatted per project choice (ruff/black); notebooks kept clean (no execution unless requested).
- Docs: if algorithms/losses change, update docstrings and relevant text.
- Security: scan diffs for accidental secrets/paths; ensure no new network calls on import.
- Performance: check tensor shapes for accidental O(N^2) changes; keep FFT/spectral ops bounded.
- Add a short reviewer note summarizing risks and coverage for theory-heavy changes.

## Performance & Reliability Notes

- Guarded policies: keep temperature/clipping defaults stable; document any deviation.
- Spectral/OceanFM blocks: retain dropout and mode limits to avoid OOM; check tensor device compatibility when adding ops.
- Diffusion models: keep safety constraints and penalties wired; never drop loss terms without rationale.
- GFlowNet/actor-critic: keep normalization constants and masking intact; avoid log-space underflow/overflow.

## Communication Protocol

- When unsure, ask: confirm dangerous commands, clarify formatter/linter choices, and surface assumptions.
- Record deviations from defaults in commit messages and PR descriptions.
- Prefer short reviewer notes summarizing risk areas and test coverage when opening PRs.

## Future Extensions (placeholders for evolutions)

- Add CLAUDE files for new domains (e.g., `experiments/CLAUDE.md`) instead of expanding root.
- Extend hooks to integrate with CI once a standard formatter/test suite is finalized.

## Lineage & Hierarchy Reminder

- This root file is authoritative; subdirectory CLAUDE.md files refine scope-specific behavior.
- On conflicts, closest CLAUDE.md to the working path overrides higher-level guidance, but never violates safety/secret rules.
