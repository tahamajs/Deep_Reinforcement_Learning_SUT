# paperAssignments — Math-First LaTeX & Code Alignment

**Purpose**: Authoring and maintaining paper drafts and math notes aligned with implemented algorithms.
**Parent Context**: Extends [../CLAUDE.md](../CLAUDE.md) with paper- and formula-specific rules.

## Core Expectations (MUST)
- **MUST** keep LaTeX equations in `paperAssignments/` consistent with the implemented code (losses, gradients, constraints, schedules).
- **MUST** define symbols explicitly (domains, shapes, norms) and keep names identical to code variables and hyperparameters.
- **MUST** update derivations when code changes (e.g., Bellman variants, KL terms, entropy bonuses, regularizers, safety constraints).
- **MUST** annotate any approximations/assumptions used in code (e.g., truncation, clipping, baselines, target networks).
- **MUST NOT** leave placeholder math; provide full equations for objectives and updates.

## Recommended Workflow
```bash
# Preview LaTeX locally (adjust command to your toolchain)
pdflatex main.tex || true
# Quick search for symbols to keep aligned
rg -n "\\(lambda|beta|alpha|gamma\\)" paperAssignments
```
- When adding new methods, first write the objective and gradients in LaTeX, then mirror them in code/notebooks.
- Keep figures/tables references stable; if code changes metrics or ablations, update captions and text accordingly.

## Structure & Consistency
- Use consistent notation across sections (e.g., \(\gamma\) for discount, \(\alpha\) for learning rate, \(\beta\) for entropy/temperature, \(\lambda\) for GAE or regularizers).
- Specify update rules (gradient steps, target networks, replay weighting) and link to the exact functions/modules implementing them.
- Document evaluation metrics and experimental setups; ensure defaults match the code/configs referenced elsewhere.
- For novel ideas, include a short derivation and note any constraints (e.g., Lipschitz bounds, safety margins) and how they are enforced in code.

## Integration with Notebooks and Assignments
- If a paper section describes an algorithm used in `CAs/` or `homeworks/`, update the corresponding README/notebook math blocks to match the LaTeX formulas.
- Keep references to data sources and environments aligned with the actual scripts/configs.
- Avoid embedding large binaries; store figures under versioned vector formats where possible and keep generation scripts in notebooks.

## Quality Checks Before Commit
- Cross-check equations against the latest code defaults and hyperparameters.
- Ensure tables/figures cite the correct metrics and experiment settings.
- Run a LaTeX build locally if feasible; otherwise, at least validate syntax for new math environments.
