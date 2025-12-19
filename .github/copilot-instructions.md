# GitHub Copilot Instructions for DRL Workspace

This file contains custom instructions for GitHub Copilot to follow when working in this Deep Reinforcement Learning (DRL) research workspace.

## Project Overview
- Multi-project research workspace (Python/PyTorch) with lecture notes, synthetic safety/flow models, and vendored research repos.
- Execution lives in notebooks; src modules are library-style and typed. External code under `external/` is vendored; hard-assignment briefs live under `assignments-hard/`.
- Each major area has its own AGENTS.md; read the nearest one before editing.

## Setup Commands
- python -m venv .venv && source .venv/bin/activate
- python -m pip install --upgrade pip
- python -m pip install torch numpy matplotlib # add extras per package AGENT
- (optional) python -m pip install -e external/mamba # only if touching mamba code
- No global build tool; run package-specific commands from sub AGENTS.
- Tests: python -m pytest assignments-hard/31_Actor-Critic_GFlowNets/tests # only if you edit those files

## Universal Conventions
- Python 3.10+, strict type hints, dataclasses, and explicit docstrings for algorithms.
- Keep execution in notebooks; modules under `src/` must stay import-safe (no side effects on import).
- Preserve long-form math/theory comments; avoid TODO/placeholder logic.
- Prefer small, isolated edits; do not mix unrelated prototypes in one commit.
- Commit messages should describe intent and scope of the touched files.

## Security & Secrets
- Never commit API keys, tokens, or private datasets. No secrets belong in the repo.
- If env vars are needed, document names in code comments and keep values in a local `.env` ignored from git.
- PDFs and slide assets are static; do not embed credentials or PII.

## Directory Structure
- `src/`: Core Python modules (library-style, typed, import-safe).
- `assignments-hard/`: Advanced RL assignments with tests.
- `external/`: Vendored libraries like mamba and Swin-Transformer.
- `notebooks/`: Demo and visualization runs.
- `paperAssignments/`: CA1-CA50 paper assignments in LaTeX.
- `homeworks/`: HW1-HW14 with solutions.
- `course_notes/`: Markdown files on RL topics.
- `guests/`: Notes on guest speakers.
- `archive/`: Old solutions and no-answer archives.
- `Other_Assisments/`: Berkeley Deep RL and other course materials.
- `Slides/`: Presentation slides.
- `summaries/`, `quizzes/`: Assessment materials.
- `Workshops/`: Workshop materials.
- And more as detailed in AGENTS.md.

## LaTeX Guidelines
- Use LaTeX for all paper assignments; compile with `pdflatex` or `xelatex` for Unicode support.
- Bibliography: Use BibTeX with `.bib` files; run `bibtex` after initial compile.
- Templates: Many assignments use IEEE or NeurIPS styles; check `report_neurips.tex` for examples.
- Figures: Generate placeholders or use scripts like `gen_placeholder_figures.py`.
- Build command: `cd paperAssignments/Assignments1-50/CAXX && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex`
- Avoid committing build artifacts (*.aux, *.log, *.pdf unless static).

## Notebook Guidelines
- All execution in `notebooks/`.
- Configure notebook before first run.
- Install packages via `notebook_install_packages` if needed.
- Avoid running notebooks in production; keep them for experimentation and demos.
- Clean metadata before committing: remove execution counts and outputs unless intentional.

## Dependencies and Installation
- Core: Python 3.10+, PyTorch, NumPy, Matplotlib.
- Optional: Install vendored packages like `pip install -e external/mamba` for mamba-related work.
- For assignments: Check individual AGENTS.md in subfolders for specific deps.
- Use virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Update pip first: `python -m pip install --upgrade pip`.

## Testing and Validation
- Run tests for assignments-hard: `python -m pytest assignments-hard/XX/tests`
- Validate code: Use type checkers like mypy on src/ modules.
- For LaTeX: Compile reports and check for errors.
- Notebook validation: Run cells and ensure outputs are as expected.
- Pre-commit: Ensure no syntax errors, import issues, or linting failures.

## Research Focus and Topics
- Deep Reinforcement Learning (DRL) with emphasis on safety, flow models, and advanced algorithms.
- Key areas: Policy gradients, actor-critic, GFlowNets, model-based RL, exploration, multi-agent, hierarchical RL, offline RL, safe RL.
- Course notes cover bandits, exploration, imitation, inverse RL, meta-learning, etc.
- Guest lectures from RL experts like Richard Sutton, Peter Dayan, etc.

## Best Practices
- Follow type hints and docstrings for all algorithms.
- Preserve mathematical comments and derivations.
- Use dataclasses for configuration and state.
- Keep commits small and focused; describe intent clearly.
- Avoid mixing unrelated changes in one commit.
- Document environment variables in comments; never commit secrets.
- For LaTeX: Use consistent citation styles; generate figures programmatically where possible.

## Troubleshooting
- Import errors: Check virtual environment activation and package installation.
- LaTeX compilation fails: Ensure BibTeX is run after initial pdflatex; check for missing packages.
- Notebook kernel issues: Reconfigure notebook or restart kernel.
- Git conflicts: Pull latest changes; resolve merges carefully.
- Performance issues: Profile code; consider GPU usage for PyTorch models.

## External Resources
- PyTorch docs: https://pytorch.org/docs/
- RL papers: Check stochastic_refs/ and course_notes/ for references.
- GitHub repos: Vendored code in external/ links to original sources.
- Course materials: Slides/, Workshops/ for additional context.

## Contribution Guidelines
- Read relevant AGENTS.md before editing.
- Test changes locally before committing.
- Update documentation if adding new features.
- Follow universal conventions for code style.
- For new assignments: Add to appropriate folder with AGENTS.md.

## Version and Changelog
- This workspace is for DRL course assignments and research.
- Updates: Regularly sync with course repo; add new assignments as completed.
- Changelog: See git log for recent changes; major updates noted in README.md.

## Definition of Done
- New/changed code import-safe with type hints and docstrings aligned to nearby theory.
- Relevant package-specific checks from sub AGENTS executed (tests if you touched them).
- No edits to vendored code unless intentionally scoped and documented; notebooks remain non-executed with clean metadata.

## Quick Find Commands (Expanded)
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

## Research Methodology Guidelines
- Follow rigorous experimental design: Use proper train/val/test splits, multiple random seeds, and statistical significance testing.
- Implement ablation studies to isolate the impact of each component.
- Log all hyperparameters, random seeds, and environment versions for reproducibility.
- Use Weights & Biases (wandb) or TensorBoard for experiment tracking.
- Document research questions, hypotheses, and expected outcomes in code comments.

## DRL Algorithm Implementation Tips
- Use stable baselines for baseline implementations, but implement custom algorithms from scratch for research.
- Implement actor-critic methods with proper entropy regularization and value function clipping.
- For off-policy methods, use replay buffers with prioritization and n-step returns.
- Ensure exploration strategies (epsilon-greedy, Boltzmann, etc.) are configurable.
- Implement gradient clipping and learning rate scheduling for stable training.

## Experiment Design and Logging
- Structure experiments with clear config files (dataclasses) for hyperparameters.
- Log metrics at multiple granularities: episode-level, step-level, and aggregate.
- Implement early stopping based on validation performance.
- Save model checkpoints and training curves for analysis.
- Use sacred or hydra for experiment configuration management.

## Paper Writing and Citation Practices
- Cite original papers for algorithms and datasets used.
- Include mathematical derivations in comments and docstrings.
- Use consistent notation across code and papers.
- Generate figures programmatically from logged data.
- Maintain a BibTeX file with all relevant references.

## Code Review and Peer Review Standards
- Ensure all functions have comprehensive docstrings with math notation.
- Validate that implementations match paper equations.
- Check for numerical stability (avoid NaNs, infs).
- Verify reproducibility with fixed seeds.
- Test edge cases and failure modes.

## Data Handling and Ethics
- Never commit sensitive or proprietary datasets.
- Use synthetic data for demos and testing.
- Implement data augmentation and preprocessing pipelines.
- Ensure fair representation in datasets.
- Document data sources and preprocessing steps.

## Reproducibility Practices
- Pin dependency versions in requirements.txt or pyproject.toml.
- Use Docker containers for complex environments.
- Provide scripts for data generation and preprocessing.
- Include random seed management in all stochastic operations.
- Share trained models and evaluation scripts.

## Advanced PyTorch Tips for RL
- Use torch.jit.script for performance-critical components.
- Implement custom autograd functions for complex losses.
- Use DataParallel or DistributedDataParallel for multi-GPU training.
- Profile with torch.profiler to identify bottlenecks.
- Implement mixed precision training with torch.cuda.amp.

## Common Pitfalls in DRL Research
- Avoid reward shaping without theoretical justification.
- Don't overfit to evaluation environments.
- Implement proper exploration vs exploitation balance.
- Watch for catastrophic forgetting in continual learning.
- Validate that learned policies generalize to unseen states.

## Integration with RL Libraries
- Use Gym/Gymnasium for environment interfaces.
- Leverage Stable-Baselines3 for baseline comparisons.
- Integrate with Ray RLlib for distributed training.
- Use PettingZoo for multi-agent environments.
- Implement custom wrappers for environment modifications.

## Specific DRL Topics and Implementation
- **Bandits**: Implement epsilon-greedy, UCB, Thompson sampling for multi-armed bandits.
- **Exploration**: Use intrinsic rewards, curiosity-driven exploration, or count-based methods.
- **Imitation Learning**: Implement behavioral cloning and DAgger for learning from demonstrations.
- **Inverse RL**: Use maximum entropy IRL or apprenticeship learning.
- **Meta-Learning**: Implement MAML or Reptile for few-shot adaptation.
- **Multi-Agent RL**: Use MADDPG or QMIX for cooperative/competitive settings.
- **Hierarchical RL**: Implement options framework or feudal networks.
- **Offline RL**: Use CQL, BCQ, or TD3+BC for learning from fixed datasets.
- **Safe RL**: Implement constrained MDPs with Lagrangian methods or safety layers.

## Evaluation and Benchmarking
- Evaluate agents on standard benchmarks: OpenAI Gym, Atari, Mujoco, Procgen.
- Track metrics: Average return, sample efficiency, stability, generalization.
- Use multiple seeds for statistical significance.
- Compare against baselines: Random, human-level, state-of-the-art.

## Hyperparameter Tuning and Optimization
- Use grid search, random search, or Bayesian optimization (e.g., Optuna).
- Tune learning rates, batch sizes, network architectures.
- Implement automated hyperparameter sweeps with wandb.
- Validate tuning on held-out validation sets.

## Documentation and Experiment Tracking
- Document all experiments with clear descriptions, hypotheses, and results.
- Use structured logging with JSON/YAML configs.
- Maintain experiment notebooks with visualizations.
- Create reproducible scripts for key results.

## Ethics and Responsible RL
- Ensure fairness: Test on diverse populations, avoid biased datasets.
- Safety: Implement safety constraints, monitor for unintended behaviors.
- Transparency: Document model decisions, provide uncertainty estimates.
- Societal Impact: Consider real-world deployment implications.

## Emerging Trends and Future Directions
- Large Language Models in RL: Using LLMs for planning and reasoning.
- Multi-Modal RL: Integrating vision, language, and action.
- Foundation Models for RL: Pre-trained models adapted to RL tasks.
- Energy-Efficient RL: Optimizing for compute and energy constraints.
- Human-AI Collaboration: RL for human-in-the-loop systems.

## Collaboration and Code Sharing
- Use GitHub for version control and collaboration.
- Create clear PR descriptions with context and testing.
- Review code for correctness, style, and documentation.
- Share models and datasets responsibly (anonymize if needed).

When generating code or suggestions, adhere to these conventions to maintain consistency in the DRL research workspace.