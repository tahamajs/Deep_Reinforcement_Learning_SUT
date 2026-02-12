# archive/ – Computer Assignments CA01–CA19

This directory preserves the main computer assignments (CAs) used in the course. Every assignment has two views:
- **Solutions/**: fully worked code, reports, and figures for each CA. Folder names follow `CAxx_<topic>`.
- **No Answer/**: question-only notebooks (`CAxx.ipynb`) for practice or grading.

## How to use
1. Pick a CA number and topic from `Solutions/`.
2. Read that CA's README (inside its folder) for dependencies and run steps.
3. If you want to attempt it yourself, start from the matching notebook in `No Answer/`.
4. Compare with the solution once finished.

## Coverage snapshot
- CA01–CA06: RL fundamentals, DP/MC/TD, DQN variants, policy gradients.
- CA07–CA12: Advanced DQN, TRPO/PPO, causal & multi-modal RL, continuous control, world models, multi-agent.
- CA13–CA19: Sample efficiency, offline/safe/robust RL, hierarchical models, foundation/quantum/neurosymbolic topics.

## Environment tips
Each solution folder ships a `requirements.txt`; install it in a fresh virtualenv before running notebooks. Some assignments need extra assets (e.g., Atari ROMs, MuJoCo); follow the per-CA README for those downloads.
