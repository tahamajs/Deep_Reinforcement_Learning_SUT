# CA27 — Report (Template)

## Title
Meta-Learning for Fast Adaptation: MAML and RL² Baselines

## Authors
Your Name(s)

## Abstract
Provide a concise summary (150–250 words) of the problem, proposed baselines, experimental setup, and the main result. Mention the key takeaways and reproducibility information (where to find code and configs).

## 1. Introduction
- Problem statement: Why fast adaptation / meta-learning matters.
- Research question and hypotheses.
- Short summary of contributions (e.g., "We compare MAML and RL² on CartPole variants with varying dynamics; we provide an import-safe implementation, rigorous tests, and reproducible experiments.").

## 2. Related Work
- Briefly discuss MAML (Finn et al., 2017) and RL² (Duan et al., 2016 / 2017) and other relevant baselines.

## 3. Methods
### 3.1 MAML
- Model architecture (MLP policy, hidden sizes)
- Inner/outer loop optimization details (inner lr, meta lr, inner steps)
- Any important implementation details (how trajectories / returns are computed; second-order gradients enabled)

### 3.2 RL²
- LSTM-based policy architecture
- How experience is encoded (obs + prev_action + prev_reward + done)
- Inner optimization (PPO-style updates per task)

### 3.3 Tasks and Metrics
- Describe the CartPole variants (gravity, mass, length ranges).
- Evaluation metrics: average return after K adaptation steps, sample efficiency curves, and variance across seeds.

## 4. Experimental Setup
- Hardware used (CPU/GPU), random seeds, hyperparameter ranges.
- How runs were conducted (meta iterations, batch sizes, seeds, checkpoints).
- File and plot naming conventions (saved under `results/` and `pictures/` with config & seed in filenames).

## 5. Results
- Present your main results with figures and short captions.
- Include tables summarizing average final returns and sample efficiency.

> NOTE: This repository includes a notebook `notebooks/meta_learning_experiment.ipynb` to reproduce the figures. The notebook is intentionally non-executed in the repo—execute locally with the environment specified in `requirements.txt`.

## 6. Discussion
- Interpret the results relative to your hypotheses.
- Discuss limitations (compute budget, small-scale experiments, missing baselines) and potential improvements.

## 7. Reproducibility Checklist
- [ ] Code for all algorithms included in `src/` and import-safe
- [ ] Tests included and passing (`pytest`)
- [ ] All experiment configs saved under `configs/`
- [ ] Figures saved to `pictures/` and results to `results/`
- [ ] Random seeds documented

## 8. Conclusion
Short summary and directions for follow-up work.

## References
- Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.
- Duan, Y., Schulman, J., Chen, X., Bartlett, P., Sutskever, I., & Abbeel, P. (2016). RL²: Fast Reinforcement Learning via Slow Reinforcement Learning.

---

Appendices: include additional hyperparameter tables, ablation studies, and any extended mathematical derivations.
