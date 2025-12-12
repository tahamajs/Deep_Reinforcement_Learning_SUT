# ---
# comments: True
# description: Foundations of imitation learning, comparing behavior cloning, DAgger, and adversarial methods (GAIL/AIRL), plus practical guidance on data collection, covariate shift, and evaluation.
# ---

# Imitation Learning

Imitation learning (IL) trains policies from expert demonstrations instead of explicit rewards. IL is valuable when rewards are hard to specify, safety is critical, or quick bootstrapping is needed before RL fine-tuning.

## Problem Setup
- Dataset \(\mathcal{D} = \{(s_i, a_i)\}\) from an expert policy \(\pi_E\).
- Goal: learn a policy \(\pi_\theta(a \mid s)\) that matches or exceeds \(\pi_E\) in the target environment.
- Challenges: **covariate shift** (train distribution differs from states visited by \(\pi_\theta\)), limited demonstrations, and noisy experts.

## Behavior Cloning (BC)
- Supervised learning objective: \(\min_\theta \mathbb{E}_{(s,a)\sim\mathcal{D}}[-\log \pi_\theta(a \mid s)]\).
- Pros: simple, stable, strong when expert is near-deterministic and data covers test states.
- Cons: compounding error from covariate shift—small mistakes move the policy into unseen states.
- Mitigations: data augmentation, dropout/regularization, early stopping, and ensembling to reduce overfitting.

## Dataset Aggregation (DAgger)
- Iteratively roll out current policy, query expert on encountered states, and aggregate new labeled data.
- Addresses covariate shift by training on the distribution induced by \(\pi_\theta\).
- Variants reduce expert burden (e.g., probabilistic mixing, learned critics to select states to query).

## Adversarial Imitation
### Generative Adversarial Imitation Learning (GAIL)
- Train a discriminator \(D\) to distinguish expert vs. policy trajectories; train \(\pi\) to fool \(D\).
- Equivalent to RL with a learned reward \(r(s,a) = -\log(1 - D(s,a))\).
- Pros: avoids explicit reward engineering, can match occupancy measures.
- Cons: GAN instability; requires on-policy or off-policy RL inner loop.

### Adversarial IRL / AIRL
- Factorizes discriminator into reward + shaping term, enabling reward recovery transferable across dynamics.
- Useful when reward extraction (not just policy matching) is desired.

## Offline / Batch IL
- When only logged demonstrations are available, combine **BC** with **offline RL regularization**:
  - **AWR/AWAC**: advantage-weighted regression with conservative policy updates.
  - **CQL/IQL**: value-based constraints to prevent extrapolation to out-of-distribution actions.
- Hybrid pipelines: pretrain with BC, then run constrained offline RL on the same dataset.

## Learning from Imperfect Data
- **Noisy experts**: use robust losses, label smoothing, or filter trajectories by performance.
- **Partial observability**: recurrent policies (LSTM/GRU) or belief-state estimation to capture history.
- **Low expert coverage**: leverage data augmentation, goal relabeling (if goals observable), or use RL fine-tuning with shaped rewards derived from demos.

## Evaluation
- **Success/return** in the target environment.
- **Behavioral similarity**: action agreement, state visitation distribution (e.g., Wasserstein distance).
- **Robustness**: performance under perturbations, domain shifts, and varying initial states.
- **Data efficiency**: success vs. number of demonstrations and expert queries.

## Practical Tips
- Start with BC; if rollouts drift, add DAgger-style aggregation or mix BC pretraining with RL fine-tuning.
- Normalize observations and, for continuous actions, standardize action targets.
- Use demonstration quality scores if available to reweight samples.
- For high-dimensional inputs (e.g., pixels), pair an encoder with IL losses; optionally distill into a smaller policy after convergence.

## References
- Pomerleau (1989) ALVINN (early BC), Ross et al. (2011) DAgger, Ho & Ermon (2016) GAIL, Fu et al. (2018) AIRL, Nair et al. (2020) AWAC.
