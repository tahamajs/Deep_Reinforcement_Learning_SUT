# ---
# comments: True
# description: Inverse reinforcement learning (IRL) fundamentals, from apprenticeship learning and max-margin IRL to maximum-entropy and adversarial (AIRL) approaches, with guidance on identifiability and practical tips.
# ---

# Inverse RL

Inverse reinforcement learning (IRL) infers a reward function that explains expert behavior. Once a reward is learned, a policy can be optimized via standard RL, enabling transfer to new dynamics or constraints.

## Problem Statement
Given expert trajectories \(\tau = (s_0, a_0, \ldots, s_T)\) from an unknown reward \(r^\star\), find \(r\) such that the optimal policy under \(r\) matches the expert occupancy distribution. IRL is **under-determined**: many rewards can induce the same behavior; regularization and priors are essential.

## Classic Approaches
### Apprenticeship Learning (AL)
- Optimize policy to match expert feature expectations \(\mathbb{E}_{\pi}[\phi(s,a)] \approx \mathbb{E}_{\pi_E}[\phi(s,a)]\).
- Use feature-based linear reward \(r(s,a) = w^\top \phi(s,a)\).
- Maximize margin between expert and learner features via projected subgradient methods.

### Max-Margin IRL
- Formulate a large-margin optimization to find weights \(w\) that separate expert trajectories from alternatives.
- Encourages rewards that make expert trajectories uniquely optimal.

### Maximum Entropy IRL (MaxEnt IRL)
- Choose the reward that maximizes the likelihood of expert trajectories under a **maximum-entropy** distribution:
\[
p(\tau \mid r) = \frac{1}{Z} \exp\Big(\sum_t r(s_t,a_t)\Big).
\]
- Avoids arbitrary assumptions about unobserved behavior; naturally handles stochastic experts.

## Deep and Adversarial IRL
### Deep MaxEnt IRL
- Replace linear rewards with neural networks; train with policy gradients or soft value iteration to estimate partition functions.
- Requires care to stabilize gradients through the soft value backup.

### Adversarial IRL (AIRL)
- Decomposes discriminator \(D\) into reward and shaping terms:
\[
D(s,a,s') = \frac{\exp(f_\theta(s,a,s'))}{\exp(f_\theta(s,a,s')) + \pi(a\mid s)},
\]
with \(f\) parameterized as \(r_\theta(s,a) + \gamma V_\psi(s') - V_\psi(s)\).
- Yields **reward functions that are transferable** across dynamics when shaping is isolated.

### GAIL vs. IRL
- GAIL imitates occupancy measures directly; AIRL recovers a reward up to shaping.
- Choose GAIL for pure imitation; choose AIRL/IRL when downstream optimization or transfer is needed.

## Identifiability and Shaping
- Rewards are defined up to **potential-based shaping**: \(r'(s,a,s') = r(s,a,s') + \gamma \Phi(s') - \Phi(s)\).
- To improve identifiability:
  - Constrain reward class (e.g., state-only, bounded norm).
  - Add sparsity or smoothness priors.
  - Penalize shaping via AIRL-style decomposition.

## Practical Tips
- Start with MaxEnt IRL for discrete/short-horizon tasks; move to AIRL for continuous control or transfer.
- Use **feature normalization** and **gradient clipping** to stabilize reward learning.
- Regularize rewards (L2 or spectral norm) to avoid degenerate, high-magnitude solutions.
- Validate learned rewards by re-optimizing a policy and comparing performance to experts under perturbations.
- When demonstrations are few, augment with behavior cloning to warm-start policy optimization and reduce RL variance.

## References
- Ng & Russell (2000) IRL; Abbeel & Ng (2004) Apprenticeship Learning; Ratliff et al. (2006) Max-Margin IRL; Ziebart et al. (2008) MaxEnt IRL; Finn et al. (2016) Guided Cost Learning; Fu et al. (2018) AIRL.
