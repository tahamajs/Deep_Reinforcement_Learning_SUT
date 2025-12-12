# ---
# comments: True
# description: Phase 2 introduces mathematical foundations and advanced policy optimization methods: dynamic programming, contraction analysis, natural gradients, TRPO/SAC, and generalization bounds.
# ---

# Introduction to RL in Depth

This phase bridges theory and practice for modern deep RL. We revisit **dynamic programming** under a metric lens (contraction mappings, Lipschitz properties), then study **policy gradients** (natural gradient, TRPO, SAC) and basic **generalization and concentration bounds** relevant to RL.

## Goals
- Formalize value- and policy-iteration as contractions with guarantees.
- Derive and compare first- and second-order policy gradient updates (natural gradient, TRPO).
- Connect statistical concentration (Hoeffding, Azuma) to RL sample complexity and regret.
- Prepare for continuous-control algorithms (DDPG, SAC) and their stability conditions.

## Dynamic Programming Foundations
### Value Iteration (VI)
- Bellman optimality operator \(\mathcal{T}\):
\[
(\mathcal{T}V)(s) = \max_a \big[r(s,a) + \gamma \mathbb{E}_{s'} V(s')\big].
\]
- \(\mathcal{T}\) is a **\(\gamma\)-contraction** in \(\|\cdot\|_\infty\); iterating converges to \(V^\star\).

### Policy Iteration (PI)
- Alternates **policy evaluation** (solve \(V^{\pi}\)) and **policy improvement** (\(\pi' = \arg\max_a Q^{\pi}(s,a)\)).
- **Modified PI / GPI**: partial evaluation, guaranteeing monotonic improvement if the evaluation error is bounded.

### Contraction and Lipschitzness
- A mapping \(F\) is a contraction if \(\|F(x)-F(y)\| \le \kappa \|x-y\|\) with \(\kappa < 1\); Banach fixed-point theorem ensures uniqueness and convergence.
- **Lipschitz dynamics/rewards** help bound approximation error when using function approximators.

## Policy Gradients
### Vanilla Policy Gradient (PG)
- Objective \(J(\theta) = \mathbb{E}_{\pi_\theta}\big[\sum_t \gamma^t r_t\big]\).
- Gradient: \(\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t]\) with advantage estimator \(\hat{A}_t\).

### Natural Policy Gradient (NPG)
- Uses **Fisher information matrix** \(F\) to precondition updates:
\[
\theta_{k+1} = \theta_k + \alpha F^{-1} \nabla_\theta J.
\]
- Invariant to reparameterization; approximated via conjugate gradients or Kronecker-factored methods.

### Trust Region Policy Optimization (TRPO)
- Solves a constrained problem: maximize surrogate \(L(\theta)\) s.t. \(\text{KL}(\pi_\theta \parallel \pi_{\theta_{\text{old}}}) \le \delta\).
- Guarantees monotonic improvement under bounded KL; implemented via CG + line search.

### Soft Actor-Critic (SAC)
- Entropy-regularized objective: maximize \(\mathbb{E}[r + \alpha \mathcal{H}(\pi(\cdot|s))]\).
- Twin critics + target networks mitigate overestimation; temperature \(\alpha\) can be tuned automatically.

### DDPG (Deterministic PG)
- Deterministic policy gradient: \(\nabla_\theta J = \mathbb{E}[\nabla_a Q(s,a)\rvert_{a=\pi_\theta(s)} \nabla_\theta \pi_\theta(s)]\).
- Requires exploration noise (e.g., OU or Gaussian) and target networks for stability.

## KL and Divergence Measures
- KL divergence measures policy shift; crucial for TRPO/PPO trust regions.
- Alternative constraints: \(\chi^2\), Wasserstein; can trade tighter bounds vs. tractability.

## Concentration and Generalization
- **Hoeffding/Azuma**: bound deviations of bounded random variables; used to derive PAC and regret bounds in bandits/MDPs.
- **Concentration of measure** enables confidence intervals for UCB/optimistic methods.
- **Regret bounds**: logarithmic for stochastic bandits (UCB/TS), \(\tilde{O}(\sqrt{T})\) for adversarial settings; in RL, regret depends on horizon \(H\), state-action cardinality, and mixing assumptions.

## Notation
- States \(s \in \mathcal{S}\), actions \(a \in \mathcal{A}\); transition \(P(s'|s,a)\); reward \(r(s,a)\); discount \(\gamma\).
- Value \(V^\pi\), action-value \(Q^\pi\), advantage \(A^\pi = Q^\pi - V^\pi\).
- Policies parameterized by \(\theta\); Fisher \(F\); temperature \(\alpha\) for entropy bonuses.

## Prerequisites
- Familiarity with basic RL (Bellman equations), linear algebra, and probability inequalities (Markov, Hoeffding).
- Comfort with automatic differentiation and optimization (SGD, line search, CG).

## Suggested Reading
- Sutton & Barto (2018) Ch. 3–4 for DP, Ch. 13 for PG; Kakade (2001) on NPG; Schulman et al. (2015) TRPO; Haarnoja et al. (2018) SAC; Dann et al. (2019) on regret bounds.