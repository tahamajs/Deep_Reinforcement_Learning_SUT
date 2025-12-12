# ---
# comments: True
# description: Practical and theoretical tools for balancing exploration and exploitation in bandits and RL, covering optimism (UCB), posterior sampling (Thompson), count-based bonuses, and modern intrinsic-motivation methods such as curiosity, RND, and information gain.
# ---

# Exploration Methods

## Why Exploration Matters
Reinforcement learning agents must discover rewarding behaviors without knowing the environment a priori. This creates the classic **exploration–exploitation trade-off**: exploit the best-known actions to maximize immediate return, or explore uncertain actions to gather information that could improve long-term performance. Good exploration reduces **regret** (gap to an optimal policy) and combats issues such as local optima, sparse rewards, and non-stationarity.

### Regret and Uncertainty
- **Instantaneous regret**: \(I_t = v_\star - q(A_t)\), where \(v_\star\) is the optimal value and \(q(A_t)\) is the value of the chosen action.
- **Cumulative regret** over \(T\) steps: \(L_T = \sum_{t=1}^T I_t\). Sublinear regret \(L_T = o(T)\) implies vanishing average regret.
- Exploration strategies differ in how they **quantify uncertainty** (confidence bounds, posterior variance, state visitation counts, prediction error) and how they trade it against estimated value.

## Classical Strategies
### \(\epsilon\)-Greedy
- **Rule**: with prob. \(1-\epsilon_t\) take the greedy action; with prob. \(\epsilon_t\) sample uniformly.
- **Decay**: \(\epsilon_t = \frac{c}{t}\) or \(\frac{c \log t}{t}\) yields logarithmic regret in bandits; constant \(\epsilon\) yields linear regret.
- **Use**: cheap baseline, useful for warm-up before switching to more directed methods.

### Softmax / Boltzmann Exploration
- Sample action \(a\) with \(\Pr(a) \propto \exp(Q(a)/\tau)\).
- Temperature \(\tau\) controls stochasticity; annealing \(\tau \downarrow 0\) recovers greedy behavior.

## Optimism in the Face of Uncertainty
### Upper Confidence Bounds (UCB)
- **Idea**: choose the action with the highest plausible upper bound.
- **Bandit UCB1**: \(A_t = \arg\max_a \hat{q}_t(a) + \sqrt{\frac{2 \ln t}{N_t(a)}}\).
- Guarantees \(O(\log T)\) regret in stochastic bandits. Extensions include UCB-V (variance-aware) and KL-UCB.

### Optimistic Initialization
- Initialize value estimates to high values so that unexplored actions look appealing.
- Effective in stationary tasks; insufficient when function approximation generalizes poorly.

## Posterior (Bayesian) Approaches
### Thompson Sampling
- Maintain a posterior over action values; sample a value function from the posterior and act greedily with respect to the sample.
- Naturally balances exploration by “probability matching” and often performs strongly in practice.

## State-Dependent Exploration in RL
### Count-Based and Pseudo-Counts
- For discrete states, add an exploration bonus \(b(s) = \frac{\beta}{\sqrt{N(s)}}\) or \(b(s,a)\).
- In high-dimensional settings, use **pseudo-counts** from density models (e.g., PixelCNN, hashing) to approximate visitation rarity.
- Objective variant: maximize \(r_{\text{total}} = r_{\text{env}} + b(s,a)\).

### Intrinsic Motivation
- **Prediction error bonuses**: reward the agent when the dynamics or reward model is surprised (e.g., ICM, curiosity).
- **Random Network Distillation (RND)**: train a predictor to match a fixed random target network; prediction error is an intrinsic reward that decays as states become familiar.
- **Information gain / Bayesian surprise**: intrinsic reward proportional to reduction in posterior entropy over dynamics or rewards, encouraging states that most shrink epistemic uncertainty.
- **Entropy regularization**: directly encourage high-entropy policies (SAC style) to prevent premature convergence.

### Directed Exploration for Continuous Control
- **Noise injection**: action noise (Gaussian, Ornstein–Uhlenbeck) for local exploration; works best with replay buffers.
- **Parameter noise**: perturb policy parameters to induce more coherent action changes across time.

## Safe and Constrained Exploration
- **Safety critics / shields**: block actions that violate constraints (e.g., cost critics, reachability analysis).
- **Risk-sensitive objectives**: CVaR, worst-case, or chance-constrained formulations to bound downside during exploration.
- **Domain randomization**: expose policies to varied dynamics in simulation to avoid brittle exploitation of narrow regimes.

## Practical Guidance
- Start simple (decaying \(\epsilon\)-greedy or entropy bonuses), then add directed bonuses if rewards are sparse.
- Use **ensembles** or **Bayesian layers** to measure epistemic uncertainty and downweight rollouts in high-uncertainty regions.
- In model-based RL, limit rollout horizons (MBPO-style) or replan frequently (MPC) to curb exploitation of model errors.
- Monitor exploration with visitation heatmaps or disagreement metrics; adjust coefficients to avoid overwhelming the task reward.

## Further Reading
- Auer et al., 2002 (UCB1), Thompson 1933 (posterior sampling), Bellemare et al., 2016 (pseudo-counts), Pathak et al., 2017 (ICM), Burda et al., 2019 (RND), and Hafner et al., 2020 (entropy-regularized exploration in world models).
