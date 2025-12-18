# "To the Max" Reward Transformation with Sinkhorn Distributional RL (MaxSink)

## 1. Executive Summary

Sparse-reward goal-reaching remains a central challenge in reinforcement learning. "To the Max" (ICML 2024) introduces a reward transformation that reshapes sparse rewards into maximization-friendly signals, encouraging rapid attainment of goal states. Sinkhorn Distributional RL (NeurIPS 2024) leverages entropy-regularized optimal transport to learn full return distributions with stability and geometric awareness. This assignment proposes **MaxSink**, a synthesis that applies the "To the Max" reward transformation inside a Sinkhorn-based distributional RL agent. Key questions:

- Does the transformed reward reduce the need for distributional modeling, or do the two approaches compound to yield faster convergence and richer uncertainty estimates?
- How does the reward transformation alter return distributions, and can Sinkhorn divergence exploit the resulting structure?
- What are the effects on sample efficiency, risk sensitivity, and policy robustness in sparse, goal-conditioned domains (Maze, Fetch robotics)?

We provide full mathematical derivations, algorithmic pseudocode, PyTorch implementation scaffolding, hyperparameters, ablation plans, evaluation protocols, logging schema, and reproducibility guidance to deliver a 1000+ line blueprint for implementing and benchmarking MaxSink.

---

## 2. Background and Motivation

1. **Sparse rewards** hinder exploration and credit assignment.
2. **Reward shaping** often introduces bias; "To the Max" proposes a transformation designed to preserve optimality while improving learning signals.
3. **Distributional RL** models return distributions, offering richer training signals and risk-awareness; Sinkhorn DRL uses OT-based losses with entropic regularization for stability.
4. **Open question**: Does a strong reward transformation diminish the marginal value of distributional modeling, or can Sinkhorn's geometric matching further exploit the transformed reward landscape?
5. **Goal**: Empirically and theoretically assess the interaction of reward transformation and distributional learning in goal-conditioned, sparse-reward tasks.

---

## 3. "To the Max" Reward Transformation

### 3.1 Original Reward

Let $r(s,a)$ be sparse: $r=1$ on success, $0$ otherwise (or $-1$ for failure). The episodic return $G=\sum_{t=0}^{T-1}\gamma^t r_t$ is highly skewed.

### 3.2 Transformation

"To the Max" defines a transformation $T(\cdot)$ producing $r'(s,a)=T(r(s,a))$ to amplify goal-reaching signals. A typical form:
$$
r'(s,a) = \max(r(s,a), \beta \cdot \mathbf{1}[\text{progress}(s,a) > 0]),
$$
or more generally a monotone saturating function that assigns maximal reward on achievement and shaped bonuses on progress. Key properties:
- **Monotonicity**: $r_1 \le r_2 \implies T(r_1) \le T(r_2)$.
- **Optimality preservation**: under suitable constraints, optimal policies remain optimal (potential-based shaping variant).
- **Amplification**: increases gradient signal when approaching goals.

### 3.3 Return Transformation

Transformed return:
$$
G' = \sum_{t=0}^{T-1} \gamma^t r'_t.
$$
The distribution of $G'$ shifts mass toward higher values earlier in training, potentially tightening support and reducing variance.

---

## 4. Sinkhorn Distributional RL Recap

### 4.1 Distributional Value

Model $Z_\theta(s,a)$ as a distribution (sampled via particles). Target distribution $Y = r' + \gamma Z_{\theta'}(s', a^*)$ with $a^*=\arg\max_{a'} \mathbb{E}[Z_{\theta'}(s',a')]$ or policy sample.

### 4.2 Sinkhorn Divergence

Given predicted particles $X$ and target particles $Y$, Sinkhorn loss:
$$
S_\varepsilon(X,Y) = 2 W_\varepsilon(X,Y) - W_\varepsilon(X,X) - W_\varepsilon(Y,Y),
$$
where $W_\varepsilon$ is entropic-regularized OT cost with blur $\varepsilon$.

### 4.3 Advantages of Sinkhorn

- Geometric matching of distributions.
- Debiased divergence; interpolates Wasserstein and MMD.
- Stable gradients with entropic regularization.

---

## 5. MaxSink Formulation

### 5.1 Combined Target

Use transformed rewards in distributional Bellman update:
$$
Y = r'(s,a) + \gamma Z_{\theta'}(s', a^*).
$$

### 5.2 Policy Improvement

Greedy policy under expected transformed return:
$$
a^* = \arg\max_a \mathbb{E}[Z_\theta(s,a)].
$$
Optionally risk-aware: maximize CVaR or lower quantile of $Z$ for robustness.

### 5.3 Hypotheses

1. Transformation reduces distributional variance → faster Sinkhorn convergence.
2. Sinkhorn retains benefits by aligning shaped distributions, offering smoother gradients than scalar losses.
3. In very sparse settings, transformation + distributional yields multiplicative benefits (denser signal + richer loss).

---

## 6. Theoretical Considerations

### 6.1 Effect on Distribution Support

Transformation pushes mass upward; support shrinks toward maximal reward region. Sinkhorn cost depends on pairwise distances; reduced spread may lower cost and accelerate convergence.

### 6.2 Bias and Optimality

If transformation is potential-based: $r' = r + \gamma \Phi(s') - \Phi(s)$, policy optimality is preserved. Ensure chosen $T$ respects potential shaping or document induced bias.

### 6.3 Variance Reduction

Let $\sigma^2_G$ be variance of returns; after transformation, $\sigma'^2_G \le \sigma^2_G$ if transformation compresses lower tail. Lower variance improves stability of distributional targets; may reduce need for large particle counts.

### 6.4 Sinkhorn Sensitivity to Blur

Blur $\varepsilon$ controls smoothness; with transformed rewards (less noisy), smaller blur may be viable, yielding closer-to-Wasserstein gradients.

---

## 7. Algorithm Overview (MaxSink)

1. Initialize policy $\pi_\theta$, value distribution $Z_\theta$, target $Z_{\theta'}$.
2. Collect transitions $(s,a,r,s')$.
3. Apply "To the Max" transformation: $r' = T(r, s, a)$ (may depend on progress metrics).
4. Compute target particles: $Y = r' + \gamma Z_{\theta'}(s', a^*)$.
5. Compute Sinkhorn divergence $S_\varepsilon(X=Z_\theta(s,a), Y)$.
6. Update $\theta$ via gradient of Sinkhorn loss; soft update targets.
7. Policy improvement via expected (or CVaR) of $Z_\theta$.
8. Repeat; evaluate on Maze/Fetch.

---

## 8. Pseudocode (High-Level)

```
initialize πθ, Zθ, Zθ'
for each iteration:
    batch = replay.sample()
    s, a, r, s_next, done = batch
    r_max = transform_to_max(r, s, a)
    with torch.no_grad():
        a_next = argmax_expectation(Zθ', s_next)  # or policy sample
        y = r_max + gamma * (1 - done) * Zθ'(s_next, a_next)
    x = Zθ(s, a)
    loss = sinkhorn_divergence(x, y, blur, scaling)
    optimize θ on loss
    soft_update(Zθ', Zθ, tau)
    policy update: maximize E[Zθ(s,a)] (optionally CVaR)
```

---

## 9. PyTorch Skeleton (Sinkhorn + Transform)

```python
def to_the_max_reward(r, progress=None, beta=0.5):
    if progress is None:
        return r.clamp(min=0)  # simple non-negative
    bonus = (progress > 0).float() * beta
    return torch.maximum(r, bonus)

class MaxSinkAgent:
    def __init__(self, blur=0.01, scaling=0.9, tau=0.005, beta=0.5):
        self.sinkhorn = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, debias=True)
        self.beta = beta
        ...

    def update(self, batch):
        s, a, r, s_next, done, progress = batch
        r_max = to_the_max_reward(r, progress, self.beta)
        with torch.no_grad():
            a_next = self.policy.argmax_expectation(s_next)
            y = r_max + self.gamma * (1 - done) * self.target_dist(s_next, a_next)  # [B, N, D]
        x = self.dist(s, a)
        loss = self.sinkhorn(x, y).mean()
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dist.parameters(), 10.0)
        self.opt.step()
        soft_update(self.target_dist, self.dist, self.tau)
```

---

## 10. Hyperparameters (Suggested)

| Component | Setting |
| --------- | ------- |
| Blur $\varepsilon$ | 0.01 (try {0.005, 0.01, 0.05}) |
| Scaling | 0.9 |
| Particles | 32 (try 16, 32, 64) |
| $\gamma$ | 0.99 (Maze), 0.995 (Fetch) |
| Target update $\tau$ | 0.005 |
| Reward beta (progress bonus) | 0.2–0.6 |
| Optimizer | Adam 3e-4 |
| Batch size | 256 |
| Grad clip | 10.0 |
| Blur decay | optional anneal from 0.02 → 0.01 |

---

## 11. Environments and Benchmarks

- **Maze Navigation**: 2D grid, sparse goal reward.
- **FetchReach / FetchPush (Sparse)**: Robotics manipulation; success signal sparse.
- **Ablation sandbox**: MiniGrid (Goal), MinAtar (for quick tests).

---

## 12. Evaluation Metrics

- Success rate vs environment steps.
- Sample efficiency: steps to 80% success.
- Return distribution statistics: mean, std, CVaR.
- Sinkhorn loss curve.
- Reward distribution before/after transformation (histograms).
- Transport plan diagnostics: cost matrix stats (optional).

---

## 13. Ablations

1. Reward: Standard vs To-the-Max.
2. Loss: Scalar RL (Huber) vs Sinkhorn.
3. Particles: 16/32/64.
4. Blur: 0.005/0.01/0.05.
5. Risk objective: mean vs CVaR for action selection.
6. Progress bonus beta: 0.0/0.2/0.4/0.6.

---

## 14. Safety and Stability

- Clip rewards to a reasonable range after transformation (e.g., [0, 1+beta]).
- Normalize inputs; standardize observations.
- Use target networks to stabilize distributional targets.
- If Sinkhorn loss NaNs: increase blur, reduce learning rate, clip gradients.
- Optional entropy regularization to avoid premature exploitation.

---

## 15. Reproducibility Checklist

- Fix seeds (torch, numpy, env).
- Log: blur, scaling, particles, beta, gamma, tau.
- Log reward histograms (pre/post transform).
- Save checkpoints (policy, dist, target).
- Version pin: torch≥2.2, geomloss, gymnasium/Fetch, Minigrid, mujoco-py.

## 16. Testing ✅

- Run unit tests with pytest:

  python -m pip install -r paperAssignments/Assignments1-50/CA8/test-requirements.txt
  pytest paperAssignments/Assignments1-50/CA8/tests

- Tests included:
  - `test_imports.py`: basic import and fallback checks (geomloss fallback) 
  - `test_config_loader.py`: verify YAML configs update `cfg` and unknown keys are ignored

## 17. Citation / Paper

If you use MaxSink or parts of this assignment in your work, please cite this assignment and the original references (Bellemare et al., Dabney et al., Feydy et al.). For questions about reproducing the experiments, open an issue or contact the assignment maintainers.


---

## 16. Logging Schema

- Scalars: `success`, `return_mean`, `return_std`, `sinkhorn_loss`, `blur`, `gamma`, `beta`, `cvar` (if used).
- Histograms: `reward_raw`, `reward_transformed`, `adv_particles`.
- Curves: success vs steps; sinkhorn loss vs updates.

---

## 17. Visualization Plan

- Reward distributions before/after transform.
- Particle clouds (projected) for $Z$ and targets across training.
- Sinkhorn loss curve.
- Success rate curves for four configs: (Std+Scalar, Std+Sinkhorn, Max+Scalar, Max+Sinkhorn).
- CVaR vs mean returns (if risk-aware).

---

## 18. Proof Sketch: Potential Shaping Compatibility

If transformation is potential-based: $r'(s,a)=r(s,a)+\gamma \Phi(s')-\Phi(s)$, then optimal policies are invariant. Choose $\Phi$ such that bonus equals progress measure. Document any deviations to clarify induced bias.

---

## 19. Interaction Between Transformation and Sinkhorn

- Transformation reduces variance → smaller OT cost, potentially faster convergence.
- OT still beneficial to align multimodal return distributions (e.g., different paths in maze).
- If transformation collapses distribution to near-deterministic, Sinkhorn advantage diminishes; verify empirically.

---

## 20. Risk Sensitivity Variant

- Use CVaR or lower quantile for action selection:
  $a^* = \arg\max_a \text{CVaR}_\alpha(Z_\theta(s,a))$.
- Hypothesis: risk-aware selection may further accelerate reaching goals with less variance.

---

## 21. Implementation Notes (Reward Transform)

```python
def compute_progress(s, goal):
    # e.g., negative distance difference
    return (prev_dist - curr_dist)

def to_the_max(r, progress, beta=0.4):
    bonus = (progress > 0).float() * beta
    return torch.maximum(r, bonus)
```
Ensure progress is computed per env; for Fetch use distance to goal; for Maze use Manhattan/Euclidean delta.

---

## 22. Minimal Training Loop (Sketch)

```python
for step in range(total_steps):
    a = policy.act(s)
    s_next, r, done, info = env.step(a)
    progress = info.get("progress", 0.0)
    r_max = to_the_max(torch.tensor([r]), torch.tensor([progress]), beta)
    replay.add(s, a, r_max, s_next, done)
    s = s_next if not done else env.reset()

    if step > start_updates and step % update_every == 0:
        batch = replay.sample(batch_size)
        update_dist_and_policy(batch)
```

---

## 23. Hyperparameter Table (Per Env)

| Env | Blur | Particles | Beta | Gamma | Batch | Tau | Notes |
| --- | ---- | --------- | ---- | ----- | ----- | --- | ----- |
| Maze | 0.01 | 32 | 0.4 | 0.99 | 256 | 0.005 | small state |
| FetchReach | 0.01 | 32 | 0.4 | 0.995 | 256 | 0.005 | dense proprio |
| FetchPush | 0.02 | 64 | 0.6 | 0.995 | 512 | 0.005 | harder sparse |

---

## 24. Ablation Matrix Template

| Config | Success@500k | Steps to 80% | Sinkhorn Loss Final | Notes |
| ------ | ------------ | ------------- | ------------------- | ----- |
| Std + Scalar | 0.55 | >500k | — | baseline |
| Std + Sinkhorn | 0.68 | 420k | 0.12 | distro only |
| Max + Scalar | 0.70 | 380k | — | transform only |
| Max + Sinkhorn | **0.82** | **300k** | 0.08 | combined |

---

## 25. Safety Checks

- Ensure $r'$ bounded; clamp to [0, 1+beta].
- If OT cost explodes, increase blur or reduce beta.
- Disable transformation for terminal transitions if it would distort success signal incorrectly.

---

## 26. Reproducibility Artifacts

- YAML configs for Maze and Fetch.
- Seed list.
- Checkpoints with policy/dist and beta/blur.
- Scripts for reward histogram and success plots.

---

## 27. Extended Mathematical Notes

### 27.1 Expected Return Shift

If $r' = r + b$, then $\mathbb{E}[G'] = \mathbb{E}[G] + b \cdot \frac{1}{1-\gamma}$. Here, progress bonus is state-dependent; analyze its expectation to quantify bias.

### 27.2 OT Gradient Sensitivity

Sinkhorn plan $P_\varepsilon = \exp((f+g-C)/\varepsilon)$; transformed rewards shift $C$ via targets $Y$; smaller spread → steeper gradients feasible (smaller $\varepsilon$).

---

## 28. Potential Failure Modes

- Over-shaping: agent exploits progress bonus without finishing → add success-only gate.
- Diminished distributional benefit: if rewards become dense, Sinkhorn marginal gain drops; still test.
- Computational overhead: OT adds cost; mitigate with fewer particles/KeOps.

---

## 29. Compute and Performance Estimates

- Maze: 1 GPU, particles 32, blur 0.01 → fast (<6h for 1M steps).
- FetchPush: 1 GPU, particles 64, blur 0.02 → moderate (~12h for 1M steps).
- KeOps/geomloss accelerates Sinkhorn; fallback to CPU slower.

---

## 30. Extensions

- **Risk-sensitive MaxSink**: CVaR action selection.
- **Diffusion rewards**: learn progress via learned distance predictor.
- **Hindsight relabeling + Max**: combine with HER; transform relabeled rewards.
- **Multi-goal**: condition policy on goal; apply transform using goal distance.

---

## 31. Logging/Plot Scripts (Outline)

- `plot_rewards.py`: histograms pre/post transform.
- `plot_success.py`: success vs steps for 4 configs.
- `plot_sinkhorn.py`: loss vs updates, blur schedule.
- `plot_particles.py`: PCA of particles vs targets over training.

---

## 32. Checklist for Experiments

- [ ] Implement reward transform in env wrapper.
- [ ] Integrate into Sinkhorn agent; ensure targets use transformed reward.
- [ ] Run four configs.
- [ ] Ablate blur/particles/beta.
- [ ] Collect plots/tables.
- [ ] Document seeds, configs, code hash.

---

## 33. Risk and Mitigation Table

| Risk | Mitigation |
| ---- | ---------- |
| Reward exploit | Gate bonus to progress + success, clamp |
| OT instability | Increase blur, reduce particles, clip gradients |
| Slow training | Lower particles, smaller nets, fewer OT iterations |
| Bias from shaping | Use potential-based bonus; report bias if not |

---

## 34. Implementation Details: OT Iterations

- Sinkhorn iterations: 20–50; fewer with blur 0.02.
- Use log-domain to avoid underflow.
- If using geomloss: set `debias=True`, adjust `scaling`.

---

## 35. Network Architecture

- Encoder (CNN/MLP) → latent.
- Particle generator: MLP outputs $N \times d$ particles per action.
- Policy: shared encoder with actor head.
- Use LayerNorm/GroupNorm for stability; avoid BN if KeOps used (BN optional).

---

## 36. Code Skeleton (Env Wrapper)

```python
class ToTheMaxWrapper(gym.Wrapper):
    def __init__(self, env, beta=0.4):
        super().__init__(env)
        self.beta = beta
        self.prev_dist = None
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_dist = info.get("dist_to_goal", None)
        return obs, info
    def step(self, action):
        obs, r, done, trunc, info = self.env.step(action)
        dist = info.get("dist_to_goal", None)
        progress = 0.0
        if dist is not None and self.prev_dist is not None:
            progress = (self.prev_dist - dist)
        bonus = self.beta if progress > 0 else 0.0
        r_max = max(r, bonus)
        self.prev_dist = dist
        info["reward_raw"] = r
        info["reward_max"] = r_max
        return obs, r_max, done, trunc, info
```

---

## 37. Action Selection Variants

- Expected value: $\mathbb{E}[Z]$.
- CVaR$_\alpha$: lower quantile mean.
- Entropy-regularized: add $\alpha \log \pi$ as in SAC; combine with distributional Q.

---

## 38. Why Sinkhorn over Quantile Huber?

- OT exploits geometry; for shaped rewards, supports move smoothly; Sinkhorn provides smoother gradients for continuous spaces.
- Debias term avoids self-similarity bias.
- For small particles, Sinkhorn can outperform quantile Huber in stability under shaped rewards.

---

## 39. Concluding Summary

MaxSink merges "To the Max" reward shaping with Sinkhorn Distributional RL to tackle sparse goal-reaching. The transformation densifies signal; Sinkhorn models the resulting return distributions with geometric fidelity. The combined approach aims to deliver faster success rates, improved sample efficiency, and richer uncertainty estimates. This README supplies the necessary theory, algorithms, code patterns, hyperparameters, ablations, and evaluation plans to implement and rigorously test MaxSink on Maze and Fetch tasks.

---

_This README is the complete blueprint for Assignment 8: integrating "To the Max" reward transformation with Sinkhorn Distributional RL. Ensure code, math, and experiments stay aligned._

---

## 75. Detailed Proof Sketch: Sinkhorn with Shaped Rewards

Assume cost matrix $C_{ij} = \|x_i - y_j\|^2$ where $y_j$ includes transformed reward. If transformation scales rewards by factor $\alpha$ and shifts by $\beta$, then cost scales as $\alpha^2$; Sinkhorn potentials scale accordingly. Debiasing term removes self-similarity offset. Contraction in Wasserstein space remains with factor $\gamma$ under bounded rewards; shaping does not break fixed point existence.

---

## 76. Formal Bias Bound for Potential-Based Shaping

If $r' = r + \gamma \Phi(s') - \Phi(s)$ and $\Phi$ bounded by $M$, value shift satisfies:
$$
V'^\pi(s) = V^\pi(s) + \Phi(s) - \gamma^T \mathbb{E}[\Phi(s_T)] \approx V^\pi(s) + \Phi(s),
$$
for episodic tasks with finite horizon $T$; infinite horizon yields $V'^\pi = V^\pi + \frac{1}{1-\gamma}(\gamma \mathbb{E}[\Phi]-\Phi)$; optimal policy invariant. Choose $\Phi=-k d(s)$ to encode progress; document $k$.

---

## 77. Additional Pseudocode: Potential-Based Transform

```python
def potential_based_reward(r, s, s_next, gamma, k=1.0, goal=None):
    if goal is None:
        return r
    phi_s = -k * dist(s, goal)
    phi_sn = -k * dist(s_next, goal)
    return r + gamma * phi_sn - phi_s
```

Use this to guarantee policy invariance; compare to heuristic max transform in ablation.

---

## 78. Sensitivity to Blur and Particles: Experiment Plan

- Run blur {0.005, 0.01, 0.02} × particles {16, 32, 64} on Maze.
- Record success, Sinkhorn loss, runtime.
- Expect diminishing returns beyond 32 particles; blur too low may destabilize with noisy rewards; transformed rewards may tolerate lower blur.

---

## 79. Reward Leakage Prevention

- Do not grant progress bonus when collision or invalid move.
- For Fetch, ensure grip failure does not produce false progress bonus.
- Mask bonus when done=True to avoid inflating terminal reward beyond intended.

---

## 80. Mixed Precision Considerations

- Sinkhorn loss uses exponentials; keep in float32 to avoid underflow; run model forward in AMP but compute loss in fp32.
- If using KeOps, follow its dtype recommendations; avoid half precision in cost computation.

---

## 81. Diagnostics: OT Plan Entropy

- Track entropy of transport plan $P$; high entropy indicates diffuse match; low entropy indicates confident alignments. With transformed rewards, expect entropy to drop as distribution sharpens.

---

## 82. Curriculum Ideas

- Start with higher blur (0.02) and anneal to 0.01 as training stabilizes.
- Start with higher beta (0.6) to jumpstart learning, anneal to 0.3 to reduce bias.
- Particle annealing: begin with 16 particles for speed; increase to 32 mid-training.

---

## 83. Multi-Goal Extension

- Condition policy and critic on goal $g$.
- Progress based on distance to $g$.
- Train with HER + MaxSink: relabel transitions with new goals, compute transformed reward accordingly, then apply Sinkhorn loss.

---

## 84. Code Modules Recommendation

- `envs/to_the_max_wrapper.py` for reward shaping.
- `agents/maxsink_agent.py` encapsulating policy, dist head, sinkhorn loss.
- `configs/maze.yaml`, `configs/fetch.yaml` for hyperparams.
- `scripts/eval.py`, `scripts/plot_rewards.py`, `scripts/plot_sinkhorn.py`.

---

## 85. Negative Control Experiment

- Use random progress signal (noise) to show Max+Sink degrades if shaping is uninformative—demonstrates reliance on correct progress metric.

---

## 86. Risk Mitigation for Over-Shaping

- Cap cumulative bonus per episode to avoid overpowering terminal success reward.
- Add small time penalty to prevent endless progress farming.

---

## 87. Additional Metrics

- OT cost components: $W(X,Y)$, self terms.
- Policy entropy (if stochastic).
- State coverage (% of grid visited).
- Goal distance over time.

---

## 88. Wall-Clock vs Performance Plot

- Plot success vs wall-clock for four configs to show efficiency gains (or losses) with Sinkhorn overhead.

---

## 89. Implementation Detail: Argmax Expectation

- For discrete Maze: compute mean per action and argmax.
- For continuous Fetch: use actor network; critic used for training only.
- Optionally sample multiple candidate actions, pick best expected Q to approximate argmax.

---

## 90. Failure Case Analysis

- If Sinkhorn underperforms scalar with Max: investigate cost scaling; maybe blur too low; distribution collapsed → switch to Huber baseline.
- If Max+Sink overshoots and oscillates: reduce beta; add entropy; increase blur.

---

## 91. Reproducibility Artifacts (Expanded)

- Save reward transform code hash.
- Save reward histograms snapshots at checkpoints.
- Include seed-specific logs to reproduce curves.

---

## 92. Statistical Significance Protocol

- For each metric, compute 95% CI via bootstrap.
- Report p-values comparing Max+Sink vs Max+Scalar.
- Include effect sizes (Cohen’s d).

---

## 93. Checklist for Camera-Ready

- [ ] Ablations complete (beta, blur, particles, loss type).
- [ ] Four-way comparison plotted with CI.
- [ ] Reward histograms included.
- [ ] Sinkhorn stability analysis (entropy, loss curves).
- [ ] Code + configs released.

---

## 94. Extended Future Work

- Apply to long-horizon manipulation (Kitchen tasks) with dense proprio + sparse goal.
- Combine with diffusion value models; use transformed rewards as conditioning.
- Investigate interaction with curriculum learning for goal difficulty.
- Evaluate on safety-constrained tasks; use distribution tails for risk-averse policies.

---

## 95. Potential Reviewer Questions & Answers

- **Does shaping bias optimality?** Use potential-based variant; quantify bound; report raw-reward evaluation.
- **Why Sinkhorn over quantile?** Geometric matching; smoother gradients; OT robustness to support mismatch.
- **Is overhead worth it?** Provide wall-clock vs performance; if modest overhead with significant gains, justify.
- **How sensitive to progress metric?** Include negative control with noisy progress; show degradation.

---

## 96. Additional Figures

- Transport plan heatmap early vs late training.
- Particle vs target scatter after PCA.
- Progress over steps distribution (how often bonus triggered).
- Blur schedule vs Sinkhorn loss.

---

## 97. Auxiliary Losses (Optional)

- Predict progress from latent to improve shaping robustness.
- Contrastive learning on latent states to improve geometry for OT.

---

## 98. Debugging Guide

- NaNs in Sinkhorn: increase blur, reduce LR, check reward bounds.
- Success stagnant: verify bonus gating; check progress metric; reduce blur if too smooth.
- OOM: reduce particles/batch; disable KeOps fallback to CPU (slower).

---

## 99. Final Remarks

MaxSink targets sparse goal-reaching by fusing informative reward transformation with distributional OT-based critics. It offers a principled, empirically testable avenue to accelerate learning while maintaining rich uncertainty estimates. With the detailed math, algorithms, code scaffolds, and evaluation plan above, practitioners can implement, debug, and benchmark MaxSink across maze and robotic domains.

---

_This README is the complete blueprint for Assignment 8: integrating "To the Max" reward transformation with Sinkhorn Distributional RL. Ensure code, math, and experiments stay aligned._

---

## 40. Extended Mathematical Analysis

### 40.1 Distribution Shift from Transformation

Let $p(G)$ be return distribution under raw rewards, $p'(G')$ under transformed rewards. If $T$ is monotone and bounded, cumulative distribution satisfies $F_{G'}(x) = F_G(T^{-1}(x))$ for invertible regions. For non-invertible (clipped) regions, mass accumulates at maxima, sharpening upper tail. Sinkhorn transports $p'(G')$ to target distributions; concentrated mass can reduce entropy of optimal plan.

### 40.2 Bias Characterization

If $T$ is not potential-based, policy optimality may shift. Bound bias in value:
$$
|V'^\pi(s) - V^\pi(s)| \le \frac{\|T(r)-r\|_\infty}{1-\gamma}.
$$
Report this bound and empirically measure returns under both reward definitions for fairness.

### 40.3 Lipschitz Properties

Assume $T$ is $L_T$-Lipschitz: $|T(r_1)-T(r_2)| \le L_T |r_1-r_2|$. Then TD targets contract with factor $\gamma L_T$; if $L_T \le 1$, transformation does not worsen contraction. Sinkhorn operator with blur $\varepsilon$ is Lipschitz in input measures; combined operator remains contractive under bounded costs.

---

## 41. Detailed Algorithm Steps (Training)

1. **Env Step**: observe $(s,a,r,s')$, compute progress, transform reward $r' = T(r, s, a, s')$.
2. **Replay Store**: store $(s,a,r',r,s',progress,done)$.
3. **Sample Batch** of size $B$.
4. **Compute Targets**:
   - Sample $a'$ via policy or argmax expectation.
   - Sample target particles $Z_{\theta'}(s', a')$.
   - Form $Y = r' + \gamma (1-d) Z_{\theta'}(s', a')$.
5. **Compute Predictions**: $X = Z_\theta(s,a)$.
6. **Loss**: $L = S_\varepsilon(X, Y)$ (mean over batch).
7. **Update Critic**: backprop $L$, clip grads.
8. **Update Policy**: maximize expected value (or CVaR) using $Z_\theta$.
9. **Target Update**: soft update $\theta'$.
10. **Logging**: record $L$, success, reward stats, particles stats.

---

## 42. Advantage and Policy Update Details

- Expected Q: $\mathbb{E}[Z] = \frac{1}{N}\sum_i z_i$.
- CVaR: sort particles; take mean of lowest $\alpha N$.
- Policy loss (deterministic): $L_\pi = -\mathbb{E}_s[\mathbb{E}[Z_\theta(s,\pi(s))]]$ (or CVaR).
- Add entropy bonus for stochastic policies if desired.

---

## 43. Sinkhorn Configuration Tips

- **Blur**: lower blur → sharper OT; increase if unstable.
- **Scaling**: <1 accelerates convergence; default 0.9.
- **Iterations**: 20–50; monitor convergence of dual potentials.
- **KeOps**: enable for large batch/particles to reduce memory.

---

## 44. Reward Transform Variants

1. **Progress bonus**: $r'=\max(r, \beta \mathbf{1}[d_\text{prev}-d_\text{curr}>0])$.
2. **Potential-based**: $r' = r + \gamma \Phi(s') - \Phi(s)$ with $\Phi=-k d(\cdot)$.
3. **Max clamp**: $r' = \min(1+\beta, \max(r, \beta \cdot \text{sigmoid}(\Delta d)))$.
4. **Hybrid**: combine progress with time penalty to discourage dithering.

---

## 45. Reward Visualization

- Plot histograms of $r$ vs $r'$ over training to confirm densification.
- Track mean/std of $r'$; ensure no saturation at maximum for all steps (would remove learning signal).

---

## 46. Experimental Design (Detailed)

- **Seeds**: 5–10 per config.
- **Steps**: Maze 1–2M, Fetch 2–3M.
- **Eval frequency**: every 10k steps (Maze), 50 episodes per eval.
- **Replay**: 1M buffer; prioritized off (keep uniform for fairness).
- **Network**: shared encoder + distribution head; policy shares encoder.

---

## 47. Metrics Table Template (Final Report)

| Env | Config | Success @1M | Steps to 80% | Return Mean | Return Std | Sinkhorn Loss Final |
| --- | ------ | ----------- | ------------- | ----------- | ---------- | ------------------- |
| Maze | Std+Scalar | 0.55 | >1M | 0.35 | 0.20 | — |
| Maze | Std+Sink | 0.68 | 0.9M | 0.42 | 0.18 | 0.12 |
| Maze | Max+Scalar | 0.70 | 0.8M | 0.44 | 0.15 | — |
| Maze | Max+Sink | **0.82** | **0.6M** | **0.55** | **0.12** | 0.08 |

---

## 48. Logging Keys (Expanded)

- `reward_raw_mean`, `reward_raw_std`, `reward_max_mean`, `reward_max_std`.
- `sinkhorn_loss`, `blur`, `particles`.
- `success_rate`, `episode_len`.
- `value_mean`, `value_std`, `cvar_mean`.
- `progress_mean`, `progress_pos_frac`.
- `target_self_loss` (W(X,X), W(Y,Y)) for diagnostics.

---

## 49. Implementation Details: Particles

- Output shape: `[B, N, d]` where $d=1$ for scalar returns; can be >1 for multi-objective but here $d=1$.
- Sample via MLP head producing $N$ scalars per action or per state-action.
- Optionally reparameterize as mean + noise to encourage spread; though Sinkhorn already handles distributional support—keep simple first.

---

## 50. Policy Extraction from Distribution

- Deterministic: pick action maximizing mean or CVaR.
- Stochastic: sample action from policy network; use Q distribution for evaluation.
- For continuous actions: actor-critic style (DDPG/SAC) where critic is distributional; actor trained on mean/CVaR.

---

## 51. Potential Edge Cases

- If progress metric noisy, reward may oscillate; smooth progress with EMA over distance.
- If all steps receive bonus, exploration trivializes; ensure bonus gated on strict improvement.
- In Fetch tasks, dense proprio signals may already help; transformation should not drown out terminal success reward—cap bonus.

---

## 52. Ablation: Blur vs Beta

Run grid over blur {0.005,0.01,0.02} and beta {0.2,0.4,0.6}. Plot success heatmap; expect moderate beta + moderate blur best; too high beta may reduce need for OT.

---

## 53. Entropy and Exploration

- Add small entropy bonus (0.01) to policy to avoid early collapse.
- For Maze, epsilon-greedy exploration may be sufficient; for Fetch, Gaussian noise on actions.
- Ensure reward shaping does not remove need for exploration; track state coverage.

---

## 54. Coverage Metrics

- State visitation entropy.
- Count-based novelty (for Maze).
- Correlate coverage with success to see if shaping + distributional improves exploration efficiency.

---

## 55. Robustness Tests

- Add observation noise; verify Max+Sink more robust due to distributional awareness.
- Vary goal positions; test generalization with same trained policy.
- Remove progress metric mid-training (ablation) to see sensitivity.

---

## 56. Compute Budget & Throughput

- Estimate OT cost: O(B*N^2) with geomloss optimized; choose N accordingly.
- Profile step time; ensure training feasible on single GPU.
- For large N, use KeOps or sliced-Sinkhorn approximation (optional).

---

## 57. Alternative Loss: Energy Distance

- As baseline, replace Sinkhorn with Energy (MMD-like) loss; compare performance to show OT advantage.

---

## 58. Statistical Testing

- Use t-tests or bootstrap CI on success rates across seeds to demonstrate significance between configs.

---

## 59. Code Quality Checklist

- Deterministic seeding.
- Assert reward bounds post-transform.
- Unit tests: transform monotonicity; progress bonus gating; sinkhorn call shapes.
- Logging consistent across runs; config saved with checkpoints.

---

## 60. Minimal Unit Tests

1. Transform monotonic: $r1<r2 \implies T(r1)\le T(r2)$.
2. Progress gate: no progress → $r'=r$ (if $r$ already maxed).
3. Sinkhorn loss finite for random clouds.
4. Particles shapes correct; gradients flow.

---

## 61. Extended Pseudocode for Sinkhorn Loss

```python
from geomloss import SamplesLoss

class SinkhornHead(nn.Module):
    def __init__(self, blur=0.01, scaling=0.9, p=2):
        super().__init__()
        self.loss = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling, debias=True)
    def forward(self, x, y):
        # x, y: [B, N, 1]
        return self.loss(x, y)
```

---

## 62. Risk-Sensitive Variant Pseudocode

```python
def cvar(q_particles, alpha=0.1):
    q_sorted, _ = torch.sort(q_particles, dim=1)
    k = max(1, int(alpha * q_sorted.size(1)))
    return q_sorted[:, :k].mean(dim=1, keepdim=True)

policy_loss = -(cvar(q_sa, alpha).mean())
```

---

## 63. Fetch Task Specifics

- Progress: decrease in end-effector-to-goal distance.
- Success: grip threshold within tolerance.
- Bonus: only if distance decreases by >epsilon to avoid tiny jitter bonuses.
- Reset progress tracker each episode.

---

## 64. Maze Task Specifics

- Progress: Euclidean/Manhattan distance to goal.
- Obstacles: ensure progress only counts if move valid; otherwise bonus disabled.
- Optional shaping: small time penalty to prevent loops.

---

## 65. Limitations and Notes

- Reward shaping may bias optimal policies; document and, if possible, use potential-based shaping to preserve optimality.
- Sinkhorn adds compute; choose particle count carefully.
- Progress signals require environment access to goal distance; not always available.

---

## 66. Future Work Directions

- Combine with HER: relabel goals, then apply To-the-Max on relabeled reward.
- Curriculum on beta: start high to speed learning, anneal down to reduce bias.
- Sliced-Wasserstein Sinkhorn to scale to higher particle counts.
- Multi-objective extension: treat success and efficiency as separate reward dimensions.

---

## 67. Reporting Structure for Paper/Slides

- Section: Motivation & theory (transformation + OT).
- Section: Algorithm (MaxSink) with diagram of reward transform + OT loop.
- Experiments: Maze, Fetch; four-config comparison; ablations (beta, blur, particles).
- Plots: success curves, reward histograms, Sinkhorn loss, CVaR vs mean.
- Table: success/steps/return metrics; statistical significance.

---

## 68. Appendix Suggestions

- Full hyperparameter grids.
- Extra plots for sensitivity.
- Proof details for potential shaping bias bound.
- Wall-clock profiling charts.

---

## 69. Integration Notes (Codebase)

- Add wrapper for reward transform; toggle via config.
- Parameterize beta, blur, particles in config file.
- Ensure replay stores raw reward for analysis; use transformed reward for training.
- Keep evaluator optionally using raw reward for fairness (report both).

---

## 70. Wall-Clock and Memory Profiling

- Log time/update, samples/sec.
- Monitor GPU memory; adjust particle count or batch size if OOM.
- KeOps can reduce memory at cost of compilation; enable flag if needed.

---

## 71. Sanity Run Plan

- MiniGrid Goal: run 200k steps; verify Max+Sink > baselines.
- Maze small: 500k steps; ensure reward histogram shows bonus but not saturated.
- FetchReach: quick run (300k) to validate progress metric.

---

## 72. Potential Negative Outcomes

- If Max+Scalar already solves tasks quickly, Sinkhorn marginal gain may be small; still report to show boundary of usefulness.
- If progress metric noisy, may harm performance; include ablation showing impact of noisy progress.

---

## 73. Open Questions

- Does OT still help when reward is dense? Maybe less; but for multimodal paths (different routes), OT could still aid learning correlated structure.
- Is CVaR action selection beneficial with shaped rewards? Test.
- How sensitive is success to blur with transformed rewards?

---

## 74. Checklist Before Finalizing

- [ ] Reward transform tested for monotonicity and bounds.
- [ ] Sinkhorn loss stable (no NaNs).
- [ ] Four configs run with seeds.
- [ ] Plots generated.
- [ ] Bias/optimality discussion included.
- [ ] Configs and code committed.

---

_This README is the complete blueprint for Assignment 8: integrating "To the Max" reward transformation with Sinkhorn Distributional RL. Ensure code, math, and experiments stay aligned._

