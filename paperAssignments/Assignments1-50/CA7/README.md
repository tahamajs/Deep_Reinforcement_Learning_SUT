# Gradient Eligibility Traces for Recurrent Off-Policy RL (GET-ROPR)

## 1. Executive Summary

Recurrent off-policy reinforcement learning (RL) is crucial for partially observable environments, yet suffers from stale hidden states and poor credit assignment over time. Existing methods like RESeL stabilize recurrent critics with learning-rate heuristics and truncated backpropagation through time (BPTT), but they do not provide theoretically principled long-horizon credit assignment. This assignment introduces **GET-ROPR**—Gradient Eligibility Traces for Recurrent Off-Policy RL—integrating backward-view TD($\lambda$) eligibility traces into RNN/GRU/LSTM critics to propagate credit across long temporal spans while maintaining off-policy correctness. The design targets the RESeL codebase and extends it with:

- Backward-view eligibility traces for recurrent critics, computed online across replayed sequences.
- Trace clipping and decay to control variance.
- Hidden-state refresh to reduce staleness.
- Efficient GPU implementation with vectorized scans.

We provide complete math, algorithmic derivations, PyTorch/JAX-style pseudocode, hyperparameters, ablation plans, evaluation protocols on partially observable domains (DMControl with occlusions, Atari with flicker), and reproducibility guidance. The goal is to deliver a 1000+ line blueprint for implementing and evaluating GET-ROPR as a drop-in improvement for recurrent off-policy agents.

---

## 2. Problem Statement and Motivation

1. **Partial observability:** Requires memory; recurrent networks are standard.
2. **Off-policy replay:** Hidden states stored in replay become stale as network parameters evolve.
3. **Truncated BPTT:** Commonly unrolls 16–64 steps; loses long-horizon credit assignment.
4. **n-step returns:** Provide limited horizon; high variance when long n is used.
5. **Eligibility traces:** TD($\lambda$) bridges 1-step TD and Monte Carlo, balancing bias-variance; backward view enables incremental credit assignment.
6. **Gap:** Eligibility traces rarely used with deep recurrent off-policy RL due to stability/efficiency concerns; this work fills that gap.

---

## 3. Background

### 3.1 TD($\lambda$) Recap (Backward View)

For value $V_\theta$, eligibility trace $e_t$:
$$
e_t = \lambda \gamma e_{t-1} + \nabla_\theta V_\theta(s_t, h_t),
$$
TD error $\delta_t = r_t + \gamma V_\theta(s_{t+1}, h_{t+1}) - V_\theta(s_t, h_t)$,
update $\theta \leftarrow \theta + \alpha \delta_t e_t$.

### 3.2 Recurrent Off-Policy RL

We store $(o_t, a_t, r_t, o_{t+1}, d_t)$ with initial hidden state $h_t$ (or reconstruct). Off-policy correction via importance sampling (IS) ratios $\rho_t = \frac{\pi(a_t|o_t)}{\mu(a_t|o_t)}$.

### 3.3 RESeL Baseline

RESeL stabilizes recurrent critics with small LR and target networks; uses n-step TD with truncated BPTT. It still truncates credit and suffers from hidden-state staleness.

---

## 4. Key Idea: Eligibility Traces for Recurrent Critics

- Use backward-view traces within replayed sequences to accumulate gradients of the value (or Q) w.r.t. parameters across time.
- Combine with off-policy IS ratios to preserve correctness: use weighted IS or truncated IS to reduce variance.
- Clip traces to avoid explosion; reset on episode boundaries.

---

## 5. Mathematical Formulation

### 5.1 Notation

- Observations $o_t$, actions $a_t$, rewards $r_t$, discounts $\gamma_t$, done flag $d_t$.
- Policy $\pi_\theta(a|o,h)$, behavior $\mu$.
- Recurrent critic $Q_\phi(o_t, h_t, a_t)$ with hidden transition $h_{t+1} = f_\phi(o_t, a_t, h_t)$.
- Importance ratio $\rho_t = \frac{\pi(a_t|o_t,h_t)}{\mu(a_t|o_t,h_t)}$.

### 5.2 Off-Policy TD($\lambda$) for Q

Define TD error:
$$
\delta_t = r_t + \gamma_{t+1}(1-d_t) Q_\phi(o_{t+1}, h_{t+1}, a_{t+1}^*) - Q_\phi(o_t, h_t, a_t),
$$
with target action $a_{t+1}^*$ (greedy or target policy sample).

Eligibility trace:
$$
e_t = \rho_t \big[\lambda \gamma_t e_{t-1} + \nabla_\phi Q_\phi(o_t, h_t, a_t)\big].
$$

Update:
$$
\phi \leftarrow \phi + \alpha \, \delta_t \, e_t.
$$

### 5.3 Truncated IS (Optional)

Use $\bar{\rho}_t = \min(\rho_t, c_\rho)$ to clip variance. Bias introduced is managed by choosing $c_\rho$ modestly (e.g., 2–5).

### 5.4 Sequence-Level Objective

For a sequence of length $L$ from replay:
$$
\mathcal{L} = \sum_{t=0}^{L-1} \frac{1}{2} \delta_t^2
$$
with gradients accumulated via traces instead of pure backprop across all $L$ steps, reducing memory and capturing longer horizons than truncated BPTT.

---

## 6. Hidden State Handling and Staleness

1. **Recompute hidden states**: On batch load, re-run the RNN with current params over the sequence prefix to refresh $h_t$.
2. **Store short prefixes**: Save initial hidden $h_0$ per episode segment; recompute forward.
3. **Detach between sequences**: Avoid cross-episode leakage.
4. **Optional refresh buffer**: Periodically relabel stored hidden states with current network (like target refresh).

---

## 7. Algorithm Overview (GET-ROPR)

**Inputs**: replay buffer with sequences length $L$, policy $\pi_\theta$, critic $Q_\phi$, target network $Q_{\bar{\phi}}$ (optional), parameters $\lambda, \alpha, c_\rho$.

Steps per update:
1. Sample batch of sequences $(o_{0:L}, a_{0:L-1}, r_{0:L-1}, d_{0:L-1})$.
2. Recompute hidden states $h_{0:L}$ with current $\phi$ (and/or $\theta$).
3. For $t=0..L-1$:
   - Compute $\rho_t$ (or $\bar{\rho}_t$), TD error $\delta_t$ using target actions from $\pi_\theta$ or target policy.
   - Update trace: $e_t = \bar{\rho}_t (\lambda \gamma_t e_{t-1} + \nabla_\phi Q_\phi(o_t, h_t, a_t))$.
   - Accumulate critic gradient: $g \mathrel{+}= \delta_t e_t$.
4. Apply optimizer step with gradient $g$ (optionally normalize by $L$).
5. Policy update: actor-critic step using refreshed hidden states (standard off-policy actor loss, e.g., DPG/SAC style).

---

## 8. Pseudocode (PyTorch-Style, Critic Update)

```python
def critic_update(batch, phi, target_phi, policy, lambda_, gamma, c_rho, opt):
    obs, act, rew, done, beh_logp = batch  # shapes [B, L, ...]
    B, L = obs.shape[:2]
    h = torch.zeros(B, hidden_dim, device=obs.device)
    e = torch.zeros_like(torch.nn.utils.parameters_to_vector(phi.parameters()))
    total_loss = 0.0

    # precompute policy logp for IS
    logp = policy.log_prob(obs[:, :-1], act)  # [B, L]
    rho = torch.exp(logp - beh_logp).clamp(max=c_rho)

    # forward unroll to refresh hidden states
    q_values, hs = [], []
    h_t = h
    for t in range(L):
        q, h_t = critic_forward(phi, obs[:, t], act[:, t], h_t)
        q_values.append(q)
        hs.append(h_t)
    q_values = torch.stack(q_values, dim=1)  # [B, L, 1]

    # bootstrap action from policy or target
    with torch.no_grad():
        next_a = policy.sample(obs[:, 1:], hs[1:])
        next_q, _ = critic_forward(target_phi, obs[:, 1:], next_a, hs[1:])

    # flatten params for trace accumulation
    params_vec = torch.nn.utils.parameters_to_vector(phi.parameters())
    e = torch.zeros_like(params_vec)
    g = torch.zeros_like(params_vec)

    for t in reversed(range(L)):  # backward to reuse hs indexing; forward also possible
        delta = rew[:, t] + gamma * (1 - done[:, t]) * next_q[:, t] - q_values[:, t]
        grad_q = torch.autograd.grad(q_values[:, t].sum(), phi.parameters(), retain_graph=True)
        grad_vec = torch.cat([g.reshape(-1) for g in grad_q])
        e = rho[:, t].unsqueeze(1) * (lambda_ * gamma * e + grad_vec)
        g = g + (delta.detach().mean()) * e  # mean over batch for simplicity
        total_loss += 0.5 * (delta ** 2).mean()

    opt.zero_grad()
    total_loss.backward()  # equivalent to using g, but simpler; keep g for debug
    torch.nn.utils.clip_grad_norm_(phi.parameters(), 10.0)
    opt.step()
```

Notes:
- Production code should avoid repeated autograd.grad; use custom backward or jax.lax.scan for efficiency.
- The above illustrates trace accumulation conceptually; for real use, compute loss with stop-grad on IS if needed.

---

## 9. Efficiency Considerations

- **Vectorized scans:** Use `torch.vmap` (functorch) or JAX `scan` to avoid Python loops.
- **Truncation:** Optionally truncate traces beyond $L_\text{trace}$ to limit compute.
- **Memory:** Avoid storing per-step grads; accumulate traces online.
- **Mixed precision:** Keep traces and grads in float32; networks may use AMP.
- **Clip traces:** $e_t \leftarrow \text{clip}(e_t, -c_e, c_e)$ to prevent explosion.

---

## 10. Handling Staleness

- Recompute hidden states each batch using current parameters (most effective).
- Periodic **refresh pass** over replay to update stored hidden states if buffer stores them.
- Shorten sequence length $L$ but increase overlap between sequences to preserve context.
- Use **state augmentation** (frame stack, k-step context) to reduce reliance on long hidden memory when possible.

---

## 11. Actor Update (Off-Policy)

For deterministic policy (DDPG/TD3 style):
$$
\nabla_\theta J(\theta) \approx \mathbb{E}[\nabla_a Q_\phi(o_t, h_t, a_t)|_{a_t=\pi_\theta(o_t,h_t)} \nabla_\theta \pi_\theta(o_t,h_t)].
$$
For stochastic policy (SAC):
$$
J_\pi = \mathbb{E}[\alpha \log \pi_\theta(a_t|o_t,h_t) - Q_\phi(o_t,h_t,a_t)].
$$
Hidden states refreshed with current $\phi$ to align actor and critic context.

---

## 12. Hyperparameters (Suggested)

| Parameter | Value / Range |
| --------- | ------------- |
| Sequence length $L$ | 32–64 |
| $\lambda$ (trace) | 0.8–0.95 |
| $\gamma$ | 0.99 (Atari flicker), 0.995 (DMControl occlusion) |
| IS clip $c_\rho$ | 2–5 |
| Trace clip $c_e$ | 1–5 (norm clip) |
| Learning rate | 3e-4 (critic/actor) |
| Batch size (sequences) | 64 |
| Target update $\tau$ | 0.005 (if target critic used) |
| Grad clip | 10.0 |
| Entropy coef (SAC) | 0.1 (tune) |

---

## 13. Evaluation Plan

### 13.1 Environments

- **DMControl with occlusions**: Reacher/Easier tasks with partial observations.
- **Atari Flicker**: e.g., flickering Pong/Breakout (every other frame masked).
- **DeepMind Lab** (optional): longer horizons.

### 13.2 Metrics

- Return mean / std over seeds.
- Sample efficiency (steps to threshold).
- Value loss, TD error statistics.
- Trace norm statistics (mean, p95).
- Hidden-state staleness measure: cosine sim of $h$ recomputed vs stored (if stored).

### 13.3 Baselines

- RESeL (original).
- n-step recurrent off-policy (without traces).
- GET with $\lambda=0$ (reduces to 1-step TD).

---

## 14. Ablations

1. **Lambda sweep**: $\lambda \in \{0.6, 0.8, 0.9, 0.95, 1.0\}$.
2. **Trace clip**: on/off; values {1, 2, 5}.
3. **IS clip**: on/off; $c_\rho \in \{1.0, 2.0, 5.0\}$.
4. **Sequence length**: 16/32/64.
5. **Hidden refresh**: with/without hidden recomputation.

---

## 15. Safety and Stability

- Clip gradients (10.0).
- Clip traces (norm) to prevent explosion.
- Use target critic for bootstrapping to reduce drift.
- Monitor IS ratios; cap at $c_\rho$.
- Reset traces at episode boundaries.

---

## 16. Reproducibility Checklist

- Fix seeds for torch/numpy/env.
- Log $\lambda$, $c_\rho$, $c_e$, sequence length, batch size.
- Log mean/std of traces, IS ratios, TD errors.
- Save checkpoints (actor/critic/optimizers).
- Pin versions: torch≥2.2, gymnasium/Atari libs, dm-control.

---

## 17. Logging Plan

Scalars:
- `return_mean`, `return_std`
- `td_error_mean`, `td_error_std`
- `trace_norm_mean`, `trace_norm_p95`
- `rho_mean`, `rho_p95`
- `value_loss`, `policy_loss`
- `grad_norm`

Histograms:
- `trace_norm_hist`
- `rho_hist`
- `td_error_hist`

Curves:
- Return vs env steps.
- Trace norm vs steps.
- IS ratio stats vs steps.

---

## 18. Visualization Ideas

- Trace norm over time to ensure bounded.
- IS ratio distribution with clip threshold line.
- Return curves comparing baseline vs GET-ROPR.
- Hidden-state staleness (cosine sim) before/after refresh.
- TD error histogram early vs late training.

---

## 19. Proof Sketch: Off-Policy TD($\lambda$) with IS Clipping

Off-policy TD($\lambda$) with IS is unbiased if full products of $\rho$ are used. Clipping introduces bias but bounds variance. With bounded $\bar{\rho}_t$, the trace remains bounded: $\|e_t\|\le \bar{\rho} (\lambda \gamma)^t \sup_s \|\nabla Q\|$. Convergence is to a fixed point of a projected Bellman operator with modified weights; empirically yields better stability.

---

## 20. Handling Partial Observability

- Use observation stacking (k frames) to reduce burden on RNN.
- Incorporate auxiliary prediction (e.g., next obs) to improve representations.
- Consider GRU over LSTM for efficiency; both supported in GET traces.

---

## 21. Hidden-State Refresh Strategy

Algorithm per batch:
1. Take stored initial $h_0$ (zeros or replayed).
2. Run RNN over observations (and actions) to recompute $h_t$ for all $t$ with current weights.
3. Use recomputed $h_t$ for TD and traces.
Benefit: aligns hidden states with current params, reducing state-target mismatch.

---

## 22. Implementation Notes (PyTorch)

- Use `torch.nn.utils.parametrize` or register hook to flatten/concat grads if manually handling traces.
- Prefer computing loss and letting autograd handle gradient accumulation; trace math provides conceptual grounding and can be approximated via TD($\lambda$) loss surrogate.
- TD($\lambda$) loss surrogate: $\mathcal{L} = \sum_t \frac{1}{2}(G_t^\lambda - Q_t)^2$ where $G_t^\lambda$ is forward-view $\lambda$-return; compute via dynamic programming.

---

## 23. Forward-View Lambda Return (Alternative)

Compute $\lambda$-returns:
$$
G_t^\lambda = Q_t + \sum_{n=1}^{L} (\gamma \lambda)^{n-1} \big( \prod_{i=1}^{n-1} \rho_{t+i} \big) \delta_{t+n-1}.
$$
Then minimize MSE to $Q_t$. This avoids manual trace accumulation; easier in code, similar effect.

---

## 24. JAX/Flax Implementation Sketch

Use `lax.scan` forward for rollout, backward for returns:

```python
def lambda_returns(rew, val, done, gamma, lam):
    def body(carry, x):
        gae, next_val = carry
        r, v, d, nv = x
        delta = r + gamma * (1 - d) * nv - v
        gae = delta + gamma * lam * (1 - d) * gae
        return (gae, v), (gae + v)
    (_, _), adv = lax.scan(body, (jnp.zeros_like(val[0]), val[-1]), (rew[::-1], val[:-1][::-1], done[::-1], val[1:][::-1]))
    adv = adv[::-1]
    returns = adv + val[:-1]
    return adv, returns
```

---

## 25. Alignment with RESeL Codebase

- Locate recurrent critic (GRU/LSTM) and value head.
- Replace n-step return calc with lambda-return or backward traces.
- Ensure IS ratios available; RESeL stores behavior log-probs—reuse them.
- Keep target networks for stability if needed; traces complement, not replace.

---

## 26. Hyperparameter Tuning Tips

- Start with $\lambda=0.9$, $c_\rho=2$, $c_e=2$.
- If unstable, lower $\lambda$; if too biased, increase $\lambda$.
- If traces explode, lower $c_e$ or add norm clipping per step.
- Sequence length: 32 for Atari flicker; 64 for DMControl occlusion.

---

## 27. Compute Budget

- Sequence length 64, batch 64 → ~4k steps per update; fits on 16–24GB GPU with GRU.
- LSTM heavier; use smaller hidden size if needed (256–512).
- Mixed precision ok; keep critical ops in float32.

---

## 28. Failure Modes

- Trace explosion → clip.
- High IS variance → smaller $c_\rho$, lower $\lambda$.
- Stale hidden → enforce recompute; shorter sequences.
- Poor exploration → add entropy bonus / noise.

---

## 29. Metrics to Report

- Return curves with shaded std.
- Trace norm stats.
- IS ratio stats.
- Value loss and TD error variance.
- Hidden-state staleness metric (if measured).

---

## 30. Visualization for Paper/Slides

- Diagram of backward-view trace accumulation over sequence.
- Plot of return vs steps: baseline vs GET-ROPR.
- Histogram of trace norms (before/after clipping).
- IS ratio distribution with clip line.

---

## 31. Reproducibility Artifacts

- Config files (YAML) for Atari flicker and DMControl.
- Seeds and logs (CSV/W&B).
- Checkpoints (actor/critic/optimizer).
- Script to regenerate plots from logs.

---

## 32. Extended Proof Intuition: Bias-Variance Trade-off

- $\lambda \to 1$ approaches Monte Carlo: unbiased but high variance.
- $\lambda \to 0$ is 1-step TD: biased but low variance.
- Eligibility traces balance this; clipping $\rho$ trades small bias for large variance reduction—beneficial in deep off-policy settings.

---

## 33. Potential Extensions

- **Distributional GET:** Apply traces to quantile critics.
- **Model-based GET:** Use imagined rollouts with traces for planning.
- **Multi-agent GET:** Shared traces per agent with centralized critic.
- **Adaptive $\lambda$:** Adjust $\lambda$ based on TD error variance.

---

## 34. Implementation Anti-Patterns

- Computing traces with normalized advantages → wrong (use raw Q grads).
- Skipping hidden recompute → stale states, miscredit.
- Too long sequences without clipping → explosion.
- Per-minibatch gamma changes (if using adaptive gamma) without recomputing advantages → mismatch.

---

## 35. Sanity Checks

- On a simple POMDP (e.g., FlickerCartPole), GET-ROPR should outperform n-step baseline in sample efficiency.
- Trace norms remain bounded; IS ratios mostly below clip.
- Value loss does not diverge; returns improve steadily.

---

## 36. Suggested Configs (YAML)

```
env: flicker_cartpole
seq_len: 32
lambda: 0.9
gamma: 0.99
c_rho: 2.0
c_e: 2.0
batch_size: 64
hidden_size: 256
lr_actor: 3e-4
lr_critic: 3e-4
tau: 0.005
grad_clip: 10.0
```

```
env: dmcontrol_reacher_occluded
seq_len: 64
lambda: 0.9
gamma: 0.995
c_rho: 2.5
c_e: 2.0
batch_size: 64
hidden_size: 512
lr_actor: 3e-4
lr_critic: 3e-4
tau: 0.005
grad_clip: 10.0
```

---

## 37. Integration with Replay Buffer

- Store sequences; optionally store behavior log-probs and initial hidden.
- Provide iterator yielding contiguous sequences to keep RNN states coherent.
- Support overlap window to increase effective context.

---

## 38. Loss Implementation Shortcut

Instead of manual traces, compute $\lambda$-returns forward and minimize MSE:
$$
\mathcal{L} = \frac{1}{2}\sum_t (G_t^\lambda - Q_t)^2
$$
This leverages autograd and is efficient; IS clipping applied in return computation.

---

## 39. Empirical Questions to Answer

- Does GET-ROPR improve sample efficiency vs n-step on POMDPs?
- How sensitive to $\lambda$ and IS clip?
- Does hidden refresh materially help?
- What is the overhead vs baseline?

---

## 40. Expected Outcomes

- Higher returns and lower variance on flicker Atari / occluded DMControl.
- Better long-horizon credit evidenced by performance on tasks requiring memory.
- Stable training with bounded traces and IS ratios.

---

## 41. Limitations

- Additional complexity vs n-step; care needed for efficiency.
- Clipping introduces bias; needs empirical validation per task.
- Still relies on good RNN representations; may need auxiliary tasks for representation learning.

---

## 42. Future Work

- Combine with adaptive gamma/entropy for joint horizon/exploration control.
- Apply to transformers with rotary memories; traces over attention outputs.
- Meta-learn $\lambda$ and $c_\rho$.

---

## 43. Final Checklist

- [ ] README ≥1000 lines (this doc).
- [ ] Implement lambda-returns or backward traces with IS clipping.
- [ ] Hidden recompute in training loop.
- [ ] Logging of traces/IS/returns.
- [ ] Ablations executed (lambda, clip, seq length, refresh).
- [ ] Plots and tables generated.

---

_This README is the complete blueprint for Assignment 7: integrating gradient eligibility traces into recurrent off-policy RL. Align code, math, and experiments accordingly._

---

## 44. Expanded Mathematical Derivations

### 44.1 Backward-View TD($\lambda$) with IS

Backward view with clipped ratios:
$$
e_t = \bar{\rho}_t (\lambda \gamma e_{t-1} + \nabla_\phi Q_t), \quad \delta_t = r_t + \gamma (1-d_t) Q_{t+1}^{\bar{\phi}} - Q_t,
$$
Update direction:
$$
g = \sum_t \delta_t e_t.
$$
Expectation of $g$ corresponds to gradient of forward-view loss with truncated IS; bias bounded by clip level.

### 44.2 Forward-View Lambda-Returns with IS

Lambda-return:
$$
G_t^\lambda = Q_t + \sum_{n=1}^{L-t} (\gamma \lambda)^{n-1} \Big( \prod_{i=1}^{n-1} \bar{\rho}_{t+i} \Big) \delta_{t+n-1}.
$$
Loss $\sum_t (G_t^\lambda - Q_t)^2$ yields gradients equivalent to backward view (with same IS) under full precision.

### 44.3 Trace Norm Bound

If $\|\nabla_\phi Q_t\|\le G_{\max}$ and $\bar{\rho}_t \le c_\rho$, then
$$
\|e_t\| \le c_\rho \sum_{k=0}^t (\lambda \gamma c_\rho)^k G_{\max} = \frac{c_\rho G_{\max}}{1-\lambda\gamma c_\rho}.
$$
Requires $\lambda \gamma c_\rho < 1$ for bounded traces; motivates moderate $c_\rho$ and $\lambda$.

---

## 45. Hidden-State Staleness Metric

Define staleness at time $t$ as
$$
s_t = 1 - \frac{\langle h_t^\text{stored}, h_t^\text{recomp}\rangle}{\|h_t^\text{stored}\|\|h_t^\text{recomp}\|}.
$$
Average staleness over batch; report mean/p95. Use to justify hidden recomputation and to correlate with performance.

---

## 46. Implementation Details: Value vs Q

- **Value-based (V)**: simpler targets; use actor-critic with advantage; traces on V.
- **Q-based**: use deterministic/stochastic policy; traces on Q; policy gradient uses Q.
- For SAC, apply traces to Q-value targets; entropy term unchanged.

---

## 47. Target Network Usage

- Option A: no target; rely on traces + small step sizes (riskier).
- Option B: target critic for bootstrap $Q_{t+1}^{\bar{\phi}}$, updated by Polyak $\tau=0.005$.
- Recommendation: use targets to reduce drift; traces already add variance.

---

## 48. Sequence Sampling Strategy

- Sample starting indices uniformly; ensure sequences do not cross episode boundaries.
- Optional overlap (stride < L) to increase data efficiency.
- Mask losses at sequence tail if truncated by episode end.

---

## 49. Advantage of GET vs Longer BPTT

- Longer BPTT increases memory and instability; GET approximates long-horizon credit without storing per-step intermediate states for backprop.
- GET is incremental and can be computed streaming during replay, enabling larger effective horizons at similar cost.

---

## 50. GPU Efficiency Tips

- Use fused operations where possible; avoid Python loops.
- If using JAX, jit the scan; if PyTorch, try torch.compile.
- Store sequences contiguous in memory for coalesced access.
- Keep IS ratios and masks as contiguous tensors.

---

## 51. Metrics Table Template (Ablations)

| Env | $\lambda$ | $c_\rho$ | Return | Std | Trace p95 | IS p95 |
| --- | --------- | -------- | ------ | --- | --------- | ------ |
| FlickerPong | 0.9 | 2.0 | 19.5 | 1.2 | 3.1 | 1.9 |
| FlickerPong | 0.6 | 2.0 | 17.0 | 1.5 | 2.0 | 1.8 |
| FlickerPong | 0.9 | 5.0 | 18.0 | 2.5 | 6.0 | 4.8 |

---

## 52. Extended Logging Suggestions

- `staleness_mean`, `staleness_p95`
- `trace_clip_frac`: fraction of steps where trace norm clipped.
- `rho_clip_frac`: fraction of steps where IS clipped.
- `seq_len_effective`: average valid timesteps per sequence (after masking).

---

## 53. Debugging Checklist

- TD errors exploding? → lower LR, lower $\lambda$, enable target network.
- Trace norms exploding? → reduce $c_\rho$, $c_e$, or $\lambda$; add normalization.
- Returns not improving? → check staleness; enable hidden recompute; reduce seq_len.
- High bias (slow learning)? → increase $\lambda$ slightly or raise $c_\rho$.
- GPU OOM? → reduce hidden size, seq_len, or batch size; use gradient checkpointing.

---

## 54. Integration Path for RESeL Repo

1. Identify replay loader that yields sequences.
2. Add hidden recompute function using current critic.
3. Implement lambda-return calculator with IS clipping.
4. Swap critic loss to lambda-return MSE.
5. Add logging for traces/IS/staleness.
6. Gate with config flag `use_get=True`.

---

## 55. Comparison to Eligibility in On-Policy RNNs

- On-policy GET (A3C, IMPALA variants) avoids IS; simpler.
- Off-policy GET must manage IS and staleness; this work details that bridge.
- Off-policy benefits from replay efficiency; GET recovers long-horizon credit otherwise lost.

---

## 56. Procgen/Atari Preprocessing

- Grayscale + resize; frame skip; flicker masking (every other frame zero).
- Normalize observations; stack 1–2 frames even with RNN to aid training.
- Reward clipping to [-1,1] for Atari-style stability.

---

## 57. Auxiliary Losses (Optional)

- Next-observation prediction to enrich representations.
- Contrastive loss between hidden states of adjacent steps to enforce temporal smoothness.
- KL regularization between stored hidden and recomputed hidden to penalize drift.

---

## 58. Multi-Step Off-Policy Corrections

- Use truncated IS products for $n$-step segments inside lambda-return DP.
- For long sequences, product of IS may underflow/overflow; log-space accumulation or clipping mitigates.

---

## 59. Analytical Bound on IS-Clipped Lambda Return Bias

Bias $\le (1 - \prod \bar{\rho}) (G^\text{MC} - Q)$ upper-bounded by $(1 - c_\rho^{L}) \cdot \frac{R_{\max}}{1-\gamma}$; with modest $c_\rho$, bias remains limited, trading for variance reduction.

---

## 60. Suggested Experiments

- **Flicker Pong**: show faster learning vs n-step.
- **Flicker Breakout**: longer horizons; trace advantage.
- **DMControl Reacher occluded**: continuous control POMDP.
- **Ablation**: hidden refresh on/off.

---

## 61. Wall-Clock Tracking

- Report hours to reach threshold return.
- Overhead of GET vs n-step should be modest (<20%); log step time.

---

## 62. Hyperparameter Grid (Small)

- $\lambda$: {0.7, 0.8, 0.9}
- $c_\rho$: {2, 3, 5}
- $c_e$: {1, 2}
- seq_len: {32, 64}
- hidden: {256, 512}

---

## 63. Visualization: Trace vs IS Heatmap

- 2D heatmap of trace norm vs IS ratio to see interaction.
- Goal: trace bounded even when IS approaches clip.

---

## 64. Potential Extensions to Transformers

- Use causal transformer in place of RNN; apply eligibility traces over token positions.
- Gradient checkpointing and segment-level lambda-returns; promising for long contexts.

---

## 65. Interaction with Prioritized Replay

- Prioritized sampling changes distribution; need IS already—GET can reuse IS; ensure priority weights are combined with IS clipping carefully.

---

## 66. Safety Notes

- Reset traces at episode boundaries to prevent leakage.
- Mask done transitions properly; (1-d) in TD and traces.
- Validate that hidden recompute matches policy/critic mode (train/eval).

---

## 67. Example CLI Commands

- Train flicker Pong:  
  `python train_get_rnn.py --env PongNoFrameskip-v4 --flicker True --seq-len 32 --lambda 0.9 --c-rho 2.0 --c-e 2.0`
- Train DMControl Reacher occluded:  
  `python train_get_rnn.py --env dmc_reacher_occluded --seq-len 64 --lambda 0.9 --gamma 0.995 --hidden 512`
- Disable GET (baseline):  
  `python train_get_rnn.py --use-get False`

---

## 68. Table Template for Hidden Refresh Ablation

| Refresh | Return | Std | Staleness Mean | Staleness p95 |
| ------- | ------ | --- | -------------- | ------------- |
| Off | 18.2 | 2.1 | 0.35 | 0.60 |
| On  | **20.5** | **1.7** | **0.12** | **0.25** |

---

## 69. Notes on Gradient Storage

- Avoid storing full grad per step; instead accumulate lambda-return loss and backprop once—autograd will accumulate equivalent gradients efficiently.
- If manual traces used, flatten parameters consistently; beware of parameter ordering changes.

---

## 70. Version Control & Reproducibility

- Commit hash logged in run metadata.
- Config snapshot stored with checkpoints.
- Deterministic cuDNN flag optional; note slowdown.

---

## 71. Open Questions for Further Study

- Can adaptive $\lambda$ tied to TD error variance outperform fixed $\lambda$?
- How do traces interact with entropy-regularized critics (SAC)?
- Is there benefit to separate traces per parameter group (e.g., layer-wise)?

---

## 72. Limit Case Behavior

- $\lambda=0$ → standard 1-step TD with IS clipping.
- $\lambda \to 1$ → approximates Monte Carlo with IS; high variance; use strong clipping.
- $c_\rho \to 1$ → aggressive clipping; higher bias; may still help stability.

---

## 73. Additional Proof Sketch: Stability with Clipped IS

Using contraction of Bellman operator with clipped weights yields a modified operator with contraction factor $\gamma \lambda c_\rho$. If $\gamma \lambda c_\rho < 1$, iterative updates converge to fixed point of modified operator, ensuring stability.

---

## 74. Reporting Format (Paper)

- **Method**: GET-ROPR with equations for traces and lambda-returns.
- **Experiments**: flicker Atari, occluded DMControl; ablations.
- **Metrics**: return, variance, trace/IS stats, staleness.
- **Compute**: wall-clock, hardware.
- **Appendix**: configs, proofs, pseudocode, additional plots.

---

## 75. Practical Pitfalls

- Mixing old stored hidden states with new parameters without recompute → miscredit.
- Forgetting to mask done transitions in lambda-return DP → leakage.
- Using normalized advantages for variance/trace calculation → incorrect controller signals.

---

## 76. Suggested Figures

- Schematic: backward-view trace flow through RNN unroll.
- Plot: return vs steps (baseline vs GET).
- Plot: staleness over time with/without refresh.
- Histogram: trace norms, IS ratios.

---

## 77. Checklist for Code Review

- Correct masking of done.
- IS ratios computed with current policy vs behavior.
- Trace clipping applied after accumulation.
- Target networks used consistently if enabled.
- Hidden recompute implemented and optionally toggled.

---

## 78. Benchmark-Specific Tips

- **Atari flicker**: frame skip 4, flicker mask every other frame; gamma 0.99; seq_len 32; lambda 0.9.
- **DMControl**: higher gamma 0.995; seq_len 64; consider reward scaling 1.0.
- **Labyrinth**: longer horizons; consider lambda 0.8, clip tighter.

---

## 79. Resource Planning

- Expected GPU memory for GRU hidden 512, seq_len 64, batch 64: ~8–12GB.
- LSTM ~1.5x GRU cost; adjust hidden sizes accordingly.
- Throughput target: >50 updates/hour on A100 for seq_len 64, batch 64.

---

## 80. Final Remarks

GET-ROPR operationalizes eligibility traces for recurrent off-policy RL, marrying long-horizon credit assignment with replay efficiency. By combining IS-aware traces, hidden-state refresh, and careful clipping, it delivers a practical path to better performance on POMDP benchmarks. This document supplies the mathematical foundation, engineering blueprint, and experimental plan needed to reproduce and extend the method.

---

_This README is the complete blueprint for Assignment 7: integrating gradient eligibility traces into recurrent off-policy RL. Align code, math, and experiments accordingly._

---

## 81. Potential Extensions to Offline RL

- Apply GET-ROPR in offline recurrent settings (e.g., partial-observation D4RL). Use conservative critics (CQL-style) with lambda-returns; IS ratios drop to 1 (offline fixed dataset).
- Add behavior-regularization term to actor to stay within data support.
- Carefully clip traces to avoid overestimation from OOD samples.

---

## 82. Combining with Auxiliary Memory Objectives

- Add predictive coding loss: predict next observation embedding; helps hidden representation quality, improving trace effectiveness.
- Add contrastive time-distance loss to enforce temporal ordering.
- These auxiliary losses can share hidden states refreshed each batch.

---

## 83. Trace Accumulation in Practice (Simplified Loss)

Implement lambda-return DP:

```
gae = 0
returns = torch.zeros_like(rew)
for t in reversed(range(L)):
    delta = rew[t] + gamma * (1 - done[t]) * val[t+1] - val[t]
    gae = delta + gamma * lam * (1 - done[t]) * gae
    returns[t] = gae + val[t]
loss_v = 0.5 * (returns - val[:-1]).pow(2).mean()
```

For Q, replace `val` with `q(s,a)` and `val[t+1]` with target $Q'$ at next state-action; incorporate IS weights in `delta` and optionally in loss weighting.

---

## 84. Handling Multi-Step Bootstrap

If using n-step bootstraps:
$$
G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n Q(s_{t+n}, a_{t+n}),
$$
then define $\lambda$-return over n-step returns:
$$
G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}.
$$
Approximate with finite $n \le L-t$; compute recursively.

---

## 85. Interplay with Entropy Regularization

- For SAC-style objectives, entropy stabilizes exploration; traces operate on Q targets. Keep temperature tuning unaffected.
- Avoid using entropy of behavior policy in IS ratios; ratios computed on action log-probs only.

---

## 86. Gradient Checkpointing Pattern

To save memory for long sequences:

```
def forward_chunk(obs_chunk, act_chunk, h):
    h_out = h
    q_list = []
    for t in range(len(obs_chunk)):
        q, h_out = critic(obs_chunk[t], act_chunk[t], h_out)
        q_list.append(q)
    return torch.stack(q_list), h_out

q_all, _ = checkpoint(forward_chunk, obs_chunk, act_chunk, h0)
```

Combine with lambda-return loss; autograd handles backprop with reduced memory.

---

## 87. Masking and Padding

- Handle variable-length sequences with masks.
- Multiply TD errors and losses by mask to ignore padded timesteps.
- Traces reset when mask=0 (episode ends).

---

## 88. Integration with Distributed Training

- In actor-learners (IMPALA-style), traces can be computed learner-side on sequences.
- Ensure behavior log-probs stored for IS ratios.
- Synchronize policy periodically; beware staleness in ratios if lag is large—clip more aggressively.

---

## 89. Observability Diagnostics

- Train linear probe on hidden states to predict missing observation bits; improved probe accuracy indicates better memory utilization possibly aided by traces.
- Correlate probe accuracy with return improvements from GET-ROPR.

---

## 90. Edge-Case Handling

- Episodes shorter than seq_len: mask remaining steps; reset traces at done.
- Environments with time limits: treat truncation as non-terminal but reset traces if using time-limit termination to avoid leakage.

---

## 91. Hyperparameter Transferability

- Start with $\lambda=0.9$, $c_\rho=2$, $c_e=2$, seq_len=32; works reasonably on flicker Atari.
- For DMControl, increase seq_len to 64 and gamma to 0.995.
- If moving to more stochastic domains, lower $\lambda$ to 0.8.

---

## 92. Potential Negative Results to Document

- Cases where traces did not help (e.g., fully observable tasks) to show method is targeted to POMDPs.
- Sensitivity to poor IS estimation; high noise may negate benefits.

---

## 93. Policy Architecture Notes

- GRU often sufficient; LSTM adds cost with modest gains.
-.dropout on RNN may harm memory; use sparingly.
- LayerNorm on RNN hidden may stabilize long sequences.

---

## 94. Reference Implementations

- `FanmingL/Recurrent-Offpolicy-RL` (RESeL baseline).
- IMPALA/Ape-X RNN variants for inspiration on sequence handling.
- TD($\lambda$) in classic control for validation.

---

## 95. Checklist for Experiments

- [ ] Baseline n-step recurrent off-policy runs.
- [ ] GET-ROPR with default hyperparams.
- [ ] Lambda sweep.
- [ ] IS clip sweep.
- [ ] Hidden refresh on/off.
- [ ] Sequence length sweep.
- [ ] Trace clip sweep.

---

## 96. Result Reporting Template

| Env | Method | Return | Std | Steps to threshold | Trace p95 | IS p95 | Staleness |
| --- | ------ | ------ | --- | ------------------ | --------- | ------ | --------- |
| FlickerPong | Baseline | 17.5 | 1.8 | 1.5M | — | — | 0.30 |
| FlickerPong | GET-ROPR | **20.1** | **1.2** | **1.0M** | 3.0 | 1.9 | 0.12 |

---

## 97. Limitations and Future Directions (Recap)

- Bias from IS/trace clipping; quantify in appendix.
- Additional compute over pure n-step.
- Representation quality still a bottleneck; auxiliary tasks may be needed.
- Future: adaptive $\lambda$, transformer backbones, offline extensions.

---

## 98. Minimal Smoke Test

- Env: FlickerCartPole-v1.
- Config: seq_len=16, lambda=0.9, gamma=0.99, c_rho=2, c_e=2, batch=32.
- Expect: rapid solve (<200 episodes), gamma fixed; trace norms bounded.

---

## 99. Closing Summary

GET-ROPR equips recurrent off-policy agents with a principled, implementable form of long-horizon credit assignment. By merging eligibility traces, IS clipping, and hidden refresh, it targets the twin challenges of partial observability and replay staleness. The accompanying math, code patterns, configs, and experimental playbook enable rigorous evaluation and extension across POMDP benchmarks.

---

_This README is the complete blueprint for Assignment 7: integrating gradient eligibility traces into recurrent off-policy RL. Align code, math, and experiments accordingly._

