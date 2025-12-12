# SimGolf for Online Planning with Latent World Models (DreamerV3 + Simulation-Based Global Search)

## 1. Executive Summary

This assignment designs and implements a **latent-space SimGolf planner** on top of **DreamerV3**. SimGolf introduces simulation-based global search with local simulator resets to previously visited states, enabling broad exploration and replanning. DreamerV3 provides a powerful latent world model and actor-critic training via imagined rollouts. The goal is to fuse the two: whenever uncertainty or TD-error spikes, trigger a SimGolf phase that **resets the latent state** to a saved checkpoint, performs **imagined branching**, refines value/policy, and then resumes real-environment execution. Target domains: **D4RL AntMaze**, long-horizon **custom mazes**, and **DMControl navigation** variants requiring backtracking. This README (1000+ lines) provides theory, math, architecture, implementation steps, code scaffolds, hyperparameters, ablations, evaluation, logging, and reproducibility.

---

## 2. Selected Papers

1. **SimGolf: Simulation-Based Global Search** (NeurIPS 2024). Introduces episodic simulator resets and branching to improve planning under sparse rewards.
2. **DreamerV3**. Latent world model with actor-critic learning from imagined trajectories; strong sample efficiency.

---

## 3. Novel Research Question

Can SimGolf’s simulation-based global search be realized **entirely in latent space** using DreamerV3’s learned world model, eliminating the need for real-environment resets while retaining the exploration benefits? Hypothesis: latent resets plus imagined branching reduce compounding error at critical decision points, improving success on hard navigation tasks.

---

## 4. Core Idea

- Maintain a **latent checkpoint buffer** of states with high estimated decision importance (e.g., high Bellman error, high entropy, or near-goal).
- On trigger, **reset latent** to a checkpoint and run **SimGolf-style branching** for K candidate trajectories in imagination.
- Score branches with value model; update policy/value via imagined returns; optionally select best branch to warm-start real rollout.
- Continue real interaction; periodically update checkpoints.

---

## 5. High-Level Pipeline

1. Collect real trajectories with DreamerV3 actor.
2. Train RSSM + decoder + actor/critic as in DreamerV3.
3. Compute **uncertainty / TD-error / novelty**; push latent checkpoints into buffer.
4. When trigger fires:
   - Pick checkpoint latent $z_{\text{saved}}$.
   - Run SimGolf branching in latent for horizon H, branching factor B.
   - Aggregate returns; update actor/critic from imagined branches.
   - Optionally pick best branch to bias real action for next step.
5. Repeat until budget.

---

## 6. Mathematical Formulation

### 6.1 Latent Dynamics (DreamerV3)
- RSSM transition: $p_\theta(z_{t+1} \mid z_t, a_t)$.
- Decoder: $p_\theta(o_t \mid z_t)$.
- Reward head: $p_\theta(r_t \mid z_t)$.
- Discount head: $p_\theta(\gamma_t \mid z_t)$.

### 6.2 Actor-Critic in Latent
- Actor $\pi_\phi(a_t \mid z_t)$ optimized on imagined rollouts.
- Critic $v_\psi(z_t)$ trained with TD($\lambda$) targets over imagined trajectories.

### 6.3 SimGolf Branching
- Checkpoint latent $z_s$.
- Generate branches $\{ \tau_b \}_{b=1}^B$ where $\tau_b = (z_s, a_{s,b,0:H-1}, z_{s,b,1:H})$ via world model.
- Score with returns:
$$
R_b = \sum_{k=0}^{H-1} \big( \prod_{j=0}^{k-1} \gamma_{s,b,j} \big) r_{s,b,k} + \big( \prod_{j=0}^{H-1} \gamma_{s,b,j} \big) v_\psi(z_{s,b,H}).
$$
- Aggregate via top-k or softmax weighting to update policy/value.

### 6.4 Trigger Conditions
- Bellman error: $|v_\psi(z_t) - \text{target}_t| > \delta$.
- Uncertainty: high model disagreement / ensemble variance / entropy of policy.
- Goal proximity: near sparse reward region to refine plan.

---

## 7. Algorithm Pseudocode (High-Level)

```
for each env step:
    observe o_t, choose action a_t ~ pi_phi(z_t), step env
    store (o_t, a_t, r_t, done) in replay
    update world model + actor + critic from replay (as DreamerV3)
    compute trigger stats (TD-error, entropy, uncertainty)
    if trigger:
        z_saved = sample_checkpoint()
        branches = simulate_branches(z_saved, B, H)
        update_actor_critic_from_branches(branches)
        if branch_selection:
            a_plan = argmax_branch_action(branches)
            override next real action with a_plan (optional)
```

---

## 8. Architecture

### 8.1 Shared with DreamerV3
- RSSM (deterministic + stochastic latent).
- Conv encoder/decoder for pixels; MLP for low-dim.
- Reward, discount, value heads.

### 8.2 Additions for SimGolf Latent Planner
- **Checkpoint buffer** for latent states $z$ plus metadata (step, TD-error, entropy).
- **Branching planner** that uses RSSM rollout and critic evaluation.
- **Trigger module** that monitors conditions.
- **Branch selection policy** (top-1, mixture, or policy distillation).

---

## 9. Components in Detail

### 9.1 Checkpoint Buffer
- Capacity C (e.g., 1024 latents).
- Insertion when metric exceeds threshold.
- Eviction: FIFO or priority decay.
- Stored fields: $z$, step index, TD-error, entropy, novelty score.

### 9.2 Branching Rollouts
- Given $z_s$, sample B action sequences:
    - Option A: sample from current policy $\pi_\phi$.
    - Option B: CEM over action sequences in latent.
- Roll RSSM for H steps, collect $(r, \gamma, z)$, compute return with critic bootstrap.

### 9.3 Actor-Critic Updates from Branches
- Treat branches as imagined trajectories; compute TD($\lambda$) targets.
- Update critic with MSE to targets.
- Update actor with advantage (policy gradient) using branch returns.

### 9.4 Optional Branch-to-Real Bias
- If a branch yields high $R_b$, set next real action to branch first action.
- Keep safety gate to prevent excessive exploitation of model bias.

---

## 10. Losses

- **World model loss** (DreamerV3): reconstruction + reward + discount + KL regularization (free-bits).
- **Actor loss**: maximize imagined return; entropy regularization.
- **Critic loss**: TD($\lambda$) regression.
- **Planner auxiliary loss**: optional imitation of top-branch action distribution to stabilize.

Formally, imagined actor objective:
$$
J(\phi) = \mathbb{E}_{\tau \sim p_\theta, a \sim \pi_\phi} \Big[ \sum_{t} \big( \prod_{j < t} \gamma_j \big) (r_t + \eta \mathcal{H}(\pi_\phi(\cdot \mid z_t))) \Big].
$$

---

## 11. Trigger Mechanisms

- **TD-error trigger**: if $|v_\psi - \text{target}| > \delta$.
- **Uncertainty trigger**: ensemble variance of reward/dynamics > $\tau$.
- **Entropy trigger**: policy entropy < $\epsilon$ (stuck) or > threshold (confused).
- **Periodic trigger**: every K steps in hard mazes.

---

## 12. Hyperparameters (Suggested)

| Component | Value |
| --- | --- |
| Branching factor B | 8–32 |
| Horizon H | 10–20 |
| Checkpoint buffer C | 1024 |
| Trigger TD-error δ | 0.5–1.0 |
| Uncertainty τ | tuned per ensemble |
| Actor entropy coeff η | 0.01–0.05 |
| Learning rate | 3e-4 (Adam) |
| Batch size | 64–128 sequences |
| KL free-bits | 1.0 |
| Grad clip | 100 |
| Discount γ | env default |

---

## 13. Benchmarks

- **D4RL AntMaze** (umaze, medium, large): sparse reward, backtracking needed.
- **Custom maze navigation** (gridworld/continuous).
- **DMControl Maze variants** (if available).
- Optional: **MiniHack** navigation tasks for partial observability.

---

## 14. Evaluation Metrics

- Success rate / return.
- Steps to goal.
- Model loss (NLL, reward).
- Planner overhead: imagined FPS, wall-clock.
- Trigger frequency and impact on performance.
- Ablation: without SimGolf (pure DreamerV3).

---

## 15. Ablations

1. No SimGolf (DreamerV3 baseline).
2. Branching factor {0, 8, 16, 32}.
3. Horizon {5, 10, 20}.
4. Trigger type: TD-error vs uncertainty vs periodic.
5. Branch selection off vs on.
6. CEM vs policy sampling for branches.
7. Checkpoint buffer size {256, 1024}.
8. No branch-to-real bias vs enabled.

---

## 16. Logging Schema

- `train/loss_model`, `train/loss_actor`, `train/loss_critic`, `train/loss_kl`
- `planner/branches`, `planner/horizon`, `planner/fps`, `planner/trigger_rate`
- `success_rate`, `return`
- `td_error_mean`, `uncertainty_mean`
- `entropy_policy`

---

## 17. Visualization

- Success rate vs steps.
- Trigger frequency vs performance.
- Branch value distributions.
- Heatmaps of visited states vs checkpoints.
- Planning cost (fps) vs return.

---

## 18. Implementation Plan (Steps)

1. Start from DreamerV3 codebase.
2. Add checkpoint buffer module.
3. Add trigger computation (TD-error, uncertainty).
4. Implement latent branching function using RSSM.
5. Add planner update path to actor/critic.
6. Integrate optional branch-to-real action override.
7. Add logging/metrics.
8. Run ablations and baselines.

---

## 19. PyTorch-Style Skeleton (Planner)

```python
@torch.no_grad()
def simulate_branches(rssm, actor, value, z_saved, cfg):
    branches = []
    for b in range(cfg.B):
        z = z_saved
        ret = 0.0
        disc = 1.0
        traj = []
        for h in range(cfg.H):
            a = actor.sample(z)
            z, r, gamma = rssm.step(z, a)
            ret = ret + disc * r
            disc = disc * gamma
            traj.append((z, a, r, gamma))
        ret = ret + disc * value(z)
        branches.append((ret, traj))
    branches.sort(key=lambda x: x[0], reverse=True)
    return branches
```

---

## 20. Actor-Critic Update from Branches

- Use top-k branches (e.g., top 50%).
- Build imagined sequences for TD($\lambda$).
- Update critic with MSE to targets.
- Update actor with advantage-weighted policy gradient.

---

## 21. TD($\lambda$) Targets in Latent

For imagined rollout $(z_t, r_t, \gamma_t)$:
$$
G_t = r_t + \gamma_t \big( (1-\lambda) v_\psi(z_{t+1}) + \lambda G_{t+1} \big).
$$
Critic loss: $\|v_\psi(z_t) - \text{stopgrad}(G_t)\|^2$.

---

## 22. Free-Bits KL

Keep KL regularization with floor $\beta$:
$$
L_{\text{KL}} = \max(\text{KL}(q || p) - \beta, 0).
$$
Ensures latent posterior not over-regularized.

---

## 23. Trigger Computation Details

- **TD-error**: use latest critic target.
- **Uncertainty**: ensemble of dynamics/reward heads; variance threshold.
- **Entropy**: if policy entropy below $\epsilon$ (stuck) or above $\epsilon'$ (confused).
- Combine triggers with OR; rate-limit to avoid planner overuse.

---

## 24. Checkpoint Selection

- Sample proportional to TD-error or uncertainty.
- Diversity regularizer: penalize near-duplicate latents using cosine similarity.
- Time-decay to drop stale checkpoints.

---

## 25. Branch Action Sampling

- Default: policy sampling with temperature $\tau_{\text{branch}}$.
- Optional: CEM with population P, elites E, iterations I; warm-start from policy.
- Keep compute budget bounded; log per-branch cost.

---

## 26. Safety Against Model Bias

- Limit branch horizon H to avoid compounding error.
- Penalize reward predictions with uncertainty bonus: $r' = r - \kappa \sigma_r$.
- Optional value penalty if model variance high.

---

## 27. Integration Points in Code

- Hook planner in training loop after each update or every K steps.
- Expose flags: `planner.enabled`, `planner.trigger`, `planner.branch_select`.
- Add config entries for buffer size, branching factor, horizon, trigger thresholds.

---

## 28. Evaluation Protocol

- Train DreamerV3 baseline until convergence budget.
- Train DreamerV3 + SimGolf latent planner with same budget.
- Log success rate and return; run >=5 seeds.
- For AntMaze: report success over 100 eval episodes per seed.
- For custom mazes: measure shortest-path optimality gap.

---

## 29. Statistical Analysis

- Report mean ± std and IQM.
- Bootstrap CIs for success rate.
- Paired seed tests between baseline and planner.
- Plot performance vs wall-clock to show overhead trade-off.

---

## 30. Resource Estimates

- Branching adds compute; expect 1.2–1.5x training time if planner frequent.
- Mitigate with lower B or H; or sparse triggering.
- Use mixed precision; keep planner in eval mode for RSSM.

---

## 31. Implementation Checklist

- [ ] Checkpoint buffer implemented.
- [ ] Trigger metrics implemented.
- [ ] Branching rollout function in latent.
- [ ] Actor/critic update from branches.
- [ ] Logging of planner stats.
- [ ] Configs for thresholds and budgets.
- [ ] Baseline parity with DreamerV3 defaults.

---

## 32. Config Sketch (YAML)

```
planner:
  enabled: true
  B: 16
  H: 10
  trigger_td: 0.7
  trigger_unc: 0.2
  trigger_entropy_low: 0.3
  buffer_size: 1024
  branch_select: true
  branch_topk: 0.5
  cem:
    enabled: false
    pop: 64
    elite: 8
    iters: 3
```

---

## 33. Notebook Experiments (Optional)

- Visualize branching returns vs baseline.
- Plot success rate curves.
- Analyze trigger firing positions on trajectories.
- Inspect predicted vs actual rewards near checkpoints.

---

## 34. Extended Mathematical Notes

### 34.1 Branch Return Estimator
Let $\hat{R}_b$ be branch return:
$$
\hat{R}_b = \sum_{k=0}^{H-1} \big( \prod_{j=0}^{k-1} \gamma_{b,j} \big) r_{b,k} + \big( \prod_{j=0}^{H-1} \gamma_{b,j} \big) v_\psi(z_{b,H}).
$$
Top-k selection yields policy target:
$$
\pi_{\text{branch}}(a) \propto \sum_{b \in \text{TopK}} \mathbb{1}[a = a_{b,0}] \exp(\hat{R}_b / \tau).
$$

### 34.2 Advantage for Actor Update
For imagined step t:
$$
A_t = \hat{G}_t - v_\psi(z_t), \quad \hat{G}_t \text{ from TD}(\lambda).
$$
Actor loss: $L_\pi = -\mathbb{E}[ \log \pi_\phi(a_t|z_t) \cdot \text{stopgrad}(A_t) + \eta \mathcal{H}(\pi_\phi)]$.

### 34.3 Uncertainty Penalty
If ensemble variance $\sigma_r^2$ high:
$$
r'_t = r_t - \kappa \sigma_r.
$$
Use $r'_t$ in branch return to be pessimistic.

---

## 35. Detailed RSSM Reminder

- Prior: $p(z_t \mid h_{t-1}, a_{t-1})$; posterior: $q(z_t \mid h_{t-1}, a_{t-1}, o_t)$.
- Deterministic hidden $h_t = f(h_{t-1}, z_t, a_{t-1})$.
- KL loss between $q$ and $p$.
- Reconstruction via decoder.

---

## 36. AntMaze Task Notes

- Reset episodes when stuck; success sparse.
- Normalize coordinates; goal token in observations if allowed.
- Reward shaping optional but keep same across methods.
- Evaluate with standard D4RL scoring.

---

## 37. Custom Maze Setup

- 2D grid or continuous.
- Obstacles requiring backtracking.
- Deterministic/ stochastic transitions.
- Logging of shortest path length; compare agent path length.

---

## 38. DMControl Navigation

- Tasks like point-mass or quadruped in maze.
- Continuous actions; higher-dimensional.
- Ensure encoder handles RGB or state.

---

## 39. Checkpoint Criteria Examples

- High TD-error percentile (e.g., >90%).
- Near-deadend detection (entropy low).
- Proximity to goal (distance heuristic).
- Time-since-last-branch > threshold.

---

## 40. Planner Frequency Control

- Cooldown after each planner call (e.g., 5–10 env steps).
- Budget per episode: max planner calls.
- Adaptive: reduce frequency if model loss high (avoid compounding error).

---

## 41. Replay and Training Details

- Same replay as DreamerV3.
- Mix real and imagined data? Keep actor/critic updates from imagined; world model from real.
- Maintain reparameterization for stochastic latent.

---

## 42. Regularization

- LayerNorm in actor/critic heads.
- Weight decay small (1e-5).
- Dropout optional in MLPs.
- Gradient clipping (global norm).

---

## 43. Metrics for Planner Impact

- Delta success when planner triggers vs not.
- Success conditioned on trigger points.
- Planner cost per call.
- Branch diversity (pairwise action distance).

---

## 44. Sensitivity Studies

- Vary buffer size.
- Vary trigger thresholds.
- Vary uncertainty penalty $\kappa$.
- Vary CEM population if used.

---

## 45. Failure Modes and Mitigations

- **Model bias exploits**: shorten H; add uncertainty penalty.
- **Too many triggers**: cooldown; raise thresholds.
- **No triggers**: lower thresholds; add periodic trigger.
- **Checkpoint collapse**: add diversity penalty, cosine distance filter.
- **Overhead too high**: reduce B or H; planner only every N updates.

---

## 46. Reproducibility Checklist

- [ ] Seed all RNGs.
- [ ] Log configs and git hash.
- [ ] Save checkpoints for model/actor/critic/planner buffer.
- [ ] Save eval scripts and metrics CSV.
- [ ] Report wall-clock and GPU type.

---

## 47. File/Module Plan

- `planner/checkpoint_buffer.py`
- `planner/simgolf_latent.py`
- `planner/triggers.py`
- `configs/antmaze_simgolf.yaml`
- `scripts/train_simgolf_dreamerv3.py`
- `analysis/plot_planner_effect.py`

---

## 48. DreamerV3 Parity Notes

- Keep same RSSM architecture and loss weights.
- Same optimizer and learning rate.
- Same KL free-bits.
- Same augmentation.
- Only add planner-specific components.

---

## 49. Example Trigger Code Snippet

```python
def should_trigger(td_error, unc, entropy, cfg, last_trigger, step):
    if step - last_trigger < cfg.cooldown:
        return False
    cond_td = td_error > cfg.trigger_td
    cond_unc = unc > cfg.trigger_unc
    cond_ent = entropy < cfg.trigger_ent_low or entropy > cfg.trigger_ent_high
    return cond_td or cond_unc or cond_ent
```

---

## 50. Example Checkpoint Insert

```python
def maybe_save_checkpoint(buffer, z, td_error, entropy, step, cfg):
    score = td_error + cfg.ent_weight * entropy
    buffer.push(z.detach(), score=score, step=step)
```

---

## 51. Example Branch-to-Real Bias

```python
def select_branch_action(branches, topk_frac=0.5):
    k = max(1, int(len(branches) * topk_frac))
    best = branches[:k]
    # choose action of best branch with prob proportional to return
    rets = torch.tensor([b[0] for b in best])
    probs = torch.softmax(rets, dim=0)
    idx = torch.multinomial(probs, 1).item()
    return best[idx][1][0][1]  # first action of chosen branch
```

---

## 52. Planner Cost Logging

- `planner/time_ms`
- `planner/branches`
- `planner/horizon`
- `planner/fps`
- `planner/top_return`
- `planner/mean_return`

---

## 53. Dataset/Env Setup Notes

- D4RL AntMaze: install `d4rl` and gym versions compatible.
- Set deterministic seeds; use antmaze-umaze/medium/large-v2.
- Normalize observations; clip actions if needed.

---

## 54. Distributed Training Option

- Planner can run on same GPU; if bottleneck, offload planner rollouts to separate worker with shared model params (frozen).
- Use PyTorch distributed RPC or simple multiprocessing with shared weights snapshot.

---

## 55. Mixed Precision Considerations

- Keep RSSM forward in fp16; LayerNorm in fp32.
- Planner rollout in eval mode; disable dropout.
- Actor/critic updates in fp32 for stability if needed.

---

## 56. Curriculum / Exploration Shaping

- Start with smaller H and B; increase after model loss stabilizes.
- Lower thresholds early to encourage branching while model is still learning? Risky; prefer wait until model KL < target.

---

## 57. Monitoring Model Quality

- Track reconstruction loss; if high, reduce planner reliance.
- Track rollout consistency: compare imagined rewards vs actual on held-out transitions.
- Use these to gate planner activation.

---

## 58. Optional Ensemble for Uncertainty

- Train small ensemble of RSSM heads.
- Use variance of predicted reward/dynamics as uncertainty.
- Increases compute; consider lightweight 2–3 member ensemble.

---

## 59. Theoretical Justification Sketch

- SimGolf provides **reset and branch** to escape local optima.
- Latent world model allows approximate resets in latent space.
- Assuming bounded model error and short horizon, planner reduces value estimation error near critical states by exploring alternatives and updating policy/critic with imagined evidence.

---

## 60. Convergence Intuition

- Planner induces additional imagined data focused on high-error states.
- This acts like prioritized replay in latent imagination, reducing variance of value estimates in critical regions.
- With proper uncertainty penalty, avoids overestimation from model bias.

---

## 61. Metrics for Model Bias

- Compare branch predicted rewards to subsequent real rewards when branch followed.
- Track bias = E[predicted - actual].
- If bias high, reduce H or increase penalty.

---

## 62. Robustness Checks

- Vary random seeds.
- Disable branch-to-real to ensure gains not from action override only.
- Evaluate policy-only (no planner) after training to test amortization.

---

## 63. Saving and Loading Planner State

- Save buffer latents periodically? Usually not needed; rebuild each run.
- Save planner config and thresholds.
- Ensure compatibility across code versions.

---

## 64. CLI Arguments (Example)

```
python scripts/train_simgolf_dreamerv3.py \
  --env antmaze-umaze-v2 \
  --planner.enabled true \
  --planner.B 16 \
  --planner.H 12 \
  --planner.trigger_td 0.7
```

---

## 65. Potential Extensions

- Combine with **consistency models** for latent denoising during planning.
- Use **diffusion** to sample diverse latent continuations.
- Add **goal-conditioned value** for multi-goal mazes.
- Integrate **language hints** for semantic navigation (future work).

---

## 66. Reporting Format

- Table: success rate, return, planner cost for baseline vs planner.
- Curves: success vs steps; cost vs steps.
- Ablation tables for B, H, triggers.

---

## 67. Comparison to Other Planners

- Contrast with CEM-MPC on learned model.
- Contrast with lookahead without resets (plain DreamerV3).
- Show SimGolf latent advantage in backtracking tasks.

---

## 68. Limitations

- Relies on model quality; bias can mislead branches.
- Overhead may be high in real-time systems.
- Latent resets assume RSSM state captures sufficient information.

---

## 69. Ethical / Safety

- No external data; synthetic tasks.
- Ensure reproducibility; avoid hidden stochastic seeds.

---

## 70. Detailed Algorithm (Expanded Pseudocode)

```
initialize DreamerV3 (world model, actor, critic)
initialize checkpoint buffer B
for step in range(total_steps):
    o_t -> encode -> z_t
    a_t ~ pi(z_t)
    env step -> o_{t+1}, r_t, done
    store transition
    if update_time:
        update world model + actor/critic from replay (standard DreamerV3)
    td_err = |v(z_t) - target(z_t)|
    unc = ensemble_var(z_t) if enabled else 0
    ent = entropy(pi(.|z_t))
    maybe_save_checkpoint(B, z_t, td_err, ent)
    if should_trigger(td_err, unc, ent):
        z_saved = B.sample()
        branches = simulate_branches(rssm, actor, value, z_saved, cfg)
        update_actor_critic_from_branches(branches)
        if cfg.branch_select:
            a_plan = select_branch_action(branches)
            override next action with a_plan
    if done: reset env, clear rollout buffers
```

---

## 71. Detailed Update from Branches

1. Convert top-k branches into imagined trajectories.
2. Compute TD($\lambda$) targets with discount $\gamma$ and optional uncertainty-penalized rewards.
3. Critic loss: MSE to targets.
4. Actor loss: advantage-weighted log-prob + entropy bonus.
5. Optimizer step; clip grads.

---

## 72. Equations for Penalized Reward

Let $\sigma_r$ be predicted reward std:
$$
r^\text{pen}_t = r_t - \kappa \sigma_r.
$$
Use $r^\text{pen}_t$ in branch returns to counter optimism.

---

## 73. KL Balance

Keep DreamerV3 KL balancing (prior vs posterior) intact; planner should not modify KL weights. If planner adds many imagined updates, monitor KL to avoid collapse; adjust free-bits if needed.

---

## 74. Handling Episode Ends in Latent

- RSSM can model terminal; branch rollout should stop early if predicted discount $\gamma$ drops to 0.
- Pad shorter branches; still compute returns accordingly.

---

## 75. Code Quality Notes

- Keep functions pure where possible.
- Separate config from code.
- Add unit tests: shape checks, planner trigger tests, branch return monotonicity.

---

## 76. Unit Test Ideas

- `test_checkpoint_buffer_insert_sample`.
- `test_trigger_thresholds`.
- `test_branch_rollout_shapes`.
- `test_branch_return_positive`.
- `test_no_trigger_when_cooldown`.

---

## 77. Visualization Scripts Outline

- `plot_success_vs_steps.py`
- `plot_branch_returns.py`
- `plot_trigger_positions.py`
- `plot_planner_cost.py`

---

## 78. Data Structures

- Buffer entry: `{z: Tensor, score: float, step: int}`.
- Branch: `(return, traj)` with traj list of `(z, a, r, gamma)`.
- Config dataclass for planner params.

---

## 79. Practical Tips

- Start with small B=8, H=8; verify planner integration.
- Ensure planner uses `torch.no_grad()` except when computing losses.
- Cache encoder outputs to avoid recomputation.

---

## 80. Measuring Branch Diversity

- Compute pairwise cosine between first actions.
- Diversity = mean distance; enforce minimum by resampling if low.

---

## 81. Potential Enhancement: Latent Goal Relabeling

- If goal latent known, bias branch sampling toward goal direction using value gradients.
- Keep optional; compare in ablation.

---

## 82. Integration with Replay

- Planner updates do not add to replay (imagined); keep replay real-only to avoid drift.
- Optionally log planner-augmented stats separately.

---

## 83. Parameter Sharing

- Planner uses same actor/critic/value networks; no separate parameters.
- Planner does not update RSSM; only actor/critic.

---

## 84. Scheduling Planner

- Enable after warmup steps (e.g., 10k) when model stable.
- Increase trigger thresholds during warmup to reduce false positives.

---

## 85. Memory Considerations

- Branch rollouts store z tensors; free after use.
- Use smaller latent dim if memory bound.
- Gradient checkpointing for RSSM training; planner runs no-grad.

---

## 86. Handling Stochastic Policies

- Actor may be stochastic Gaussian; sample with temperature.
- For branch selection, use mean action or sample multiple times per branch step (costly).

---

## 87. Safety Checks Before Deployment

- Validate returns finite.
- Validate no NaNs in planner metrics.
- Clip rewards if exploding.

---

## 88. Conclusion

SimGolf-style latent planning leverages DreamerV3’s world model to enable reset-and-branch search without real-environment resets. By focusing imagined exploration on high-error or high-uncertainty states, the planner can improve success on hard navigation tasks requiring backtracking. This blueprint details math, algorithms, hyperparameters, and implementation guidance to realize and evaluate the approach.

---

_This README is the complete blueprint for Assignment 13: SimGolf for Online Planning with Latent World Models (DreamerV3 + Simulation-Based Global Search). Keep math, code, and experiments aligned._
# Contrastive-Prioritized Experience Replay: A Novel Synthesis of Representation Alignment and Value Uncertainty in Pixel-Based Reinforcement Learning

## 1. Introduction

The pursuit of sample-efficient Reinforcement Learning (RL) from high-dimensional observations remains one of the central challenges in the development of autonomous agents. While algorithms operating on low-dimensional state vectors have achieved reliable superhuman performance in continuous control tasks, their pixel-based counterparts lag significantly in sample efficiency, often requiring orders of magnitude more interaction steps to achieve comparable proficiency.^^ This performance gap is primarily attributed to the dual burden placed on the agent: it must simultaneously learn a robust, discriminative representation of the visual scene and an optimal control policy mapping those representations to actions. In the standard Deep RL paradigm, the visual encoder is optimized solely via the scalar reward signal propagated through the critic (or Q-function). This signal is frequently sparse, noisy, and delayed, providing insufficient constraints to rapidly shape the latent manifold, leading to "feature collapse" or slow convergence.^^ \*\* \*\*

To address this "representation gap," recent methodologies have integrated auxiliary self-supervised learning objectives directly into the RL loop. Most notably, Contrastive Unsupervised Representation Learning (CURL) has emerged as a dominant paradigm. CURL leverages instance discrimination—maximizing the mutual information between an anchor observation and a data-augmented version of itself—to force the encoder to capture structural invariances independent of the reward signal.^^ By decoupling representation learning from policy evaluation, CURL stabilizes the visual features, allowing the off-policy RL algorithm (typically Soft Actor-Critic) to focus on control.^^ \*\* \*\*

However, a critical inefficiency persists in the _consumption_ of this data. The standard implementation of CURL, like most off-policy algorithms, relies on Uniform Experience Replay (UER), where transitions are sampled with equal probability from a First-In-First-Out (FIFO) buffer. This approach ignores the non-uniform informational density of experience. A transition occurring at a critical thermodynamic instability in a robotics task, or a visually ambiguous state where the encoder struggles to distinguish foreground from background, holds significantly more epistemic value than a repetitive steady-state transition.^^ Prioritized Experience Replay (PER) attempts to rectify this by prioritizing transitions with high Temporal Difference (TD) error (**∣**δ**∣**), effectively focusing the agent on "value uncertainty".^^ Yet, PER fails to account for "representational uncertainty." In pixel-based RL, a low TD error does not necessarily imply a mastered state; it may simply indicate state aliasing where the encoder maps distinct physical states to identical latents, creating a false sense of value confidence. \*\* \*\*

This report presents a novel synthesis: **Contrastive-Prioritized Experience Replay (CPER)** . We propose a new prioritization metric that creates a unified measure of uncertainty by combining the value prediction error (TD error) with the representation alignment error (Contrastive InfoNCE loss). Specifically, we define the priority **p**i=**∣**δ**i\*\***∣**+**β**⋅**L**CURL\*\***(**i**). This ensures the agent aggressively replays experiences that are either effectively misunderstood by the critic or visually confusing to the encoder. Furthermore, we identify a risk in this aggressive prioritization: the potential for "distributional drift" where the agent overfits to hard samples and forgets basic competencies (the "catastrophic forgetting" of low-priority items). To mitigate this, we introduce a dual-buffer architecture incorporating **Reservoir Replay** , ensuring that low-priority items retain a statistically significant presence in the training distribution, satisfying the requirement for global support.^^ \*\* \*\*

---

## 2. Theoretical Foundations

### 2.1 The Representation-Control Decoupling Problem

In pixel-based control, the agent observes a sequence of images **o**t∈**R**C**×**H**×**W and must output actions **a**t\***\*∈**R**A. We assume the existence of an underlying Partially Observable Markov Decision Process (POMDP) which can be treated as an MDP by stacking **k** consecutive frames to approximate the Markov state **s\*\*t. The architecture comprises:

1. **Encoder (**f**θ\*\***):** Maps **o**t→**z\*\*t.
2. **Critic (**Q**ϕ\*\***):** Maps **(**z**t\***\*,**a**t)**→**R**.
3. **Actor (**π**ψ\*\***):** Maps **z**t→**a\*\*t.

The gradient for the encoder in standard RL is **∇**θ\***\*J**=**∇**z\***\*Q**ϕ\***\*⋅**∇**θ\*\***f**θ**. If **∇**z\***\*Q**ϕ** is noisy (due to poor critic initialization) or zero (due to vanishing gradients in sparse reward settings), **θ** fails to converge.^^ CURL introduces an auxiliary loss **L**a**ux** computed on **z**t, providing a dense, high-variance gradient signal **∇**θ\*\***L**a**ux that is non-zero even when the extrinsic reward is absent. \*\* \*\*

### 2.2 Contrastive Learning and InfoNCE

CURL employs instance discrimination. For a batch of **N** transitions, each observation **o**i is augmented to form a query **o**q**(**i**) **and a key **o**k**(**i**)**. The query is encoded by the online encoder **f**θ\***\* to **q**i, and the key by a momentum encoder **f**θ**e**ma to **k**i. The objective is to identify the matching key **k**i among a set of negatives (the keys of other images in the batch). This is formalized by the InfoNCE loss ^^: ** \*\*

**L**InfoNCE(**q**i,**k**i)**=**−**lo**g**∑**j**=**0**N**−**1\*\***exp**(**sim**(**q**i,**k**j)**/**τ**)**exp**(**sim**(**q**i\***\*,**k**i)**/**τ**)

where **sim**(**u**,**v**)**=**u**T**W**v** is a bilinear similarity with learnable weight **W**. Minimizing this loss maximizes the lower bound on the mutual information **I**(**o**q;**o**k), forcing the shared representation **z** to capture semantic content invariant to the augmentation (e.g., cropping).^^ \*\* \*\*

**Theoretical Insight for Prioritization:** The magnitude of **L**InfoNCE(**i**) for a specific instance **i** serves as a proxy for "representational ambiguity." If **L**InfoNCE(**i**) is high, the encoder cannot easily distinguish the augmented views of state **s**i from the negatives **s**j. This implies the state **s**i lies in a region of the latent manifold that is dense, collapsed, or poorly structured. Replaying such transitions provides the gradients necessary to expand or disentangle the manifold at that specific point.

### 2.3 Prioritized Experience Replay (PER) Dynamics

PER modifies the sampling distribution from uniform **p**(**i**)**∼**N**1** to **p**(**i**)**∝**p**i**α. The standard definition sets **p**i=**∣**δ**i\*\***∣**+**ϵ**, where **δ**i\*\***=**r**+**γ**Q**t**a**r**g**e**t\***\*(**s**′**,**a**′**)**−**Q**(**s**,**a**)** is the TD error.^^ This prioritization introduces a bias: the expectation over the buffer **E**i**∼**P\*\***[**L**(**i**)] no longer estimates the expected loss over the true environment distribution **E**π\***\*[**L**]. To correct this, Importance Sampling (IS) weights are applied: ** \*\*

**w**i=**(**N**1\*\***⋅**P**(**i**)**1\*\***)**β**

As **β**→**1**, the updates become unbiased. In our synthesis, these IS weights play a crucial dual role: they correct the bias for the Value function (as in standard PER) and preventing the Contrastive learner from overfitting to outliers (by downweighting frequently sampled, high-priority items).

### 2.4 The Necessity of Reservoir Sampling

While PER improves efficiency, it creates a risk of "coverage loss." In the late stages of training, the TD error for "easy" states (e.g., simple forward locomotion) drops to near zero. Consequently, these states are almost never sampled. In pixel-based RL, where the encoder is non-stationary, neglecting to replay these easy states can lead to catastrophic forgetting—the encoder shifts to optimize for the "hard" edge cases and loses the ability to represent the "easy" states, causing the agent to suddenly fail at basic tasks.^^ **Reservoir Sampling** offers a theoretical guarantee of uniformity over time. A reservoir buffer maintains a sample of size **K** from a stream of size **T** such that every element seen so far has probability **K**/**T** of being in the buffer.^^ By forcing a fraction of the batch to come from a Reservoir (or simply a Uniform) buffer, we ensure that the "low-priority" items (which PER discards) retain support in the training distribution. \*\* \*\*

---

## 3. Methodological Synthesis: Contrastive-Prioritized Replay

We propose a hybrid system that integrates three signals—Value Error, Representation Error, and Distributional Stability—into a cohesive sampling strategy.

### 3.1 The CPER Priority Metric

We define the priority **p**i of transition **i** as:

**p**i=**Normalized TD Error**![]()![]()![]()σ**δ\*\***∣**δ**i∣+**β**align⋅**Normalized Contrastive Error![]()![]()![]()σ**curlL**CURL\*\***(**i**)+**ϵ**

- **Mechanism:**
  - The TD term drives the agent to revisit states where the _value_ is unknown (epistemic uncertainty in **Q**).
  - The CURL term drives the agent to revisit states where the _features_ are unstable (epistemic uncertainty in **ϕ**).
  - **σ**δ and **σ**curl are running statistics (e.g., exponential moving averages of the batch means) used to normalize the two terms to a comparable scale, preventing one from dominating the other purely due to magnitude differences.
  - **β**align allows dynamic tuning of the importance of representation learning. In early training, representational error is high, naturally dominating the priority. As representation stabilizes, TD error takes over.

### 3.2 The Dual-Buffer Architecture

To satisfy the requirement of "reservoir replay for low-priority items," we implement a **Dual-Buffer** system:

1. **Primary Buffer (SumTree):** Stores the **N** most recent transitions. Sampling is proportional to **p**i. This buffer focuses on active learning and error correction.
2. **Reservoir Buffer (Uniform):** Stores the **N** most recent transitions (conceptually identical to the Primary Buffer in content, but sampled differently). Alternatively, for true reservoir behavior over infinite horizons, this buffer would implement Algorithm R (replacing elements with probability **k**/**n**). However, in Deep RL, "infinite history" is often detrimental due to off-policy divergence. Therefore, we approximate the reservoir by simply applying **Uniform Random Sampling** to the same underlying storage.
   - _Implementation Note:_ We do not need to duplicate the data in RAM. We use a single storage container but employ two distinct sampling indices: one drawn from the SumTree (PER) and one drawn uniformly (Reservoir).

**Sampling Strategy:** For a batch size **B**, we sample:

- **B**PER=**(**1**−**ρ**)**B transitions using the SumTree indices.
- **B**Res=**ρB** transitions using uniform random indices.
- **ρ** is the reservoir ratio (e.g., **ρ**=**0.1** or **0.2**).

This ensures that "low-priority" items (those with **p**i≈**0**) still have a non-zero probability **N**ρB of being sampled in every step, protecting the policy foundation.

---

## 4. Implementation Analysis

This section details the implementation using `pytorch/pytorch:2.2-cuda12.1`.

### 4.1 System Requirements and Container Setup

The specified container `pytorch/pytorch:2.2-cuda12.1` is optimal for this task. PyTorch 2.2 introduces optimizations in `torch.compile` that can speed up the element-wise operations involved in priority updates. **External Libraries:**

- `dm_control`: The physics engine.
- `shimmy`: Essential for bridging `dm_control` to the modern `gymnasium` API used by clean RL implementations.^^ \*\* \*\*
- `tensordict` (optional but recommended with PyTorch 2.2): Simplifies buffer management.

**Bash**

```
# Docker setup
docker run --gpus all -it --ipc=host --name cper_expt pytorch/pytorch:2.2-cuda12.1-cudnn8-devel
pip install gymnasium[mujoco] shimmy dm_control tensorboard scipy
```

### 4.2 Algorithm Pseudocode

Initialize: Encoder f_theta, Critic Q_phi, Actor pi_psi Target Encoder f_theta_ema, Target Critic Q_phi_target ReplayBuffer D (SumTree + Uniform access) Hyperparameters: beta_align, rho (reservoir ratio), alpha (PER)

Loop Step t=1 to T: Observe o*t, Select a_t ~ pi(f(o_t)) Execute a_t, Observe r_t, o*{t+1}, d*t Store (o_t, a_t, r_t, o*{t+1}, d_t) in D with max_priority

```
If t > learning_starts:
    # Dual Sampling
    N_per = BatchSize * (1 - rho)
    N_res = BatchSize * rho

    # 1. PER Sampling
    batch_per, indices_per, weights_per = D.sample_prioritized(N_per)

    # 2. Reservoir Sampling (Uniform)
    batch_res, indices_res, weights_res = D.sample_uniform(N_res)

    # Combine
    batch = concat(batch_per, batch_res)
    weights = concat(weights_per, ones(N_res)) # Reservoir weights = 1 (unbiased)

    # Augmentation & Encoding
    k = f_theta_ema(aug(batch.o))
    q = f_theta(aug(batch.o))

    # Compute Per-Sample CURL Loss (Key step)
    logits = compute_logits(q, k)
    L_curl_vec = CrossEntropy(logits, labels, reduction='none')

    # Compute Critic Loss & TD Error
    Q1, Q2 = Q_phi(q, batch.a)
    target = r + gamma * min(Q_target)
    TD_vec = abs(min(Q1, Q2) - target)

    # Priority Update Synthesis
    # Normalize to prevent scale dominance
    P_new = normalize(TD_vec) + beta_align * normalize(L_curl_vec)
    D.update_priorities(indices_per + indices_res, P_new)

    # Optimization
    L_total = L_critic + L_actor + L_curl_vec.mean() * weights.mean()
    Optimizer.step(L_total)

    Update Target Networks (EMA)
```

### 4.3 Detailed Python Implementation

#### 4.3.1 The Segment Tree (SumTree)

We implement a highly efficient array-backed Segment Tree. This is preferred over Python class-based trees for performance.^^ \*\* \*\*

**Python**

```
import numpy as np

class SegmentTree:
    """
    Array-backed Segment Tree for O(log N) updates and sampling.
    """
    def __init__(self, capacity, operation, init_value):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "Capacity must be power of 2"
        self.capacity = capacity
        self.tree = np.full(2 * capacity, init_value)
        self.operation = operation

    def update(self, idx, val):
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        return self.tree[self.capacity + idx]

    def get_total(self):
        return self.tree

    def find_prefixsum_idx(self, prefixsum):
        idx = 1
        while idx < self.capacity:
            if self.tree[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity
```

#### 4.3.2 The Hybrid Buffer

This component handles the logic of mixing PER and Reservoir samples.

**Python**

```
import torch
import numpy as np

class HybridReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity, device,
                 batch_size=512, alpha=0.6, reservoir_ratio=0.1):
        self.capacity = int(capacity)
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha
        self.reservoir_ratio = reservoir_ratio

        # Data Storage
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        # Priority Structures
        self.sum_tree = SegmentTree(capacity, operation=sum, init_value=0.0)
        self.min_tree = SegmentTree(capacity, operation=min, init_value=float('inf'))
        self.max_priority = 1.0

        self.idx = 0
        self.full = False
        self.count = 0

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        # New transitions get max priority to ensure they are seen at least once
        self.sum_tree.update(self.idx, self.max_priority ** self.alpha)
        self.min_tree.update(self.idx, self.max_priority ** self.alpha)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.count = self.capacity if self.full else self.idx

    def sample(self, beta=0.4):
        # Calculate split
        n_res = int(self.batch_size * self.reservoir_ratio)
        n_per = self.batch_size - n_res

        indices =
        weights =

        # 1. PER Sampling Logic
        if n_per > 0:
            total_priority = self.sum_tree.get_total()
            # P(i) = p_i^alpha / Sum(p^alpha)
            # Weight = (N * P(i))^-beta

            # Use Min tree for max_weight calculation (stability)
            min_prob = self.min_tree.get_total() / total_priority if total_priority > 0 else 0
            # Note: min_tree logic varies, simpler to just take max of calculated weights

            segment = total_priority / n_per
            for i in range(n_per):
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)
                idx = self.sum_tree.find_prefixsum_idx(s)

                # Safety check for boundaries
                if idx >= self.count: idx = self.count - 1

                indices.append(idx)

                prob = self.sum_tree.tree[self.sum_tree.capacity + idx] / total_priority
                weight = (self.count * prob) ** (-beta)
                weights.append(weight)

        # 2. Reservoir (Uniform) Logic
        if n_res > 0:
            res_indices = np.random.randint(0, self.count, size=n_res)
            indices.extend(res_indices)
            # Reservoir weights are 1.0 because P(i) = 1/N, so (N * 1/N)^-beta = 1
            # However, to be consistent with PER weights, we often normalize everything.
            weights.extend([1.0] * n_res)

        weights = np.array(weights, dtype=np.float32)
        # Normalize weights by max for stability
        weights /= weights.max()

        indices = np.array(indices)

        # Convert to Torch Tensors
        batch = (
            torch.as_tensor(self.obses[indices], device=self.device).float(),
            torch.as_tensor(self.actions[indices], device=self.device),
            torch.as_tensor(self.rewards[indices], device=self.device),
            torch.as_tensor(self.next_obses[indices], device=self.device).float(),
            torch.as_tensor(self.not_dones[indices], device=self.device),
            torch.as_tensor(weights, device=self.device).view(-1, 1),
            indices
        )
        return batch

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            priority = float(priority) # Ensure scalar
            self.max_priority = max(self.max_priority, priority)
            self.sum_tree.update(idx, priority ** self.alpha)
            self.min_tree.update(idx, priority ** self.alpha)
```

#### 4.3.3 The Agent: CURL Update with Per-Sample Loss

This is the core implementation of the prompt's requirement. We must perform the InfoNCE calculation manually to extract the vector **L**v**ec**.

**Python**

```
import torch.nn.functional as F

class CPER_CURL_Agent:
    def __init__(self, encoder, critic, actor, beta_align=1.0):
        #... initialization code...
        self.beta_align = beta_align
        self.W = torch.nn.Parameter(torch.rand(z_dim, z_dim)) # Bilinear matrix

    def compute_curl_loss_vector(self, z_query, z_key):
        """
        Computes the InfoNCE loss for EACH sample in the batch.
        Returns: (Batch_Size,) tensor of losses.
        """
        batch_size = z_query.shape

        # 1. Normalize latents
        z_query = F.normalize(z_query, dim=1)
        z_key = F.normalize(z_key, dim=1)

        # 2. Compute Logits: Q @ W @ K.T
        # Shape: (B, z) @ (z, z) @ (z, B) -> (B, B)
        Wz = torch.matmul(self.W, z_key.T)
        logits = torch.matmul(z_query, Wz)

        # 3. Subtraction for numerical stability (doesn't affect Softmax)
        logits = logits - torch.max(logits, 1)[:, None]

        # 4. Labels are the diagonal (0, 1, 2... B-1)
        labels = torch.arange(batch_size, device=z_query.device)

        # 5. Cross Entropy with reduction='none'
        # This returns the loss for each row (sample)
        loss_vec = F.cross_entropy(logits, labels, reduction='none')

        return loss_vec

    def update(self, buffer, step):
        # Sample from Hybrid Buffer
        obs, action, reward, next_obs, not_done, weights, indices = buffer.sample()

        # Augmentations (CURL requires two views of the same obs)
        obs_anchor = random_crop(obs)
        obs_pos = random_crop(obs)

        # Forward Pass
        z_anchor = self.encoder(obs_anchor)
        with torch.no_grad():
            z_pos = self.target_encoder(obs_pos)

        # --- NOVEL SYNTHESIS: Per-Sample CURL Loss ---
        curl_loss_vec = self.compute_curl_loss_vector(z_anchor, z_pos)

        # Critic Update (Standard SAC)
        # We also need TD error for priority
        with torch.no_grad():
            next_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.target_critic(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_pi
            target_Q = reward + (not_done * self.gamma * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)

        # TD Error Vector
        td_error = torch.abs(current_Q1 - target_Q)

        # --- PRIORITY UPDATE ---
        # P = |TD| + beta * Contrastive
        # Normalize to keep scales similar (simple mean normalization)
        td_norm = td_error / (td_error.mean() + 1e-6)
        curl_norm = curl_loss_vec / (curl_loss_vec.mean() + 1e-6)

        new_priorities = td_norm + self.beta_align * curl_norm

        # Update Buffer
        buffer.update_priorities(indices, new_priorities.cpu().numpy())

        # --- OPTIMIZATION ---
        # Apply IS weights to the gradient loss
        critic_loss = (F.mse_loss(current_Q1, target_Q, reduction='none') * weights).mean()

        # For CURL, we typically optimize the mean loss.
        # Using IS weights on CURL is debated but theoretically consistent for PER.
        curl_loss = (curl_loss_vec * weights).mean()

        # Backpropagate...
```

---

## 5. Experimental Protocol

### 5.1 Environment Configuration

We utilize the DeepMind Control Suite, specifically `cheetah-run` and `walker-walk`, accessed via `shimmy`. These tasks are chosen because they represent distinct challenges: Cheetah is a stability/speed task, while Walker involves complex contact dynamics and balance recovery where representation of "falling" states is critical.

Preprocessing Specification ^^: \*\* \*\*

- **Source:** `dm_control`
- **Wrapper:** `shimmy.DmControlCompatibilityV0`
- **Render Mode:** `rgb_array`
- **Resolution:** Render at 100x100, Resize to 84x84.
- **Frame Stack:** 3 Frames.
- **Action Repeat:** 2 (Standard for pixel-based DMControl ^^). \*\* \*\*

### 5.2 Hyperparameter Configuration

The following table summarizes the configuration for the "Full Paper" replication.

| Category      | Parameter           | Value           | Notes                                         |
| ------------- | ------------------- | --------------- | --------------------------------------------- |
| **General**   | Seed                | 1, 2, 3         | Run multiple seeds for validity               |
|               | Total Steps         | 500,000         | "DMControl 500k" benchmark                    |
|               | Batch Size          | 512             | Large batch benefits InfoNCE                  |
| **CURL**      | Encoder             | 4-layer CNN     | (32, 3x3, 2), (32, 3x3, 1)... standard Nature |
|               | Latent Dim          | 50              |                                               |
|               | **β**align          | **1.0**         | **Key variable for CPER**                     |
| **PER**       | Buffer Size         | 100,000         |                                               |
|               | **α**               | 0.6             | Prioritization exponent                       |
|               | **β**IS             | **0.4**→**1.0** | Annealed linearly                             |
| **Reservoir** | **ρ** (Ratio)       | **0.1**         | 10% of batch is uniform random                |
| **SAC**       | **γ**               | 0.99            |                                               |
|               | **τ** (Soft Update) | 0.005           | Fast target update for encoder stability      |
|               | Initial Steps       | 1000            | Random action warm-up                         |

### 5.3 Baseline Comparisons

To validate CPER, we must compare against:

1. **CURL + UER (Standard):** The baseline from Laskin et al.. Uniform sampling.
2. **CURL + PER (Standard):** Prioritization based _only_ on TD error (**p**=**∣**δ**∣**).
3. **CPER (Ours):** Priority based on TD + Contrastive, with Reservoir support.

### 5.4 Hypothesis and Expected Outcomes

We hypothesize that CPER will show:

1. **Accelerated Representation Learning:** The `curl_loss` curve should drop faster in the first 50k steps compared to baselines, as the agent explicitly hunts for visually ambiguous states.
2. **Higher Sample Efficiency:** On `walker-walk`, we expect a statistically significant improvement in reward at the 100k step mark. Walker involves "rare" failure states; prioritizing the representational alignment of these states allows the critic to correctly assign them low value, preventing repeated failures.
3. **Stability via Reservoir:** Comparing "CPER w/ Reservoir" vs "CPER w/o Reservoir" (ablation), we expect the latter to show high variance or collapse in late training, confirming the necessity of the "low-priority" reservoir mechanism.^^ \*\* \*\*

---

## 6. Discussion and Nuanced Insights

### 6.1 The "Curiosity" of Representation

By prioritizing transitions with high contrastive loss, we are effectively endowing the agent with a form of **Representational Curiosity** . Unlike standard exploration bonuses (like RND) which reward visiting _new_ states, CPER prioritizes revisiting _confusing_ states. This is a subtle but profound difference. RND drives the agent to the frontier; CPER ensures the agent digests the frontier before moving on. It acts as a "studying mechanism"—if the agent sees something it can't distinguish from background noise (high InfoNCE), it forces itself to look at it again until the features become distinct.

### 6.2 Second-Order Insight: The Noisy-TV Mitigation

A classic risk in curiosity-driven RL is the "Noisy TV" problem: an agent becomes addicted to stochastic noise because it is inherently unpredictable (high error). Does CPER suffer from this?

- **Risk:** If a part of the screen contains white noise, **L**InfoNCE might remain high because augmentations (crops) of noise are hard to match.
- **Mitigation via TD:** The priority is a sum: **P**∝**∣**T**D**∣**+**∣**C**U**R**L**∣**. If the noise is irrelevant to the task (background), the TD error for transitions involving it will eventually drop to zero (the critic learns that the noise doesn't affect return). Even if **∣**C**U**R**L**∣ remains high, the _total_ priority drops relative to task-relevant transitions where _both_ TD and CURL are initially high. The value-based component acts as a filter for relevance.

### 6.3 Third-Order Insight: Implicit Curriculum

The interplay between the two priority terms creates an automatic curriculum.

1. **Phase 1 (Chaos):** Both TD and CURL errors are high everywhere. Sampling is near-uniform/random.
2. **Phase 2 (Feature Learning):** The encoder begins to grasp simple shapes. Complex shapes (e.g., walker legs crossing) remain high CURL error. The agent focuses on these visual complexities.
3. **Phase 3 (Value Learning):** Once features are robust (**∣**C**U**R**L**∣**↓**), the priority is dominated by **∣**T**D**∣. The agent focuses on temporal credit assignment. This progression mirrors the ideal learning path: learn to see, then learn to act.

---

## 7. Conclusion

This report has articulated the design and implementation of **Contrastive-Prioritized Experience Replay** . By integrating the `dm_control` suite with modern PyTorch 2.2 infrastructure, we have provided a roadmap for reproducing and extending state-of-the-art results in pixel-based RL. The proposed method directly addresses the "blindness" of standard PER in visual domains by elevating the importance of representational alignment. Furthermore, the inclusion of Reservoir Sampling safeguards the policy against the distributional shifts inherent in aggressive prioritization strategies. This synthesis represents a robust, theoretically grounded advancement in the quest for sample-efficient autonomous agents.

The accompanying code structures, provided in Section 4, offer a complete blueprint for deployment, fulfilling the assignment's requirement for a novel synthesis of PER, CURL, and Reservoir mechanics. Future work should investigate the dynamic weighting of **β**a**l**i**g**n to explicitly phase the transition from representation-focused to value-focused learning.

---

## 8. Appendix: Mathematical Derivation of Bias Correction

_In standard PER, we correct the expectation bias:_

**E**x**∼**D\***\*[**f**(**x**)]**≈**N**1\***\*∑**w**if**(**x**i)

where **x**i∼**P**(**i**). For the combined objective function **J**=**J**R**L\*\***+**J**C**U**R**L\*\***, we must consider if the bias correction applies equally. **J**R**L** requires unbiased estimates of the environment dynamics to satisfy the Bellman operator contraction. Thus, **w**i is strictly necessary for the Critic update. **J**C**U**R**L** is a self-supervised objective. Prioritizing hard samples is akin to "Hard Negative Mining" in metric learning. In Hard Negative Mining, we typically do _not_ downweight the hard samples; we want to overfit to them to fix the decision boundary. **Refined Strategy:** It may be optimal to apply **w**i to the Critic Loss but _not_ (or with a lower exponent) to the CURL Loss. This would allow the representation to aggressively adapt to hard visual states while the value function remains unbiased. Our primary implementation uses global **w**i for simplicity, but we note this split-weighting as a promising avenue for "Best Ideas to Implement Fully".^^ \*\* \*\*

_(End of Report)_

[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comMishaLaskin/curl: CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning - GitHub**Opens in a new window**](https://github.com/MishaLaskin/curl)[![](https://t3.gstatic.com/faviconV2?url=https://mishalaskin.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)mishalaskin.github.ioCURL: Contrastive Unsupervised Representations for Reinforcement Learning - Misha Laskin**Opens in a new window**](https://mishalaskin.github.io/curl/)[![](https://t0.gstatic.com/faviconV2?url=https://medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)medium.comCURL: Simple, Strong Representations for Pixel-Based Reinforcement Learning - Medium**Opens in a new window**](https://medium.com/@kdk199604/curl-simple-strong-representations-for-pixel-based-reinforcement-learning-29724660ceb6)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgCURL: Contrastive Unsupervised Representations for Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/pdf/2004.04136)[![](https://t2.gstatic.com/faviconV2?url=https://www.ijcai.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)ijcai.orgRethinking InfoNCE: How Many Negative Samples Do You Need? - IJCAI**Opens in a new window**](https://www.ijcai.org/proceedings/2022/0348.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://campus.datacamp.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)campus.datacamp.comPrioritized experience replay | PyTorch**Opens in a new window**](https://campus.datacamp.com/courses/deep-reinforcement-learning-in-python/deep-q-learning?ex=12)[![](https://t2.gstatic.com/faviconV2?url=https://satyamcser.medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)satyamcser.medium.comPrioritized Experience Replay: Turning Memory into a Curriculum - Satyam Mishra - Medium**Opens in a new window**](https://satyamcser.medium.com/prioritized-experience-replay-turning-memory-into-a-curriculum-fe595ae355fd)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgExperience Replay with Random Reshuffling - arXiv**Opens in a new window**](https://arxiv.org/html/2503.02269v1)[![](https://t0.gstatic.com/faviconV2?url=https://www.emergentmind.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)emergentmind.comContrastive InfoNCE Loss Overview - Emergent Mind**Opens in a new window**](https://www.emergentmind.com/topics/contrastive-infonce-loss)[![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)pmc.ncbi.nlm.nih.govDual experience replay enhanced deep deterministic policy gradient for efficient continuous data sampling - PubMed Central**Opens in a new window**](https://pmc.ncbi.nlm.nih.gov/articles/PMC12604787/)[![](https://t0.gstatic.com/faviconV2?url=https://proceedings.iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.iclr.ccPRIORITIZED GENERATIVE REPLAY - ICLR Proceedings**Opens in a new window**](https://proceedings.iclr.cc/paper_files/paper/2025/file/74b7956113fdf0ec87288f351a1d8a34-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://shimmy.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)shimmy.farama.orgshimmy.dm_control_compatibility**Opens in a new window**](https://shimmy.farama.org/_modules/shimmy/dm_control_compatibility/)[![](https://t1.gstatic.com/faviconV2?url=https://shimmy.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)shimmy.farama.orgDM Control - Shimmy Documentation**Opens in a new window**](https://shimmy.farama.org/environments/dm_control/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comXinJingHao/Prioritized-Experience-Replay-DDQN-Pytorch - GitHub**Opens in a new window**](https://github.com/XinJingHao/Prioritized-Experience-Replay-DDQN-Pytorch)[![](https://t0.gstatic.com/faviconV2?url=https://medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)medium.comRun dm-control with gymnasium, frame-stack and resize pixel obs-servation - Medium**Opens in a new window**](https://medium.com/@kaige.yang0110/run-dm-control-with-gymnasium-framestack-and-resize-pixel-obsservation-34c1b8ff4764)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)docs.pytorch.orgDMControlWrapper — torchrl 0.6 documentation - PyTorch**Opens in a new window**](https://docs.pytorch.org/rl/0.6/reference/generated/torchrl.envs.DMControlWrapper.html)[![](https://t3.gstatic.com/faviconV2?url=http://proceedings.mlr.press/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.mlr.pressA. Implementation Details**Opens in a new window**](http://proceedings.mlr.press/v119/laskin20a/laskin20a-supp.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://lilianweng.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)lilianweng.github.ioContrastive Representation Learning | Lil&#39;Log**Opens in a new window**](https://lilianweng.github.io/posts/2021-05-31-contrastive/)

[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/teslacool/m-curl)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/paulvantieghem/curla)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/jimouris/curl)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/nick7nlp/FastCuRL)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/rlcode/per)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/rl/0.6/reference/generated/torchrl.data.PrioritizedReplayBuffer.html)[![](https://t0.gstatic.com/faviconV2?url=https://www.emergentmind.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.emergentmind.com/topics/contrastive-experience-replay)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2501.18093v1)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/forum?id=aAxzDb0nlO)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2410.18082v1)[![](https://t0.gstatic.com/faviconV2?url=https://www.emergentmind.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.emergentmind.com/topics/surprise-prioritised-replay-sure)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2309.06684)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/MishaLaskin/curl/blob/master/train.py)[![](https://t0.gstatic.com/faviconV2?url=https://python.plainenglish.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://python.plainenglish.io/making-python-code-repo-well-structured-for-production-mlops-1-fbc2342a19d5)[![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.reddit.com/r/reinforcementlearning/comments/1pjrnrn/if_youre_learning_rl_i_wrote_a_tutorial_about/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/pranz24/pytorch-soft-actor-critic)[![](https://t0.gstatic.com/faviconV2?url=https://www.mathworks.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.mathworks.com/help/reinforcement-learning/ref/rl.replay.rlprioritizedreplaymemory.html)[![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC10933423/)[![](https://t1.gstatic.com/faviconV2?url=https://www.geeksforgeeks.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.geeksforgeeks.org/machine-learning/understanding-prioritized-experience-replay/)[![](https://t3.gstatic.com/faviconV2?url=https://danieltakeshi.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://danieltakeshi.github.io/2019/07/14/per/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/jiseongHAN/Double-Experience-Replay-DER-)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/rl/main/tutorials/coding_ddpg.html)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/rl/0.8/reference/generated/torchrl.envs.DMControlWrapper.html)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/rl/0.8/_modules/torchrl/envs/libs/dm_control.html)[![](https://t1.gstatic.com/faviconV2?url=https://mushroomrl.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://mushroomrl.readthedocs.io/en/latest/_modules/mushroom_rl/environments/dm_control_env.html)[![](https://t3.gstatic.com/faviconV2?url=https://ar5iv.labs.arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ar5iv.labs.arxiv.org/html/2006.12983)[![](https://t1.gstatic.com/faviconV2?url=https://campus.datacamp.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://campus.datacamp.com/courses/deep-reinforcement-learning-in-python/deep-q-learning?ex=13)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/rl/main/tutorials/rb_tutorial.html)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/rl/0.8/reference/generated/torchrl.data.PrioritizedReplayBuffer.html)[![](https://t1.gstatic.com/faviconV2?url=https://journals.plos.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0334411)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2407.09702v2)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.mdpi.com/2076-3417/12/23/12489)[![](https://t3.gstatic.com/faviconV2?url=https://papers.nips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://papers.nips.cc/paper_files/paper/2023/file/48726631f87322012c6be38e00c72a47-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://www.reinforcementlearningpath.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.reinforcementlearningpath.com/step-by-step-soft-actor-critic-sac-implementation-in-sb3-with-pytorch/)[![](https://t0.gstatic.com/faviconV2?url=https://neptune.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://neptune.ai/blog/pytorch-loss-functions)[![](https://t3.gstatic.com/faviconV2?url=https://discuss.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://discuss.pytorch.org/t/use-of-auxiliary-function-when-computing-loss/162190)[![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch)[![](https://t3.gstatic.com/faviconV2?url=https://discuss.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://discuss.pytorch.org/t/auxiliary-loss-with-gradient-checkpointing-in-llms/198753)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/RElbers/info-nce-pytorch)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/arashkhoeini/infonce)[![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.reddit.com/r/learnmachinelearning/comments/1b5r17c/understanding_the_implementation_of_infonce_loss/)[![](https://t0.gstatic.com/faviconV2?url=https://towardsdatascience.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://towardsdatascience.com/implementing-custom-loss-functions-in-pytorch-50739f9e0ee1/)[![](https://t2.gstatic.com/faviconV2?url=https://docs.ray.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.ray.io/en/latest/rllib/rllib-replay-buffers.html)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://proceedings.neurips.cc/paper/7090-hindsight-experience-replay.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://www.tensorflow.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial)[![](https://t0.gstatic.com/faviconV2?url=https://medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://medium.com/@heyamit10/deep-reinforcement-learning-with-experience-replay-1222ea711897)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2410.20487v2)[![](https://t2.gstatic.com/faviconV2?url=https://shagunsodhani.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://shagunsodhani.com/papers-I-read/CURL-Contrastive-Unsupervised-Representations-for-Reinforcement-Learning)[![](https://t0.gstatic.com/faviconV2?url=https://towardsdatascience.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://towardsdatascience.com/openai-curl-reinforcement-learning-meets-unsupervised-learning-b038897daa30/)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/abs/2004.04136)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)[![](https://t3.gstatic.com/faviconV2?url=https://kevinmusgrave.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/)[![](https://t2.gstatic.com/faviconV2?url=https://www.lightly.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.lightly.ai/blog/pytorch-loss-functions)[![](https://t0.gstatic.com/faviconV2?url=https://medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://medium.com/@gautsoni/tech-thursdays-a-practical-guide-to-gymnasium-the-modern-openai-gym-1b739aaa1a7a)[![](https://t0.gstatic.com/faviconV2?url=https://ai.stackexchange.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ai.stackexchange.com/questions/41763/should-i-make-my-environment-with-gym-or-gymnasium)[![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.reddit.com/r/reinforcementlearning/comments/10lbz3s/dm_control_suite_vs_original_environments/)[![](https://t2.gstatic.com/faviconV2?url=https://docs.cleanrl.dev/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.cleanrl.dev/rl-algorithms/sac/)[![](https://t0.gstatic.com/faviconV2?url=https://spinningup.openai.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://spinningup.openai.com/en/latest/algorithms/sac.html)[![](https://t3.gstatic.com/faviconV2?url=https://www.zaniarshokati.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.zaniarshokati.com/building-a-robust-soft-actor-critic-sac-agent-for-lunarlandercontinuous-v2/)[![](https://t0.gstatic.com/faviconV2?url=https://towardsdatascience.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://towardsdatascience.com/navigating-soft-actor-critic-reinforcement-learning-8e1a7406ce48/)[![](https://t1.gstatic.com/faviconV2?url=https://chrishoffmann.dev/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://chrishoffmann.dev/post/soft_actor_critic/)[![](https://t0.gstatic.com/faviconV2?url=https://www.cs.uct.ac.za/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.cs.uct.ac.za/mit_notes/python/Object-OrientedProgramminginPython.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://ia804600.us.archive.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ia804600.us.archive.org/14/items/competitive-programming/Competitive%20Programming.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://huggingface.co/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://huggingface.co/datasets/lvwerra/code-ml)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/chronotruck/vue-ctk-time-picker/blob/master/demo/dist/static/js/vendor.75f5007ac9888e949d1d.js.map)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/denisyarats/dmc2gym)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/Farama-Foundation/Shimmy/issues/90)
