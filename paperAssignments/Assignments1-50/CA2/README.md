# CrossQ with Sophia Optimization: High-UTD Continuous Control Without Target Networks

## 0. TL;DR
Replace Adam with the second-order Sophia optimizer in CrossQ to push Update-To-Data (UTD) ratios beyond 1 while maintaining stability. Batch Normalization (BN) plus Sophia’s curvature-aware scaling should allow fast, target-network-free TD learning on MuJoCo tasks with >5k reward in <1M steps.

## 1. Source Papers and Repos
- **CrossQ: Batch Normalization in Deep Reinforcement Learning** (ICLR 2024) — GitHub: https://github.com/adityab/CrossQ  
- **Sophia: A Scalable Stochastic Second-Order Optimizer** (ICLR 2024) — GitHub: https://github.com/Liuhong99/Sophia  

## 2. Research Gap / Novel Question
CrossQ stabilizes bootstrapping without target networks via BN, but Adam gradients become unstable as UTD rises (≥5), limiting sample-efficiency gains. Hypothesis: Sophia’s diagonal Hessian preconditioning will (a) keep value-scale drift bounded under BN, (b) enable UTD 10–20 without divergence, and (c) accelerate convergence on hard continuous-control benchmarks.

## 3. Mathematical Formulation
### 3.1 CrossQ Critic Loss (single critic; extend to twin critics if desired)
\[
\mathcal{L}_{\text{CrossQ}}(\theta)
= \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}
\Big[\big(Q_\theta(s,a)-y\big)^2\Big],
\\
y = r + \gamma \max_{a'} Q_\theta(s', a')
\]
BN is applied to concatenated current/next batches to share statistics.

### 3.2 Sophia-G Update (per-parameter)
Let \(g_t = \nabla_\theta \mathcal{L}_t\), \(h_t \approx \text{diag Hessian}\) (EMA).
\[
 m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,
\quad
 h_t = \beta_2 h_{t-1} + (1-\beta_2) \hat{h}_t
\]
\[
 \theta_{t+1} = \theta_t - \eta \cdot
\text{clip}\!\left(\frac{m_t}{\max(\gamma h_t, \epsilon)}, \delta\right)
\]
where \(\hat{h}_t\) is a mini-batch Hessian-diagonal estimator (e.g., squared gradients), \(\gamma\) scales curvature, and \(\delta\) clips extreme steps. BN stats must stay consistent during Hessian estimation; freeze or synchronize running means during \(\hat{h}_t\) computation.

### 3.3 UTD and BN Stability
- UTD = gradient steps per env step. BN keeps scale invariant across concatenated \([s, s']\); Sophia’s curvature term counteracts BN-induced non-stationarity at high UTD.

## 4. Algorithm Outline
1. Sample batch \(B = \{(s,a,r,s')\}\).
2. Build concatenated tensors: \(X = [s; s']\) → BN → features.
3. Critic forward → \(Q_\theta(s,a)\), target \(y\).
4. Compute loss \(\mathcal{L}\); backprop to get \(g_t\).
5. Update Hessian diag estimate \(\hat{h}_t\); apply Sophia step.
6. Actor (TD3-style): delay policy update; policy loss \(-\mathbb{E}_{s} Q_\theta(s, \pi_\phi(s))\).
7. Repeat `UTD` times per env step; periodically update BN running stats from mixed (s, s') mini-batches.

## 5. Implementation Blueprint (PyTorch)
- **Base repo:** fork `adityab/CrossQ`.
- **Key file touchpoints:**
  - `agent.py` or equivalent: inject Sophia optimizer class; expose `--utd`, `--bn_momentum`, `--sophia_gamma`, `--sophia_eps`, `--sophia_clip`.
  - `networks.py`: ensure BN layers accept concatenated batches (`(2N, feature_dim)`), and support `eval()` freeze when estimating Hessian diag if needed.
  - `replay_buffer.py`: unchanged; but ensure sampler can yield large batches for stable BN stats (e.g., batch 1024).
- **New module:** `optim/sophia.py`
  - Implements Sophia-G with EMA of gradient and Hessian diag.
  - Supports mixed-precision safe ops; uses `torch.clamp` for stability.
- **Logging:** track `||g||`, `||h||`, BN running mean/var drift, and UTD × loss curves.

## 6. Experimental Protocol
### 6.1 Environments
- MuJoCo v4: Humanoid-v4, Ant-v4, Walker2d-v4.
- Control suite alternative: DMControl Cheetah/Walker for cross-check.

### 6.2 Hyperparameters (starting grid)
- `utd`: {1, 5, 10, 20}
- Batch size: 1024
- Replay size: 1e6
- Discount γ: 0.99
- Policy delay: 2 (if TD3-style)
- BN momentum: 0.05
- Sophia: `beta1=0.965`, `beta2=0.99`, `gamma=0.1`, `eps=1e-12`, `clip=1.0`, `lr_critic=3e-4`, `lr_actor=3e-4`

### 6.3 Ablations
- Adam vs Sophia at UTD ∈ {1,5,10,20}
- BN on/off (expect failure without BN at UTD>1)
- Hessian estimation frequency: every step vs every k=2/4 steps
- BN stats handling: train-mode vs frozen during Hessian pass

### 6.4 Metrics
- Episode return (mean, IQM) vs env steps
- Divergence rate (NaN/inf in Q or loss blow-up)
- Gradient and Hessian norms
- Wall-clock to 5k Humanoid reward

## 7. Engineering Notes
- **BN + Hessian:** When estimating \(\hat{h}_t\), run BN in eval or synchronize stats to avoid double-drift.
- **Mixed precision:** Prefer `torch.cuda.amp` for forward/backward; keep Hessian accumulations in FP32.
- **Replay freshness:** Large UTD benefits from prioritized or recency-biased sampling; optional.
- **Target networks:** Keep disabled (CrossQ design); if instability remains at UTD 20, allow a slow-EMA target as a safety fallback.

## 8. Docker / Reproducibility
Use a pinned CUDA image:
```
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install mujoco==3.1.5 gymnasium[mujoco]==0.29.1 wandb hydra-core==1.3.2
```
Seed all RNGs; log config + git SHA; run ≥3 seeds per setting (prefer 5) and report IQM with bootstrap CIs (`rliable`).

## 9. Success Criteria
- Stable training at UTD 10 on Humanoid-v4 without target networks.
- ≥5k reward within 1M env steps; improvement over CrossQ+Adam at same UTD.
- No BN/stat explosions; finite losses across runs.

## 10. Checklist
- [ ] Implement `SophiaG` optimizer.
- [ ] Integrate optimizer switch & hyperparams CLI.
- [ ] Ensure BN uses concatenated (s,s') batches; add unit test for stat consistency.
- [ ] Add logging for grad/Hessian norms and BN drift.
- [ ] Run ablations (Adam vs Sophia; UTD sweep).
- [ ] Aggregate results with IQM + CIs; produce table + learning curves.

