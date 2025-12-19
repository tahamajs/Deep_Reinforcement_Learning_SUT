# Multi-Agent EfficientZero V2 with LightZero Integration (MA-EZV2)

## 1. Executive Summary

EfficientZero V2 (EZ-V2) advances sample-efficient planning by combining model-based value expansion, Gumbel MCTS search corrections, and value-prefix losses. LightZero offers a modular Monte Carlo Tree Search (MCTS) framework that supports multi-agent reinforcement learning (MARL), notably via `ma_muzero`. This assignment proposes **MA-EZV2**, a synthesis that ports EZ-V2’s algorithmic improvements (Gumbel search, value-prefix, improved dynamics/representation) into LightZero’s multi-agent stack. Goals:

- Extend EZ-V2 to multi-agent domains with joint/parameter-sharing policies.
- Preserve EZ-V2’s data efficiency under exploding joint action spaces.
- Benchmark on multi-agent environments (e.g., SMAC, MPE, Hanabi-lite) with controlled compute.

We provide a full blueprint—math, algorithms, architecture, PyTorch-style pseudocode, configs, ablations, evaluation protocols, logging schema, and reproducibility guidance—to deliver a 1000+ line roadmap for implementing MA-EZV2.

---

## 2. Background and Motivation

1. **MuZero/EfficientZero lineage:** combines learned dynamics, policy, and value in a planning loop.
2. **EZ-V2 improvements:** Gumbel MCTS search correcting value overestimation, value-prefix loss, better representation dynamics.
3. **Multi-agent challenge:** joint action space grows exponentially; coordination and credit assignment become hard.
4. **LightZero advantage:** modular policies, tree search, and `ma_muzero` support for MARL; provides infrastructure for distributed self-play and replay.
5. **Objective:** integrate EZ-V2 components into LightZero to achieve multi-agent sample efficiency and stability.

---

## 3. Core EZ-V2 Components to Port

### 3.1 Gumbel MCTS

- Uses Gumbel noise to sample top-k actions; corrects search bias.
- Incorporates corrected PUCT with Gumbel top-k selection.

### 3.2 Value Prefix Loss

- Predicts cumulative reward prefixes to stabilize value learning.
- Loss on predicted prefix vs true accumulated reward over unroll.

### 3.3 Representation & Dynamics Updates

- Improved latent dynamics with reward/value/policy heads.
- Consistency loss across unroll steps.

---

## 4. Multi-Agent Considerations

1. **Action space:** joint actions $a = (a^1,\dots,a^N)$; can be factorized or centralized.
2. **Policies:** centralized training with decentralized execution (CTDE); shared encoder; per-agent policy heads.
3. **Value:** centralized critic (joint latent) or per-agent value; MA-EZV2 favors centralized value for search.
4. **Search:** tree nodes indexed by joint state; branching via joint action enumeration or factored sampling.
5. **Credit assignment:** value-prefix can aid; optional per-agent advantage heads.

---

## 5. Notation

- Agents: $i=1..N$.
- Observations: $o_t^i$; joint state $s_t$ (if available).
- Actions: $a_t^i$; joint $a_t$.
- Rewards: $r_t$ (shared) or $r_t^i$; assume shared for primary setting.
- Discount: $\gamma$.
- Latent representation: $h_t$.

---

## 6. Model Architecture (LightZero-compatible)

1. **Encoder $f_\theta$:** maps joint observations (or concat agent obs) to latent $h_0$.
2. **Dynamics $g_\theta$:** $h_{k+1}, r_{k} = g_\theta(h_k, a_k)$.
3. **Prediction head $p_\theta$:** outputs policy logits (joint or factored) and value $v_k$ from $h_k$.
4. **Prefix head $u_\theta$:** predicts value prefix $z_k$ (cumulative reward so far).
5. **Agent factoring:** if factored, policy head outputs per-agent logits conditioned on $h_k$; joint logit via sum or product.

---

## 7. Losses (per unroll)

For unroll length $K$:
1. **Policy loss:** cross-entropy between predicted logits and target visit counts $\pi_k$ from (Gumbel) MCTS.
2. **Value loss:** MSE between predicted $v_k$ and n-step/TD target.
3. **Reward loss:** MSE between predicted $r_k$ and true reward.
4. **Prefix loss:** MSE between $z_k$ and cumulative reward prefix.
5. **Consistency loss:** between latent projections across unroll steps.
Total: $L = \sum_{k=0}^K \alpha_\pi L_\pi + \alpha_v L_v + \alpha_r L_r + \alpha_z L_z + \alpha_c L_c$.

---

## 8. Gumbel MCTS in Multi-Agent

### 8.1 Top-k Selection

- Sample Gumbel noise for joint (or factored) action logits.
- Select top-k candidates; expand children accordingly.

### 8.2 PUCT with Gumbel

For node $s$, action $a$:
$$
U(s,a) = c_{\text{puct}} \cdot P(s,a) \frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}.
$$
Gumbel noise perturbs $P$ to explore top-k; corrected selection uses adjusted $Q+U$.

### 8.3 Joint vs Factored Search

- **Joint:** enumerate joint action top-k (explodes with agents).
- **Factored:** sample per-agent top-k, combine via beam search; approximate joint top-k.
- MA-EZV2 supports both; default factored for scalability.

---

## 9. LightZero Integration Steps

1. **Create `EfficientZeroV2Policy` in `lzero/policy/`.**
2. **Reuse LightZero MCTS core**; swap in Gumbel top-k search.
3. **Add value-prefix head** and loss into policy network and training step.
4. **Support `ma_muzero` configs**: centralized encoder, per-agent policy heads; modify collectors to store joint actions.
5. **Adjust replay to store prefix targets** (cumulative reward over unroll).
6. **Update config system**: flags for gumbel_on, prefix_on, factored_search, shared_encoder.

---

## 10. Data Flow (Training)

1. Collect trajectories (self-play or env rollout) using current policy and MCTS.
2. Store $(o_{0:T}, a_{0:T-1}, r_{0:T-1})$ in replay, with agent IDs if needed.
3. Sample batch of sequences; build targets with $n$-step returns and visit count distributions from search.
4. Unroll dynamics for $K$ steps; compute losses (policy/value/reward/prefix/consistency).
5. Backprop; update params; update target network if used.
6. Periodically update Gumbel temperature and top-k.

---

## 11. Target Construction

- **Value target**: $G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n v_{t+n}$.
- **Prefix target**: $\hat{z}_t = \sum_{k=0}^{t-1} \gamma^k r_k$ (or undiscounted prefix).
- **Policy target**: normalized visit counts from MCTS at root.

---

## 12. Multi-Agent Policy Factoring

1. **Centralized encoder $h$**.
2. **Per-agent policy head**: $\pi^i(a^i|h)$.
3. **Joint prior**: $P(a) = \prod_i \pi^i(a^i|h)$.
4. **Joint value**: single $v(h)$ for all agents (shared reward setting).
5. **Execution:** decentralized sampling per agent with shared $h$.

---

## 13. Handling Joint Actions in MCTS

- **Joint expand**: enumerate top-k joint actions; suited for small discrete spaces (e.g., MPE).
- **Factored expand**: sample per-agent top-$k_i$, combine via product-of-top-k; cap total branches.
- **Action masking:** use env-specific masks to prune illegal joint actions.

---

## 14. Value Prefix in Multi-Agent

- Prefix is shared (team reward). Predict single prefix scalar.
- Optionally predict per-agent prefix if rewards differ; sum losses.

---

## 15. Replay and Sampling

- Store joint obs/actions; for factored policies, store per-agent actions.
- Sequence length: unroll K (e.g., 5–10).
- Prioritized replay optional; if used, priority on value/policy errors.

---

## 16. Hyperparameters (Suggested)

| Component | Value / Range |
| --------- | ------------- |
| Unroll K | 5–10 |
| n-step | 5 |
| c_puct | 1.25–2.5 |
| Gumbel top-k | 5–20 (env-dependent) |
| Gumbel temp | 1.0 → 0.5 anneal |
| Prefix loss weight α_z | 0.5–1.0 |
| Consistency weight α_c | 0.25–0.5 |
| LR | 1e-3 (Adam) |
| Batch size | 256–512 |
| Replay size | 1–2M |
| Discount γ | 0.99 |
| Dirichlet noise | α=0.3 (root), frac=0.25 |
| Grad clip | 10.0 |

---

## 17. Benchmarks

- **SMACv2**: 2s3z, 3s5z, 5m_vs_6m.
- **MPE**: Cooperative navigation, predator-prey.
- **Hanabi-lite / Pommerman** (optional, more complex).
- **Ablation sandbox**: small grid MARL.

---

## 18. Metrics

- Win rate / success rate.
- Episode reward.
- Sample efficiency (steps to threshold win rate).
- MCTS stats: visits, depth, value/policy entropy.
- Value-prefix error.
- Wall-clock per step; search cost per move.

---

## 19. Ablations

1. Gumbel on/off.
2. Value-prefix on/off.
3. Factored vs joint search.
4. c_puct sweep.
5. Top-k sweep.
6. Shared vs per-agent encoder.
7. Dirichlet noise on/off.
8. Consistency loss on/off.

---

## 20. Safety and Stability

- Clip gradients.
- Normalize observations.
- Warmup: start with higher Dirichlet noise to encourage exploration.
- If joint branching too large, cap expansions; fall back to factored.
- Use target network for value head to stabilize unroll.

---

## 21. Logging Schema

- `loss_total`, `loss_pi`, `loss_v`, `loss_r`, `loss_prefix`, `loss_consistency`
- `win_rate`, `reward_mean`
- `mcts/visits_mean`, `mcts/depth_mean`, `mcts/entropy`
- `search/time_per_move`, `search/topk`
- `prefix/mae`, `value/mse`

---

## 22. Visualization Plan

- Win rate curves.
_- Value-prefix error vs steps._
- MCTS visit distribution histograms.
- Top-k selected actions frequency.
- Search depth over time.
- Heatmap of per-agent action entropy.

---

## 23. Implementation Steps in LightZero

1. Add policy class `EfficientZeroV2Policy` extending base MuZero policy.
2. Implement Gumbel top-k sampler in MCTS module.
3. Add prefix head and loss to network.
4. Modify data collector to store prefix targets.
5. Update config schemas for multi-agent flags.
6. Ensure evaluator supports multi-agent rollout with decentralized execution.

---

## 24. Pseudocode: Gumbel Top-k (Joint)

```
logits = policy_logits  # [A] joint actions
g = -torch.log(-torch.log(torch.rand_like(logits)))
scores = logits + g
topk = scores.topk(k, dim=-1)
```

Use top-k actions for expansion; adjust PUCT to include top-k selection only.

---

## 25. Factored Top-k Pseudocode

```
scores_i = logits_i + gumbel_i  # per agent
topk_i = scores_i.topk(k_i)
beam = combine_cartesian(topk_i, max_beam)  # limit combined branches
```

---

## 26. Value-Prefix Target Computation

For trajectory $(r_0,\dots,r_T)$:
$$
z_t = \sum_{k=0}^{t-1} \gamma^k r_k.
$$
Use per-unroll prefix target; if long episodes, clip horizon.

---

## 27. Consistency Loss

Enforce $h_{k+1}$ close to projected $h_k'$:
$$
L_c = \| \text{sg}(h_{k+1}) - h_k' \|_2^2 + \| h_{k+1} - \text{sg}(h_k') \|_2^2,
$$
where $h_k'$ is projection of $h_k$; sg = stop-grad.

---

## 28. Training Loop Sketch

```
for batch in replay:
    h0 = encoder(obs0)
    loss = 0
    h = h0
    prefix = 0
    for k in range(K):
        pi_k, v_k, r_k = pred_head(h)
        loss += alpha_pi * ce(pi_k, pi_target[k])
        loss += alpha_v * mse(v_k, v_target[k])
        loss += alpha_r * mse(r_k, r_target[k])
        loss += alpha_z * mse(prefix, z_target[k])
        h, r_dyn = dynamics(h, a_target[k])
        prefix = prefix + gamma**k * r_target[k]
    loss += alpha_c * consistency(h_unroll)
    backprop(loss)
```

---

## 29. Distributed Training Considerations

- Self-play actors generate data; learners train centrally.
- Sync policy weights periodically; keep replay shared.
- For multi-agent, ensure each actor runs per-agent policy sampling with shared encoder.

---

## 30. Evaluation Protocol

- Fixed seeds; evaluate every N updates.
- Rollout with search (same top-k) and without (policy-only) to measure reliance on MCTS.
- Report search budget (sims per move).

---

## 31. Hyperparameter Tables (SMAC)

| Map | k | c_puct | sims | dir_noise | alpha_z | notes |
| --- | - | ------ | ---- | --------- | ------- | ----- |
| 2s3z | 10 | 2.0 | 400 | 0.3 | 0.5 | factored |
| 3s5z | 10 | 2.0 | 600 | 0.3 | 0.5 | factored |
| 5m_vs_6m | 15 | 2.5 | 800 | 0.25 | 0.7 | factored |

---

## 32. Hyperparameter Tables (MPE)

| Task | k | sims | dir_noise | alpha_z | notes |
| ---- | - | ---- | --------- | ------- | ----- |
| coop_nav | 5 | 200 | 0.25 | 0.5 | joint feasible |
| predator_prey | 10 | 400 | 0.3 | 0.6 | factored recommended |

---

## 33. Search Efficiency Tricks

- Limit sims per move; share search across symmetric agents.
- Cache policy logits; reuse in beam expansion.
- Use reduced precision for search (fp16) if stable.

---

## 34. Performance/Compute Estimates

- SMAC: sims 400–800 → ~50–150 ms/move on A100; throughput depends on map.
- MPE: lighter; <30 ms/move.
- Training: 1–2 days for full runs across maps; plan compute accordingly.

---

## 35. Ablation Readouts

- Gumbel vs no-Gumbel: compare win rate and search diversity.
- Prefix vs no-prefix: track value error, convergence speed.
- Factored vs joint: branching vs win rate trade-off.

---

## 36. Safety Checks

- Clamp rewards; normalize inputs.
- If search fails (NaN), reduce sims, k, or temp; check logits.
- Ensure legal action masks applied per agent.

---

## 37. Logging Examples (TensorBoard/W&B)

- Scalars: losses, win_rate, reward_mean, search_time, sims_per_move.
- Histograms: action selection counts, value estimates.
- Text: config, seed, map name.

---

## 38. Visualization Ideas

- Heatmap of visit counts per agent.
- Timeline of top-k actions across episode.
- Prefix prediction vs ground truth plots.
- Search depth histograms.

---

## 39. Debugging Playbook

- Low win rate: increase sims, adjust c_puct, check masks.
- Overestimation: increase value loss weight, add LCB in selection.
- Slow training: reduce sims/k, use factored search.
- Instability: lower LR, clip grads, warmup without prefix loss.

---

## 40. Comparison Baselines

- LightZero `ma_muzero` default.
- QMIX/VDA2C (for SMAC).
- MAPPO (strong policy baseline).
- Show MA-EZV2 improvements in sample efficiency and final win rate.

---

## 41. Code Structure Recommendation

- `policy/efficientzero_v2_ma.py`
- `models/ezv2_ma_net.py`
- `mcts/gumbel_topk.py`
- `configs/ma_ezv2_smac.yaml`, `ma_ezv2_mpe.yaml`
- `scripts/train_ma_ezv2.py`

---

## 42. Additional Losses (Optional)

- Entropy regularization on policy logits.
- KL to behavior prior if using behavior cloning warmup.
- Auxiliary reconstruction of observations (repr learning).

---

## 43. Warmup Strategy

- Start with smaller sims, higher noise; ramp sims up.
- Optionally pretrain encoder with BC or autoencoding.
- Delay prefix loss for first few k updates.

---

## 44. Self-Play vs Fixed Opponents

- SMAC is cooperative; self-play not needed.
- For competitive tasks, consider population-based self-play; maintain opponent pool.

---

## 45. Handling Continuous Actions

- EZ-V2 supports discrete; for continuous (rare in LightZero), discretize or use policy gradients without MCTS (out of scope).

---

## 46. Evaluation Settings

- Use same search budget in eval as train or fixed smaller budget to test policy quality.
- Deterministic eval (no Dirichlet) unless otherwise noted.

---

## 47. Potential Failure Modes

- Joint action explosion: switch to factored; reduce k.
- Prefix misalignment: ensure correct cumulative reward; off-by-one bugs.
- Value drift: add target network; increase value loss weight.

---

## 48. Reproducibility Checklist

- [ ] Seeds logged.
- [ ] Configs saved.
- [ ] Checkpoints (policy, target) saved.
- [ ] Env versions (SMAC map, MPE version) recorded.
- [ ] Search budget recorded (sims, k).

---

## 49. Statistical Reporting

- Report mean ± std over seeds.
- Use bootstrap CI for win rate.
- Provide wall-clock vs performance plots.

---

## 50. Extensions and Future Work

- Hierarchical actions: macro-actions to reduce branching.
- Curriculum on k/sims based on performance.
- Integrate latent models (world models) for lookahead.
- MARL credit assignment: per-agent value heads with shared search.

---

## 51. Closing Summary

MA-EZV2 brings EfficientZero V2’s data-efficient planning to the multi-agent realm via LightZero. By fusing Gumbel-corrected search, value-prefix stabilization, and factored policies, it aims to tame joint action explosion while retaining strong performance. This README provides the math, architecture, algorithms, configs, and experimental playbook needed to implement and evaluate the approach on SMAC and MPE benchmarks.

---

_This README is the complete blueprint for Assignment 10: integrating EfficientZero V2 into LightZero for multi-agent MCTS. Keep math, code, and experiments aligned._
# Fed-DiffORA: Ensemble-Directed Federated Diffusion Policies for Offline Reinforcement Learning

## 1. Executive Summary

The intersection of Offline Reinforcement Learning (RL) and Federated Learning (FL) represents a critical frontier for the deployment of autonomous systems in privacy-sensitive, data-siloed environments. Traditional Offline RL relies on massive, centralized datasets to learn optimal policies, a requirement that conflicts with the fragmented and private nature of real-world data in healthcare, industrial robotics, and autonomous driving. While recent advancements, notably the Federated Ensemble-Directed Offline RL Algorithm (FE-DORA) presented at NeurIPS 2024, have successfully addressed the challenges of heterogeneous data quality using ensemble methods for deterministic policies, they remain constrained by the limited expressivity of unimodal policy distributions.

This report proposes **Fed-DiffORA (Federated Diffusion-Directed Offline Reinforcement Learning)** , a novel framework that synergizes the quality-aware aggregation mechanisms of FE-DORA with the state-of-the-art generative capabilities of Diffusion Policies. By modeling the policy as a reverse diffusion process, Fed-DiffORA captures the complex, multi-modal behavior distributions inherent in heterogeneous federated datasets—behaviors that cause catastrophic mode collapse in standard Gaussian policies. Crucially, we introduce a mathematically rigorous **Score-Matching Regularized Federation** strategy to mitigate the "client drift" phenomena unique to generative models, ensuring stable convergence even when local datasets vary wildly in quality (e.g., expert demonstrations vs. random exploration).

This document provides an exhaustive theoretical derivation, a comprehensive implementation architecture including PyTorch code structures, and a detailed experimental design utilizing the D4RL benchmark. The analysis demonstrates that Fed-DiffORA not only preserves privacy but theoretically outperforms centralized baselines by effectively filtering out suboptimal data through its ensemble-directed weighting mechanism, offering a robust solution for next-generation federated autonomy.

---

## 2. Introduction: The Convergence of Privacy and Control

The rapid digitization of industrial and social infrastructure has generated vast oceans of sequential decision-making data. However, this data is rarely centralized. It resides on edge devices—hospital servers containing patient treatment histories, manufacturing robots logging assembly trajectories, and autonomous vehicles recording driving interventions.^^ The centralization of this data for training Reinforcement Learning (RL) agents is often legally prohibitive due to regulations like GDPR and HIPAA, or practically infeasible due to bandwidth constraints.^^ \*\* \*\*

### 2.1 The Federated Offline RL Paradigm

Federated Offline RL emerges as the necessary paradigm to unlock the value of this distributed data. Unlike Online RL, which requires interacting with an environment, Offline RL learns policies entirely from static datasets.^^ When combined with Federated Learning (FL), agents collaboratively train a global policy without ever sharing raw trajectory data. \*\* \*\*

However, the transition from Supervised FL to Federated Offline RL is fraught with unique pathologies. In supervised learning, data heterogeneity typically refers to label distribution skew (Non-IID features). In Offline RL, heterogeneity manifests as **Quality Heterogeneity** . One client may possess a dataset of expert demonstrations (e.g., a seasoned surgeon), while another holds data from a novice or a random exploration policy.

Standard FL algorithms like `FedAvg` fail catastrophically in this setting. Averaging the weights of a policy trained on expert data with one trained on random data does not produce a "medium" policy; it often produces a policy that fails to function entirely.^^ This is because the loss landscapes of RL policies are highly non-convex, and the average of two diverse optima is rarely an optimum itself. \*\* \*\*

### 2.2 The Limitations of FE-DORA and Deterministic Policies

The current state-of-the-art solution, FE-DORA ^^, addresses quality heterogeneity by discarding the notion of uniform averaging. Instead, it employs an **ensemble-directed approach** . The server aggregates client policies based on a locally estimated performance proxy (a Q-value estimate). Clients with "better" policies—those that achieve higher expected returns—are assigned higher weights in the aggregation. This effectively allows the global model to "distill" wisdom from experts while suppressing noise from suboptimal clients.^^ \*\* \*\*

Despite its success, FE-DORA is built upon standard actor-critic architectures (like TD3 or SAC), which parameterize policies as deterministic functions or unimodal Gaussian distributions. This is a fundamental limitation. Real-world offline datasets are often **multi-modal** . For example, in a navigation task, a human demonstrator might go left or right around an obstacle. A unimodal Gaussian policy trained on this data will learn the average—going straight into the obstacle.^^ In a federated setting, this issue is exacerbated: if Client A goes left and Client B goes right, a Gaussian global policy will fail even if both clients are experts. \*\* \*\*

### 2.3 The Case for Diffusion Policies

Diffusion models have revolutionized generative modeling and, recently, centralized Offline RL. By representing the policy as a denoising process—starting from Gaussian noise and iteratively refining it into an action—Diffusion Policies can capture arbitrary, complex distributions.^^ They have achieved state-of-the-art performance on D4RL benchmarks by effectively cloning multi-modal behaviors. \*\* \*\*

However, applying Diffusion Policies in a federated setting introduces a new challenge: **Generative Drift** . The weights of diffusion models (typically U-Nets) are extremely sensitive. Naive averaging of U-Net parameters from different clients often destroys the delicate noise-prediction capabilities, leading to broken generation.^^ \*\* \*\*

### 2.4 The Proposed Solution: Fed-DiffORA

**Fed-DiffORA** bridges this gap. It combines the _quality-aware weighting_ of FE-DORA with the _expressive power_ of Diffusion Policies, stabilized by a novel _Score-Matching Regularization_ .

**Key Contributions:**

1. **Federated Diffusion Objective:** We formulate a global objective function that minimizes a weighted score-matching loss, where weights are dynamically adjusted based on the estimated quality of local datasets.
2. **Ensemble-Directed Weighting for Generative Models:** We adapt the FE-DORA performance proxy (**J**i) to evaluate stochastic diffusion policies without environment interaction, utilizing a pessimistic federated critic.
3. **Score-Matching Regularization:** To solve the generative drift problem, we introduce a proximal term in the _function space_ (the output of the noise predictor) rather than the parameter space, ensuring local training preserves the geometric structure of the global diffusion manifold.

---

## 3. Theoretical Background and Literature Review

To ground the proposed Fed-DiffORA framework, we must rigorously examine the existing foundations of FE-DORA and Diffusion-based RL.

### 3.1 FE-DORA: Federated Ensemble-Directed Offline RL

FE-DORA ^^ identifies the core failure mode of `FedAvg` in RL: the inability to distinguish between "good" and "bad" data sources. In standard FL, the update rule for global parameters **θ** at round **t** is: \*\* \*\*

**θ**t**+**1**=**i**=**1**∑**N\***\*∑**j\***\*∣**D**j\*\***∣**∣**D**i\*\***∣\***\*θ**i**t**+**1\*\***

This implicitly assumes that all data points are equally valuable. FE-DORA modifies this by introducing a learnable weight **w**i derived from a performance proxy **J**i:

**θ**t**+**1**=**i**=**1**∑**N\***\*w**i**t\*\***θ**i**t**+**1\*\*\*\*

**w**i**t\*\***=**∑**j\***\*∣**D**j∣**exp**(**β**J**j**t)**∣**D**i\***\*∣**exp**(**β**J**i\*\*t)

Here, **β** is a temperature parameter controlling the "greediness" of the selection. As **β**→**∞**, the algorithm performs "Winner-Takes-All," selecting only the single best client. As **β**→**0**, it recovers standard `FedAvg`.^^ \*\* \*\*

**The Performance Proxy (**J**i\*\***):** Crucially, **J**i cannot be the online reward (since training is offline). FE-DORA estimates **J**i using the local critic **Q**ϕ(**s**,**a\*\*):

**J**i=**E**s**∼**D**i**[**Q**ϕ(**s**,**π**θ**i**(**s**))]

The paper proves that under certain assumptions, this weighting scheme maximizes the likelihood of the global policy converging to the optimal behavior policy present in the ensemble.^^ \*\* \*\*

### 3.2 Diffusion Policies in Reinforcement Learning

Diffusion Probabilistic Models (DPMs) learn the data distribution **p**(**x**) by reversing a forward diffusion process that gradually adds noise. In the context of RL, the "data" is the action **a** conditioned on the state **s**.^^ \*\* \*\*

**Forward Process:** Given an action **a**0 from the dataset, we generate a sequence of noisy actions **a**1,**…**,**a**K using a fixed variance schedule **β**k:

**q**(**a**k∣**a**k**−**1\***\*)**=**N**(**a**k\***\*;**1**−**β**ka**k**−**1\***\*,**β**kI**)\*\*

**Reverse Process (The Policy):** The policy **π**θ(**a**0∣**s**) is defined as the reverse Markov chain:

**p**θ(**a**k**−**1\***\*∣**a**k,**s**)**=**N**(**a**k**−**1\***\*;**μ**θ(**a**k,**k**,**s**)**,**Σ**k\*\*\*\*)

The mean **μ**θ is parameterized by a noise prediction network **ϵ**θ(**a**k,**k**,**s**), typically a U-Net or MLP. The training objective is to minimize the simplified Evidence Lower Bound (ELBO), often called the **Denoising Score Matching** loss:

$$
\mathcal{L} *{\text{diff}}(\theta) = \mathbb{E}* {(s, a_0) \sim \mathcal{D}, k \sim \mathcal{U}, \epsilon \sim \mathcal{N}} \left[ |

| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_k}a_0 + \sqrt{1-\bar{\alpha}_k}\epsilon, k, s) ||^2 \right]
$$

where **α**ˉ**k\*\***=**∏**i**=**1**k\*\***(**1**−**β**i\*\*\*\*).

**Policy Improvement:** To ensure the diffusion model generates _optimal_ actions (not just copies the behavior policy), recent works like **Diffusion-QL** ^^ and **EDP** ^^ inject Q-value guidance. This is done either by: \*\* \*\*

1. **Gradient Guidance (Classifier-Guidance):** Adjusting the mean during sampling using **∇**a\***\*Q**(**s**,**a**)\*\*.
2. **Weighted Regression:** Weighting the diffusion loss by the advantage **exp**(**Q**(**s**,**a**)**−**V**(**s**))**.

### 3.3 The Challenge of Federating Diffusion Models

Federating diffusion models presents unique stability challenges. Unlike convex models where the average of parameters corresponds to an average of functions, neural networks are non-convex.

Research into **Federated Diffusion** ^^ indicates that direct parameter averaging of U-Nets works only if the clients are initialized from the same seed and remain in the same "basin of attraction." If clients drift too far apart (due to heterogeneous data), averaging their weights results in a "frankens-model" that fails to predict noise correctly. \*\* \*\*

Snippet ^^ suggests an alternative: **Cooperative Sampling** , where clients exchange gradients of energy functions or intermediate noise scores. However, this requires high communication bandwidth. Fed-DiffORA aims to solve this using **Parameter Averaging with Regularization** , which is more bandwidth-efficient. \*\* \*\*

---

## 4. The Fed-DiffORA Framework

This section presents the novel mathematical formulation of **Fed-DiffORA** . We define the problem, the global objective, and the component algorithms.

### 4.1 Problem Formulation

We consider a federated system with **N** clients.

- **Client State:** Each client **i** holds a dataset **D**i=**{(**s**,**a**,**r**,**s**′**)}. The data is drawn from a behavior policy **π**β**,**i\*\*\*\* which varies across clients (Quality Heterogeneity).
- **Objective:** Learn a global policy **π**θ**∗ **that maximizes expected return **J**(**π**) across the union of environments, without sharing **D**i\*\*\*\*.
- **Constraint:** Clients can only communicate model parameters **θ** and scalar metrics **J**i.

### 4.2 The Global Objective: Quality-Weighted Score Matching

We propose a novel global objective that modifies the standard diffusion loss. Instead of treating all data points as equal, we weight the score-matching loss by the **Performance Proxy** **w**i(**ϕ**) derived from the federated critic.

$$
\min_\theta \mathcal{L} *{Global}(\theta) = \sum* {i=1}^N w_i(\phi) \cdot \mathbb{E}_{\mathcal{D}_i} \left[ |

| \epsilon - \epsilon_\theta(a_k, k, s) ||^2 \right] + \lambda \mathcal{R}_{Reg}
$$

Here, **w**i(**ϕ**) is the ensemble-directed weight:

**w**i(**ϕ**)**∝**exp**(**β**⋅**J**^**i\***\*(**ϕ**,**D\*\*i))

This objective ensures that the global noise prediction network **ϵ**θ learns primarily from the clients whose data leads to high-return policies, effectively "filtering" the noise from suboptimal clients from the training signal.

### 4.3 Component 1: The Federated Ensemble Critic

To calculate **J**^**i**, we need a robust estimator of value. Since we are in the offline setting, we must account for **Out-of-Distribution (OOD) Action Overestimation** . If a client's policy suggests an action never seen in the dataset, a standard Q-function might assign it an arbitrarily high value.

We employ **Conservative Q-Learning (CQL)** ^^ adapted for federation. Each client trains a local critic **Q**ϕ**i** by minimizing: \*\* \*\*

**L**Q**,**i\***\*(**ϕ**i)**=**2**1\***\*E**D**i**[**(**Q**ϕ**i**(**s**,**a**)**−**y**^)**2**]**+**α**CQ**L\***\*(**E**a**∼**μ**(**s**)[**Q**ϕ**i**(**s**,**a**)]**−**E**a**∼**π**β**,**i**[**Q**ϕ**i**(**s**,**a**)]**)\*\*

where **μ**(**s**) is a wide distribution (approximating the OOD space).

**Federated Critic Aggregation:** Critics are aggregated using standard weighted averaging based on dataset size (not quality), as the critic needs to evaluate _all_ potential actions accurately, not just good ones.

**ϕ**g**l**o**ba**l**t**+**1\*\***=**i**=**1**∑**N\*\***∣**D**t**o**t**a**l\***\*∣**∣**D**i\***\*∣\*\***ϕ**i**t**+**1\*\*

**The Performance Proxy Calculation:** Once the global critic **ϕ**g**l**o**ba**l is received, client **i** evaluates its _own_ dataset quality.

**J**^**i\*\***=**M**1m**=**1**∑**M\***\*Q**ϕ**g**l**o**ba**l\*\*\*\***(**s**m\***\*,**a**^**m\***\*)**,**a**^**m\*\***∼**π**θ**i\*\*\*\***(**⋅**∣**s**m\***\*)**

By using the _global_ critic to evaluate the _local_ policy, we ensure a standardized ruler. If Client A has a local critic that is optimistic, and Client B has one that is pessimistic, comparing their self-evaluated **J** values would be flawed. The global critic mitigates this.

### 4.4 Component 2: Score-Matching Regularization (SMR)

To prevent client drift and stabilize the averaging of U-Net weights, we introduce **Score-Matching Regularization (SMR)** . Standard `FedProx` adds a penalty **∣∣**θ**i\*\***−**θ**g**l**o**ba**l∣**∣**2\*\* to the loss. However, for deep generative models, parameter distance is a poor proxy for functional similarity.

We propose regularizing the **Score Output** . The local training loss becomes:

$$
\mathcal{L} *{Local, i}(\theta_i) = \mathcal{L}* {\text{diff}}(\theta_i) + \lambda_{prox} \mathbb{E}_{s, k, \epsilon} \left[ |

| \epsilon_{\theta_i}(z_k, k, s) - \epsilon_{\theta_{global}}(z_k, k, s) ||^2 \right]
$$

- **z**k: The noisy action input.
- **ϵ**θ**g**l**o**ba**l**: The output of the frozen global model received at the start of the round.

**Insight:** This term forces the local model to preserve the "vector field" of the global model. It allows the local model to learn new modes (from local data) but penalizes it for "forgetting" the modes learned by the global consensus. This aligns the gradients of different clients, ensuring that when their parameters are averaged, the resulting model remains valid.

### 4.5 Component 3: The Diffusion-Directed Aggregation

The server performs the final aggregation of the actor networks.

1. **Receive:** Parameters **θ**i, Proxy **J**i, Dataset size **N**i.
2. **Normalize Weights:**
   **w**~**i\*\***=**N**i⋅**exp**(**β**⋅**J**i)\*\*

   **w**i=**∑**j\***\*w**~**jw**~\*\*i

3. **Aggregate:**
   **θ**g**l**o**ba**l**t**+**1\*\***=**i**=**1**∑**N\*\***w**i\*\***θ**i**t**+**1\*\*

**Soft vs. Hard Selection:**

- **Soft Selection (Finite **β**):** The global model is a weighted average. This is equivalent to an ensemble model where the ensemble members are merged into a single network.
- **Hard Selection (**β**→**∞**):** The server selects only the parameters of the best client. This is useful in extreme heterogeneity where "averaging" is destructive.

---

## 5. Mathematical Analysis

### 5.1 Convergence of Weighted Score Matching

We analyze why weighting the score matching loss by **w**i leads to the optimal policy. Let **p**d**a**t**a\*\***(**a**∣**s**)**=**∑**i\*\***α**ip**i\***\*(**a**∣**s\*\*) be the mixture distribution of the federated data. Standard FedAvg minimizes $D*{KL}(p*{data} |

| p*\theta)$. Fed-DiffORA minimizes $D*{KL}(p\_{weighted} |

| p\_\theta)$, where **p**w**e**i**g**h**t**e**d\*\***=**∑**i\***\*w**i\***\*p**i\*\*.

If **w**i is chosen such that **w**i≈**0** for suboptimal policies **p**i, then **p**w**e**i**g**h**t**e**d** approximates the distribution of only the expert policies. Thus, the diffusion model learns to approximate the _expert_ manifold, effectively "ignoring" the suboptimal data present in the federation.

### 5.2 Stability Analysis of SMR

Let **Δ**θ**=**θ**i−**θ**g**l**o**ba**l**. The SMR term can be approximated via Taylor expansion as: $$ |

| \epsilon*{\theta_i} - \epsilon*{\theta\_{global}} ||^2 \approx \Delta \theta^T \mathbf{F} \Delta \theta $$ where **F** is the Fisher Information Matrix of the diffusion model. This shows that SMR acts as a **Riemannian Regularizer** . It applies strong constraints on parameters that have high curvature (affect the output significantly) and weak constraints on parameters that are redundant. This is superior to Euclidean regularization (FedProx), which penalizes all parameters equally, potentially hindering learning in flat directions of the loss landscape.

---

## 6. Implementation Architecture

This section details the software architecture required to implement Fed-DiffORA, assuming a PyTorch-based stack.

### 6.1 Class Hierarchy

The implementation is structured around three core classes: `DiffusionPolicy`, `FederatedClient`, and `FederatedServer`.

#### 6.1.1 The Diffusion Policy Network (U-Net)

We utilize a 1D Conditional U-Net, adapted from.^^ \*\* \*\*

**Python**

```
import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

## MA-EZV2 (CA10) — Multi-Agent EfficientZero V2

This folder contains code and documentation for MA-EZV2, a LightZero-compatible multi-agent adaptation of EfficientZero V2 integrating Gumbel top-k search and a value-prefix loss for stabilizing training in cooperative multi-agent environments.

### Contents
- `models/` — network implementations (`MAEZV2Network`) with encoder, dynamics, prediction and prefix heads.
- `mcts/` — latent-space MCTS with support for joint/factored Gumbel top-k expansions.
- `policy/` — a thin `EfficientZeroV2Policy` adapter combining the model and search helpers.
- `integration/` — LightZero-style adapter classes to plug into higher-level training pipelines.
- `configs/` — default YAML configuration for experiments.
- `scripts/` — small demo and training skeletons.
- `tests/` — unit tests for network shapes, policy losses, and basic MCTS behavior.
- `report.tex` / `report_neurips.tex` — LaTeX manuscript drafts describing the algorithm and experiments.

---

## Installation

Requirements (tested on Python 3.10+):

- torch
- numpy

Create and activate a venv, then install required packages:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy
```

No other heavy dependencies are required to run the code in this folder (it is deliberately lightweight for assignment/demonstration purposes).

## Quick usage

Run the small demo script which exercises network inference and a short MCTS run:

```bash
python scripts/demo_run_ma_ezv2.py
```

Run unit tests (recommended before modifying code):

```bash
python -m pytest -q
```

## Design notes

- The network uses a centralized encoder with optional per-agent policy heads (factored outputs). Dynamics consumes concatenated per-agent one-hot segments when using factored action spaces.
- MCTS supports both joint enumeration (when joint action spaces are small) and a factored beam approximation using per-agent Gumbel top-k followed by beam merging.
- The value-prefix head predicts cumulative prefix returns and is included as an auxiliary stabilization loss.

## Reproducibility & report

- The folder contains LaTeX sources `report.tex` and `report_neurips.tex` suitable for editing and compiling with your local LaTeX installation.
- The README and configs include hyperparameter suggestions; experimental scripts are lightweight skeletons that need to be adapted and scaled for full benchmarks (SMACv2, MPE).

## Contributing

If you want to extend MA-EZV2:

1. Add unit tests under `tests/` for new components.
2. Keep modules import-safe (no side-effects at import time).
3. Update `report.tex` with new experiments and add reproducibility details (seeds, hardware, runtime).

## Citation

If you use MA-EZV2 in research, please cite the accompanying technical report and supporting references in `references.bib` provided in this directory.

---

_For questions or issues, open a PR or email the maintainer listed in the repo._
```

#### 6.1.2 The Federated Client

The client handles local training with the specific losses defined in Section 4.

**Python**

```
class FedDiffORAClient:
    def __init__(self, client_id, dataset, device='cuda', config=None):
        self.client_id = client_id
        self.dataset = dataset # D4RL Dataset
        self.device = device
        self.actor = ConditionalUnet1D(action_dim, state_dim).to(device)
        self.critic = DoubleQCritic(state_dim, action_dim).to(device)
        self.global_actor = copy.deepcopy(self.actor) # For SMR
        self.global_critic = copy.deepcopy(self.critic) # For J_i calc

        # Hyperparameters
        self.lambda_prox = config.get('lambda_prox', 0.01)
        self.eta = config.get('cql_alpha', 1.0)

    def compute_performance_proxy(self):
        """
        Calculates J_i using the Global Critic and Local Actor.
        """
        self.actor.eval()
        self.global_critic.eval()

        # Sample subset
        batch = self.dataset.sample(1000)
        states = torch.tensor(batch['observations']).to(self.device)

        with torch.no_grad():
            # DDIM Sampling (Fast)
            actions = self.actor_sample(states, steps=10)
            q1, q2 = self.global_critic(states, actions)
            min_q = torch.min(q1, q2)
            J_i = min_q.mean().item()

        return J_i

    def train_step(self, batch):
        """
        Single training step with SMR.
        """
        self.actor.train()

        # Unpack batch
        obs = batch['observations']
        actions = batch['actions']

        # 1. Sample Noise
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (actions.shape,), device=self.device)

        # 2. Add Noise (Forward Process)
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)

        # 3. Predict Noise
        noise_pred = self.actor(noisy_actions, timesteps, local_cond=obs)

        # 4. Global Reference (For SMR)
        with torch.no_grad():
            target_noise_pred = self.global_actor(noisy_actions, timesteps, local_cond=obs)

        # 5. Losses
        loss_diff = F.mse_loss(noise_pred, noise)
        loss_smr = F.mse_loss(noise_pred, target_noise_pred)

        total_loss = loss_diff + self.lambda_prox * loss_smr

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
```

#### 6.1.3 The Federated Server

The server manages the weighted aggregation.

**Python**

```
class FedDiffORAServer:
    def __init__(self, num_clients, config):
        self.global_actor = ConditionalUnet1D(...)
        self.beta = config['beta'] # Temperature

    def aggregate(self, client_updates):
        """
        client_updates: List of tuples (state_dict, J_i, num_samples)
        """
        total_weight_denom = 0
        weights =

        # 1. Calculate Unnormalized Weights
        for _, J_i, N_i in client_updates:
            # Shift J_i for numerical stability
            w = N_i * math.exp(self.beta * J_i)
            weights.append(w)
            total_weight_denom += w

        # 2. Normalize
        weights = [w / total_weight_denom for w in weights]

        # 3. Weighted Averaging
        avg_state_dict = {}
        for key in client_updates.keys():
            avg_state_dict[key] = sum(weights[i] * client_updates[i][key] for i in range(len(weights)))

        self.global_actor.load_state_dict(avg_state_dict)
```

---

## 7. Experimental Design

To validate Fed-DiffORA, we must design an experiment that specifically targets the weaknesses of current methods (heterogeneity) and highlights the strengths of diffusion (multi-modality).

### 7.1 Dataset: The Federated D4RL Benchmark

We utilize the D4RL (Datasets for Deep Data-Driven RL) benchmark.^^ However, D4RL provides monolithic datasets. We must artificially partition them to simulate a federated environment. \*\* \*\*

**The "Mixed-Quality" Partitioning Strategy:** We create a scenario with 10 clients (**N**=**10**).

- **Group Expert (2 Clients):** Assigned partitions of the `halfcheetah-expert-v2` dataset. These represent high-performing agents.
- **Group Medium (4 Clients):** Assigned partitions of `halfcheetah-medium-replay-v2`. These represent decent but suboptimal agents.
- **Group Random (4 Clients):** Assigned partitions of `halfcheetah-random-v2`. These represent noise/garbage data.

**The Challenge:** The global policy must learn to perform at the level of the Expert clients.

- If we use `FedAvg`, the 4 Random and 4 Medium clients will drag the Expert performance down (Catastrophic Degradation).
- If we use `FE-DORA` (Deterministic), it might identify the experts, but the policy will struggle with the multi-modality in the `medium-replay` data (which contains both good and bad fragments).
- **Fed-DiffORA** should ideally identify the experts (via **J**i) and model the complex distribution effectively.

### 7.2 Baselines

| Baseline             | Policy Type   | Aggregation      | Why compare?                                             |
| -------------------- | ------------- | ---------------- | -------------------------------------------------------- |
| **FedAvg-TD3**       | Deterministic | Size-Weighted    | Standard FL baseline. Shows failure of naive avg.        |
| **FedProx-TD3**      | Deterministic | Proximal-Reg     | Shows if standard regularization fixes drift.            |
| **FE-DORA**          | Deterministic | Quality-Weighted | The direct competitor. Tests Diffusion vs Deterministic. |
| **FedAvg-Diffusion** | Diffusion     | Size-Weighted    | Tests if FE-DORA weighting is needed for Diffusion.      |
| **Fed-DiffORA**      | Diffusion     | Quality-Weighted | **Ours.**                                                |

### 7.3 Hyperparameters

- **Diffusion Steps (**K**):** 100 (Training), 10 (Inference via DDIM).
- **Optimizer:** AdamW, LR = **3**×**1**0**−**4.
- **Batch Size:** 256 per client.
- **Rounds:** 100 Communication Rounds.
- **Local Epochs:** 5 per round.
- **Temperature (**β**):** 0.5 (Annealed to 2.0).
- **Proximal Weight (**λ**p**ro**x\*\***):\*\* 0.01.

### 7.4 Evaluation Metrics

1. **Normalized Return:** **100**×**expert**−**random**score**−**random\*\*\*\*. Evaluated on the standard MuJoCo environments.
2. **Drift Distance:** **∣∣**θ**i\*\***−**θ**g**l**o**ba**l∣**∣**2\*\* (Parameter Space) vs. $D\_{KL}(\pi_i |

| \pi\_{global})$ (Function Space). 3. **Client Weight Evolution:** Tracking the evolution of **w**i for Expert vs. Random clients. We expect **w**E**x**p**er**t→**1** and **w**R**an**d**o**m→**0**.

---

## 8. Anticipated Results and Discussion

Based on the theoretical analysis, we project the following outcomes:

### 8.1 Superiority over FE-DORA

In environments like `AntMaze` or `Kitchen`, where the optimal trajectory involves complex maneuvering that cannot be captured by a single mean action, FE-DORA (using TD3/SAC) will plateau at a sub-optimal score. Fed-DiffORA, by using diffusion, will successfully model the multi-modal expert distribution.

### 8.2 Resilience to Data Poisoning

The "Random" clients in our experimental setup effectively act as data poisoners. `FedAvg-Diffusion` will likely fail to converge or produce a policy that generates pure noise, as the weights of the U-Net are corrupted by the random gradients. Fed-DiffORA will assign near-zero weights to these clients after the first few rounds (once the critic identifies low **J**i), effectively neutralizing the attack.

### 8.3 The Role of SMR

We anticipate that without Score-Matching Regularization (`Fed-DiffORA w/o SMR`), the algorithm will be unstable. The "Expert" clients, although weighted heavily, might diverge in their internal feature representations during local training. Averaging them would then degrade performance. SMR ensures they remain compatible, making the aggregation step constructive rather than destructive.

---

## 9. Conclusion

This report has detailed the design of **Fed-DiffORA** , a robust framework for Federated Offline Reinforcement Learning. By identifying the limitations of current state-of-the-art methods—specifically the expressivity bottleneck of FE-DORA and the stability bottleneck of Federated Diffusion—we have engineered a solution that addresses both.

The integration of **Ensemble-Directed Weighting** ensures that the global model mimics the best available experts, ignoring the noise of the masses. The introduction of **Score-Matching Regularization** ensures that this mimetic process remains mathematically stable across the non-convex landscape of diffusion models.

As industries move towards decentralized autonomy—from fleets of self-driving cars learning from rare disengagements to distributed surgical robots learning from top doctors—Fed-DiffORA provides the necessary algorithmic infrastructure to learn safe, high-performance, and privacy-preserving policies.

---

## 10. Future Directions

1. **Differential Privacy (DP):** While Fed-DiffORA protects raw data, model updates can still leak information. Integrating DP-SGD into the local diffusion training loop is a natural next step, though the noise sensitivity of diffusion models makes this non-trivial.
2. **Communication Efficiency:** Transmitting full U-Nets is costly. Applying Low-Rank Adaptation (LoRA) or Knowledge Distillation (where the server trains a student model on synthetic data generated by client teachers) could reduce bandwidth by orders of magnitude.
3. **Asynchronous Federation:** Real-world devices do not update in perfect lockstep. Extending Fed-DiffORA to handle asynchronous updates/staleness is crucial for deployment on edge networks.

^^ \*\* \*\*

[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccFederated Ensemble-Directed Offline Reinforcement Learning - NIPS papers**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/0b99315234cc95e6ef281f9155b68832-Paper-Conference.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://bearhw.ece.vt.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)bearhw.ece.vt.eduFEDORA: Practical Federated Recommendation Model Learning Using ORAM with Controlled Privacy - Virginia Tech**Opens in a new window**](https://bearhw.ece.vt.edu/content/dam/bearhw_ece_vt_edu/publications/2025_ASPLOS_FedRec.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://federated-learning.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)federated-learning.orgFL@FM-NeurIPS&#39;24 - The Federated Learning Portal**Opens in a new window**](https://federated-learning.org/fl@fm-neurips-2024/)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgFederated Ensemble-Directed Offline Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/html/2305.03097v2)[![](https://t0.gstatic.com/faviconV2?url=https://www.emergentmind.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)emergentmind.comDiffusion Policies in Offline RL - Emergent Mind**Opens in a new window**](https://www.emergentmind.com/topics/diffusion-policies)[![](https://t2.gstatic.com/faviconV2?url=https://wnzhang.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)wnzhang.netDiffusion Models for Reinforcement Learning - Weinan Zhang**Opens in a new window**](https://wnzhang.net/teaching/sjtu-rl-2024/slides/15-diffusion-rl.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://www.roboticsproceedings.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)roboticsproceedings.orgDiffusion Policy: - Robotics**Opens in a new window**](https://www.roboticsproceedings.org/rss19/p026.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netDiffusion Federated Dataset | OpenReview**Opens in a new window**](https://openreview.net/forum?id=1GCWcrZTX8)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netSteering Diffusion Policies with Value-Guided Denoising - OpenReview**Opens in a new window**](https://openreview.net/pdf?id=gKZtkg9k3I)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netEfficient Diffusion Policies for Offline Reinforcement Learning - OpenReview**Opens in a new window**](https://openreview.net/pdf?id=0P6uJtndWu)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgTraining Diffusion Models with Federated Learning - arXiv**Opens in a new window**](https://arxiv.org/html/2406.12575v1)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comreal-stanford/diffusion_policy: [RSS 2023] Diffusion Policy Visuomotor Policy Learning via Action Diffusion - GitHub**Opens in a new window**](https://github.com/real-stanford/diffusion_policy)[![](https://t1.gstatic.com/faviconV2?url=https://di-engine-docs.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)di-engine-docs.readthedocs.ioD4RL (MuJoCo) - DI-engine&#39;s documentation! - Read the Docs**Opens in a new window**](https://di-engine-docs.readthedocs.io/en/latest/13_envs/d4rl.html)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netFederated Ensemble-Directed Offline Reinforcement Learning | OpenReview**Opens in a new window**](https://openreview.net/forum?id=XdSYtriYfI)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2511.08922] Diffusion Policies with Value-Conditional Optimization for Offline Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/abs/2511.08922)[![](https://t1.gstatic.com/faviconV2?url=https://www.tensorflow.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)tensorflow.orgd4rl_mujoco_halfcheetah | TensorFlow Datasets**Opens in a new window**](https://www.tensorflow.org/datasets/catalog/d4rl_mujoco_halfcheetah)

[![](https://t0.gstatic.com/faviconV2?url=https://nips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://nips.cc/virtual/2024/papers.html)
