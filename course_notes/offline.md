# ---
# comments: True
# description: Offline (batch) reinforcement learning concepts, dataset shift challenges, and practical algorithms such as BCQ, BRAC, CQL, IQL, and TD3+BC for stable learning without environment interaction.
# ---

# Offline RL

Offline RL learns policies **without further environment interaction**, using a fixed dataset \(\mathcal{D} = \{(s,a,r,s')\}\) collected by some behavior policy \(\mu\). This setting is powerful when exploration is unsafe or expensive, but introduces severe **distribution shift**: learned policies may query out-of-distribution (OOD) actions/states where value estimates are unreliable.

## Key Challenges
- **Extrapolation error**: Q-functions overestimate values for actions rarely or never seen in \(\mathcal{D}\).
- **Distribution shift**: Policy improvement moves away from \(\mu\), compounding error in bootstrapped targets.
- **Coverage**: If \(\mathcal{D}\) lacks support for optimal actions, performance is upper-bounded by the best in-data behavior.

## Core Approaches
### Conservative / Penalized Q-Learning
- **CQL**: Adds a conservative term to push Q-values down on OOD actions:
\[
\mathcal{L}_{\text{CQL}} = \mathcal{L}_{\text{TD}} + \alpha \Big(\mathbb{E}_{a \sim \pi}[Q(s,a)] - \mathbb{E}_{a \sim \mu}[Q(s,a)]\Big),
\]
approximated via sampled actions. Reduces overestimation and prevents greedy policies from exploiting spurious high Q.
- **IQL**: Avoids explicit behavior cloning by regressing Q-values toward expectile values of returns, then distilling a policy via advantage-weighted regression. Stable and performant without importance sampling.

### Behavior-Regularized Policy Improvement
- **BCQ**: Trains a generative model (VAE) of behavior actions; policy selects near-behavior actions and uses a perturbation model for refinement.
- **BRAC**: Penalizes divergence between \(\pi\) and \(\mu\) (KL or MMD) during actor updates.
- **TD3+BC / AWR / AWAC**: Adds behavior-cloning or advantage-weighted regression terms to the actor loss to keep actions close to the dataset support while still improving return.

### Model-Based Offline RL
- Learn a dynamics model, generate imagined rollouts **only near the data distribution** (short horizons, uncertainty penalties).
- Use **conservative value estimation** on model rollouts (e.g., MOPO/MOReL) by adding penalties for model-uncertain states.

## Practical Tips
- Start with **TD3+BC** or **IQL** for continuous control baselines; they balance simplicity and robustness.
- Monitor divergence: action likelihood under the behavior model or KL to \(\mu\).
- Calibrate uncertainty: ensembles for Q-functions or dynamics; penalize high-variance predictions.
- Limit policy updates (small step sizes, target networks) and prefer **short rollout horizons** if using models.
- Normalize rewards and observations; ensure replay buffers are **frozen** to avoid on-policy contamination.
- Evaluate with multiple seeds; offline metrics (e.g., behavior cloning loss, estimated return) are weak—deploy cautiously.

## When to Use Offline RL vs. IL
- If returns are available in the dataset and coverage is adequate, offline RL can outperform pure imitation by improving over suboptimal experts.
- If demonstrations are narrow and high quality, **BC or DAgger-style IL** may be safer; combine with conservative RL to further improve.

## References
- Fujimoto et al. (2019) BCQ; Kumar et al. (2019) BRAC; Kumar et al. (2020) CQL; Kostrikov et al. (2021) IQL; Fujimoto & Gu (2021) TD3+BC; Yu et al. (2020) MOPO; Kidambi et al. (2020) MOReL.
