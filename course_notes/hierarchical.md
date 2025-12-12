# ---
# comments: True
# description: Overview of hierarchical reinforcement learning (HRL) with temporal abstraction, the options framework, goal-conditioned policies, and modern algorithms such as MAXQ, FeUdal, HIRO, and HAC, plus guidance on subgoal discovery and stability.
# ---

# Hierarchical RL

Hierarchical RL (HRL) introduces **temporal abstraction** to break long-horizon tasks into reusable skills. By composing high-level decisions with low-level controllers, HRL can improve exploration, credit assignment, and sample efficiency.

## Why Temporal Abstraction?
- **Long horizons**: Primitive actions make credit assignment and exploration difficult.
- **Reusable skills**: Subpolicies (options) capture common behaviors (navigate, grasp, open).
- **Sparse rewards**: Subgoals provide denser feedback and structure.

## The Options Framework
An **option** \(\omega = (I_\omega, \pi_\omega, \beta_\omega)\):
- **Initiation set** \(I_\omega\): states where the option can start.
- **Intra-option policy** \(\pi_\omega(a \mid s)\): low-level controller.
- **Termination** \(\beta_\omega(s)\): probability the option ends in state \(s\).

Planning occurs in a **semi-MDP** where actions are options. Option-value learning uses:
\[
Q(s,\omega) \leftarrow Q(s,\omega) + \alpha \big[r + \gamma^{k} \max_{\omega'} Q(s',\omega') - Q(s,\omega)\big],
\]
where \(k\) is the option duration.

### Advantage Actor–Critic with Options
- High-level policy selects options; low-level policies run until termination.
- Gradients decompose into high-level (over options) and low-level (inside an option) components.

## Goal-Conditioned HRL
### Goal-Conditioned Policies (GCPs)
- Policy \(\pi(a \mid s, g)\) conditioned on goal \(g\).
- Enables **universal value functions** \(Q(s,a,g)\) that generalize across goals.

### Hindsight Relabeling (HER)
- For sparse rewards, relabel failed trajectories with achieved states as goals.
- HRL variants (e.g., HIRO, HAC) rely on hindsight to train subgoals efficiently.

## Canonical Algorithms
- **MAXQ**: Decomposes value functions into task/subtask hierarchies; encourages reusable subtasks.
- **Options-Critic**: Learns option policies, terminations, and the high-level policy end-to-end with policy gradients.
- **FeUdal Networks (FuN)**: Manager emits goal vectors; worker maximizes intrinsic reward by matching goals.
- **HIRO**: Two-level hierarchy; high level outputs continuous subgoals; lower level maximizes intrinsic reward to reach them. Uses hindsight to stabilize learning.
- **HAC (Hierarchical Actor-Critic)**: Multi-level controllers with subgoal testing transitions to stabilize subgoal feasibility.
- **Hierarchical DQN / h-DQN**: High-level selects goals; low-level uses DQN to achieve them.

## Subgoal Discovery
- **Graph / bottleneck analysis**: Identify states that connect dense regions.
- **Diversity / empowerment**: Learn skills that maximize state-space coverage (DIAYN-style).
- **Spectral methods**: Use Laplacian or successor features to propose meaningful subgoals.
- **Human priors / demonstrations**: Seed options with expert-specified waypoints.

## Training Challenges
- **Non-stationarity**: Lower levels change while higher levels learn; mitigate with hindsight relabeling, off-policy corrections, or slower updates for higher levels.
- **Credit assignment across levels**: Use hierarchical returns or shaped intrinsic rewards for subgoals.
- **Subgoal feasibility**: Penalize unreachable goals; use subgoal testing transitions (HAC) or model-based reachability checks.
- **Termination collapse**: Regularize \(\beta_\omega\) (entropy, minimum duration) to avoid trivial options.

## Practical Tips
- Start with **two levels** (manager + worker); more levels increase instability.
- If goals are geometric, clamp subgoals to reachable radii; anneal intrinsic reward scales to balance with task reward.
- Use replay buffers per level; relabel subgoals in hindsight to keep data consistent.
- For continuous control, add a **distance-based intrinsic reward** for subgoal completion and optionally a shaping term for speed.

## References
- Sutton, Precup, Singh (1999) Options; Dietterich (2000) MAXQ; Bacon et al. (2017) Options-Critic; Vezhnevets et al. (2017) FeUdal Networks; Nachum et al. (2018) HIRO; Levy et al. (2019) HAC.
