# ---
# comments: True
# description: Multi-agent reinforcement learning basics, covering cooperative, competitive, and mixed settings, CTDE paradigms (MADDPG/MAPPO), value decomposition (VDN/QMIX), non-stationarity, and credit assignment.
# ---

# Multi-Agent RL

Multi-agent RL (MARL) studies multiple learners interacting in a shared environment. Interactions can be **cooperative**, **competitive**, or **mixed**, leading to non-stationary dynamics from each agent’s perspective.

## Problem Settings
- **Fully cooperative**: agents share a team reward (e.g., SMAC).
- **Fully competitive / zero-sum**: agents optimize opposing returns (e.g., two-player games).
- **General-sum / mixed**: partially aligned interests (e.g., traffic control, resource allocation).
- **Communication**: explicit messages may be allowed or learned implicitly via actions.

## Core Challenges
- **Non-stationarity**: other agents’ policies change during learning, breaking the Markov property.
- **Credit assignment**: decomposing team rewards to individuals.
- **Scalability**: joint action spaces grow exponentially.
- **Partial observability**: each agent sees only local observations.

## Centralized Training, Decentralized Execution (CTDE)
Train with access to global information, execute with local observations:
- **MADDPG**: deterministic actors per agent; centralized critics conditioned on joint observations/actions.
- **MAPPO**: PPO with centralized value functions; decentralized stochastic policies.
- **HATRPO/HAPPO**: trust-region variants with hierarchical or per-agent constraints.

## Value Decomposition
Factor team Q-values into per-agent terms to enable decentralized policies:
- **VDN**: \(Q_{tot} = \sum_i Q_i\).
- **QMIX**: monotonic mixing network ensures \(\arg\max Q_{tot}\) aligns with individual argmaxes.
- **QTRAN/QPLEX**: relax monotonicity for greater expressiveness.

## Exploration and Stability
- **Role randomization** and **parameter noise** help avoid symmetric equilibria.
- **Population-based training / self-play**: maintain opponent pools to avoid overfitting; crucial in competitive settings.
- **Entropy regularization** or intrinsic bonuses can mitigate coordination failures.

## Communication and Coordination
- Learned communication channels (e.g., CommNet, TarMAC) allow agents to share latent messages.
- **Differentiable inter-agent communication**: end-to-end training with attention or graph neural networks to propagate information.
- Regularize bandwidth or use discrete bottlenecks to prevent overfitting to dense messaging.

## Evaluation
- Report both **team return** and **individual consistency** across seeds and partner policies.
- In competitive settings, evaluate against held-out opponents and measure exploitability or NashConv.
- Measure **generalization**: performance with unseen partners or different numbers of agents.

## Practical Tips
- Start with CTDE baselines (MAPPO for discrete/stochastic, MADDPG for continuous actions).
- Normalize observations per agent; share parameters when agents are homogeneous.
- Use replay buffers with fingerprints (episode step, policy version) to reduce non-stationarity.
- Clip or bound gradient norms; tune mixing network capacity to avoid overfitting.

## References
- Lowe et al. (2017) MADDPG; Rashid et al. (2018) QMIX; Sunehag et al. (2017) VDN; Yu et al. (2021) MAPPO; Vinyals et al. (2019) AlphaStar for large-scale self-play.
