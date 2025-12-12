# ---
# comments: True
# description: Meta-reinforcement learning principles, including context-based methods (PEARL), recurrent RL^2, gradient-based adaptation (MAML, RL^2-MAML), and evaluation considerations.
# ---

# Meta-RL

Meta-reinforcement learning trains agents that **adapt quickly** to new tasks drawn from a distribution \(\mathcal{T}\). Instead of maximizing return on a single task, the objective is to minimize expected regret after limited interaction on unseen tasks.

## Problem Setup
- Task distribution \(\mathcal{T}\) with transitions \(P_T\) and rewards \(r_T\).
- Meta-training optimizes parameters to produce fast adaptation when exposed to a few episodes or trajectories from a new task.
- Metrics: **few-shot return**, sample efficiency during adaptation, and robustness to task shift.

## Paradigms
### Recurrent / Context-Based (RL\(^2\))
- Embed history \(h_t = (s_0,a_0,r_0,\ldots,s_t)\) via RNN; the hidden state acts as an **implicit belief** over the task.
- Single policy \(\pi_\theta(a_t \mid s_t, h_t)\) adapts on the fly without gradient updates.
- Works well for stationary but partially observed task identity; needs rich training diversity to generalize.

### Latent Context Inference (e.g., PEARL)
- Learn a latent variable \(z\) that summarizes task identity from a small context set of transitions.
- Policy conditions on \(z\): \(\pi(a\mid s, z)\). Inference network \(q_\phi(z\mid \text{context})\) is trained with amortized variational inference.
- Encourages **structured uncertainty** and explicit exploration targeted at reducing task uncertainty.

### Gradient-Based Meta-Learning (MAML)
- Learn initialization \(\theta\) such that a few inner-loop gradient steps on task \(T\) produce high-performing parameters \(\theta'\).
- Inner update: \(\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_T(\theta)\).
- Outer loop optimizes \(\theta\) via gradients through the inner step(s).
- RL variants: TRPO/PPO-style surrogates, value-based MAML, and off-policy extensions (e.g., Reptile, Meta-SAC).

### Exploration-Aware Meta-RL
- Explicitly optimize **adaptive exploration**: first-episode behavior should gather informative data for rapid improvement.
- Use information gain bonuses, trajectory-level objectives, or planner-in-the-loop adaptation (MBML).

## Practical Considerations
- **Task diversity**: Meta-training must cover the variation expected at test time; otherwise, adaptation overfits.
- **Stability**: Gradient-through-gradient can be unstable; use value baselines, KL penalties, or truncated backprop.
- **Credit assignment**: For RL\(^2\), ensure sequences are long enough for hidden states to carry task info; reset hidden states between tasks.
- **Off-policy data**: Replay buffers with task labels enable sample-efficient meta-updates; importance sampling or conservative value estimation mitigates bias.

## Evaluation
- Few-shot return after \(k\) adaptation steps/episodes.
- Robustness to task shift (out-of-distribution tasks).
- Ablations: adaptation horizon, context size, latent dimension, and exploration temperature.
- Compute adaptation speed vs. asymptotic performance to ensure no regression on easy tasks.

## References
- Wang et al. (2016) RL\(^2\); Finn et al. (2017) MAML; Rakelly et al. (2019) PEARL; Gupta et al. (2018) Meta-World benchmark; Zintgraf et al. (2020) VariBAD.