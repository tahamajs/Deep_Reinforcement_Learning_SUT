# Distributional State-Corrected Action Suppression (Dist-SCAS): A Unified Framework for Risk-Sensitive Offline Reinforcement Learning

## 1. Introduction: The Convergence of Safety, Distribution, and Dynamics

The domain of Offline Reinforcement Learning (RL) stands at a pivotal intersection of statistical learning theory, optimal control, and generative modeling. The fundamental promise of offline RL—the ability to learn optimal policies from static, previously collected datasets without the risks or costs of online interaction—is tempered by the formidable challenge of distributional shift. When an agent attempts to maximize a value function learned from finite data, it inevitably encounters the "extrapolation error," where the value estimator erroneously assigns high utility to actions or states outside the support of the training distribution (Out-of-Distribution, or OOD). This phenomenon, often referred to as the "deadly triad" when combined with bootstrapping and function approximation, necessitates mechanisms that enforce conservatism.

Traditional approaches to enforcing conservatism have largely bifurcated into two distinct schools of thought: **value-based regularizers** , such as Conservative Q-Learning (CQL) or Implicit Q-Learning (IQL), which penalize Q-values for OOD actions ^^; and **policy-constraint methods** , such as Behavior Cloning (BC) or TD3+BC, which explicitly tether the learned policy to the behavior policy **π**β.^^ While effective, these scalar-value approaches often suffer from a reductionist view of uncertainty. They typically conflate _epistemic uncertainty_ (uncertainty due to lack of data) with _aleatoric uncertainty_ (inherent stochasticity in the environment). Consequently, a scalar penalty may suppress a highly promising action simply because the environment is noisy, or conversely, fail to penalize a confident but erroneous extrapolation in a sparse-data regime. \*\* \*\*

This report proposes and details the implementation of a novel, comprehensive framework: **Distributional State-Corrected Action Suppression (Dist-SCAS)** . This method synthesizes the geometric state-correction capabilities of the recently proposed SCAS framework ^^ with the statistical robustness of Distributional Reinforcement Learning, specifically focusing on lower-quantile maximization.^^ \*\* \*\*

The SCAS framework addresses a critical gap in prior literature: the distinction between OOD actions and OOD states. While CQL effectively suppresses OOD actions on known states, it provides no guarantee that a valid action will not transition the agent into an OOD region of the state space where the value function is undefined. SCAS introduces a model-based regularizer to align the agent's induced transition dynamics with a "Value-Aware State Transition" distribution derived from the dataset.^^ \*\* \*\*

Concurrently, Distributional RL posits that modeling the full distribution of returns, **Z**(**s**,**a**), rather than merely the expectation **Q**(**s**,**a**), provides a richer signal for decision-making.^^ By optimizing a policy to maximize a lower quantile (e.g., the 10th percentile or **τ**=**0.1**) of the return distribution, the agent naturally avoids actions with heavy-tailed downside risks—often a proxy for OOD uncertainty—without the need for the computationally expensive negative sampling required by CQL.^^ \*\* \*\*

Dist-SCAS represents a unified theory of "Safety through Geometry and Statistics." By rigorously deriving the mathematical interaction between the SCAS state-correction term and the distributional Bellman operator, this report offers a blueprint for an algorithm that is not only robust to OOD actions and states but also capable of disentangling risk from opportunity. The following sections provide an exhaustive analysis of the theoretical underpinnings, a detailed derivation of the requisite loss functions, a comprehensive guide to code implementation including network architectures and hyperparameter selection, and a strategic validation plan utilizing the D4RL benchmark suite.

---

## 2. Theoretical Foundations and Literature Synthesis

To understand the necessity and mechanics of Dist-SCAS, one must first deconstruct the limitations of current state-of-the-art methods and the specific theoretical advances offered by SCAS and Distributional RL.

### 2.1 The Pathology of Extrapolation Error in Scalar Offline RL

The core objective of standard RL is to maximize the expected cumulative discounted return:

**J**(**π**)**=**E**τ**∼**π\*\***[**t**=**0**∑**∞γ**t**r**(**s**t,**a**t)**]

In the offline setting, we only have access to a dataset **D**=**{(**s**,**a**,**r**,**s**′**)} sampled from a behavior policy **π**β. Standard off-policy algorithms like Soft Actor-Critic (SAC) or TD3 approximate the Q-function by minimizing the Bellman error:

**L**Q(**θ**)**=**E**(**s**,**a**,**r**,**s**′**)**∼**D\***\*[**(**Q**θ\***\*(**s**,**a**)**−**(**r**+**γ**Q**θ**′(**s**′**,**π**(**s**′**)))**)**2**]

The pathology arises in the maximization step **π**(**s**)**=**arg**max**aQ**θ\*\***(**s**,**a**)**. Since the Q-function is trained only on samples from **π**β\*\***, its estimates for actions **a**∈**/**supp**(**π**β\*\***)\*\* are unconstrained and often arbitrarily high due to function approximation noise. The maximization operator exploits these errors, leading the policy towards OOD actions with hallucinated high values.

**Conservative Q-Learning (CQL)** addresses this by adding a regularizer that minimizes Q-values for OOD actions while maximizing them for data actions.^^ \*\* \*\*

**L**CQ**L\*\***(**θ**)**=**α**(**E**s**∼**D**,**a**∼**μ**(**a**∣**s**)[**Q**(**s**,**a**)]**−**E**s**∼**D**,**a**∼**π**β(**a**∣**s**)[**Q**(**s**,**a**)]**)**+**L**B**e**ll**man\*\***

While effective, CQL introduces a complex hyperparameter **α** (the penalty weight) and requires sampling from an approximate inverse distribution **μ**(**a**∣**s**), which is computationally expensive and difficult to tune. Furthermore, CQL operates on the expectation **E**[**Q**], ignoring the variance or shape of the value distribution, which contains critical information about safety.

### 2.2 The Case for Distributional Reinforcement Learning

Distributional RL replaces the scalar Q-function **Q**(**s**,**a**) with a random variable **Z**(**s**,**a**) representing the full distribution of returns. The Distributional Bellman Operator **T**π is defined as:

**Z**(**s**,**a**)**=**D**R**(**s**,**a**)**+**γ**Z**(**s**′**,**A**′**)**where**A**′**∼**π**(**⋅**∣**s**′**)**,**s**′**∼**P**(**⋅**∣**s**,**a**)**

The divergence between standard and distributional RL is profound in the offline setting.

| Feature              | Scalar RL (e.g., CQL, IQL)      | Distributional RL (e.g., QR-DQN, TQC)                 |
| -------------------- | ------------------------------- | ----------------------------------------------------- |
| **Value Estimate**   | Mean:**E**[**Z**(**s**,**a**)]  | Full Distribution:**P**(**Z**(**s**,**a**)**≤**z**)** |
| **Uncertainty**      | Ignored or Implicit             | Explicitly modeled via variance/quantiles             |
| **Risk Sensitivity** | Risk-Neutral                    | Configurable (e.g., CVaR, Lower Quantile)             |
| **OOD Handling**     | Requires explicit penalty terms | Naturally handled via pessimistic quantile selection  |

**Lower Quantile Maximization:** Recent works like Lower Expectile Q-learning (LEQ) ^^ and Truncated Quantile Critics (TQC) ^^ demonstrate that maximizing a lower quantile (e.g., the **α**-quantile where **α**≈**0.1**) acts as a robust pessimism operator. If an action is OOD, the epistemic uncertainty regarding its outcome results in a high-variance return distribution. Consequently, the lower tail of this distribution drops significantly. By maximizing the lower tail, the agent naturally prefers actions where the outcome is both high-value and certain (low variance), effectively filtering OOD actions without explicit negative sampling.^^ \*\* \*\*

### 2.3 The SCAS Framework: State Correction vs. Action Suppression

While Distributional RL addresses value uncertainty, it does not strictly prevent the agent from drifting into OOD states via valid actions—a subtle but critical distinction. **State Correction and Action Suppression (SCAS)** ^^ argues that OOD actions are merely the symptom; the disease is the OOD state visitation. \*\* \*\*

SCAS unifies state correction and action suppression by forcing the learned policy's induced transition distribution **P**π**(**s**′**∣**s**) to align with a "target" in-distribution (ID) transition **P**v**a(**s**′**∣**s**). This target distribution is derived from the dataset but re-weighted to favor high-value transitions. The SCAS objective is: $$ \max\_\pi \mathbb{E} _{s \sim \mathcal{D}} [Q(s, \pi(s))] - \lambda D_ {KL}(P^\pi(\cdot|s) |

| P\_{va}(\cdot|s)) $$ The key insight from SCAS research is that by regularizing the _outcome_ (next state) rather than just the _input_ (action), the algorithm becomes robust to dynamics shifts and "delusional" policies that exploit model errors.^^ \*\* \*\*

---

## 3. Mathematical Framework of Dist-SCAS

Dist-SCAS merges these philosophies. We define the objective not as maximizing the expected return subject to state constraints, but as maximizing the **risk-adjusted return** (lower quantile) subject to **geometric state-correction constraints** .

### 3.1 The Distributional Bellman Operator under State Correction

Let the return distribution be approximated by a set of **N** quantiles, **θ**=**{**θ**1\*\***,**…**,**θ**N}**, such that **q**τ**i**(**s**,**a**)**=**θ**i**. The standard distributional projection minimizes the Wasserstein distance (approximated by Quantile Huber Loss) between the current distribution and the target distribution **Y\*\*:

**Y**(**s**,**a**)**=**r**+**γ**Z**(**s**′**,**π**(**s**′**))

In Dist-SCAS, the policy **π** used in the target calculation is the **Distributional-SCAS Policy** , which we denote **π**d**sc**a**s**.

### 3.2 The Risk-Adjusted Objective

We define the risk-adjusted value **Q**r**i**s**k\*\***(**s**,**a**)** as the Condition Value at Risk (CVaR) at level **α\*\*:

**Q**r**i**s**k\*\***(**s**,**a**)**=**CVaR**α\*\***(**Z**(**s**,**a**))**=**E**[**Z**(**s**,**a**)**∣**Z**(**s**,**a**)**≤**F**Z**−**1\*\***(**α**)]\*\*

In a discrete quantile approximation with sorted quantiles **q**1≤**q**2≤**⋯**≤**q**N, this is efficiently computed as the mean of the lowest **k**=**⌊**α**N**⌋ quantiles:

**Q**r**i**s**k\*\***(**s**,**a**)**≈**k**1\*\***i**=**1**∑**k\***\*q**τ**i**(**s**,**a**)\*\*

This term replaces the standard **Q**(**s**,**a**) in the policy improvement step.

### 3.3 The Value-Aware State Transition Distribution (**P**v**a**)

To implement the SCAS regularizer, we must rigorously define **P**v**a\*\***(**s**′**∣**s**). Following the SCAS literature ^^, **P**v**a** is a distribution supported on the dataset **D** that assigns higher probability to transitions leading to high-value states. ** \*\*

**P**v**a\*\***(**s**′**∣**s**)**∝**P**D(**s**′**∣**s**)**exp**(**β**V**(**s**′\*\*))

However, we do not have access to the true manifold of **P**D. We approximate this using a **Conditional Variational Autoencoder (CVAE)** or a **Diffusion Model** trained on the weighted dataset. Let **f**d**y**n(**s**,**a**) be the environment dynamics (approximated by a learned model). The SCAS loss minimizes the distance between the predicted next state and the manifold of **P**v**a**.

The explicit SCAS loss derived for this framework is:

**L**SC**A**S(**π**)**=**E**s**∼**D**,**a**∼**π**(**⋅**∣**s**)[**s**t**a**r**g**e**t**′∼**P**v**a****(**⋅**∣**s**)**min****∥**f**d**y**n(**s**,**a**)**−**s**t**a**r**g**e**t**′∥**2**2]\*\*

This form essentially pulls the action **a** such that the resulting state **f**d**y**n(**s**,**a**) lies close to high-value, in-distribution transitions.

### 3.4 The Unified Dist-SCAS Optimization Problem

Combining the components, the optimization problem for the policy parameters **ϕ** is:

**\phi^\* = \arg\max*\phi \mathbb{E}*{s \sim \mathcal{D}} \left**

Crucially, the regularization parameter **λ** is not a static hyperparameter. In Dist-SCAS, we propose an **adaptive uncertainty-weighted regularization** . We utilize the inter-quartile range (IQR) of the return distribution, **Δ**Z(**s**,**a**)**=**q**0.75\*\***−**q**0.25\*\*, as a proxy for aleatoric uncertainty.

**λ**(**s**,**a**)**=**λ**ba**se⋅**(**1**+**σ**(**Δ**Z\*\***(**s**,**a**)))\*\*

This novel formulation ensures that state correction is enforced more strictly when the value estimate is uncertain or the environment is noisy, bridging the gap between value-based and model-based constraints.

---

## 4. Implementation Strategy and Architecture

Implementing Dist-SCAS requires a sophisticated architecture that handles quantile regression, policy optimization, and dynamics modeling simultaneously. This section details the necessary components, network structures, and algorithmic flow.

### 4.1 Network Architectures

The system is composed of three primary neural entities: the **Distributional Critic** , the **Actor** , and the **State-Correction Model** .

#### 4.1.1 The Distributional Critic (Quantile Network)

We employ a Monotonic Quantile Network (MQN) or a standard MLP with independent quantile heads. Given the continuous action space of D4RL tasks, we follow the architecture of TQC.^^ \*\* \*\*

- **Input:** Concatenation of State **s**∈**R**d**s** and Action **a**∈**R**d**a**.
- **Hidden Layers:** 3 layers of 512 units with ReLU activation. This increased capacity (vs. standard 256) is necessary to model the full distribution complexities.
- **Output:** A vector of size **N** (number of quantiles). We recommend **N**=**25** or **N**=**50**.
- **Ensembling:** To further stabilize training, we use an ensemble of **M**=**2** or **M**=**5** critic networks, similar to TQC. The risk-adjusted value is computed over the pooled quantiles of the ensemble.

#### 4.1.2 The Actor (Policy Network)

- **Input:** State **s**.
- **Structure:** MLP with 2 layers of 256 units.
- **Output:** Parameters for a `TanhGaussian` distribution (mean **μ**, log-std **σ**).
- **Activation:** `Mish` or `ReLU`. Recent benchmarks suggest `Mish` provides smoother gradients for actor networks in offline RL.

#### 4.1.3 The Dynamics/State-Correction Model

This component is unique to SCAS. We implement it as a **Probabilistic Ensemble of Neural Networks (PENN)** , similar to MOPO (Model-Based Offline Policy Optimization) ^^, but used solely for calculating the regularization loss, not for rollout generation. \*\* \*\*

- **Input:** State **s**, Action **a**.
- **Output:** Gaussian distribution params for next state **N**(**μ**s**′,**Σ**s**′).
- **Training:** Supervised learning on dataset transitions **{(**s**,**a**,**s**′**)} maximizing likelihood.

Alternatively, for the "Value-Aware" target **P**v**a**, we train a separate **Action-Conditioned VAE (AC-VAE)** or a **Conditional Diffusion Model** that learns to reconstruct **s**′ given **s** and _high-value_ indicators. A simpler, computationally efficient proxy is to use the nearest-neighbor distance in the embedding space of the dynamics model to the dataset's next states.

### 4.2 Detailed Algorithmic Walkthrough

The training process involves an interleaved update of the Critic, the Actor, and the Dynamics model.

**Algorithm: Dist-SCAS**

**1. Initialization:**

- Initialize Ensemble Critics **{**Z**θ**1**,**…**,**Z**θ**M**}**.
- Initialize Actor **π**ϕ.
- Initialize and pre-train Dynamics Model **f**^d**y**n (or VAE) on **D**.
- Target networks **θ**i**′,**ϕ**′** initialized.

**2. Critic Training (Quantile Regression):**

- Sample batch **B**=**{(**s**,**a**,**r**,**s**′**,**d**)} from **D**.
- Compute Target Quantiles:
  - Sample next action **a**′**∼**π**ϕ**′(**s**′**)**.
  - Fetch quantiles from target critics: **z**j**,**k\***\*=**Z**θ**j**′**(**s**′**,**a**′**)**k\*\***.
  - Pool all quantiles from all **M** critics.
  - Sort pooled quantiles and truncate (optional, TQC style) or keep all.
  - Apply Bellman: **y**j**,**k\***\*=**r**+**γ**(**1**−**d**)**z**j**,**k\*\***.
- Update Critics:
  - Minimize Quantile Huber Loss **L**κ between current estimates **Z**θ**i**(**s**,**a**) and targets **y**.

**3. Actor Training (Risk-Sensitive SCAS):**

- Generate actions **a**^**∼**π**ϕ\*\***(**s**)\*\* (using reparameterization trick).
- **Distributional Value Step:**
  - Query critics to get quantiles **{**q**1\*\***,**…**,**q**N}** for **(**s**,**a**^\*\*).
  - Calculate **Q**r**i**s**k\*\***=**⌊**α**N**⌋**1\*\***∑**i**=**1**⌊**α**N**⌋\*\***q**(**i**)\*\*** (average of lower tail).
- **SCAS Regularization Step:**
  - Predict next state **s**^**′**=**f**^d**y**n(**s**,**a**^**)**.
  - Find nearest neighbor **s**NN**′** in batch (or query VAE).
  - Calculate **L**SC**A**S=**∥**s**^**′**−**s**NN**′∥**2**.
- **Total Loss:**
  - **L**a**c**t**or\*\***=**−**Q**r**i**s**k+**λ**⋅**L**SC**A**S+**β**H**(**π\*\*) (entropy term).
  - Backpropagate **∇**ϕ\***\*L**a**c**t\*\*or.

**4. Updates:**

- Soft update target networks.

---

## 5. Implementation Details and Code Analysis

This section provides specific guidance on translating the theory into PyTorch code, addressing common pitfalls in distributional RL implementation.

### 5.1 The Quantile Huber Loss Implementation

The Quantile Huber Loss is the engine of the critic. It is asymmetric and requires careful broadcasting to be efficient.

**Python**

```
import torch
import torch.nn.functional as F

def quantile_huber_loss_f(quantiles, target_quantiles, tau):
    """
    Args:
        quantiles: (batch_size, n_quantiles) - Estimation
        target_quantiles: (batch_size, n_quantiles) - Target
        tau: (batch_size, n_quantiles) - Quantile centroids
    Returns:
        loss: Scalar
    """
    # Expand dims for pairwise difference
    # (batch_size, n_quantiles, 1) - (batch_size, 1, n_quantiles)
    # Result: (batch_size, n_quantiles, n_quantiles)
    u = target_quantiles.unsqueeze(1) - quantiles.unsqueeze(2)

    # Huber Loss component
    abs_u = torch.abs(u)
    kappa = 1.0 # Threshold for Huber loss
    huber_loss = torch.where(
        abs_u <= kappa,
        0.5 * u.pow(2),
        kappa * (abs_u - 0.5 * kappa)
    )

    # Asymmetric Quantile Weighting
    # diff represents the indicator function |tau - I(u<0)|
    diff = torch.abs(tau.unsqueeze(2) - (u < 0).float().detach())

    # Combined Loss
    loss = diff * huber_loss

    # Sum over target quantiles (dim=2), mean over current quantiles (dim=1)
    return loss.sum(dim=2).mean(dim=1).mean()
```

**Critical Detail:** The `tau` (quantile centroids) must be centered. For **N** quantiles, **τ**i=**N**i**−**0.5\*\*\*\*. Using `u < 0` requires detaching from the graph to prevent gradient flow through the indicator function, although PyTorch logic operators usually don't track gradients anyway. Explicit `.detach()` is safer.

### 5.2 The Risk-Sensitive Actor Loss

To backpropagate through the quantile sorting, we rely on the property that `torch.sort` is differentiable with respect to the values.

**Python**

```
def get_lower_cvar(quantiles, alpha=0.1):
    """
    Computes the CVaR at level alpha.
    """
    batch_size, n_quantiles = quantiles.shape
    k = int(alpha * n_quantiles)

    # Sort quantiles (ascending)
    sorted_quantiles, _ = torch.sort(quantiles, dim=1)

    # Take the lowest k quantiles
    lower_tail = sorted_quantiles[:, :k]

    # Return the mean of the lower tail
    return lower_tail.mean(dim=1, keepdim=True)
```

**Architectural Nuance:** The choice of **α** is critical.

- **α**≈**0.1**: Extremely conservative. Good for `medium-replay` datasets with high noise.
- **α**≈**0.25**: Balanced. Good for `medium-expert`.
- **α**=**1.0**: Risk-neutral (equivalent to mean).

### 5.3 Implementing the SCAS Regularizer

We need a differentiable dynamics model. A simple MLP trained with MSE loss often suffices for the regularizer, as we only need the _gradient direction_ to pull the actor towards valid transitions.

**Python**

```
class SCAS_Regularizer(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Residual dynamics model: s' = s + f(s,a)
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, state_dim)
        )

    def forward(self, state, action):
        return state + self.dynamics(torch.cat([state, action], dim=1))

    def compute_loss(self, state, action, dataset_next_states):
        """
        Computes distance between predicted next state and nearest dataset state.
        In practice, finding the exact NN in the whole dataset is slow.
        Approximation: Use the next_state from the current batch.
        """
        predicted_next = self.forward(state, action)

        # Simple Proxy: Force consistency with the batch's actual transition
        # This acts as a behavior cloning regularization on the dynamics level
        # A more advanced version would search a local neighborhood or VAE latent space.
        mse_loss = F.mse_loss(predicted_next, dataset_next_states)
        return mse_loss
```

**Advanced SCAS Implementation:** To truly capture "Value-Aware" transitions, the `dataset_next_states` passed to this loss should ideally be re-sampled from the dataset based on high advantage scores, or the loss should be weighted by the advantage **A**(**s**,**a**) of the batch sample.

---

## 6. Experimental Validation: Datasets and Protocol

To rigorously validate Dist-SCAS, we utilize the **D4RL (Datasets for Deep Data-Driven Reinforcement Learning)** benchmark.^^ The selection of specific environments is crucial to demonstrate the method's unique benefits. \*\* \*\*

### 6.1 D4RL Environment Selection

| Environment                     | Dataset Quality            | Challenge                                               | Dist-SCAS Hypothesis                                                                                                                                                                               |
| ------------------------------- | -------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hopper/Walker2d/HalfCheetah** | `Medium-Expert`            | Multi-modal data (optimal + suboptimal policies mixed). | **Distributional Advantage:** Mean-based methods average the modes (leading to mediocre behavior). Dist-SCAS can isolate the "expert" mode via quantile manipulation and SCAS keeps it consistent. |
| **AntMaze**                     | `Umaze`, `Medium`, `Large` | Sparse rewards, stitching required.                     | **SCAS Advantage:** These are navigation tasks where OOD state visitation is fatal (agent gets lost). SCAS forces the agent to "connect the dots" of valid transitions.                            |
| **Kitchen / Franka**            | `Mixed`, `Partial`         | High-dimensional, realistic robotics.                   | **Unified Advantage:** Complex dynamics require rigorous state correction; sparse data requires distributional pessimism.                                                                          |

### 6.2 Evaluation Metrics

Beyond the standard **Normalized Score** , we propose tracking specific metrics to validate the internal mechanics of Dist-SCAS:

1. **OOD Action Rate:** Measure the distance of **π**(**s**) from the behavior **π**β(**s**) (estimated via BC). Dist-SCAS should show lower divergence in high-variance states.
2. **OOD State Visitation:** During evaluation rollouts, track the minimum distance of the agent's state **s**t to the training dataset **D**. SCAS should explicitly minimize this metric compared to CQL.
3. **Value Estimation Bias:** Compare estimated **Q**r**i**s**k** vs. actual discounted returns. Dist-SCAS should show a _negative bias_ (underestimation) that correlates with state novelty.
4. **Quantile Spread:** Monitor the gap between **q**0.9 and **q**0.1. A collapsing spread indicates high confidence; a widening spread indicates effective uncertainty capture.

### 6.3 Baseline Comparisons

To prove efficacy, Dist-SCAS must be benchmarked against:

- **CQL (Conservative Q-Learning):** The standard for OOD action suppression.
- **IQL (Implicit Q-Learning):** The standard for expectile-based (scalar) offline RL.
- **SCAS (Original):** To demonstrate the lift provided by the Distributional Critic.
- **TQC (Truncated Quantile Critics):** To demonstrate the lift provided by the SCAS State Correction regularizer in the offline setting.

---

## 7. Ablation Studies and Expected Insights

A rigorous "new paper" implementation requires ablation studies to dissect the contribution of each component.

### 7.1 Ablation: Scalar vs. Distributional SCAS

- **Setup:** Replace the Quantile Critic with a standard ensemble of 2 scalar Q-networks (min-clipping). Keep the SCAS regularizer.
- **Expected Outcome:** The scalar version will perform well on dense data but struggle on `Medium-Replay` or `AntMaze` where the variance of returns is high. The scalar penalty is too uniform; the distributional lower-quantile penalty is adaptive.

### 7.2 Ablation: SCAS vs. No State Correction

- **Setup:** Remove **L**SC**A**S (**λ**=**0**). This essentially reduces the algorithm to an Offline TQC/DSAC.
- **Expected Outcome:** The agent may achieve high scores on simple tasks (Hopper) but will fail catastrophically on **AntMaze** . Without SCAS, the agent will select actions that theoretically maximize the lower quantile but physically transport the agent to undefined regions of the maze (OOD states), leading to policy collapse.

### 7.3 Ablation: Quantile Level **α**

- **Setup:** Vary **α**∈**{**0.05**,**0.1**,**0.25**,**0.5**}**.
- **Expected Outcome:**
  - **α**=**0.5** (Median): Too risky, high OOD rates.
  - **α**=**0.05**: Too conservative, agent freezes (similar to high **α** in CQL).
  - **α**=**0.1**: Sweet spot for offline safety.

---

## 8. Conclusion

**Dist-SCAS** represents a coherent evolution in Offline Reinforcement Learning. By acknowledging that safety in offline RL is a dual problem—requiring both **geometric adherence** to the training manifold (via SCAS) and **statistical pessimism** in value estimation (via Distributional RL)—this framework offers a robust solution to the extrapolation error.

The implementation details provided here, specifically the use of the Quantile Huber Loss for critic training and the derivation of the CVaR-based actor update with dynamics regularization, constitute a complete roadmap for deploying this novel algorithm. The theoretical synergy suggests that Dist-SCAS is particularly well-suited for the most challenging regimes of offline RL: sparse rewards, multi-modal data, and complex dynamics, as exemplified by the AntMaze and Robotics workloads in the D4RL benchmark. Future work lies in the automated tuning of the SCAS regularization parameter **λ** using the distributional variance signal, effectively creating a self-regulating agent that tightens its geometric constraints in the face of statistical uncertainty.

---

## 9. Mathematical Deep Dive: Dist-SCAS Objective and Gradients

**Notation:** Dataset $\mathcal{D}$ over $(s,a,r,s',d)$; $d$ is terminal flag. Policy $\pi_\phi$, behavior $\pi_\beta$, critic ensemble $\{Z_{\theta_m}\}_{m=1}^M$ with $N$ quantiles, dynamics/state-correction model $f_\psi$.

### 9.1 CVaR Risk from Quantile Critics

Pooled quantiles $\tilde{q}_{1:MN}(s,a)$ sorted ascending define

$$
Q_\text{risk}(s,a) = \frac{1}{k}\sum_{i=1}^k \tilde{q}_i(s,a), \quad k=\lfloor \alpha MN \rfloor.
$$

### 9.2 Distributional Bellman Targets

For each critic $m$:

$$
y_j = r + \gamma(1-d)\, z'_j,\quad z'_j \sim Z_{\theta'_m}(s', a'),\ a' \sim \pi_{\phi'}(s').
$$

Quantile Huber with $\tau_i=\frac{i-0.5}{N}$:

$$
\mathcal{L}_\text{crit}(\theta_m)=\mathbb{E}\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^N \rho_\kappa(y_j - q^{(m)}_i)\, \big|\tau_i-\mathbb{1}[y_j<q^{(m)}_i]\big|.
$$

### 9.3 State-Correction (SCAS) Term

Residual dynamics $f_\psi(s,a)=s+\delta_\psi(s,a)$; value-aware target $\tilde{s}'\sim \nu(\cdot|s)$ (nearest in-batch or VAE):

$$
\mathcal{L}_\text{SCAS}(\phi,\psi)=\mathbb{E}_{s,a}\big\|f_\psi(s,a)-\tilde{s}'\big\|_2^2.
$$

### 9.4 Adaptive Weight via Interquartile Range

Let $\Delta Z = q_{0.75}-q_{0.25}$ from pooled quantiles. Set

$$
\lambda(s,a)=\lambda_\text{base}\big(1+\sigma(\Delta Z)\big),
$$

where $\sigma$ is sigmoid, amplifying penalties where aleatoric spread is high.

### 9.5 Actor Objective

$$
\mathcal{L}_\text{actor}(\phi)= - Q_\text{risk}(s,a_\phi) + \lambda(s,a_\phi)\|f_\psi(s,a_\phi)-\tilde{s}'\|_2^2 + \beta \mathcal{H}(\pi_\phi(\cdot|s)),
$$

with $a_\phi$ from reparameterized Gaussian policy.

### 9.6 Contraction Sketch

CVaR-projected Bellman operator is a $\gamma$-contraction in truncated Wasserstein; the convex SCAS penalty preserves contraction, ensuring a fixed point under bounded rewards/dynamics.

---

## 10. Algorithm (Step-by-Step Pseudocode)

1. Init critics $Z_{\theta_m}$, targets $\theta'_m$, actor $\pi_\phi$, target $\phi'$, dynamics $f_\psi$.
2. Pretrain $f_\psi$ on $(s,a,s')$.
3. Loop:
   - Sample batch $B$.
   - $a' \leftarrow \pi_{\phi'}(s')$; targets $y = r + \gamma(1-d) z'$ from target critics.
   - Critic update: minimize quantile Huber to $y$ for all $m$.
   - Actor: $a_\phi \leftarrow \pi_\phi(s)$; pooled $\tilde{q}$ → $Q_\text{risk}$; compute $\lambda(s,a_\phi)$; SCAS loss with $f_\psi$ vs batch $s'$; minimize $\mathcal{L}_\text{actor}$.
   - Optional: refresh $f_\psi$ on batch.
   - Polyak update $\theta',\phi'$.
4. Eval: deterministic policy; D4RL score + OOD metrics.

---

## 11. PyTorch Reference Snippets

### 11.1 Quantile Critic Ensemble

```
class QuantileMLP(nn.Module):
    def __init__(self, s_dim, a_dim, n_q=50, hid=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, n_q)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))
```

### 11.2 CVaR Helper

```
def cvar_tail(qs: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    q_sorted, _ = torch.sort(qs, dim=1)
    k = max(1, int(alpha * q_sorted.size(1)))
    return q_sorted[:, :k].mean(dim=1, keepdim=True)
```

### 11.3 SCAS Regularizer

```
class SCASReg(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.dyn = nn.Sequential(
            nn.Linear(s_dim + a_dim, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish(),
            nn.Linear(256, s_dim)
        )

    def forward(self, s, a):
        return s + self.dyn(torch.cat([s, a], dim=-1))

    def loss(self, s, a, s_next):
        return F.mse_loss(self.forward(s, a), s_next)
```

### 11.4 Quantile Huber Loss

```
def quantile_huber(pred, target, taus, kappa=1.0):
    u = target.unsqueeze(1) - pred.unsqueeze(2)  # [B, N, N]
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
    weight = (taus.unsqueeze(2) - (u.detach() < 0).float()).abs()
    return (weight * huber).sum(dim=2).mean(dim=1).mean()
```

---

## 12. Hyperparameters (Suggested Defaults)

| Component       | Setting               | Value                |
| --------------- | --------------------- | -------------------- |
| Quantiles       | $N$                   | 50                   |
| Critics         | $M$                   | 2 (option 5)         |
| CVaR level      | $\alpha$              | 0.1                  |
| LR              | Critic/Actor          | $3\mathrm{e}{-4}$    |
| Batch           |                       | 256                  |
| Target $\tau$   |                       | 0.005                |
| SCAS $\lambda$  | $\lambda_\text{base}$ | 1.0                  |
| Entropy $\beta$ |                       | 0.2                  |
| Discount        | $\gamma$              | 0.99 (0.995 AntMaze) |
| Dyn hidden      |                       | 256                  |

---

## 13. Evaluation Protocol (D4RL)

- Envs: hopper/walker/halfcheetah (medium-replay, medium-expert), antmaze (umaze, medium, large), kitchen-mixed.
- Metrics: D4RL normalized score, OOD action rate (distance to BC), OOD state distance (NN to dataset), value bias, quantile spread.
- Seeds: ≥5; report mean ± std.

---

## 14. Ablations

1. Remove SCAS ($\lambda=0$) → expect AntMaze failure, higher OOD states.
2. Scalar critics (twin Q) → higher OOD action rate, less safety.
3. CVaR sweep $\alpha \in \{0.05,0.1,0.25,0.5\}$.
4. Ensemble size $M \in \{1,2,5\}$.
5. Dynamics quality: MLP vs VAE vs diffusion target selector.

---

## 15. Practical Tips

- Normalize states; clip rewards to [-10,10] for mujoco to stabilize tails.
- If quantiles collapse: lower LR, increase $M$, or raise $kappa$.
- If policy freezes: reduce $\lambda_\text{base}$ or cap $\lambda(s,a)$.
- AntMaze: set $\gamma=0.995$, add light Gaussian action noise during eval to avoid deadlocks.

---

## 16. Reproducibility Checklist

- Fix seeds (PyTorch, NumPy, gym).
- Log hyperparameters, taus, lambda schedule, dataset name, seed.
- Save checkpoints (actor, critics, targets, optimizers) every 100k steps.
- Store evaluation rollouts and OOD metrics with scores.

---

## 17. Extension Ideas

- **CrossQ**: BN-stabilized high-UTD updates with quantile critics.
- **Sophia-G**: curvature-aware optimizer for actor.
- **Diffusion targets**: sample $\tilde{s}'$ from high-value manifolds.
- **Spectral norm**: constrain critic Lipschitz constants.

---

## 18. Minimal Training Skeleton (Outline)

```
for step in range(T):
    batch = replay.sample()
    with torch.no_grad():
        a_next = actor_t(batch.s_next)
        target_qs = [c_t(batch.s_next, a_next) for c_t in target_critics]
        pooled = torch.sort(torch.cat(target_qs, dim=1), dim=1)[0]
        y = batch.r + gamma * (1 - batch.d) * pooled
    loss_c = sum(quantile_huber(c(batch.s, batch.a), y, taus) for c in critics)
    a = actor(batch.s)
    qs = [c(batch.s, a) for c in critics]
    pooled = torch.sort(torch.cat(qs, dim=1), dim=1)[0]
    q_risk = cvar_tail(pooled, alpha)
    lambda_sa = lambda_base * (1 + torch.sigmoid(iqr(pooled)))
    scas = scas_reg.loss(batch.s, a, batch.s_next)
    loss_a = (-q_risk + lambda_sa * scas + beta * entropy(actor, batch.s)).mean()
```

---

## 19. Notebook Visualization Plan

- Training curves: critic loss, SCAS loss, CVaR value, OOD metrics.
- Quantile fan plot for fixed $(s,a)$ over training.
- AntMaze rollouts with state-density overlay vs dataset.
- Scatter of IQR vs $\lambda(s,a)$ to show adaptive penalty behavior.

---

## 20. Proof Sketch: CVaR Bias Under OOD

For sub-Gaussian $Z$ with mean $\mu$ and std $\sigma$, $\text{CVaR}_\alpha(Z) \le \mu - \sigma \frac{\phi(\Phi^{-1}(\alpha))}{\alpha}$. OOD inflates epistemic $\sigma$, shrinking CVaR and discouraging uncertain actions without explicit behavior constraints.

---

## 21. Dataset Notes

- Hopper/Walker: moderate rewards; action L2 penalty optional.
- HalfCheetah: add tanh squashing; consider target entropy -3.
- AntMaze: longer horizon; use $\gamma=0.995$ and larger replay (2M) if memory allows.
- Kitchen: increase hidden sizes (1024) and add LayerNorm.

---

## 22. Reporting Guidance

- Equations (Section 9) aligned with code.
- Tables: D4RL scores ± std, OOD metrics, ablations.
- Figures: quantile fan, AntMaze trajectories, SCAS loss vs score.
- Appendix: hyperparameters, compute budget, failure cases.

---

## 23. End-to-End Checklist

- [ ] README ≥1000 lines with math + code (this file).
- [ ] Training implements CVaR actor + adaptive SCAS.
- [ ] Dynamics model trained and plugged into regularizer.
- [ ] Evaluation scripts for D4RL + OOD metrics.
- [ ] Ablations executed and logged.

---

## 24. Closing Remarks

Distributional State-Corrected Action Suppression combines pessimistic distributional value learning with geometry-aware state regularization. By optimizing lower-tail returns and aligning predicted transitions to in-distribution manifolds, it addresses offline RL's twin failures: OOD action overestimation and OOD state drift. The derivations, pseudocode, and hyperparameters here are intended to be immediately actionable for reproducing robust results on the hardest D4RL benchmarks.

---

## 25. Experimental Matrix and Logging Plan

**Matrix (rows = env, cols = settings):**

| Env            | Base | +SCAS | +SCAS+Adaptive $\lambda$ | +SCAS+Adaptive $\lambda$+Ensemble5 | +Sophia-G |
| -------------- | ---- | ----- | ------------------------ | ---------------------------------- | --------- |
| Hopper-m       | ✓    | ✓     | ✓                        | ✓                                  | ✓         |
| Walker2d-m     | ✓    | ✓     | ✓                        | ✓                                  | ✓         |
| HalfCheetah-m  | ✓    | ✓     | ✓                        | ✓                                  | ✓         |
| AntMaze-umaze  | ✓    | ✓     | ✓                        | ✓                                  | ✓         |
| AntMaze-medium | ✓    | ✓     | ✓                        | ✓                                  | ✓         |
| Kitchen-mixed  | ✓    | ✓     | ✓                        | ✓                                  | ✓         |

**Logging (per step / episode):**

- Critic loss, SCAS loss, CVaR value, entropy.
- IQR, $\lambda(s,a)$ statistics (mean, p95).
- OOD action distance (to BC logits or Gaussian KL).
- OOD state distance (NN in dataset embedding).
- Replay buffer coverage (state density histogram).
- D4RL score (eval every 100k gradient steps).

---

## 26. Safety and Robustness Considerations

- **Risk floor:** Optionally cap CVaR to not exceed median to avoid optimism drift.
- **Action clipping:** Enforce env action bounds; penalize saturation in actor loss.
- **Dynamics uncertainty:** Track ensemble variance of $f_\psi$; inflate $\lambda$ when variance is high (state-uncertainty aware SCAS).
- **Stochasticity:** For stochastic envs (kitchen), increase particle count for critics to capture tails.
- **Numerical stability:** Use `float32` with AMP; clamp log-std in actor to [-20, 2]; add small $\epsilon$ in Huber.

---

## 27. Failure Modes and Mitigations

- **Mode collapse in quantiles:** Increase ensemble, raise kappa, or add dropout in critic hidden layers.
- **Over-regularization:** If performance stalls, decay $\lambda_\text{base}$ over steps or cap $\lambda(s,a)$ to 5× base.
- **Dynamics drift:** Periodically re-pretrain $f_\psi$ on replay or add L2 weight decay.
- **AntMaze dead ends:** Add small Gaussian noise to actions during eval; use longer target update interval (every 2 steps).
- **Kitchen instability:** Use LayerNorm in critics/actor; increase hidden to 1024; reduce LR to 1e-4.

---

## 28. Implementation Checklist (Codebase-Level)

- [ ] Replay buffer stores $(s,a,r,s',d)$ tensors (float32) and supports large batches (256/512).
- [ ] Actor outputs mean/log-std; sampled via reparameterization with tanh squashing.
- [ ] Critic ensemble forward returns list of quantile tensors [B,N]; pooled via `torch.sort`.
- [ ] Tau vector precomputed: `taus = (torch.arange(N, device)/N + 0.5/N).view(1,N)`.
- [ ] SCAS regularizer module with `.loss` API.
- [ ] Adaptive $\lambda$: compute IQR on pooled quantiles, apply sigmoid, scale by $\lambda_\text{base}$.
- [ ] Target updates: Polyak with $\tau=0.005`; ensure no_grad.
- [ ] Evaluation script computing D4RL normalized score and OOD metrics.
- [ ] Config system (YAML/argparse) exposing alpha, lambda_base, gamma, N, M, kappa, tau_update, batch, seed.
- [ ] Logging (TensorBoard/W&B) for metrics listed in Section 25.

---

## 29. Sample YAML Configs

**hopper-medium-expert.yaml**

```
env: hopper-medium-expert-v2
seed: 0
gamma: 0.99
alpha_cvar: 0.1
n_quantiles: 50
n_critics: 2
lambda_base: 1.0
lr_actor: 3e-4
lr_critic: 3e-4
batch_size: 256
target_tau: 0.005
entropy_beta: 0.2
use_adaptive_lambda: true
use_amp: true
```

**antmaze-medium-diverse.yaml**

```
env: antmaze-medium-diverse-v2
seed: 0
gamma: 0.995
alpha_cvar: 0.1
n_quantiles: 50
n_critics: 2
lambda_base: 2.0
lr_actor: 3e-4
lr_critic: 3e-4
batch_size: 512
target_tau: 0.005
entropy_beta: 0.1
use_adaptive_lambda: true
use_amp: true
```

---

## 30. Extended Proof Sketch: Bias Control via SCAS

Let $s'_\pi = f_\psi(s,\pi(s))$ and $\tilde{s}'$ from dataset manifold. The SCAS penalty enforces $\|s'_\pi - \tilde{s}'\|^2 \le \epsilon$. For Lipschitz critic $L_Q$, the bias in target value due to state shift satisfies

$$
|Q_\text{risk}(s'_\pi,a') - Q_\text{risk}(\tilde{s}',a')| \le L_Q \|s'_\pi - \tilde{s}'\| \le L_Q \sqrt{\epsilon}.
$$

Thus bounding SCAS loss bounds value extrapolation bias, complementing CVaR pessimism on action uncertainty.

---

## 31. Additional Ablations to Strengthen Paper Claims

- **Adaptive vs fixed $\lambda$**: show improved score-variance trade-off.
- **IQR vs variance for $\lambda$**: test robustness to heavy-tailed returns.
- **Behavior-clone warm-start**: initialize actor from BC; measure impact on AntMaze success.
- **Particle count**: $N\in\{16,32,50,80\}$; plot score vs compute.
- **Entropy anneal**: linear decay of $\beta$; inspect exploration vs stability.

---

## 32. Qualitative Analyses

- **State visitation heatmaps**: Compare coverage to dataset states (PCA/UMAP).
- **Action overlap**: KL between $\pi_\phi$ and $\pi_\beta$ across states.
- **Return distribution plots**: Histograms of predicted returns for chosen actions vs OOD actions.
- **Trajectories**: Visualize AntMaze paths, highlighting OOD excursions avoided by Dist-SCAS.

---

## 33. Engineering Notes for Large-Scale Runs

- Use `torch.compile` (PyTorch 2.2+) to accelerate critic forward/sort if available.
- Enable AMP for critic/actor; keep quantile loss in float32 to avoid underflow.
- Gradient clipping at 10.0 for both actor and critics.
- Pin memory + prefetch to speed dataloader for large buffers.
- Periodic evaluator process to avoid blocking training loop.

---

## 34. Future Research Directions

- **SCAS with Consistency Models**: use consistency-trained dynamics for smoother manifolds.
- **Hybrid MBPO**: roll out short model trajectories gated by SCAS; add to buffer with penalty.
- **Uncertainty-guided data pruning**: remove high IQR transitions to test robustness.
- **Multi-task offline RL**: extend SCAS to task-conditioned datasets; use task embeddings in $\lambda$.
- **Safety constraints**: add cost dimension and optimize joint CVaR over return and cost.

---

## 35. Author Checklist for Submission

- [ ] All equations consistent with code defaults (alpha=0.1, kappa=1.0).
- [ ] Tables: full matrix (Section 25) + ablations (Sections 14, 31).
- [ ] Figures: heatmaps, fan plots, trajectories, OOD metrics.
- [ ] Appendix: configs, hyperparameters, compute budget, additional proofs.
- [ ] Reproducibility: seeds, code links, commit hash.

---

## 36. Minimal Eval Script Sketch

```
def evaluate(policy, env, episodes=10):
    scores, ood_actions, ood_states = [], [], []
    for _ in range(episodes):
        s, done = env.reset(), False
        ep_actions, ep_states = [], []
        while not done:
            a = policy.act_eval(s)
            s_next, r, done, info = env.step(a)
            ep_actions.append(a)
            ep_states.append(s)
            s = s_next
        scores.append(info.get("episode_score", 0.0))
        ood_actions.append(compute_action_kl(ep_actions))
        ood_states.append(min_dataset_nn(ep_states))
    return {
        "score": np.mean(scores),
        "score_std": np.std(scores),
        "ood_action": np.mean(ood_actions),
        "ood_state": np.mean(ood_states),
    }
```

---

## 37. Extended Notebook Cells (to be written in `notebooks/main.ipynb`)

- **Data inspection:** plot dataset action/state histograms, reward distribution.
- **Training loop cell:** pseudocode mirroring Section 18 with `tqdm` progress.
- **Metric dashboards:** live plots for losses, CVaR, OOD metrics.
- **Frontier visualization:** for AntMaze, overlay paths on map; for mujoco, plot return distributions.
- **Ablation runner:** loop over configs, log to separate folders.

---

## 38. Risks and Mitigations Table

| Risk               | Cause                  | Mitigation                           |
| ------------------ | ---------------------- | ------------------------------------ |
| NaN loss           | overflow in sort/Huber | clip rewards, lower LR, AMP autocast |
| Over-conservatism  | high $\lambda$         | cap $\lambda$, schedule decay        |
| Under-penalization | low IQR                | floor $\lambda$ at base value        |
| Dynamics overfit   | small buffer           | dropout/weight decay; ensemble       |
| Eval variance      | stochasticity          | 10 seeds for AntMaze                 |

---

## 39. Open Questions

- Best proxy for $\nu(\cdot|s)$: NN vs VAE vs diffusion?
- Does adaptive $\lambda$ slow learning early? Consider warmup with fixed $\lambda$.
- How does CVaR interact with entropy temperature tuning? Joint schedule may help.
- Could SCAS be applied in online RL as exploration regularizer?

---

## 40. Final Notes

Dist-SCAS is designed for practitioners needing robust offline RL under severe distributional shift. By combining distributional pessimism (CVaR) with geometric state correction (SCAS), it offers a principled way to tame extrapolation while retaining performance. The expanded sections above provide the operational detail—math, code patterns, configs, ablations, and diagnostics—to implement, evaluate, and extend the method in a research setting.

---

## 41. Implementation FAQ

- **Do I need target networks?** Yes—use Polyak with $\tau=0.005$; CrossQ-style removal of targets is risky offline.
- **Can I use TD3-style clipped double Q?** Not for quantiles; instead use ensemble pooling to reduce overestimation.
- **How many particles for CVaR?** 50 is a good default; fewer may underestimate tails.
- **Can SCAS use k-NN instead of dynamics?** For small state spaces, yes; for high-dim, learn $f_\psi$.
- **What about actor update frequency?** 1:1 with critic works; try delayed actor every 2 steps if unstable.

---

## 42. Unit Tests (Recommended)

- **Quantile loss finite:** random tensors → loss finite, gradients non-NaN.
- **CVaR monotonicity:** increasing all quantiles should increase CVaR.
- **SCAS loss zero:** when predicted next == target next, loss == 0.
- **Adaptive $\lambda$ bounds:** verify $\lambda \ge \lambda_\text{base}$ and grows with IQR.
- **Target update:** after Polyak, parameters change but remain close (norm diff decreases).

---

## 43. Compute Budget Estimates

- Hopper/Walker/HalfCheetah: 1 GPU (V100/A100), ~4–6 hours for 1M gradient steps with AMP.
- AntMaze medium: 1–2 GPUs, ~10–14 hours; batch 512; gamma 0.995 increases horizon.
- Kitchen: 2 GPUs recommended; larger nets; expect ~12–16 hours.
- Storage: Replay buffers up to ~5–10 GB; checkpoints every 100k steps (~50–100 MB each).

---

## 44. Reproduction Steps (Minimal)

1. Install deps: `pip install torch geomloss gymnasium[d4rl] mujoco-py`.
2. Download D4RL datasets (handled on first env reset).
3. Train: `python train_dist_scas.py --config hopper-medium-expert.yaml`.
4. Eval: `python eval_dist_scas.py --checkpoint ckpt_1M.pt --env hopper-medium-expert-v2`.
5. Run ablations: loop configs from Section 29; log to `runs/{env}/{config}`.
6. Report: aggregate scores, OOD metrics, plots (Sections 25, 31).

---

## 45. Benchmark-Specific Hyperparameters

| Env                  | $\gamma$ | $\alpha$ | $\lambda_\text{base}$ | Batch | Entropy $\beta$ | Notes                        |
| -------------------- | -------- | -------- | --------------------- | ----- | --------------- | ---------------------------- |
| Hopper-medium        | 0.99     | 0.1      | 1.0                   | 256   | 0.2             | standard                     |
| Hopper-medium-expert | 0.99     | 0.1      | 1.0                   | 256   | 0.2             | keep LR 3e-4                 |
| Walker2d-medium      | 0.99     | 0.1      | 1.0                   | 256   | 0.2             | grad clip 10                 |
| HalfCheetah-medium   | 0.99     | 0.1      | 0.8                   | 256   | 0.2             | clip reward [-10,10]         |
| AntMaze-umaze        | 0.995    | 0.1      | 2.0                   | 512   | 0.1             | larger batch, longer horizon |
| AntMaze-medium       | 0.995    | 0.1      | 2.5                   | 512   | 0.1             | stronger SCAS                |
| Kitchen-mixed        | 0.99     | 0.1      | 1.5                   | 512   | 0.1             | hidden 1024, LayerNorm       |

---

## 46. Adaptive $\lambda$ Pseudocode

```
def compute_lambda(pooled_qs, lambda_base):
    q_sorted, _ = torch.sort(pooled_qs, dim=1)  # [B, MN]
    q25 = q_sorted[:, int(0.25 * q_sorted.size(1))]
    q75 = q_sorted[:, int(0.75 * q_sorted.size(1))]
    iqr = q75 - q25
    lam = lambda_base * (1 + torch.sigmoid(iqr))
    return lam.unsqueeze(1)  # broadcast over actions if needed
```

---

## 47. Symbol Glossary (for consistency with code)

| Symbol    | Meaning              | Code variable               |
| --------- | -------------------- | --------------------------- |
| $s,a$     | state, action        | `s`, `a`                    |
| $r$       | reward               | `r`                         |
| $d$       | done flag            | `d`                         |
| $\gamma$  | discount             | `gamma`                     |
| $\alpha$  | CVaR level           | `alpha_cvar`                |
| $N$       | quantiles per critic | `n_quantiles`               |
| $M$       | number of critics    | `n_critics`                 |
| $\kappa$  | Huber threshold      | `kappa`                     |
| $\lambda$ | SCAS weight          | `lambda_base` / `lambda_sa` |
| $\beta$   | entropy coeff        | `entropy_beta`              |
| $\tau$    | target Polyak        | `target_tau`                |

---

## 48. Implementation Timeline (Suggested)

- Day 1: Setup, replay/dataloader, baseline actor/critic scaffolding.
- Day 2: Quantile loss wired, CVaR actor update, target updates.
- Day 3: SCAS dynamics model, adaptive $\lambda$, initial runs on hopper-medium.
- Day 4: AntMaze configuration, logging OOD metrics, fix stability issues.
- Day 5: Ablations (lambda off, scalar critics, alpha sweep), collect results.
- Day 6: Visualizations, notebooks, tables, write-up alignment.

---

## 49. Data Structures

Replay element:

```
{
  "obs": float32[s_dim],
  "action": float32[a_dim],
  "reward": float32[1],
  "next_obs": float32[s_dim],
  "done": bool,
}
```

Batch tensors:

- `obs` [B, s_dim], `actions` [B, a_dim], `rewards` [B,1], `next_obs` [B, s_dim], `dones` [B,1].
- Optional: store behavior log-probs to compute action KL OOD metric.

---

## 50. Licensing and Attribution

Reference implementations: IQN/FQF/TQC repos (MIT/BSD), SCAS GitHub (check license). Ensure derivative code preserves original notices. Cite all papers listed in citations; no proprietary assets included.

---

### Citations

^^ \*\* \*\*

[![](https://t0.gstatic.com/faviconV2?url=https://papers.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)papers.neurips.ccConservative Q-Learning for Offline Reinforcement Learning**Opens in a new window**](https://papers.neurips.cc/paper_files/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://cs224r.stanford.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)cs224r.stanford.eduCS224R Spring 2025 Homework 3 Offline RL(Updated) - CS 224R Deep Reinforcement Learning**Opens in a new window**](https://cs224r.stanford.edu/material/CS224r_Homework3_updated.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgA Clean Slate for Offline RL - arXiv**Opens in a new window**](https://arxiv.org/html/2504.11453v1)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)researchgate.netOffline Reinforcement Learning with OOD State Correction and OOD Action Suppression | Request PDF - ResearchGate**Opens in a new window**](https://www.researchgate.net/publication/397195773_Offline_Reinforcement_Learning_with_OOD_State_Correction_and_OOD_Action_Suppression)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccOffline Reinforcement Learning with OOD State Correction and OOD Action Suppression - NIPS papers**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/a9f3457fa97f106f1756885237787789-Paper-Conference.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netModel-based Offline Reinforcement Learning with Lower Expectile Q-Learning**Opens in a new window**](https://openreview.net/forum?id=OATPSB5JK1)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netMODEL-BASED OFFLINE REINFORCEMENT LEARNING WITH LOWER EXPECTILE Q-LEARNING - OpenReview**Opens in a new window**](https://openreview.net/pdf/68a0713a71679e2a82d9b8b9cb139bd5f6c3f963.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netUNCERTAINTY-AWARE DISTRIBUTIONAL OFFLINE RE- INFORCEMENT LEARNING - OpenReview**Opens in a new window**](https://openreview.net/pdf?id=NHb6mbD99v)[![](https://t0.gstatic.com/faviconV2?url=https://ojs.aaai.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)ojs.aaai.orgQUOTA: The Quantile Option Architecture for Reinforcement Learning - AAAI Publications**Opens in a new window**](https://ojs.aaai.org/index.php/AAAI/article/view/4527/4405)[![](https://t0.gstatic.com/faviconV2?url=https://www.emergentmind.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)emergentmind.comTruncated Quantile Critic (TQC) Algorithm - Emergent Mind**Opens in a new window**](https://www.emergentmind.com/topics/truncated-quantile-critic-tqc-algorithm)[![](https://t0.gstatic.com/faviconV2?url=https://www.econstor.eu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)econstor.euGradient-based reinforcement learning for dynamic quantile - EconStor**Opens in a new window**](https://www.econstor.eu/bitstream/10419/331350/1/1933272287.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)researchgate.net(PDF) Tailoring Portfolio Choice via Quantile-Targeted Policies - ResearchGate**Opens in a new window**](https://www.researchgate.net/publication/396790047_Tailoring_Portfolio_Choice_via_Quantile-Targeted_Policies)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgVariational OOD State Correction for Offline Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/pdf/2505.00503)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comthu-rllab/SCAS: NeurIPS 2024 - GitHub**Opens in a new window**](https://github.com/maoyixiu/SCAS)[![](https://t0.gstatic.com/faviconV2?url=https://papers.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)papers.neurips.ccCORL: Research-oriented Deep Offline Reinforcement Learning Library**Opens in a new window**](https://papers.neurips.cc/paper_files/paper/2023/file/62d2cec62b7fd46dd35fa8f2d4aeb52d-Paper-Datasets_and_Benchmarks.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)researchgate.netRAMAC: Multimodal Risk-Aware Offline Reinforcement Learning and the Role of Behavior Regularization | Request PDF - ResearchGate**Opens in a new window**](https://www.researchgate.net/publication/396223636_RAMAC_Multimodal_Risk-Aware_Offline_Reinforcement_Learning_and_the_Role_of_Behavior_Regularization)[![](https://t3.gstatic.com/faviconV2?url=https://www.kaggle.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)kaggle.comQuantile Regression DQN - RL - Kaggle**Opens in a new window**](https://www.kaggle.com/code/auxeno/quantile-regression-dqn-rl)

[![](https://t0.gstatic.com/faviconV2?url=https://nips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://nips.cc/virtual/2024/papers.html)[![](https://t0.gstatic.com/faviconV2?url=https://iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://iclr.cc/virtual/2024/events/spotlight-posters)[![](https://t0.gstatic.com/faviconV2?url=https://proceedings.iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://proceedings.iclr.cc/paper_files/paper/2024/file/43d7bc009cf5171e7af77a91ee4bb890-Paper-Conference.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://proceedings.iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://proceedings.iclr.cc/paper_files/paper/2024/file/10a3b1c30b8cceb507b9e8ddcc9a1a6a-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2511.11973v1)[![](https://t1.gstatic.com/faviconV2?url=https://kwanyoungpark.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://kwanyoungpark.github.io/LEQ/)[![](https://t0.gstatic.com/faviconV2?url=http://staff.ustc.edu.cn/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](http://staff.ustc.edu.cn/~lszhuang/Doc/2024-IJCNN-CIDQL.pdf)[![](https://t2.gstatic.com/faviconV2?url=http://ieeexplore.ieee.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](http://ieeexplore.ieee.org/iel8/35/11103457/11103474.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2504.03804v1)[![](https://t3.gstatic.com/faviconV2?url=https://research-blog.vballoli.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://research-blog.vballoli.com/posts/offline-rl/)[![](https://t2.gstatic.com/faviconV2?url=https://dblp.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://dblp.org/pid/32/1593-1)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/pdf?id=4l4Gfc1B6E)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/search?term=%F0%9F%98%87%20CXCOCO.com%20%F0%9F%98%88%20order%20cocaine%20Darwin)[![](https://t0.gstatic.com/faviconV2?url=https://grokipedia.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://grokipedia.com/page/Q-learning)[![](https://t0.gstatic.com/faviconV2?url=https://iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://iclr.cc/media/iclr-2025/Slides/29841.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2407.00699v2)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/pdf/2502.10792)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/389091349_Tackling_the_Zero-Shot_Reinforcement_Learning_Loss_Directly)[![](https://t2.gstatic.com/faviconV2?url=https://docs.ray.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.ray.io/en/latest/rllib/rllib-offline.html)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/abs/2406.12205)[![](https://t1.gstatic.com/faviconV2?url=https://elischolar.library.yale.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://elischolar.library.yale.edu/cgi/viewcontent.cgi?article=3871&context=cowles-discussion-paper-series)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/pdf/2111.03788)[![](https://t0.gstatic.com/faviconV2?url=https://docs.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.pytorch.org/rl/stable/tutorials/coding_dqn.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch)[![](https://t0.gstatic.com/faviconV2?url=https://spinningup.openai.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://spinningup.openai.com/en/latest/algorithms/sac.html)[![](https://t2.gstatic.com/faviconV2?url=https://docs.cleanrl.dev/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://docs.cleanrl.dev/rl-algorithms/sac/)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2310.05858v4)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2004.14547v3)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/xtma/dsac)[![](https://t0.gstatic.com/faviconV2?url=https://findingtheta.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://findingtheta.com/blog/mastering-robotic-manipulation-with-reinforcement-learning-tqc-and-ddpg-for-fetch-environments)[![](https://t2.gstatic.com/faviconV2?url=https://sb3-contrib.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2405.02576v1)[![](https://t3.gstatic.com/faviconV2?url=https://tik-db.ee.ethz.ch/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://tik-db.ee.ethz.ch/file/f03ba7159424072225fd718afc222272/)[![](https://t1.gstatic.com/faviconV2?url=https://futur.upc.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://futur.upc.edu/RIS/acces_obert/ade/RGVwYXJ0YW1lbnQgZCdFbmdpbnllcmlhIGRlIFNpc3RlbWVzLCBBdXRvbcOgdGljYSBpIEluZm9ybcOgdGljYSBJbmR1c3RyaWFs)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.mdpi.com/2076-3417/15/4/1798)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.mdpi.com/2073-8994/17/5/638)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2309.14471)[![](https://t2.gstatic.com/faviconV2?url=https://liner.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://liner.com/review/controlling-overestimation-bias-with-truncated-mixture-continuous-distributional-quantile-critics)[![](https://t3.gstatic.com/faviconV2?url=https://compression.stanford.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://compression.stanford.edu/sites/g/files/sbiybj26591/files/media/file/report-reinforcement_learning.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2510.19271v1)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/395474480_Generalizing_Beyond_Suboptimality_Offline_Reinforcement_Learning_Learns_Effective_Scheduling_through_Random_Data)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2509.10303v1)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/349195345_Risk-Averse_Offline_Reinforcement_Learning)[![](https://t2.gstatic.com/faviconV2?url=http://emergingtrends.stanford.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](http://emergingtrends.stanford.edu/files/original/478c41df78a2ab89c4342e1c7979fdcdd02fd6d1.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://support.sas.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://support.sas.com/resources/papers/proceedings/proceedings/sugi30/213-30.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/377503816_Quantile_Regression_Model_and_Its_Application_Research)[![](https://t0.gstatic.com/faviconV2?url=https://www.ibm.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.ibm.com/docs/SSLVMB_sub/statistics_mainhelp_ddita/spss/regression/idh_quantile.html)[![](https://t2.gstatic.com/faviconV2?url=https://pmc.ncbi.nlm.nih.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://pmc.ncbi.nlm.nih.gov/articles/PMC12571108/)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/forum?id=NHb6mbD99v)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2510.02695v1)[![](https://t3.gstatic.com/faviconV2?url=https://www.arxivdaily.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.arxivdaily.com/thread/53596)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2403.17646v1)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/attachment?id=g3IaQTqzeq&name=pdf)[![](https://t3.gstatic.com/faviconV2?url=https://ieeexplore.ieee.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ieeexplore.ieee.org/iel8/11046192/11045962/11046305.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://www.research.unipd.it/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.research.unipd.it/retrieve/e14fb270-5765-3de1-e053-1705fe0ac030/tesi_definitiva_Daniel_Cunico.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://iclr.pangram.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://iclr.pangram.com/reviews?query=&submission_number=&sort_by=submission_id_hash&sort_dir=asc&page=2&prediction_filter=Fully+AI-generated%2CHeavily+AI-edited%2CModerately+AI-edited%2CLightly+AI-edited%2CFully+human-written&rating_filter=5%2C1%2C6%2C9%2C4%2C2&confidence_filter=1%2C2%2C3%2C4%2C5)
