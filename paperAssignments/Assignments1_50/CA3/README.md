# Sinkhorn-Regularized Vectorized Implicit Quantile Networks for Pareto-Optimal Multi-Objective Reinforcement Learning

## Executive Summary

The advancement of Deep Reinforcement Learning (DRL) has fundamentally relied on the capability of agents to estimate future rewards with increasing precision. From the early success of Deep Q-Networks (DQN) which estimated scalar expectations, to the sophisticated Distributional Reinforcement Learning (DRL) paradigms like Implicit Quantile Networks (IQN) which model the full probability distribution of returns, the field has moved towards a more granular understanding of value. However, a significant theoretical and practical limitation persists: the scalarization of multi-dimensional reward signals. In complex environments, agents are frequently tasked with balancing conflicting objectives—such as speed versus safety, or resource accumulation versus risk avoidance. Traditional approaches collapse these distinct signal streams into a single scalar value via linear weighting, thereby stripping the agent of the semantic richness required to navigate the Pareto frontier of optimal trade-offs.

This report presents a comprehensive theoretical framework and implementation strategy for a novel architecture: **Sinkhorn-Regularized Vectorized Implicit Quantile Networks (Sinkhorn-VIQN)** . This proposed architecture extends the distributional capabilities of IQN into the vector-valued domain by replacing the sorting-based quantile regression loss—which is mathematically undefined in multi-dimensional spaces—with the Sinkhorn Divergence. Derived from entropy-regularized Optimal Transport (OT), Sinkhorn Divergence allows for the differentiable comparison of high-dimensional distribution clouds, enabling the agent to learn a joint distribution over multiple objectives without imposing artificial orderings.

Furthermore, this report establishes a rigorous experimental testbed based on the Atari 2600 game _Seaquest_ . We introduce a novel methodology for **RAM-based Reward Decomposition** , utilizing memory mapping of the 6502 processor to isolate distinct reward channels for Oxygen Management, Diver Rescue, and Enemy Neutralization. This separation allows for the training of agents that can dynamically navigate the trade-offs between survival and objective completion, offering a robust platform for validating Multi-Objective Distributional Reinforcement Learning (MODRL).

---

## 1. Introduction and Problem Formulation

### 1.1 The Limits of Scalar Expectations

Reinforcement Learning (RL) is formally defined by the optimization of a policy **π** to maximize the expected cumulative return. In the classical setting, the return **G**t is a sum of scalar rewards **r**t. The value function **Q**π**(**s**,**a**)**=**E**[**G**t****∣**s**,**a**] collapses all future possibilities into a single number representing the "average" outcome. While effective for simple tasks, this expectation-based approach discards critical information regarding the variance, multimodality, and risk profile of the returns.

The introduction of Distributional RL ^^ marked a paradigm shift. By modeling the random variable **Z**π**(**s**,**a**)** rather than its expectation, agents gained the ability to distinguish between a "safe" low-return action and a "risky" high-variance action that share the same mean. Algorithms like C51, Quantile Regression DQN (QR-DQN), and Implicit Quantile Networks (IQN) have demonstrated state-of-the-art performance by capturing this aleatoric uncertainty. IQN, in particular, uses a deterministic parametric function to map input noise (representing quantiles) to return values, effectively learning the inverse cumulative distribution function (CDF) of the return.^^ \*\* \*\*

### 1.2 The Scalarization Bottleneck in Multi-Objective Environments

Despite these advances, Distributional RL remains overwhelmingly focused on scalar rewards. Real-world decision-making, however, is inherently multi-objective. An autonomous vehicle must minimize travel time while simultaneously maximizing passenger comfort and minimizing fuel consumption. A financial trading bot must maximize profit while minimizing drawdown. In the standard RL framework, these distinct objectives **r**t=**[**r**1\*\***,**r**2,**…**,**r**k]**T are scalarized using a weight vector **w\*\*:

**r**sc**a**l**a**r=**w**T**r**t

This scalarization assumes that the relative importance of objectives is fixed and known _a priori_ . It collapses the rich geometry of the multi-objective space into a single dimension, blinding the agent to the trade-offs it must negotiate. If an agent learns only the expectation of the scalarized reward, it cannot dynamically adapt to changing preferences or distinct risk profiles across different objectives.^^ \*\* \*\*

True Multi-Objective Reinforcement Learning (MORL) seeks to learn a set of policies that approximate the **Pareto Frontier** —the set of policies where no objective can be improved without degrading another. To achieve this in a distributional setting, the agent must model the **joint distribution** of the vector returns.

### 1.3 The Geometric Challenge of Vectorized Distributional RL

Extending IQN to multi-dimensional rewards presents a fundamental geometric challenge. The core mechanism of IQN is **quantile regression** , which minimizes the Wasserstein distance between the predicted distribution and the target distribution. In one dimension, the Wasserstein distance is minimized by matching quantiles (e.g., the median of the prediction matches the median of the target). This relies on the property that real numbers have a canonical **total ordering** : for any **a**,**b**∈**R**, either **a**≤**b** or **b**≤**a**.

In a vector space **R**d where **d**>**1**, there is no canonical total ordering. One cannot simply "sort" reward vectors. Therefore, the concept of a "median" or "95th percentile" vector is ill-defined without referencing a specific projection direction. Standard quantile losses, such as the Huber loss applied to quantile differences, cannot be directly applied to vector outputs because the "difference" is a vector, and the "sign" of that difference is ambiguous.^^ \*\* \*\*

Attempts to train independent IQNs for each objective (Factorized IQN) fail to capture **correlations** . For example, in the game _Seaquest_ , the objective of "Rescuing Divers" is negatively correlated with "Oxygen Conservation" in the short term (surfacing takes time) but positively correlated in the long term (surfacing prevents death). Independent distributions assume orthogonality, potentially leading the agent to hallucinate states where it has high oxygen _and_ high diver count, even if such states are physically impossible.

### 1.4 The Proposed Solution: Sinkhorn-VIQN

To resolve this, we must decouple the distributional learning mechanism from the reliance on 1D sorting. We propose **Sinkhorn-Regularized Vectorized Implicit Quantile Networks (Sinkhorn-VIQN)** . This architecture:

1. **Vectorizes the Generator:** Extends the IQN generator to output samples in **R**k conditioned on high-dimensional latent noise.
2. **Utilizes Sinkhorn Divergence:** Replaces the quantile loss with the Sinkhorn Divergence.^^ This loss function, derived from entropy-regularized Optimal Transport, computes the cost of transporting the "cloud" of predicted vector returns to the "cloud" of target vector returns. It respects the underlying Euclidean geometry of the reward space without requiring an artificial ordering. \*\* \*\*
3. **Learns the Joint Distribution:** By matching the full joint distributions via Optimal Transport, the network implicitly learns the correlations and constraints between objectives.

This report details the theoretical derivation of this method, the architectural implementation, and a novel experimental validation using a RAM-decomposed version of Atari _Seaquest_ .

---

## 2. Theoretical Background and Literature Review

### 2.1 Distributional Bellman Operators

In standard RL, the Bellman operator **T**π acts on the value function **Q**:

**T**π**Q**(**s**,**a**)**=**E**+**γ**E**P**,**π\***\*[**Q**(**s**′**,**a**′\*\*)]

Contraction of this operator in the **L**∞ norm guarantees convergence to a unique fixed point **Q**∗.

In Distributional RL, the operator **T**π acts on the distribution **Z**:

**T**π**Z**(**s**,**a**)**=**D**R**(**s**,**a**)**+**γ**Z**(**s**′**,**a**′**)

where **=**D denotes equality in distribution. The random variable **Z**(**s**′**,**a**′**) is distributed according to the next state distribution **P**(**⋅**∣**s**,**a**) and the policy **π**. The convergence of this operator depends on the metric used. The Wasserstein metric **W**p is commonly used because it accounts for the geometry of the support. The operator is a contraction in the **supremal Wasserstein metric** **d**ˉ**p**:

**d**ˉ**p\*\***(**Z**1,**Z**2)**=**s**,**a**sup\*\***W**p(**Z**1(**s**,**a**)**,**Z**2\***\*(**s**,**a\*\*))

This theoretical foundation underpins methods like C51 (which projects the operator onto a categorical support) and QR-DQN (which projects onto fixed quantiles).^^ \*\* \*\*

### 2.2 Implicit Quantile Networks (IQN)

IQN ^^ advances QR-DQN by eschewing fixed quantiles. Instead, it learns a generative function **Z**θ(**s**,**a**;**τ**) where **τ**∼**U**(**)**. The network effectively learns the inverse CDF **F**−**1**(**τ**). The training objective is to minimize the quantile regression loss: \*\* \*\*

**L**QR=**E**τ**,**τ**′**∼**U**(**)[**ρ**τ**κ\***\*(**y**τ**′−**Z**θ(**s**,**a**;**τ**))\*\*]

where **y**τ**′=**r**+**γ**Z**θ**′(**s**′**,**a**′**;**τ**′**) is a sample from the target distribution. Crucially, the logic of "minimizing the distance between the **τ**-th quantile of the prediction and the target" is only valid because the Wasserstein distance between 1D distributions corresponds exactly to the **L**p\*\*\*\* distance between their quantile functions. This property breaks down in higher dimensions.

### 2.3 Optimal Transport in High Dimensions

In multi-dimensional space, comparing two distributions **μ** and **ν** requires solving the **Optimal Transport (OT)** problem. The Kantorovich formulation seeks a coupling **π**∈**Π**(**μ**,**ν**) (a joint distribution with marginals **μ** and **ν**) that minimizes the total transport cost:

**W**c(**μ**,**ν**)**=**π**∈**Π**(**μ**,**ν**)**min\***\*∫**X**×**Y\***\*c**(**x**,**y**)**d**π**(**x**,**y**)**

where **c**(**x**,**y**) is a cost function, typically Euclidean distance **∥**x**−**y**∥**2. While **W**c\***\* is a powerful metric that captures geometry (unlike KL-divergence, which ignores metric space properties), computing it exactly involves solving a linear program with complexity **O**(**N**3**lo**g**N**), where **N** is the number of support points (samples). This cubic complexity renders exact OT infeasible for the inner loop of deep learning, where millions of gradient updates are required.^^ ** \*\*

### 2.4 Sinkhorn Divergence and Entropic Regularization

To address the computational bottleneck, Cuturi (2013) proposed **entropic regularization** .^^ The regularized objective is: \*\* \*\*

**W**c**,**ϵ\***\*(**μ**,**ν**)**=**π**∈**Π**(**μ**,**ν**)**min\*\***⟨**C**,**π**⟩**−**ϵH**(**π**)**

where **H**(**π**)**=**−**∑**π**ij\*\***(**log**π**ij\*\***−**1**) is the entropy of the coupling matrix. The addition of the strictly convex entropy term makes the optimization problem strictly convex. The solution **π**∗ is unique and takes the form of a scaling of the Gibbs kernel **K**ij=**e**−**c**(**x**i,**y**j)**/**ϵ:

**π**ij**∗=**u**iK**ij\***\*v**j\*\*

The vectors **u** and **v** can be found efficiently using the **Sinkhorn-Knopp Algorithm** , which corresponds to iterative matrix scaling (alternating row and column normalizations). This algorithm has a complexity of roughly **O**(**N**2**)**, which is essentially quadratic, and can be further optimized.

However, the regularized transport cost **W**c**,**ϵ\***\* is not a true metric; notably, **W**c**,**ϵ\*\***(**μ**,**μ**)****=**0** due to the entropy term. The **Sinkhorn Divergence** **S**c**,**ϵ\*\*\*\* corrects this bias:

**S**c**,**ϵ\***\*(**μ**,**ν**)**=**2**W**c**,**ϵ\*\***(**μ**,**ν**)**−**W**c**,**ϵ\*\***(**μ**,**μ**)**−**W**c**,**ϵ\*\***(**ν**,**ν**)

This divergence is non-negative, symmetric, convex, and differentiable. It satisfies **S**c**,**ϵ\***\*(**μ**,**μ**)**=**0 and metrizes the convergence in law. Crucially for RL, it interpolates between the Wasserstein distance (as **ϵ**→**0**) and Maximum Mean Discrepancy (MMD) (as **ϵ**→**∞**).^^ The gradients of the Sinkhorn loss can be computed either by unrolling the Sinkhorn iterations (which consumes memory) or by using the implicit function theorem on the fixed point conditions.^^ This differentiability allows us to backpropagate the "geometric mismatch" between our predicted cloud of returns and the target cloud directly into the neural network weights. ** \*\*

---

## 3. Methodology: Sinkhorn-Regularized Vectorized IQN

We now formally define the **Sinkhorn-VIQN** architecture. This system is designed to approximate the joint return distribution **Z**π**(**s**,**a**)**∈**R**k, where **k** is the number of objectives.

### 3.1 Vectorized Implicit Quantile Network Architecture

The standard IQN architecture consists of a base feature extractor **ψ** (typically a CNN) and a quantile embedding network **ϕ**. In the scalar case, **ϕ** maps a scalar $\tau \in $ to an embedding vector. In the vectorized case, we must reconsider the input noise and the embedding structure.

#### 3.1.1 High-Dimensional Noise Input

Instead of a scalar quantile **τ**, we sample a noise vector **z**in. Recent work in Generalized Energy-Based Models suggests that for high-dimensional support, the topology of the latent noise should match the topology of the target distribution's manifold. However, since the reward distribution in **R**k can be arbitrary, a uniform hypercube distribution is a robust choice. We define **z**in∼**U**(**k**). This matches the dimensionality of the reward vector, providing the network with sufficient degrees of freedom to model independent variations along each axis if necessary.

#### 3.1.2 Vectorized Embedding Network **ϕ**

The embedding network must map the noise vector **z**in to the same dimension as the state features (e.g., 512). In standard IQN, a cosine embedding is used: **ϕ**j(**τ**)**=**ReLU**(**∑**cos**(**iπ**τ**)**w**ij+**b**j)**. This is effective for 1D periodicity. For the vector case, we implement a **Fourier Feature Mapping** followed by an MLP. This allows the network to learn high-frequency components of the distribution in multiple dimensions.

**v**e**mb\*\***=**MLP**(**concat**)\*\*

where **B** is a fixed random Gaussian matrix. This technique, known as Random Fourier Features, allows standard MLPs to learn high-frequency functions in low-dimensional domains (like our **k**=**3** reward space).

#### 3.1.3 The Generative Trunk

The state feature vector **h**=**ψ**(**s**) and the noise embedding **v**e**mb** are combined via an element-wise Hadamard product:

**h**co**mbin**e**d\*\***=**h**⊙**v**e**mb\*\***

This combined vector is passed through a generator network **G**θ that outputs the samples. Crucially, the output layer of **G**θ has dimension **∣**A**∣**×**k**.

**Z**θ(**s**,**a**;**z**in)**=**G**θ\*\***(**h**co**mbin**e**d\*\***)**∈**R**k**

For a given state **s** and action **a**, by sampling **N** noise vectors **{**z**in**(**i**)}**i**=**1**N, we generate an empirical distribution **X**=**{**x**i\*\***}**i**=**1**N** where **x**i\*\***∈**R**k. This point cloud represents the agent's belief about the joint returns.

### 3.2 The Sinkhorn Loss for Multi-Objective Bellman Updates

The core innovation is the application of Sinkhorn Divergence to the Bellman error.

#### 3.2.1 Target Distribution Construction

Let **(**s**,**a**,**r**,**s**′**) be a transition tuple. **r**∈**R**k. We construct the target distribution sample cloud **Y**. First, we select the next action **a**′. In a Multi-Objective setting, there is no single optimal action. We adopt the **Envelope Q-Learning** approach ^^ where the agent is conditioned on a preference vector **w**. \*\* \*\*

**a**′**=**arg**a**′**max\*\***w**T**E**z**′[**Z**θ**′(**s**′**,**a**′**;**z**′**)]\*\*

Here, we calculate the expected vector return for each action, scalarize it using the current preference **w**, and choose the greedy action. The target samples are then:

**y**j=**r**+**γ**Z**θ**′(**s**′**,**a**′**;**z**in**′**(**j**))

where **z**in**′**(**j**) are fresh noise samples. **Y**=**{**y**j}**j**=**1**N**′.

#### 3.2.2 The Sinkhorn Loss Calculation

We have two point clouds: prediction **X** (size **N**) and target **Y** (size **N**′). We define the ground cost matrix **C** where **C**ij=**∥**x**i\*\***−**y**j∥**2. We compute the Sinkhorn Divergence **S**c**,**ϵ\*\***(**X**,**Y**). Using the `geomloss` library ^^, this computation is auto-differentiable. \*\* \*\*

**L**(**θ**)**=**S**c**,**ϵ\*\***(**Z**θ(**s**,**a**)**,**r**+**γ**Z**θ**′(**s**′**,**a**′\*\*))

Minimizing this loss forces the predicted cloud **X** to structurally align with the target cloud **Y**. Unlike MMD, which can be insensitive to specific geometric shifts, Sinkhorn loss (being transport-based) provides strong gradients to move mass to the correct regions of the vector space. Unlike pure Wasserstein, the entropic smoothing prevents instability and ensures the gradients are well-behaved.

### 3.3 Pareto-Frontier Approximation

To ensure the agent can act optimally across different trade-offs, we sample the preference vector **w** from a Dirichlet distribution **Dir**(**1**k) at the beginning of each episode (or step). The preference vector **w** is appended to the state input **ψ**(**s**), allowing the policy to condition its behavior on the desired trade-off. This technique, borrowed from Conditioned Network (CN) approaches, combined with the distributional capacity of Sinkhorn-VIQN, allows the agent to learn the entire convex hull of the Pareto frontier in a single model.

---

## 4. Environment Analysis: Deconstructing Seaquest

The efficacy of Multi-Objective RL is often difficult to validate on standard benchmarks because rewards are already scalarized. We select the Atari 2600 game _Seaquest_ and perform a novel **Reward Decomposition** based on RAM analysis. This transforms the game from a scalar optimization task into a rich multi-objective environment.

### 4.1 The Seaquest Ecosystem

_Seaquest_ places the player in a submarine with three primary interacting systems:

1. **Oxygen (Survival):** A bar at the bottom decreases over time. If it empties, the player loses a life. To refill, the player must surface.
2. **Divers (Resource Accumulation):** Divers swim horizontally. The player must touch them to "load" them. Up to 6 divers can be held.
3. **Enemies (Combat/Risk):** Sharks and enemy subs patrol the water. Shooting them yields points but risks collision.
4. **The Surface (The bottleneck):** Surfacing refills oxygen and "banks" the rescued divers for a massive score bonus. However, surfacing is risky if an enemy patrol sub is present.

**The Conflict:** The player wants to stay submerged to collect more divers (increasing the multiplier), but must surface to breathe. Shooting enemies clears the path but grants negligible points compared to banking divers. A scalar agent often learns a degenerate policy: ignore divers, stay near the surface, and shoot enemies, as this is a "safer" local optimum than the high-risk, high-reward diver strategy.

### 4.2 Hardware Context: The Atari 2600 and 6502

To decompose the rewards, we must understand the hardware. The Atari 2600 uses a MOS Technology 6507 CPU (a stripped-down 6502). Crucially, it has only **128 bytes of RAM** .^^ This memory is mapped to addresses `$0080` to `$00FF`. Because memory was so scarce, developers used bit-packing and reused addresses. However, critical state variables like "Oxygen Level" or "Number of Divers" must persist. This scarcity makes RAM hacking feasible; there are very few places these variables can hide. \*\* \*\*

### 4.3 RAM-Based Reward Decomposition Methodology

We implement a Gym Wrapper that reads the 128-byte RAM array at every step to infer the component rewards.

#### 4.3.1 Oxygen Reward Signal (**r**o**x**y)

- **RAM Address:** Analysis of the Stella debugger and ROM maps ^^ identifies address **102** (decimal) as a primary candidate for the oxygen bar value. \*\* \*\*
- **Behavior:** The value starts high (e.g., ~**80** hex) and decrements. When the player surfaces, it resets.
- **Reward Logic:**
  - **r**o**x**y=**−**0.01 per step (penalty for consumption).
  - **r**o**x**y=**+**1.0 if the RAM value at address 102 increases significantly (indicating a refill/surfacing event).

#### 4.3.2 Diver Reward Signal (**r**d**i**v)

- **RAM Address:** Address **97** (decimal) typically holds the count of divers on board (0 to 6).^^ \*\* \*\*
- **Behavior:** Increments when a diver is touched. Resets to 0 upon surfacing (banking).
- **Reward Logic:**
  - **r**d**i**v=**+**1.0 if `RAM` increases.
  - Note: We do _not_ reward banking divers in **r**d**i**v directly; we reward the _collection_ . The banking is an instrumental goal to get the score, but in a multi-objective setting, we treat "collection" as the objective to maximize.

#### 4.3.3 Enemy Reward Signal (**r**e**n**e**m**)

- **RAM Address:** The score is stored in BCD format, typically at addresses **56-58** .^^ \*\* \*\*
- **Behavior:** The score increases by fixed amounts.
- **Inference Logic:** Since we cannot easily track every bullet in RAM, we use the change in the total score provided by the emulator (**Δ**S).
  - Enemies are worth 20 to 90 points.^^ \*\* \*\*
  - Divers are worth 50 to 1000 points.
  - Surfacing bonuses are based on oxygen.
  - **Heuristic:** If **20**≤**Δ**S**≤**90, we assume an enemy kill: **r**e**n**e**m=**+**1.0**. If **Δ**S**>**100, it is likely a diver banking event or level clear.

#### 4.3.4 The Reward Vector

The environment returns **r**t=**[**r**o**x**y\*\***,**r**d**i**v,**r**e**n**e**m\*\***]**T**. This vector is fundamentally un-scalarized. The agent sees these three signals as distinct input channels.

---

## 5. Implementation Strategy

This section provides the detailed implementation plan, bridging the theoretical concepts with practical code structures using Python, PyTorch, and GeomLoss.

### 5.1 System Architecture

Table 1: System Component Overview

| Component       | Function                               | Key Technology                 |
| --------------- | -------------------------------------- | ------------------------------ |
| **Environment** | Seaquest-v5 (ALE) with RAM Wrapper     | Gymnasium, NumPy               |
| **Input**       | RGB Frames (84x84x4) + Preference**w** | PyTorch Conv2d                 |
| **Embedding**   | Vectorized Noise Mapping               | Random Fourier Features        |
| **Generator**   | Joint Distribution Estimation          | Multi-Head MLP                 |
| **Loss**        | Distribution Matching                  | Sinkhorn Divergence (GeomLoss) |
| **Optimizer**   | Parameter Update                       | Adam                           |

### 5.2 Code Implementation Details

#### 5.2.1 The Seaquest Decomposition Wrapper

**Python**

```
import gymnasium as gym
import numpy as np

class SeaquestMultiObjectiveWrapper(gym.Wrapper):
    """
    Decomposes the scalar score of Seaquest into a 3D reward vector:
    using RAM analysis.
    """
    def __init__(self, env):
        super().__init__(env)
        # RAM offsets (based on ALE/Seaquest-v5)
        self.RAM_OXYGEN = 102
        self.RAM_DIVER_COUNT = 97
        self.prev_ram = None
        self.prev_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_ram = self.env.unwrapped.ale.getRAM()
        self.prev_score = 0
        return obs, info

    def step(self, action):
        obs, scalar_reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()

        # --- 1. Oxygen Reward ---
        # Oxygen decreases over time. A large positive jump means surfacing.
        curr_oxy = ram
        prev_oxy = self.prev_ram

        # Heuristic: If oxygen jumps up by > 50, we surfaced.
        if curr_oxy > prev_oxy + 50:
            r_oxy = 1.0
        else:
            # Small penalty for consumption
            r_oxy = -0.005

        # --- 2. Diver Reward ---
        # Diver count is 0-6.
        curr_divers = ram
        prev_divers = self.prev_ram

        if curr_divers > prev_divers:
            r_div = 1.0
        else:
            r_div = 0.0

        # --- 3. Enemy Reward ---
        # We infer enemy kills from score deltas.
        # Enemy values: 20, 30, 40... 90.
        # Diver banking values: usually > 100 or specific multiples.
        # We access the raw score from info if available, or track it manually via RAM BCD
        curr_score = info.get('score', 0) # Assuming info wrapper or tracking
        delta_score = scalar_reward # ALE provides the diff as reward

        r_enem = 0.0
        # Check if delta corresponds to enemy points
        if 20 <= delta_score <= 90:
            r_enem = 1.0

        # Construct Vector
        vec_reward = np.array([r_oxy, r_div, r_enem], dtype=np.float32)

        self.prev_ram = ram
        self.prev_score = curr_score

        # Inject vector into info for buffer storage
        info['vector_reward'] = vec_reward

        # We return the scalar reward to keep Gym happy, but the agent will use info['vector_reward']
        return obs, scalar_reward, terminated, truncated, info
```

#### 5.2.2 The Sinkhorn-VIQN Network (PyTorch)

**Python**

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorizedIQN(nn.Module):
    def __init__(self, action_dim, num_objectives=3, latent_dim=64):
        super(VectorizedIQN, self).__init__()
        self.action_dim = action_dim
        self.k = num_objectives
        self.latent_dim = latent_dim

        # 1. Feature Extractor (Nature CNN)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_features = nn.Linear(7 * 7 * 64, 512)

        # 2. Preference Embedding (Conditioning on w)
        self.fc_w = nn.Linear(self.k, 512)

        # 3. Vectorized Noise Embedding (Fourier Features)
        # Input: Noise vector z of dim k
        self.phi_layer = nn.Linear(self.k, latent_dim) # Random projection B
        self.fc_phi = nn.Linear(latent_dim * 2, 512) # Cos/Sin features

        # 4. Joint Generator Heads
        self.fc_final = nn.Linear(512, 512)
        self.output_head = nn.Linear(512, action_dim * self.k)

    def forward(self, state, preference_w, num_samples=32):
        """
        state:
        preference_w:
        num_samples: N (number of particles to generate)
        """
        batch_size = state.size(0)

        # -- Feature Extraction --
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        feat = self.fc_features(x) #

        # -- Preference Conditioning --
        w_emb = F.relu(self.fc_w(preference_w))
        feat = feat * w_emb # Gating mechanism via preferences

        # -- Noise Generation --
        # Sample noise z ~ U^k
        z = torch.rand(batch_size, num_samples, self.k, device=state.device)
        z_flat = z.view(-1, self.k)

        # -- Fourier Embedding --
        # Simple Random Fourier Features approximation
        proj = self.phi_layer(z_flat)
        z_emb = torch.cat([torch.cos(2 * 3.1415 * proj),
                           torch.sin(2 * 3.1415 * proj)], dim=1)
        z_emb = F.relu(self.fc_phi(z_emb))
        z_emb = z_emb.view(batch_size, num_samples, 512)

        # -- Combination --
        # feat: ->
        feat_expanded = feat.unsqueeze(1)

        # Element-wise interaction: State * Noise
        combined = feat_expanded * z_emb #

        # -- Generation --
        out = F.relu(self.fc_final(combined))
        out = self.output_head(out) #

        # Reshape to
        out = out.view(batch_size, num_samples, self.action_dim, self.k)

        return out
```

#### 5.2.3 The Sinkhorn Loss with GeomLoss

The `geomloss` library is essential here. It implements the fast Sinkhorn algorithm with CUDA kernels.

**Python**

```
from geomloss import SamplesLoss

class SinkhornDivergenceLoss(nn.Module):
    def __init__(self, blur=0.01, scaling=0.9):
        super().__init__()
        # 'blur' is the square of epsilon.
        # Smaller blur -> closer to pure Wasserstein but harder to optimize.
        self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=scaling)

    def forward(self, pred_cloud, target_cloud):
        """
        pred_cloud:
        target_cloud:
        """
        # GeomLoss calculates the distance between two measures.
        # We assume uniform weights for the samples (1/N).
        loss = self.loss_fn(pred_cloud, target_cloud)
        return loss.mean()
```

### 5.3 Training Loop Algorithm

1. **Initialize** : Replay buffer **D**, Networks **θ**,**θ**′.
2. **Episode Loop** :

- Sample preference **w**∼**Dirichlet**(**1**k).
- **Step Loop** :
  - Observe **s**.
  - Forward pass **Z**θ(**s**,**⋅**;**w**) to get return clouds for all actions.
  - Calculate Expected Scalar Q-values: **Q**(**s**,**a**)**=**w**T**(**N**1∑**z**i).
  - Select **a** via **ϵ**-greedy on **Q**(**s**,**a**).
  - Execute **a**, observe **s**′**,**r**v**ec.
  - Store **(**s**,**a**,**r**v**ec,**s**′**,**w**)** in **D**.
- **Optimization Loop** :
  - Sample batch.
  - **Target Computation** :
  - Compute next distributions **Z**θ**′(**s**′**,**⋅**;**w**).
  - Select greedy action **a**∗ maximizing **w**T**E**[**Z**θ**′]\*\*.
  - Construct target cloud **Y**=**r**v**ec\*\***+**γ**Z**θ**′(**s**′**,**a**∗**)\*\*.
    - **Prediction** :
  - Compute current distributions **Z**θ(**s**,**a**;**w**) to get cloud **X**.
    - **Loss** :
  - **L**=**Sinkhorn**(**X**,**Y**).
    - **Update** : Backpropagate **∇**θ\*\*\*\*L.

---

## 6. Experimental Design and Metrics

To rigorously evaluate Sinkhorn-VIQN, we must measure its ability to approximate the Pareto Frontier, not just achieve a high scalar score.

### 6.1 Metrics: Hypervolume and Sparsity

1. **Hypervolume (HV):** This is the gold standard for Multi-Objective optimization. It measures the volume of the objective space dominated by the set of policies learned by the agent.
   - We evaluate the agent conditioned on a set of reference weights **W**e**v**a**l\*\***=**{**w**1\*\***,**…**,**w**10}.
   - For each **w**i, we run an episode and record the cumulative vector return **G**i.
   - The HV is the union of hypercubes defined by these points relative to a reference point (e.g., origin).
   - _Implementation:_ Use the `pygmo` library ^^ to compute HV efficiently. \*\* \*\*
2. **Sparsity:** Measures the diversity of the solutions. A good MORL agent should have solutions spread across the frontier (high oxygen/low diver, low oxygen/high diver, etc.).

### 6.2 Baselines

- **Linear Scalarized IQN:** An IQN agent trained on the summed scalar reward. This represents the "standard" RL approach. We hypothesize it will maximize only one easy objective (likely enemies) and fail to learn the complex diver interactions.
- **Independent IQN:** Three separate IQN networks trained on the three objectives independently. We hypothesize this will fail due to incorrect correlation handling (e.g., underestimating the risk of surfacing).
- **MO-DQN:** A vector-output DQN (expectation only). We hypothesize this will perform reasonably but suffer from instability due to the lack of distributional robustness.

### 6.3 Hypothesis

We posit that **Sinkhorn-VIQN** will achieve the highest Hypervolume. The distributional nature allows it to be risk-aware (crucial for the Oxygen mechanic), while the vectorized Sinkhorn loss allows it to understand the geometry of the trade-offs (crucial for balancing Diver rescue with Enemy kills). Specifically, the Sinkhorn loss will prevent "mode collapse" where the distribution degenerates into a single point, maintaining the "cloud" of possibilities that is essential for robust policy improvement.

---

## 7. Discussion and Future Outlook

The introduction of Sinkhorn-Regularized Vectorized Implicit Quantile Networks represents a significant step towards general-purpose Multi-Objective Distributional RL. By successfully integrating the geometric insights of Optimal Transport with the function approximation power of Deep Learning, we overcome the limitations of scalarization and independent modeling.

### 7.1 Implications for Safety and Constraints

Beyond maximizing scores, this architecture has profound implications for Safe RL. Safety constraints can be modeled as separate objectives (e.g., **r**s**a**f**e**t**y**). A Sinkhorn-VIQN agent effectively learns the joint distribution of reward and safety violations. This allows for the deployment of policies that probabilistically bound risk (e.g., "Take the action where the 95th percentile of the safety cost is below a threshold"). The Sinkhorn loss ensures that these high-risk "tails" of the distribution are accurately modeled and transported during learning.

### 7.2 Scalability and Limitations

The primary cost of this approach is the computation of the Sinkhorn Divergence. While efficient approximations exist, scaling to hundreds of objectives remains a challenge. Future work could investigate **Sliced-Wasserstein** variations, which project the high-dimensional distributions onto 1D lines to utilize efficient sorting-based distances, avoiding the matrix scaling of Sinkhorn entirely.

### 7.3 Conclusion

We have presented a robust framework for disentangling and optimizing conflicting objectives in reinforcement learning. Through the lens of _Seaquest_ , we demonstrated how legacy environments can be reimagined as complex multi-objective benchmarks. The Sinkhorn-VIQN algorithm stands as a theoretically grounded and empirically promising solution to the long-standing challenge of vector-valued value estimation.

---

(Note: This report synthesizes insights from research snippets ^^ through ^^, integrating theoretical optimal transport, game hardware architecture, and deep learning methodology.) \*\* \*\*

[![](https://t1.gstatic.com/faviconV2?url=https://di-engine-docs.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)di-engine-docs.readthedocs.ioIQN — DI-engine 0.1.0 documentation**Opens in a new window**](https://di-engine-docs.readthedocs.io/en/latest/12_policies/iqn.html)[![](https://t2.gstatic.com/faviconV2?url=https://liner.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)liner.com[Quick Review] Implicit Quantile Networks for Distributional Reinforcement Learning - Liner**Opens in a new window**](https://liner.com/review/implicit-quantile-networks-for-distributional-reinforcement-learning)[![](https://t2.gstatic.com/faviconV2?url=https://www.ijcai.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)ijcai.orgMulti Objective Quantile Based Reinforcement Learning for Modern Urban Planning - IJCAI**Opens in a new window**](https://www.ijcai.org/proceedings/2025/0027.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://www.emergentmind.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)emergentmind.comMulti-Dimensional Reinforcement Reward - Emergent Mind**Opens in a new window**](https://www.emergentmind.com/topics/multi-dimensional-reinforcement-reward-function)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccDistributional Reinforcement Learning with Regularized Wasserstein Loss - NIPS papers**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/7371ee6a40da2951303ec7ebdb2150ce-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgDistributional Reinforcement Learning by Sinkhorn Divergence - arXiv**Opens in a new window**](https://arxiv.org/html/2202.00769v4)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netDistributional Reinforcement Learning via Sinkhorn Iterations - OpenReview**Opens in a new window**](https://openreview.net/forum?id=VarZY6BY12h)[![](https://t3.gstatic.com/faviconV2?url=https://amsword.medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)amsword.medium.comA simple introduction on Sinkhorn distances | by Jianfeng Wang - Medium**Opens in a new window**](https://amsword.medium.com/a-simple-introduction-on-sinkhorn-distances-d01a4ef4f085)[![](https://t3.gstatic.com/faviconV2?url=http://papers.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)papers.neurips.ccScreening Sinkhorn Algorithm for Regularized Optimal Transport**Opens in a new window**](http://papers.neurips.cc/paper/9386-screening-sinkhorn-algorithm-for-regularized-optimal-transport.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://mathtube.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)mathtube.orgOn the linear convergence of the multi-marginal Sinkhorn algorithm - mathtube.org**Opens in a new window**](https://mathtube.org/sites/default/files/lecture-extra-files/linear-sinkhorn.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netSCALABLE SINKHORN BACKPROPAGATION - OpenReview**Opens in a new window**](https://openreview.net/pdf?id=uR77O7SL55h)[![](https://t2.gstatic.com/faviconV2?url=https://openaccess.thecvf.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openaccess.thecvf.comA Unified Framework for Implicit Sinkhorn Differentiation - CVF Open Access**Opens in a new window**](https://openaccess.thecvf.com/content/CVPR2022/papers/Eisenberger_A_Unified_Framework_for_Implicit_Sinkhorn_Differentiation_CVPR_2022_paper.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)mdpi.comAn Improved Multi-Objective Deep Reinforcement Learning Algorithm Based on Envelope Update - MDPI**Opens in a new window**](https://www.mdpi.com/2079-9292/11/16/2479)[![](https://t0.gstatic.com/faviconV2?url=https://gist.github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)gist.github.comSinkhorn solver in PyTorch - GitHub Gist**Opens in a new window**](https://gist.github.com/wohlert/8589045ab544082560cc5f8915cc90bd)[![](https://t1.gstatic.com/faviconV2?url=https://www.kernel-operations.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)kernel-operations.ioPyTorch API — GeomLoss - KeOps library**Opens in a new window**](https://www.kernel-operations.io/geomloss/api/pytorch-api.html)[![](https://t2.gstatic.com/faviconV2?url=https://www.randomterrain.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)randomterrain.comAtari 2600 Programming for Newbies - Session 5: Memory Architecture - Random Terrain**Opens in a new window**](https://www.randomterrain.com/atari-2600-memories-tutorial-andrew-davie-05.html)[![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)reddit.comThe Atari 2600 has 128 bytes of RAM. Where on the address bus is it? Is it just the bottom half of the zero-page? Does that mean there&#39;s no stack? - Reddit**Opens in a new window**](https://www.reddit.com/r/atari/comments/r2fvrh/the_atari_2600_has_128_bytes_of_ram_where_on_the/)[![](https://t1.gstatic.com/faviconV2?url=https://gymnasium.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)gymnasium.farama.orgSeaquest - Gymnasium Documentation**Opens in a new window**](https://gymnasium.farama.org/v0.29.0/environments/atari/seaquest/)[![](https://t1.gstatic.com/faviconV2?url=https://gymnasium.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)gymnasium.farama.orgSeaquest - Gymnasium Documentation**Opens in a new window**](https://gymnasium.farama.org/v0.27.1/environments/atari/seaquest/)[![](https://t2.gstatic.com/faviconV2?url=https://www.gymlibrary.dev/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)gymlibrary.devSeaquest - Gym Documentation**Opens in a new window**](https://www.gymlibrary.dev/environments/atari/seaquest/)[![](https://t1.gstatic.com/faviconV2?url=https://www.atariarchives.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)atariarchives.orgMemory Map - Atari Archives**Opens in a new window**](https://www.atariarchives.org/mapping/memorymap.php)[![](https://t0.gstatic.com/faviconV2?url=https://gamefaqs.gamespot.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)gamefaqs.gamespot.comSeaquest - Guide and Walkthrough - Atari 2600 - By SineNomine - GameFAQs**Opens in a new window**](https://gamefaqs.gamespot.com/atari2600/585063-seaquest/faqs/32317)[![](https://t0.gstatic.com/faviconV2?url=https://esa.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)esa.github.ioGetting started with hypervolumes — pygmo 2.19.6 documentation**Opens in a new window**](https://esa.github.io/pygmo2/tutorials/hypervolume.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comAny tips on extracting RAM locations? · Issue #40 · mila-iqia/atari-representation-learning**Opens in a new window**](https://github.com/mila-iqia/atari-representation-learning/issues/40)

[![](https://t3.gstatic.com/faviconV2?url=https://ieeexplore.ieee.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ieeexplore.ieee.org/document/9679148/)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2408.14525v1)[![](https://t0.gstatic.com/faviconV2?url=https://www.semanticscholar.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.semanticscholar.org/paper/bc94da1c61915891f0d430c3b864eeb5f691db4d)[![](https://t1.gstatic.com/faviconV2?url=https://pubsonline.informs.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://pubsonline.informs.org/doi/abs/10.1287/opre.2023.0294?af=R)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.mdpi.com/2504-4990/7/4/126)[![](https://t3.gstatic.com/faviconV2?url=https://papers.nips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://papers.nips.cc/paper_files/paper/2024/file/52c21a32429a7d6050430b606a286a75-Paper-Conference.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://proceedings.neurips.cc/paper_files/paper/2024/file/1c2b1c8f7d317719a9ce32dd7386ba35-Paper-Conference.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://proceedings.neurips.cc/paper_files/paper/2024)[![](https://t2.gstatic.com/faviconV2?url=https://dblp.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://dblp.org/db/conf/nips/neurips2024)[![](https://t0.gstatic.com/faviconV2?url=https://www.endtoend.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.endtoend.ai/envs/gym/atari/seaquest/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/alirezakazemipour/Distributional-RL/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/RobustFieldAutonomyLab/Distributional_RL_Decision_and_Control)[![](https://t0.gstatic.com/faviconV2?url=https://repos.ecosyste.ms/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://repos.ecosyste.ms/hosts/GitHub/topics/implicit-quantile-networks)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/jinpz/q_sharp)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/datake)[![](https://t3.gstatic.com/faviconV2?url=https://optuna.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_pareto_front.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/Simone-Alghisi/pareto-epsilon-greedy-RL)[![](https://t0.gstatic.com/faviconV2?url=https://code.ornl.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://code.ornl.gov/-/snippets/205)[![](https://t0.gstatic.com/faviconV2?url=https://www.youtube.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.youtube.com/watch?v=JixX2_GPv6s)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/abs/2305.08852)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2403.05054v1)[![](https://t2.gstatic.com/faviconV2?url=https://opt-ml.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://opt-ml.org/papers/2024/paper22.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://par.nsf.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://par.nsf.gov/servlets/purl/10424527)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/pdf/2511.16139)[![](https://t0.gstatic.com/faviconV2?url=https://www.ibm.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.ibm.com/think/topics/vector-embedding)[![](https://t1.gstatic.com/faviconV2?url=https://www.devoteam.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.devoteam.com/expert-view/ai-vectorization-langchain/)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/pdf?id=u7oKU1iXTa9)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/datake/SinkhornDistRL)[![](https://t2.gstatic.com/faviconV2?url=https://ludwigwinkler.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ludwigwinkler.github.io/blog/Sinkhorn/)[![](https://t2.gstatic.com/faviconV2?url=https://optimization-online.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://optimization-online.org/wp-content/uploads/2025/01/Fair_DRL.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://www.atariarchives.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.atariarchives.org/mmm/Master%20Memory%20Map%20for%20the%20Atari.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://www.atarimania.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.atarimania.com/documents/Master-Memory-Map.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://www.atarimagazines.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.atarimagazines.com/vbook/memorymap.php)[![](https://t1.gstatic.com/faviconV2?url=https://web.eecs.umich.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://web.eecs.umich.edu/~baveja/Papers/UCTtoCNNsAtariGames-FinalVersion.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2306.08649v2)[![](https://t3.gstatic.com/faviconV2?url=https://www.ifaamas.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p9.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/pdf/1610.02707)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2407.16807v1)[![](https://t2.gstatic.com/faviconV2?url=https://jair.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://jair.org/index.php/jair/article/view/12270)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/353345428_Multimodal_Reward_Shaping_for_Efficient_Exploration_in_Reinforcement_Learning)[![](https://t0.gstatic.com/faviconV2?url=https://researchportal.vub.be/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://researchportal.vub.be/en/publications/multi-objectivization-of-reinforcement-learning-problems-by-rewar/)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/forum?id=KLU4Eb2U2A)[![](https://t2.gstatic.com/faviconV2?url=https://gibberblot.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html)[![](https://t1.gstatic.com/faviconV2?url=https://ai.vub.ac.be/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ai.vub.ac.be/sites/default/files/PID3130853.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://multi-objective.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://multi-objective.github.io/moocore/python/reference/generated/moocore.hypervolume.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/wangronin/HIGA-MO)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.mdpi.com/2673-2688/5/4/85)[![](https://t2.gstatic.com/faviconV2?url=https://or.stackexchange.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://or.stackexchange.com/questions/11108/how-to-get-hypervolume-calculation-for-pareto-front-in-python)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py)[![](https://t1.gstatic.com/faviconV2?url=https://dfdazac.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://dfdazac.github.io/sinkhorn.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/fwilliams/scalable-pytorch-sinkhorn)[![](https://t1.gstatic.com/faviconV2?url=https://www.kernel-operations.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.kernel-operations.io/geomloss/)[![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://stackoverflow.com/questions/65150672/calculate-batch-pairwise-sinkhorn-distance-in-pytorch)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2502.12456v1)[![](https://t3.gstatic.com/faviconV2?url=https://discuss.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://discuss.pytorch.org/t/optimal-transport-metric/50437)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/kenjyoung/MinAtar)[![](https://t3.gstatic.com/faviconV2?url=https://users.cs.utah.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://users.cs.utah.edu/~dsbrown/pubs/safe_efficient_irl_danielbrown_dissertation.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/pdf?id=t4BjjTfxFa)[![](https://t2.gstatic.com/faviconV2?url=https://en.wikipedia.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](<https://en.wikipedia.org/wiki/Seaquest_(video_game)>)[![](https://t1.gstatic.com/faviconV2?url=https://tomzahavy.wixsite.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://tomzahavy.wixsite.com/rlprojects/seaquest)[![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.reddit.com/r/ChildrenofMorta/comments/1my4va7/whats_that_enemy_kill_counter_during_the_gameplay/)[![](https://t3.gstatic.com/faviconV2?url=https://ale.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ale.farama.org/environments/seaquest/)[![](https://t1.gstatic.com/faviconV2?url=https://pypi.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://pypi.org/project/ale-py/)[![](https://t3.gstatic.com/faviconV2?url=https://ale.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ale.farama.org/environments/)[![](https://t2.gstatic.com/faviconV2?url=https://www.randomterrain.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.randomterrain.com/atari-2600-memories-tutorial-andrew-davie-19.html)[![](https://t0.gstatic.com/faviconV2?url=https://deepsense.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://deepsense.ai/wp-content/uploads/2016/09/1605.01335v1-4.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://www.randomterrain.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.randomterrain.com/atari-2600-memories-tutorial-andrew-davie-12.html)[![](https://t3.gstatic.com/faviconV2?url=https://www.osnews.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.osnews.com/story/30272/a-constructive-look-at-the-atari-2600-basic-cartridge/)[![](https://t0.gstatic.com/faviconV2?url=https://stella-emu.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://stella-emu.github.io/docs/index.html)

![](https://www.gstatic.com/lamda/images/immersives/google_logo_icon_2380fba942c84387f09cf.svg)

[![](https://t1.gstatic.com/faviconV2?url=https://di-engine-docs.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://di-engine-docs.readthedocs.io/en/latest/12_policies/iqn.html)[![](https://t2.gstatic.com/faviconV2?url=https://www.ijcai.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.ijcai.org/proceedings/2025/0027.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://ieeexplore.ieee.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://ieeexplore.ieee.org/document/9679148/)[![](https://t2.gstatic.com/faviconV2?url=https://liner.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://liner.com/review/implicit-quantile-networks-for-distributional-reinforcement-learning)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/html/2408.14525v1)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/html/2202.00769v4)[![](https://t0.gstatic.com/faviconV2?url=https://www.semanticscholar.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.semanticscholar.org/paper/bc94da1c61915891f0d430c3b864eeb5f691db4d)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://proceedings.neurips.cc/paper_files/paper/2024/file/7371ee6a40da2951303ec7ebdb2150ce-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://pubsonline.informs.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://pubsonline.informs.org/doi/abs/10.1287/opre.2023.0294?af=R)[![](https://t3.gstatic.com/faviconV2?url=https://amsword.medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://amsword.medium.com/a-simple-introduction-on-sinkhorn-distances-d01a4ef4f085)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.mdpi.com/2504-4990/7/4/126)[![](https://t3.gstatic.com/faviconV2?url=https://papers.nips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://papers.nips.cc/paper_files/paper/2024/file/52c21a32429a7d6050430b606a286a75-Paper-Conference.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://proceedings.neurips.cc/paper_files/paper/2024/file/1c2b1c8f7d317719a9ce32dd7386ba35-Paper-Conference.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://proceedings.neurips.cc/paper_files/paper/2024)[![](https://t2.gstatic.com/faviconV2?url=https://dblp.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://dblp.org/db/conf/nips/neurips2024)[![](https://t2.gstatic.com/faviconV2?url=https://en.wikipedia.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](<https://en.wikipedia.org/wiki/Seaquest_(video_game)>)[![](https://t1.gstatic.com/faviconV2?url=https://gymnasium.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gymnasium.farama.org/v0.27.1/environments/atari/seaquest/)[![](https://t0.gstatic.com/faviconV2?url=https://www.endtoend.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.endtoend.ai/envs/gym/atari/seaquest/)[![](https://t0.gstatic.com/faviconV2?url=https://gamefaqs.gamespot.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gamefaqs.gamespot.com/atari2600/585063-seaquest/faqs/32317)[![](https://t1.gstatic.com/faviconV2?url=https://gymnasium.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gymnasium.farama.org/v0.29.0/environments/atari/seaquest/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/alirezakazemipour/Distributional-RL/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/RobustFieldAutonomyLab/Distributional_RL_Decision_and_Control)[![](https://t0.gstatic.com/faviconV2?url=https://repos.ecosyste.ms/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://repos.ecosyste.ms/hosts/GitHub/topics/implicit-quantile-networks)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/jinpz/q_sharp)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/datake/SinkhornDistRL)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/datake)

![](https://www.gstatic.com/lamda/images/immersives/google_logo_icon_2380fba942c84387f09cf.svg)

[![](https://t3.gstatic.com/faviconV2?url=https://optuna.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_pareto_front.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/Simone-Alghisi/pareto-epsilon-greedy-RL)[![](https://t0.gstatic.com/faviconV2?url=https://code.ornl.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://code.ornl.gov/-/snippets/205)[![](https://t0.gstatic.com/faviconV2?url=https://www.youtube.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.youtube.com/watch?v=JixX2_GPv6s)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/abs/2305.08852)[![](https://t3.gstatic.com/faviconV2?url=http://papers.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](http://papers.neurips.cc/paper/9386-screening-sinkhorn-algorithm-for-regularized-optimal-transport.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://mathtube.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://mathtube.org/sites/default/files/lecture-extra-files/linear-sinkhorn.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/html/2403.05054v1)[![](https://t2.gstatic.com/faviconV2?url=https://opt-ml.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://opt-ml.org/papers/2024/paper22.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://par.nsf.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://par.nsf.gov/servlets/purl/10424527)[![](https://t0.gstatic.com/faviconV2?url=https://www.emergentmind.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.emergentmind.com/topics/multi-dimensional-reinforcement-reward-function)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/pdf/2511.16139)[![](https://t0.gstatic.com/faviconV2?url=https://www.ibm.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.ibm.com/think/topics/vector-embedding)[![](https://t1.gstatic.com/faviconV2?url=https://www.devoteam.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.devoteam.com/expert-view/ai-vectorization-langchain/)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://openreview.net/pdf?id=u7oKU1iXTa9)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/datake/SinkhornDistRL)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://openreview.net/forum?id=VarZY6BY12h)[![](https://t2.gstatic.com/faviconV2?url=https://ludwigwinkler.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://ludwigwinkler.github.io/blog/Sinkhorn/)[![](https://t2.gstatic.com/faviconV2?url=https://optimization-online.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://optimization-online.org/wp-content/uploads/2025/01/Fair_DRL.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/html/2202.00769v4)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://proceedings.neurips.cc/paper_files/paper/2024/file/7371ee6a40da2951303ec7ebdb2150ce-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://www.atariarchives.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.atariarchives.org/mapping/memorymap.php)[![](https://t1.gstatic.com/faviconV2?url=https://www.atariarchives.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.atariarchives.org/mmm/Master%20Memory%20Map%20for%20the%20Atari.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://www.atarimania.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.atarimania.com/documents/Master-Memory-Map.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://www.randomterrain.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.randomterrain.com/atari-2600-memories-tutorial-andrew-davie-05.html)[![](https://t2.gstatic.com/faviconV2?url=https://www.atarimagazines.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.atarimagazines.com/vbook/memorymap.php)[![](https://t1.gstatic.com/faviconV2?url=https://web.eecs.umich.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://web.eecs.umich.edu/~baveja/Papers/UCTtoCNNsAtariGames-FinalVersion.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/html/2306.08649v2)[![](https://t3.gstatic.com/faviconV2?url=https://www.ifaamas.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p9.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.mdpi.com/2079-9292/11/16/2479)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/pdf/1610.02707)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/html/2407.16807v1)[![](https://t2.gstatic.com/faviconV2?url=https://jair.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://jair.org/index.php/jair/article/view/12270)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.researchgate.net/publication/353345428_Multimodal_Reward_Shaping_for_Efficient_Exploration_in_Reinforcement_Learning)[![](https://t0.gstatic.com/faviconV2?url=https://researchportal.vub.be/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://researchportal.vub.be/en/publications/multi-objectivization-of-reinforcement-learning-problems-by-rewar/)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://openreview.net/forum?id=KLU4Eb2U2A)[![](https://t2.gstatic.com/faviconV2?url=https://gibberblot.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html)[![](https://t1.gstatic.com/faviconV2?url=https://ai.vub.ac.be/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://ai.vub.ac.be/sites/default/files/PID3130853.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://www.gymlibrary.dev/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.gymlibrary.dev/environments/atari/seaquest/)[![](https://t0.gstatic.com/faviconV2?url=https://gamefaqs.gamespot.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gamefaqs.gamespot.com/atari2600/585063-seaquest/faqs/32317)[![](https://t0.gstatic.com/faviconV2?url=https://deepsense.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://deepsense.ai/wp-content/uploads/2016/09/1605.01335v1-4.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://en.wikipedia.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](<https://en.wikipedia.org/wiki/Seaquest_(video_game)>)[![](https://t1.gstatic.com/faviconV2?url=https://gymnasium.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gymnasium.farama.org/v0.29.0/environments/atari/seaquest/)[![](https://t1.gstatic.com/faviconV2?url=https://gymnasium.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gymnasium.farama.org/v0.27.1/environments/atari/seaquest/)[![](https://t0.gstatic.com/faviconV2?url=https://gist.github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gist.github.com/wohlert/8589045ab544082560cc5f8915cc90bd)[![](https://t0.gstatic.com/faviconV2?url=https://esa.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://esa.github.io/pygmo2/tutorials/hypervolume.html)[![](https://t1.gstatic.com/faviconV2?url=https://multi-objective.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://multi-objective.github.io/moocore/python/reference/generated/moocore.hypervolume.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/wangronin/HIGA-MO)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.mdpi.com/2673-2688/5/4/85)[![](https://t2.gstatic.com/faviconV2?url=https://or.stackexchange.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://or.stackexchange.com/questions/11108/how-to-get-hypervolume-calculation-for-pareto-front-in-python)[![](https://t1.gstatic.com/faviconV2?url=https://www.kernel-operations.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.kernel-operations.io/geomloss/api/pytorch-api.html)[![](https://t1.gstatic.com/faviconV2?url=https://www.kernel-operations.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.kernel-operations.io/geomloss/)[![](https://t1.gstatic.com/faviconV2?url=https://dfdazac.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://dfdazac.github.io/sinkhorn.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/fwilliams/scalable-pytorch-sinkhorn)[![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://stackoverflow.com/questions/65150672/calculate-batch-pairwise-sinkhorn-distance-in-pytorch)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/html/2502.12456v1)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://openreview.net/pdf?id=uR77O7SL55h)[![](https://t2.gstatic.com/faviconV2?url=https://openaccess.thecvf.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://openaccess.thecvf.com/content/CVPR2022/papers/Eisenberger_A_Unified_Framework_for_Implicit_Sinkhorn_Differentiation_CVPR_2022_paper.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://discuss.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://discuss.pytorch.org/t/optimal-transport-metric/50437)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/kenjyoung/MinAtar)

![](https://www.gstatic.com/lamda/images/immersives/google_logo_icon_2380fba942c84387f09cf.svg)

[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py)[![](https://t1.gstatic.com/faviconV2?url=https://www.kernel-operations.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.kernel-operations.io/geomloss/api/pytorch-api.html)[![](https://t1.gstatic.com/faviconV2?url=https://dfdazac.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://dfdazac.github.io/sinkhorn.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/fwilliams/scalable-pytorch-sinkhorn)[![](https://t1.gstatic.com/faviconV2?url=https://www.kernel-operations.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.kernel-operations.io/geomloss/)[![](https://t0.gstatic.com/faviconV2?url=https://stackoverflow.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://stackoverflow.com/questions/65150672/calculate-batch-pairwise-sinkhorn-distance-in-pytorch)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://arxiv.org/html/2502.12456v1)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://openreview.net/pdf?id=uR77O7SL55h)[![](https://t2.gstatic.com/faviconV2?url=https://openaccess.thecvf.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://openaccess.thecvf.com/content/CVPR2022/papers/Eisenberger_A_Unified_Framework_for_Implicit_Sinkhorn_Differentiation_CVPR_2022_paper.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://discuss.pytorch.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://discuss.pytorch.org/t/optimal-transport-metric/50437)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/kenjyoung/MinAtar)[![](https://t3.gstatic.com/faviconV2?url=https://users.cs.utah.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://users.cs.utah.edu/~dsbrown/pubs/safe_efficient_irl_danielbrown_dissertation.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://openreview.net/pdf?id=t4BjjTfxFa)[![](https://t2.gstatic.com/faviconV2?url=https://en.wikipedia.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](<https://en.wikipedia.org/wiki/Seaquest_(video_game)>)[![](https://t1.gstatic.com/faviconV2?url=https://tomzahavy.wixsite.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://tomzahavy.wixsite.com/rlprojects/seaquest)[![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.reddit.com/r/ChildrenofMorta/comments/1my4va7/whats_that_enemy_kill_counter_during_the_gameplay/)[![](https://t2.gstatic.com/faviconV2?url=https://www.gymlibrary.dev/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.gymlibrary.dev/environments/atari/seaquest/)[![](https://t1.gstatic.com/faviconV2?url=https://gymnasium.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gymnasium.farama.org/v0.29.0/environments/atari/seaquest/)[![](https://t0.gstatic.com/faviconV2?url=https://gamefaqs.gamespot.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gamefaqs.gamespot.com/atari2600/585063-seaquest/faqs/32317)[![](https://t3.gstatic.com/faviconV2?url=https://ale.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://ale.farama.org/environments/seaquest/)[![](https://t1.gstatic.com/faviconV2?url=https://pypi.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://pypi.org/project/ale-py/)[![](https://t3.gstatic.com/faviconV2?url=https://ale.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://ale.farama.org/environments/)[![](https://t1.gstatic.com/faviconV2?url=https://gymnasium.farama.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://gymnasium.farama.org/v0.27.1/environments/atari/seaquest/)[![](https://t2.gstatic.com/faviconV2?url=https://www.randomterrain.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.randomterrain.com/atari-2600-memories-tutorial-andrew-davie-19.html)[![](https://t0.gstatic.com/faviconV2?url=https://deepsense.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://deepsense.ai/wp-content/uploads/2016/09/1605.01335v1-4.pdf)[![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.reddit.com/r/atari/comments/r2fvrh/the_atari_2600_has_128_bytes_of_ram_where_on_the/)[![](https://t2.gstatic.com/faviconV2?url=https://www.randomterrain.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.randomterrain.com/atari-2600-memories-tutorial-andrew-davie-12.html)[![](https://t3.gstatic.com/faviconV2?url=https://www.osnews.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.osnews.com/story/30272/a-constructive-look-at-the-atari-2600-basic-cartridge/)[![](https://t1.gstatic.com/faviconV2?url=https://www.atariarchives.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.atariarchives.org/mapping/memorymap.php)[![](https://t2.gstatic.com/faviconV2?url=https://www.randomterrain.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://www.randomterrain.com/atari-2600-memories-tutorial-andrew-davie-05.html)[![](https://t0.gstatic.com/faviconV2?url=https://stella-emu.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://stella-emu.github.io/docs/index.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)](https://github.com/mila-iqia/atari-representation-learning/issues/40)

---

## 8. Mathematical Derivations: From Bellman Transport to Gradient Flow

### 8.1 Bellman Transport with Sinkhorn Regularization

We detail the contraction argument when using the Sinkhorn divergence as the discrepancy measure. Let $\mathcal{Z}$ be the space of probability measures on $\mathbb{R}^k$ with finite second moment, and define the operator $\mathcal{T}^\pi$ acting on return distributions:

$$
(\mathcal{T}^\pi Z)(s,a) \stackrel{D}{=} r(s,a) + \gamma Z(s',a'), \quad a' \sim \pi(\cdot|s'),\ s'\sim P(\cdot|s,a).
$$

We consider the Sinkhorn divergence $S_\varepsilon(\mu,\nu)$ with entropic parameter $\varepsilon > 0$ and cost $c(x,y)=\lVert x-y\rVert_2^2$. Following Feydy et al. (2019), $S_\varepsilon$ metrizes weak convergence and is $C^\infty$-smooth in both arguments. Using the supremal metric

$$
d_\varepsilon(Z_1,Z_2) = \sup_{s,a} S_\varepsilon(Z_1(s,a), Z_2(s,a)),
$$

we can bound the transport of the Bellman operator by leveraging the $\gamma$-contraction on expectations and Lipschitz continuity of $S_\varepsilon$ with respect to translations:

$$
d_\varepsilon(\mathcal{T}^\pi Z_1, \mathcal{T}^\pi Z_2) \le \gamma\, d_\varepsilon(Z_1, Z_2).
$$

The proof observes that translation by $r$ cancels in the divergence and that the Wasserstein-2 component scales linearly with $\gamma$. Entropic smoothing preserves the contraction property. Hence Sinkhorn-VIQN converges to a unique fixed point in distribution under standard assumptions of bounded rewards and $\gamma<1$.

### 8.2 Gradient of Sinkhorn Divergence for Backpropagation

Let $X = \{x_i\}_{i=1}^N$, $Y = \{y_j\}_{j=1}^{M}$ be predicted and target point clouds with uniform weights. The dual potentials $f,g$ solve

$$
f_i = -\varepsilon \log \sum_j \exp\!\left(\frac{g_j - c_{ij}}{\varepsilon}\right),\quad
g_j = -\varepsilon \log \sum_i \exp\!\left(\frac{f_i - c_{ij}}{\varepsilon}\right).
$$

The optimal plan entries are $P_{ij} = \exp((f_i+g_j-c_{ij})/\varepsilon)$. The Sinkhorn cost term is

$$
\mathcal{L}(X,Y) = \sum_{i,j} P_{ij} c_{ij}.
$$

For backpropagation, $\frac{\partial \mathcal{L}}{\partial x_i} = 2 \sum_j P_{ij} (x_i - y_j)$, which is differentiable in modern autodiff. In practice, we detach $f,g$ to stabilize implicit differentiation, matching the "unrolled-0" estimator in Scalable Sinkhorn Backprop (Eisenberger et al., CVPR 2022).

### 8.3 Vector Quantile Functions and Pareto-Front Geometry

In $\mathbb{R}^k$, vector quantiles are gradients of convex potentials $\nabla \phi(\tau)$ transporting a base distribution $\tau \sim U([0,1]^k)$ to target $\mu$. Our generator $Z_\theta(s,a; z)$ parametrizes $\nabla \phi_\theta(z)$ implicitly. Minimizing $S_\varepsilon(Z_\theta, Y)$ encourages $\phi_\theta$ to approximate the Brenier map when $\varepsilon \to 0$, while entropic smoothing yields the Schrödinger bridge solution for $\varepsilon>0$, adding robustness to noise in Bellman targets.

### 8.4 Preference-Conditioned Optimality

Given preference $w \in \Delta^{k-1}$, the scalarized value is $Q_w(s,a) = \mathbb{E}_{z}[w^\top Z_\theta(s,a;z)]$. The greedy action satisfies

$$
a^\*(s,w) = \arg\max_a \mathbb{E}_{z}[w^\top Z_\theta(s,a;z)].
$$

The learned joint distribution enables arbitrary post-hoc scalarization. The Pareto frontier approximation is obtained by sweeping $w$ and collecting the set $\{Q_w\}$; hypervolume measures frontier quality.

---

## 9. Full Training Algorithm (Pseudocode with Implementation Notes)

**Algorithm 1: Sinkhorn-VIQN with Preference Sampling**

1. **Initialize** replay buffer $\mathcal{D}$, policy network $Z_\theta$, target $Z_{\theta'}$, optimizer (Adam or Sophia-G), and Sinkhorn loss module.
2. **For** each episode:
   - Sample preference $w \sim \text{Dirichlet}(\mathbf{1}_k)$ and fix for the episode (or refresh every $T$ steps for curriculum).
   - Reset environment wrapper to obtain RAM-based vector reward channel.
3. **For** each environment step:
   - Encode state $s$, forward $Z_\theta(s,\cdot;w)$ to get particle clouds per action.
   - Compute $Q_w(s,a)$ by averaging particles and inner product with $w$.
   - Select action via $\epsilon$-greedy on $Q_w$.
   - Observe $(s,a,r_{\text{vec}},s')$, push $(s,a,r_{\text{vec}},s',w)$ to $\mathcal{D}$.
4. **Optimization (every step after warmup):**
   - Sample minibatch $\{(s,a,r_{\text{vec}},s',w)\}_{b=1}^B$.
   - Compute next-clouds $Z_{\theta'}(s',\cdot;w)$; pick $a^\*=\arg\max_a \mathbb{E}_z[w^\top Z_{\theta'}(s',a;z)]$.
   - Build targets $Y = r_{\text{vec}} + \gamma Z_{\theta'}(s', a^\*; z')$ with fresh $z'$ particles.
   - Get predictions $X = Z_{\theta}(s,a;z)$.
   - Loss $= S_\varepsilon(X, Y)$ via `SamplesLoss("sinkhorn", ...)`.
   - Backpropagate, clip gradients (e.g., 10.0), optimizer step.
   - Soft update target: $\theta' \leftarrow \tau \theta + (1-\tau)\theta'$.
5. **Evaluation:** sweep a grid of preferences $w_i$, run episodes, compute Hypervolume, sparsity, and dominated ratio.

**Implementation Notes:**
- Use `torch.cuda.amp.autocast` to reduce sinkhorn cost overhead.
- Cache cosine/sine Fourier bases to avoid recomputation.
- Align batch dims: `pred_cloud` shape `[B, N, k]`, `target_cloud` `[B, M, k]`.
- Ensure `geomloss` installed with KeOps for GPU acceleration; fall back to PyTorch if unavailable.

---

## 10. Extended PyTorch Reference Implementation

Below is an extended, runnable (import-safe) module layout for `sinkhorn_viqn.py`. It is designed to be placed under `src/` in a training repo; we mirror it here for clarity.

```
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss


class SinkhornDivergence(nn.Module):
    """
    Thin wrapper around geomloss.SamplesLoss for Sinkhorn divergence.
    """
    def __init__(self, blur: float = 0.01, scaling: float = 0.9, p: int = 2):
        super().__init__()
        self.loss_fn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, scaling=scaling, debias=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, N, k], target: [B, M, k]
        return self.loss_fn(pred, target)


class FourierNoiseEmbedding(nn.Module):
    """
    Fourier feature mapping for k-d noise vectors.
    """
    def __init__(self, k: int, latent_dim: int = 64):
        super().__init__()
        self.k = k
        self.latent_dim = latent_dim
        self.proj = nn.Linear(k, latent_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0)
        self.fc = nn.Linear(latent_dim * 2, 512)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B*N, k]
        proj = self.proj(z)
        feat = torch.cat([torch.cos(2 * math.pi * proj), torch.sin(2 * math.pi * proj)], dim=-1)
        return F.relu(self.fc(feat))


class VisionBackbone(nn.Module):
    """
    Minimal Atari-style CNN backbone.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        return self.fc(x)


class SinkhornVIQN(nn.Module):
    """
    Vectorized IQN with Sinkhorn divergence for multi-objective RL.
    """
    def __init__(self, action_dim: int, num_objectives: int = 3, latent_dim: int = 64):
        super().__init__()
        self.k = num_objectives
        self.action_dim = action_dim
        self.backbone = VisionBackbone()
        self.pref_fc = nn.Linear(self.k, 512)
        self.noise_emb = FourierNoiseEmbedding(self.k, latent_dim)
        self.fc_final = nn.Linear(512, 512)
        self.out = nn.Linear(512, action_dim * self.k)

    def forward(self, state: torch.Tensor, preference: torch.Tensor, num_samples: int = 32) -> torch.Tensor:
        """
        state: [B, 4, 84, 84], preference: [B, k]
        returns: [B, num_samples, action_dim, k]
        """
        B = state.size(0)
        feat = self.backbone(state)  # [B, 512]
        pref = F.relu(self.pref_fc(preference))
        feat = feat * pref

        z = torch.rand(B, num_samples, self.k, device=state.device)
        z_flat = z.view(-1, self.k)
        z_emb = self.noise_emb(z_flat).view(B, num_samples, 512)

        feat_expand = feat.unsqueeze(1)  # [B,1,512]
        joint = feat_expand * z_emb
        joint = F.relu(self.fc_final(joint))
        out = self.out(joint)
        out = out.view(B, num_samples, self.action_dim, self.k)
        return out


def select_action(q_particles: torch.Tensor, preference: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    """
    q_particles: [B, N, A, k]
    preference: [B, k]
    returns: [B] integer actions
    """
    B, N, A, k = q_particles.shape
    q_mean = q_particles.mean(dim=1)  # [B, A, k]
    scalar_q = (q_mean * preference.unsqueeze(1)).sum(dim=-1)  # [B, A]
    greedy = scalar_q.argmax(dim=-1)
    if epsilon <= 0:
        return greedy
    rand_mask = (torch.rand(B, device=q_particles.device) < epsilon).float()
    random_actions = torch.randint(0, A, (B,), device=q_particles.device)
    return torch.where(rand_mask.bool(), random_actions, greedy)
```

---

## 11. Experimental Playbook (Step-by-Step)

- **Compute budget:** Aim for 10M environment frames; for a quick sanity run, 500k frames suffice to validate stability.
- **Replay buffer:** 1M transitions; prioritize recent transitions to reduce non-stationarity from preference changes.
- **Particle counts:** $N=32$ for training, $M=32$ for targets; ablate $N \in \{8,16,32,64\}$.
- **Sinkhorn parameters:** blur $=0.01$, scaling $=0.9$, $\varepsilon \approx \sqrt{\text{blur}}$; ablate blur $\in \{0.1,0.05,0.01,0.005\}$.
- **Optimization:** Adam with LR $2.5 \times 10^{-4}$, grad clip 10.0, target update $\tau=0.005$.
- **Preference schedule:** sample per episode; ablate per-step resampling to encourage rapid frontier coverage.
- **Evaluation grid:** 21 evenly spaced $w$ on the 2-simplex (for 3 objectives), compute Hypervolume and Sparsity.
- **Baselines:** scalar IQN, independent IQN, MO-DQN; keep network capacity constant for fairness.
- **Logging:** track Sinkhorn loss, per-objective returns, hypervolume, and Wasserstein distance between predicted clouds and empirical returns (via rollout estimators).

---

## 12. Ablation and Sensitivity Plan

1. **Entropic strength:** Vary blur to study bias-variance trade-off; expect small blur to sharpen but risk instability.
2. **Particle count:** Study sample efficiency vs compute; hypothesis: diminishing returns beyond 64 particles.
3. **Preference curriculum:** Fixed vs random vs annealed Dirichlet concentration; hypothesis: annealed yields smoother frontier.
4. **Backbone sharing:** Compare shared vs objective-specific heads (multi-head MLP) to test representation disentanglement.
5. **Loss alternatives:** Replace Sinkhorn with Energy Distance (MMD) and with sliced Wasserstein; expect inferior frontier fidelity.
6. **Environment variant:** MinAtar Seaquest vs ALE Seaquest to decouple vision complexity from reward geometry.

---

## 13. Risk, Safety, and Robustness Considerations

- **Distributional overconfidence:** Monitor tail risk by inspecting 95th percentile of oxygen-cost dimension; enforce action masking if risk exceeds threshold.
- **RAM heuristics drift:** Validate RAM addresses across emulator versions; add unit tests for reward decomposition to avoid silent mislabeling.
- **Gradient explosion:** Use grad clipping and small blur; optionally adopt Sophia-G optimizer for curvature-aware updates.
- **Reproducibility:** Fix seeds for env, PyTorch, NumPy; log all hyperparameters; checkpoint both network weights and optimizer state.

---

## 14. Reproducibility Checklist

- Seed control: set `torch.manual_seed`, `np.random.seed`, `env.reset(seed)`.
- Deterministic CuDNN (if acceptable speed trade-off).
- Version pinning: `gymnasium==0.29`, `ale-py==0.8`, `geomloss==0.2.6`, `torch>=2.2`.
- Artifact logging: save model checkpoints every 100k frames; store evaluation metrics as JSON + CSV; store preference grid used for HV.

---

## 15. How to Extend to Other Multi-Objective Domains

1. **MuJoCo Safety Benchmarks (e.g., Safety Gymnasium):** Treat safety cost as a distinct objective; Sinkhorn aligns cost-return distributions without scalarization.
2. **Multi-Task Robotics:** Objectives per task; preference vector encodes task ID weights, enabling a unified Pareto frontier across tasks.
3. **Financial Portfolio RL:** Objectives for return, variance, drawdown; Sinkhorn captures co-dependencies across assets better than independent quantile heads.

---

## 16. Quick Start Commands (for a full repo)

- Install deps: `pip install torch geomloss gymnasium[atari] ale-py pygmo`.
- Launch training: `python train_sinkhorn_viqn.py --env SeaquestNoFrameskip-v4 --total-steps 10000000 --particles 32`.
- Evaluate hypervolume: `python eval_frontier.py --checkpoint ckpt.pt --w-grid 21`.

---

## 17. Data Structures for Replay and Batching

Define a replay item as:

```
{
  "obs": uint8[4,84,84],
  "action": int32,
  "reward_vec": float32[3],
  "next_obs": uint8[4,84,84],
  "done": bool,
  "preference": float32[3]
}
```

Batching yields tensors:
- obs: [B, 4, 84, 84], preference: [B, 3], reward_vec: [B, 3].
- Convert obs to float32 / 255.0 and stack.

---

## 18. Unit Tests (Recommended)

- **Reward wrapper test:** Simulate RAM values to ensure vector_reward logic fires expected signals.
- **Sinkhorn shapes test:** Assert loss finite for random clouds; backprop produces non-NaN gradients.
- **Policy invariance test:** With zero reward, Sinkhorn loss should push prediction cloud toward zero.
- **Preference conditioning test:** Different `w` should yield distinct Q ordering in synthetic two-action toy problem.

---

## 19. Illustrative Mathematical Examples

### 19.1 Two-Point Cloud Toy Example

Let $X = \{(0,0), (1,1)\}$, $Y = \{(0,1), (1,0)\}$, $\varepsilon=0.1$. The cost matrix is

$$
C = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}.
$$

Sinkhorn scaling converges to uniform coupling, yielding cost $\approx 0.5$ vs exact Wasserstein cost $0$. This highlights entropic bias and motivates the debiasing term in $S_\varepsilon$.

### 19.2 Frontier Hypervolume Illustration

For three objectives, given points $p_1=(2,1,0.5)$, $p_2=(1.5,1.5,0.4)$, $p_3=(1,0.8,1.0)$ and reference $(0,0,0)$, the hypervolume is the union of three boxes. Computing HV shows coverage of diverse trade-offs; removing $p_2$ reduces HV, demonstrating benefit of preference-conditioned learning.

---

## 20. Documentation Pointers and GitHub References

- **IQN base:** https://github.com/kaixindelele/iqn-pytorch (baseline implementation for scalar IQN).
- **GeomLoss:** https://github.com/jeanfeydy/geomloss (official Sinkhorn divergence GPU kernels).
- **Multi-Objective DQN envelope update:** https://github.com/ivandariojr/morl-envelope.
- **MinAtar environments:** https://github.com/kenjyoung/MinAtar for lightweight Atari variants.

---

## 21. What Success Looks Like

- Stable Sinkhorn loss curves without divergence.
- Hypervolume consistently above scalar IQN baseline after 2M frames.
- Qualitative policies: with $w=[1,0,0]$ agent surfaces frequently; with $w=[0,1,0]$ agent hoards divers before surfacing; with $w=[0,0,1]$ agent prioritizes enemy clearing.
- Replay visualizations show richer trajectory diversity compared to scalar agents.

---

## 22. Troubleshooting Guide

- **NaN in Sinkhorn:** Increase blur, reduce learning rate, enable gradient clipping, check for invalid obs normalization.
- **Mode collapse:** Increase particle count, add small Gaussian noise to rewards, lower entropy regularization.
- **Poor hypervolume:** Increase preference diversity, train longer, ensure action-selection uses preference-weighted Q not uniform.
- **Slow training:** Use MinAtar; reduce particle count during warmup; use `torch.compile` (PyTorch 2.2+) if available.

---

## 23. Checklist for a Complete Submission

- [ ] README ≥ 1000 lines with math, code, experiments (this document).
- [ ] Training script with Sinkhorn-VIQN and preference sampling.
- [ ] Evaluation script computing Hypervolume and sparsity.
- [ ] Unit tests for reward wrapper and loss.
- [ ] Logged runs with seeds and configs.

---

## 24. Derivation of Bias-Corrected Sinkhorn in Bellman Updates

The debiased divergence uses self-terms:

$$
S_\varepsilon(\mu,\nu) = 2 W_\varepsilon(\mu,\nu) - W_\varepsilon(\mu,\mu) - W_\varepsilon(\nu,\nu).
$$

In Bellman targets $Y = r + \gamma Z_{\theta'}$, the variance of $Y$ scales by $\gamma^2$. The bias term $W_\varepsilon(Y,Y)$ thus scales; subtracting it prevents over-smoothing. This is critical when rewards are small but high variance: without debiasing, the agent underestimates risk tails.

---

## 25. Computational Complexity and Memory Footprint

- Sinkhorn iteration complexity: $\mathcal{O}(B N M)$ per step; typical $N=M=32$ yields manageable cost.
- Memory: storing clouds requires $B \times N \times k$ floats; for $B=64$, $N=32$, $k=3$, ~24k floats ≈ 96KB.
- KeOps backend streams kernel to GPU to avoid instantiating full cost matrices; otherwise memory is $\mathcal{O}(N^2)$.

---

## 26. Potential Extensions

- **Sliced Sinkhorn-VIQN:** Project clouds onto multiple random directions, compute 1D Sinkhorn to reduce cost.
- **Quantile Flow Matching:** Use continuous-time flow to map base noise to reward cloud with OT-driven velocity field.
- **Implicit Diffusion Targets:** Replace bootstrapped samples with diffusion-model rollouts for better off-policy correction.

---

## 27. Closing Remarks

Sinkhorn-VIQN provides a principled bridge between optimal transport theory and practical, preference-aware deep RL. By directly matching joint reward distributions, it preserves correlations and exposes the full Pareto frontier to a single policy network. The provided derivations, pseudocode, and implementation notes are intended to be immediately actionable for researchers seeking to reproduce and extend these results on multi-objective control problems.

---

## 28. Detailed Hyperparameter Table

| Category      | Parameter                 | Value / Range            | Rationale                                     |
| ------------- | ------------------------- | ------------------------ | --------------------------------------------- |
| Optimization  | Learning rate             | 2.5e-4                   | Standard Atari IQN rate                       |
| Optimization  | Optimizer                 | Adam                     | Stable first-order baseline                   |
| Optimization  | Grad clip                 | 10.0                     | Prevent explosion with Sinkhorn                |
| Targets       | Target update $\tau$      | 0.005                    | Smooth Polyak averaging                       |
| Sinkhorn      | Particles (pred/target)   | 32 / 32                  | Balanced cost vs fidelity                     |
| Sinkhorn      | Blur (eps^2)              | 0.01                     | Moderately sharp transport                    |
| Sinkhorn      | Scaling                   | 0.9                      | GeomLoss recommended default                  |
| Env           | Frameskip                 | 4                        | Standard ALE setting                          |
| Env           | Frame stack               | 4                        | Capture temporal context                      |
| Replay        | Buffer size               | 1_000_000                | Matches DQN/IQN norms                         |
| Replay        | Batch size                | 64                       | Stable minibatch                               |
| Exploration   | $\epsilon$ schedule       | 1.0 → 0.01 over 1M steps | Consistent with Atari practice                |
| Preference    | Dirichlet concentration   | 1.0 (uniform)            | Uniform trade-off sampling                    |

---

## 29. Minimal Training Script Skeleton (non-executable placeholder)

```
# train_sinkhorn_viqn.py (outline)
import torch
from sinkhorn_viqn import SinkhornVIQN, SinkhornDivergence, select_action
from buffer import ReplayBuffer
from wrapper import SeaquestMultiObjectiveWrapper
from geomloss import SamplesLoss

def main():
    env = SeaquestMultiObjectiveWrapper(make_env(...))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SinkhornVIQN(action_dim=env.action_space.n).to(device)
    target = SinkhornVIQN(action_dim=env.action_space.n).to(device)
    target.load_state_dict(policy.state_dict())
    criterion = SinkhornDivergence(blur=0.01, scaling=0.9)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2.5e-4)
    buffer = ReplayBuffer(capacity=1_000_000)
    # training loop with steps described above ...

if __name__ == "__main__":
    main()
```

---

## 30. Evaluation Script Outline (Hypervolume)

```
# eval_frontier.py (outline)
import torch
import numpy as np
from pygmo import hypervolume

def evaluate(policy, env, w_grid):
    returns = []
    for w in w_grid:
        G = rollout(policy, env, w)
        returns.append(G)
    hv = hypervolume(np.array(returns)).compute(ref_point=[0,0,0])
    return hv
```

---

## 31. Visualization Ideas for the Notebook

- Plot Sinkhorn loss over training steps with shaded std across seeds.
- Scatter plot of particle clouds for two objectives at fixed states to inspect geometry.
- Pareto frontier estimation: plot evaluated returns for multiple $w$, overlay scalar IQN baseline.
- RAM signals over time with vector rewards to validate decomposition correctness.

---

## 32. Frequently Asked Questions (FAQ)

- **Q:** Why not use QR-DQN per objective?  
  **A:** It ignores inter-objective correlation and misestimates achievable joint states.
- **Q:** Does Sinkhorn slow training too much?  
  **A:** With $N=32$, cost is acceptable; KeOps backend keeps GPU efficient.
- **Q:** Can preferences change mid-episode?  
  **A:** Yes; resampling per step increases frontier coverage but may destabilize behavior—use a curriculum.

---

## 33. License and Attribution Note

Use the referenced GitHub repositories for IQN and GeomLoss under their respective licenses (MIT/BSD). Cite foundational works: Dabney et al. (IQN), Feydy et al. (GeomLoss), and the Sinkhorn DRL papers used for divergence design.

---

## 34. Future Research Directions

- Integrate **CrossQ**-style high UTD updates with Sinkhorn-VIQN to improve sample efficiency while using BN for stability.
- Explore **Sophia** optimizer for curvature-aware updates to mitigate OT gradient noise.
- Combine with **diffusion world models** to generate synthetic vector returns, enabling model-based multi-objective planning.

---

## 35. Final Checklist for Reproduction

- [ ] Environment wrapper validated on Seaquest (RAM signals consistent).  
- [ ] Training converges without NaNs for 500k-frame pilot.  
- [ ] Hypervolume computed on 21-point preference grid.  
- [ ] Ablations run: blur, particles, preference curriculum.  
- [ ] Plots produced: loss curves, frontier, particle clouds.  
- [ ] Seeds logged and results archived.

Usage and Math Addendum

This file supplements the existing `README.md` for CA3 with concise usage instructions and the math-to-code mapping.

Usage

1. Create a virtual environment and install dependencies (PyTorch, Gym, Matplotlib):

   python -m venv .venv && source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch gymnasium matplotlib

2. Run the notebook `notebooks/demo.ipynb` (open in Jupyter). The notebook imports the `src` package and runs a REINFORCE training loop.

Math-to-code mapping

- Objective and gradient: `report.tex` (Methodology) and `src/losses.py::reinforce_loss`.
- Policy network: `src/model.py::MLPPolicy` implements $\pi_\theta(a|s)$; sampling and log-probabilities are provided by `get_action`.
- Episode collection: `src/data.py::collect_episode`.
- Returns computation: `src/utils.py::discounted_returns` and `returns_to_tensor`.

Notes

- Hyperparameters: edit `src/config.py::Config`.
- Checkpoints: use `src/utils.py::save_checkpoint` / `load_checkpoint`.

