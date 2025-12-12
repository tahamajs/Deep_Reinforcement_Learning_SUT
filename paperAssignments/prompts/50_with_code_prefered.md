# Strategic Research Roadmap: Next-Generation Reinforcement Learning Architectures (2025-2026)

**To:** Research Staff, Principal Investigators, and PhD Candidates **From:** Director of Advanced Research **Date:** December 12, 2025 **Subject:** Directive on Strategic Research Assignments (Assignments 1–50)

## 1. Strategic Overview and Research Imperatives

The field of Reinforcement Learning (RL) stands at a critical inflection point. The era of optimizing simple model-free baselines on stationary benchmarks has largely concluded. Our institute’s mandate for the 2025-2026 cycle is to pioneer the convergence of distinct, high-impact methodologies that have emerged from the premier conferences—NeurIPS, ICML, and ICLR—between 2021 and 2025. We are witnessing a phase transition where the disparate subfields of Distributional RL, World Models, Diffusion Processes, and Neuro-Symbolic reasoning are collapsing into a unified paradigm of highly structured, sample-efficient, and generalization-capable decision-making systems.

The analysis of recent literature suggests three dominant trends that will define our research agenda. First, the **integration of generative modeling into control** , specifically through Diffusion Models and Transformers, is reshaping how agents imagine futures and represent policies. Second, the **optimization landscape of RL is being reconsidered** , moving away from standard Adam-based updates toward second-order methods and spectral regularization to combat the pervasive issues of plasticity loss and non-stationarity. Third, **structure and symbol manipulation** are returning to the forefront, driven by the need for interpretability and the reasoning capabilities of Large Language Models (LLMs) embedded within RL loops.

This document delineates 50 PhD-level research assignments. These are not merely implementation tasks; they are rigorous scientific inquiries designed to synthesize complementary advancements into novel algorithms. Each assignment requires deep engagement with the mathematical underpinnings of the selected papers, a disciplined approach to software engineering via Dockerized reproducibility, and an evaluation strategy that meets the statistical rigor of the `rliable` protocols. The focus is strictly on code implementation and mathematical rigor.

---

## Theme I: Distributional and Value-Based Reinforcement Learning

The foundational hypothesis of this theme is that the stability and sample efficiency of RL agents are fundamentally limited by how they estimate the distribution of future returns. We move beyond scalar expectations toward sophisticated distributional metrics and optimization landscapes.

### Assignment 1: Sinkhorn Divergence in Latent World Models

**Selected Papers:**

1. **Sinkhorn Distributional Reinforcement Learning** (NeurIPS 2024) ^^ \*\* \*\*
   - _GitHub:_ (https://github.com/datake/SinkhornDistRL)
2. **DreamerV3: Mastering Diverse Domains through World Models** (ICLR 2024) ^^ \*\* \*\*
   - _GitHub:_ [https://github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3)
3. **Distributional Reinforcement Learning** ^^ \*\* \*\*

**Novel Research Question:** Current state-of-the-art world models like DreamerV3 rely on minimizing the Kullback-Leibler (KL) divergence between the stochastic posterior and the prior transition dynamics. However, KL divergence is notoriously unstable when distributions have disjoint supports or are highly multi-modal, a common occurrence in complex stochastic environments. The research question posits: Can the Sinkhorn divergence, which provides a geometrically meaningful metric between distributions via optimal transport, replace the KL divergence in the latent space of DreamerV3 to improve robustness and sample efficiency in environments with stochastic, disjoint transition dynamics?

**Implementation Plan:** The implementation necessitates a fork of the `danijar/dreamerv3` JAX repository. The core refactoring targets the `RSSM` (Recurrent State Space Model) class within `jax/models.py`. The standard ELBO (Evidence Lower Bound) objective function must be modified. Specifically, the `kl_loss` function, which currently computes the analytic KL between Gaussian distributions, will be replaced by a `sinkhorn_loss` module derived from the `SinkhornDistRL` codebase.

The mathematical formulation requires defining the Sinkhorn distance **W**ε(**μ**,**ν**) between the posterior **q**ϕ(**z**t∣**x**t) and the prior **p**θ(**z**t∣**z**t**−**1\***\*,**a**t**−**1\*\***). The cost function **c**(**x**,**y**) will be the Euclidean distance in the latent space. The objective becomes minimizing the Sinkhorn divergence:

**L**d**y**n=**2**W**ε\*\***(**q**,**p**)**−**W**ε\*\***(**q**,**q**)**−**W**ε\*\***(**p**,**p**)\*\*

where **W**ε is approximated using the Sinkhorn-Knopp algorithm for **L** iterations. The computation of this transport plan is computationally intensive; thus, the implementation must utilize JAX's `vmap` and `jit` compilation to execute the iterative scaling on the GPU efficiently. A Docker container based on `ghcr.io/nvidia/jax:2024-03` is required to ensure consistent CUDA kernel versions for the transport operations.

**Evaluation Strategy:** The evaluation will focus on the **Atari 100k** benchmark, specifically targeting games with high stochasticity and sparse rewards such as _Frostbite_ , _Seaquest_ , and _Montezuma’s Revenge_ . The primary metric is the Interquartile Mean (IQM) of human-normalized scores. Ablation studies must compare the stability of the latent state reconstructions (measured by reconstruction error on hold-out trajectories) between the KL-based DreamerV3 and the Sinkhorn-DreamerV3, verifying the hypothesis that Sinkhorn loss prevents posterior collapse in multi-modal scenarios.

### Assignment 2: CrossQ with Sophia Optimization for Continuous Control

**Selected Papers:**

1. **CrossQ: Batch Normalization in Deep Reinforcement Learning** (ICLR 2024) ^^ \*\* \*\*
   - _GitHub:_ [https://github.com/adityab/CrossQ](https://github.com/adityab/CrossQ)
2. **Sophia: A Scalable Stochastic Second-order Optimizer** (ICLR 2024) ^^ \*\* \*\*
   - _GitHub:_ ([https://github.com/Liuhong99/Sophia](https://github.com/Liuhong99/Sophia))

**Novel Research Question:** CrossQ represents a paradigm shift by removing target networks and utilizing Batch Normalization (BN) to constrain value estimates, thereby enabling an Update-To-Data (UTD) ratio of 1. However, Batch Normalization introduces non-stationarity and dependencies within the batch that can complicate the optimization landscape. The research question investigates whether the Sophia optimizer, which utilizes a diagonal Hessian estimator to adapt to the local curvature of the loss landscape, can stabilize CrossQ updates further. Specifically, we hypothesize that Sophia will allow CrossQ to maintain stability at significantly higher UTD ratios (e.g., UTD=10 or 20) than the standard implementation, unlocking greater sample efficiency.

**Implementation Plan:** This project requires integrating the Sophia optimizer into the CrossQ codebase. The researcher must first implement the `SophiaG` optimizer class in PyTorch, ensuring it correctly handles the state updates for the diagonal Hessian estimate. The `CrossQ` agent (found in `CrossQ/agent.py`) typically uses Adam. This must be replaced with Sophia.

The mathematical synthesis involves applying Sophia's update rule to the CrossQ objective. The CrossQ loss is:

**L**C**ross**Q(**θ**)**=**E**D\*\***[(**Q**θ(**s**,**a**)**−**(**r**+**γ**a**′**max****Q**θ(**s**′**,**a**′**))**)**2**]\*\*

Sophia updates parameters **θ** using an exponential moving average (EMA) of the gradient **m**t and the Hessian diagonal **h**t:

**θ**t**+**1\***\*=**θ**t−**η**⋅**clip**(**max**{**γ**h**t\***\*,**ϵ**}**m**t**,**δ**)

Crucially, the Hessian estimate **h**t must be computed on the same mini-batch used for the gradient, or a sampled subset. The implementation must ensure that the Batch Normalization statistics in CrossQ are correctly synchronized or frozen during the curvature estimation steps of Sophia to prevents statistical drift.

**Evaluation Strategy:** The primary benchmarks are the **MuJoCo** continuous control tasks (Humanoid-v4, Ant-v4, Walker2d-v4). The evaluation strategy involves a rigorous ablation study comparing CrossQ+Adam versus CrossQ+Sophia across a sweep of UTD ratios **{**1**,**5**,**10**,**20**}**. Success is defined as CrossQ+Sophia achieving a higher asymptotic return or faster convergence (fewer environment steps) at high UTD ratios where CrossQ+Adam diverges or exhibits performance collapse.

### Assignment 3: Exclusively Penalized Q-Learning (EPQ) with Rational Activations

**Selected Papers:**

1. **Exclusively Penalized Q-learning (EPQ)** (NeurIPS 2024) ^^ \*\* \*\*
   - _GitHub:_ ([https://github.com/DmitryRyumin/ICML-2025-Papers](https://github.com/DmitryRyumin/ICML-2025-Papers)) (General reference for context)
2. **Rational Activation Functions in Reinforcement Learning** (arXiv 2024) ^^ \*\* \*\*

**Novel Research Question:** Rational Activation Functions (RAFs) have been shown to significantly improve plasticity and approximation capabilities in RL agents but suffer from "activation explosion" leading to instability. EPQ reduces bias in offline RL by selectively penalizing Q-values only in Out-of-Distribution (OOD) regions. The synthesis question is: Can EPQ's selective penalty mechanism implicitly regularize the unbounded growth of Rational Activations? We hypothesize that the EPQ penalty, when applied to states with exploded Q-values (a symptom of activation explosion), will provide the necessary negative gradient signal to suppress the coefficients of the rational polynomials, enabling stable training of highly expressive offline agents.

**Implementation Plan:** The researcher will build upon a standard Offline RL repository such as `CORL` or `d3rlpy`. The first step is implementing a `RationalActivation` module in PyTorch, defined as **f**(**x**)**=**P**(**x**)**/**Q**(**x**), where **P** and **Q** are learnable polynomials. This module will replace ReLU or Tanh activations in the Critic networks.

The EPQ loss function must be implemented to replace the standard CQL loss. The mathematical formulation for the combined objective is:

**L**(**θ**)**=**E**(**s**,**a**)**∼**D\*\***[(**Q**θ(**s**,**a**)**−**y**)**2**]**+**α**E**s**∼**D**,**a**∼**π**(**s**)[**I**(**Q**θ(**s**,**a**)**>**τ**)**⋅**Q**θ(**s**,**a**)]**+**λ**∣∣**c**∣**∣**2**2**

Here, **I** is the indicator function that activates the penalty only when the Q-value exceeds a threshold **τ** (e.g., the empirical max Q-value in the batch). The term **∣∣**c**∣**∣**2**2 represents a regularization on the coefficients of the rational function. The implementation must carefully tune **α** and **τ** to ensure the penalty engages specifically when activation explosion begins to manifest.

**Evaluation Strategy:** Benchmarks will include **D4RL** datasets, specifically the challenging `antmaze-medium-diverse` and `hopper-medium-expert` tasks. The evaluation will compare the Normalized Average Return of EPQ+Rational against EPQ+ReLU and CQL+Rational. Detailed analysis tables should report the average magnitude of activations in the penultimate layer to quantitatively demonstrate the suppression of activation explosion.

### Assignment 4: Q-Value Regularized Transformer (QT) with Mamba Backbone

**Selected Papers:**

1. **Q-value Regularized Transformer for Offline Reinforcement Learning** (ICML 2024) ^^ \*\* \*\*
   - _GitHub:_ (https://github.com/charleshsc/QT)
2. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (2024) ^^ \*\* \*\*
   - _Code Reference:_ [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

**Novel Research Question:** The QT architecture successfully combines Decision Transformers with Q-learning to stitch suboptimal trajectories. However, the quadratic complexity **O**(**N**2**)** of the Transformer attention mechanism severely limits the context length, typically to a few dozen steps. The Mamba architecture, based on Selective State Space Models (SSMs), offers linear time complexity **O**(**N**). The research question is: Can replacing the Transformer backbone in QT with a Mamba block enable the agent to utilize extremely long histories (10,000+ steps) for credit assignment in sparse-reward offline tasks, thereby outperforming attention-based architectures in long-horizon stitching?

**Implementation Plan:** The codebase requires cloning `charleshsc/QT` and identifying the `DecisionTransformer` class. The researcher must replace the GPT-2 style attention blocks with `MambaBlock` modules from the `mamba_ssm` library. This involves refactoring the input embeddings to match Mamba's expected dimensions and ensuring the causal masking inherent in Mamba aligns with the RL sequence modeling (States, Actions, Rewards).

The mathematical formulation for the QT loss remains similar, but the underlying representation **h**t is now evolved via the discretized SSM dynamics:

**h**t=**A**ˉ**t\*\***h**t**−**1\*\***+**B**ˉ**t\*\***x\*\*t

**y**t=**C**h**t**

Crucially, the Q-value regularization term **L**M**a**x**imi**ze**Q** must be applied to the output embeddings **y**t. The training loop must be adjusted to handle the recurrent state `hs` of the Mamba model if training on sequences longer than GPU memory allows (via chunking), or simply training on very large `seq_len` (e.g., 4096) which is feasible with Mamba. The environment requires a Docker container with CUDA 11.8+ and specific NVCC versions to compile the Mamba CUDA kernels.

**Evaluation Strategy:** The primary benchmark is **D4RL AntMaze** (Large and Ultra-diverse datasets), where long-horizon credit assignment is critical and rewards are sparse. Evaluation metrics will include the Success Rate and the Stitching Ability (performance on medium-replay datasets). A comparative table should present Training FLOPs, Inference Latency, and Score for QT-Transformer vs. QT-Mamba across increasing context lengths (**L**=**20**,**100**,**1000**,**4000**).

### Assignment 5: Value-Evolutionary-Based RL (VEB-RL) with Elephant Activations

**Selected Papers:**

1. **Value-Evolutionary-Based Reinforcement Learning** (ICML 2024) ^^ \*\* \*\*
   - _GitHub:_ ([https://github.com/yeshenpy/VEB-RL](https://github.com/yeshenpy/VEB-RL))
2. **Elephant Activation Functions** (arXiv 2025) ^^ \*\* \*\*

**Novel Research Question:** VEB-RL utilizes genetic algorithms to maintain a population of value functions, effectively mitigating the risk of getting trapped in local optima. However, evolutionary methods are notoriously sample inefficient, and populations often suffer from "feature collapse" where all agents converge to similar representations. The research proposes integrating "Elephant" activation functions, which are designed to induce sparse gradients and representations, into the VEB-RL population. Can the sparsity induced by Elephant activations maintain higher diversity within the population's feature space, thereby accelerating the evolutionary search and improving sample efficiency?

**Implementation Plan:** Using the `yeshenpy/VEB-RL` repository, the researcher must locate the network definitions in `models.py`. The standard ReLU activations will be replaced by the Elephant activation function. The mathematical definition of the Elephant activation **ϕ**(**x**) involves a specific saturation and sparsity mechanism ^^, often taking the form: \*\* \*\*

**ϕ**(**x**)**=**{**x**0if **x**>**0**if **x**≤**0\*\***+\*\*sparse_perturbation

The implementation must also introduce a metric for **Population Diversity** , calculated as the mean cosine distance between the weight vectors or feature maps of the elite agents in the population.

**Evaluation Strategy:** Experiments will be conducted on **MinAtar** (Seaquest, Breakout) and **MuJoCo Sparse** environments. The evaluation will track two key metrics: Sample Efficiency (number of steps to reach threshold performance) and Population Diversity over generations. Results should be presented in a table comparing VEB-RL(ReLU) vs. VEB-RL(Elephant), highlighting the correlation between sustained diversity and final performance.

### Assignment 6: Adaptive Discount Factors in PPO via Variance Control

**Selected Papers:**

1. **Reinforcement Learning with Adaptive Discount Factor** (2024) ^^ \*\* \*\*
2. **Proximal Policy Optimization (PPO)** ^^ \*\* \*\*

**Novel Research Question:** The discount factor **γ** determines the effective horizon of the agent. A fixed **γ** is often suboptimal; a low **γ** stabilizes early training but leads to myopia, while a high **γ** captures long-term value but increases variance. We propose an adaptive **γ** controller for PPO that adjusts **γ** based on the **variance of the Generalized Advantage Estimation (GAE)** . Can such a mechanism automatically anneal the agent from a stable, myopic learner to a far-sighted optimizer without manual tuning?

**Implementation Plan:** The researcher will modify a standard PPO implementation (e.g., `cleanrl`). The rollout buffer collection phase must be updated to calculate the empirical variance of the advantage estimates **A**^**t** within a batch.

**γ**t**+**1\***\*=**γ**t+**α**⋅**(**σ**t**a**r**g**e**t−**Var**(**A**^**t\***\*))**

The GAE calculation loop must be refactored to accept a dynamic **γ**t. The logic suggests that if variance is low, **γ** can be increased to extend the horizon. If variance exceeds a threshold **σ**t**a**r**g**e**t**, **γ** is decreased to stabilize the updates.

**Evaluation Strategy:** Benchmarks include **MuJoCo** (Hopper, HalfCheetah) and **Procgen** . The evaluation will compare PPO with fixed **γ**=**0.99** against PPO with Adaptive-**γ**. Metrics include the learning curve stability (variance of returns across seeds) and asymptotic performance.

### Assignment 7: Gradient Eligibility Traces in Recurrent Off-Policy RL

**Selected Papers:**

1. **Deep Reinforcement Learning with Gradient Eligibility Traces** (2025) ^^ \*\* \*\*
2. **Efficient Recurrent Off-Policy RL (RESeL)** (NeurIPS 2024) ^^ \*\* \*\*
   - _GitHub:_ ([https://github.com/FanmingL/Recurrent-Offpolicy-RL](https://github.com/FanmingL/Recurrent-Offpolicy-RL))

**Novel Research Question:** Recurrent Off-Policy RL suffers from the "staleness" problem where hidden states stored in the replay buffer become outdated as the network evolves. While RESeL addresses this with specific learning rates, it does not fully solve the credit assignment issue. The assignment is to integrate **Gradient Eligibility Traces (GET)** , specifically the backward-view formulation compatible with streaming, into the training of RNN-based critics (GRU/LSTM). Can GET provide a more theoretically sound mechanism for credit assignment in recurrent off-policy learning than simple n-step returns?

**Implementation Plan:** Using the `FanmingL/Recurrent-Offpolicy-RL` repository, the researcher will modify the critic update. Standard BPTT (Backpropagation Through Time) truncates gradients. GET maintains a trace **e**t that accumulates gradients:

**e**t=**λγ**e**t**−**1\*\***+**∇**θ\***\*V**(**s**t)

The implementation requires refactoring the loss computation to use this trace for updates, effectively implementing a **TD**(**λ**) mechanism for deep recurrent networks. This must be done efficiently on the GPU, potentially requiring a custom CUDA kernel or JAX `scan` function.

**Evaluation Strategy:** The method will be evaluated on **POMDP-MuJoCo** (Partially Observable MuJoCo), where memory is essential. The comparison will be between RESeL (baseline) and RESeL+GET. Key metrics are sample efficiency and the ability to solve tasks with significant memory dependencies (e.g., Delayed Ant).

### Assignment 8: "To the Max" Reward Transformation with Sinkhorn Distributional RL

**Selected Papers:**

1. **To the Max: Reinventing Reward in Reinforcement Learning** (ICML 2024) ^^ \*\* \*\*
   - _GitHub:_ ([https://github.com/veviurko/To-the-Max](https://github.com/veviurko/To-the-Max))
2. **Sinkhorn Distributional Reinforcement Learning** (NeurIPS 2024) ^^ \*\* \*\*

**Novel Research Question:** "To the Max" proposes a specific transformation of the reward function to encourage reaching goal states. Distributional RL explicitly models the randomness of returns. The research question investigates the interaction between these two concepts: Does the "To the Max" reward transformation alter the return distribution in a way that makes Distributional RL (specifically Sinkhorn DRL) redundant, or does the combination provide a multiplicative benefit in sparse reward goal-seeking tasks?

**Implementation Plan:** The researcher will implement the reward transformation **R**′**(**s**,**a**)** defined in "To the Max" within the `SinkhornDistRL` repository.

**R**′**(**s**,**a**)**=**Transformation**(**R**(**s**,**a**))

The agent will then attempt to learn the distribution of this _transformed_ return. The study requires training agents with four configurations: Standard Reward + Scalar RL, Standard Reward + Sinkhorn RL, Max Reward + Scalar RL, and Max Reward + Sinkhorn RL.

**Evaluation Strategy:** Benchmarks: **Maze** navigation tasks and **Fetch** robotics tasks (sparse rewards). The evaluation will analyze the convergence speed and the shape of the learned value distributions (visualized via density plots).

---

## Theme II: Model-Based Reinforcement Learning

This theme focuses on the construction and utilization of internal models of the environment. The frontier lies in improving the fidelity of these models through diffusion and transformers, and scaling them to multi-agent and limited-data regimes.

### Assignment 9: Diffusion World Models with DIAMOND's Diffusion Prior

**Selected Papers:**

1. **DIAMOND: DIffusion As a Model Of eNvironment Dreams** (NeurIPS 2024) ^^ \*\* \*\*
   - _GitHub:_ [https://github.com/diamond-wm/diamond](https://github.com/diamond-wm/diamond)
2. **Diffusion World Model: Future Modeling Beyond Step-by-Step Rollout** (ICML 2024) ^^ \*\* \*\*

**Novel Research Question:** DIAMOND utilizes a diffusion model to simulate environment dynamics but relies on an autoregressive generation process (predicting **s**t**+**1\***\* from **s**t), which is slow for long-horizon planning. The Diffusion World Model (DWM) paper proposes predicting _multi-step_ futures concurrently. The research question is: Can we enhance DIAMOND by integrating the "Future Modeling" architecture from DWM, modifying the diffusion U-Net to denoise a block of **K** future frames simultaneously? This would theoretically accelerate the "dreaming" process (planning) by a factor of **K\*\*.

**Implementation Plan:** Fork the `diamond-wm/diamond` repository. The core task is to modify the U-Net architecture. Instead of accepting an input tensor of shape **(**B**,**C**,**H**,**W**)**, the model must process **(**B**,**K**,**C**,**H**,**W**)** or a stacked equivalent **(**B**,**K**×**C**,**H**,**W**)**. The diffusion loss function must be updated to denoise the entire trajectory segment: $$ \mathcal{L} = \mathbb{E}\_{t, \epsilon, \tau} [ |

| \epsilon - \epsilon*\theta(x*{t+1:t+K}^\tau, \tau, x*t, a*{t:t+K-1}) ||^2 ] $$ The agent's planning loop in `agent.py` must be refactored to query the model once for every **K** steps of imagination.

**Evaluation Strategy:** Benchmarks: **Atari 100k** , specifically visually complex games. The primary metrics are **Wall-clock Training Time** (samples per second) and **Inference Speed** (imagined steps per second), alongside the standard Human Normalized Score to ensure no degradation in policy quality.

### Assignment 10: EfficientZero V2 with LightZero Integration for Multi-Agent Systems

**Selected Papers:**

1. **EfficientZero V2: Mastering Discrete and Continuous Control** (ICML 2024) ^^ \*\* \*\*
   - _GitHub:_ (https://github.com/Shengjiewang-Jason/EfficientZeroV2)
2. **LightZero: A Unified Benchmark for MCTS** (NeurIPS 2023) ^^ \*\* \*\*
   - _GitHub:_ [https://github.com/opendilab/LightZero](https://github.com/opendilab/LightZero)

**Novel Research Question:** EfficientZero V2 (EZ-V2) sets the state-of-the-art for limited data control in single-agent domains. LightZero provides a modular MCTS framework that supports multi-agent reinforcement learning (MARL). The assignment is to port EZ-V2's specific algorithmic improvements—specifically the Gumbel search corrections and the value prefix loss—into the LightZero framework to create a **Multi-Agent EfficientZero V2** . Does the improved sample efficiency of EZ-V2 translate to the multi-agent domain, where the joint action space grows exponentially?

**Implementation Plan:** Clone the `LightZero` repository. Create a new policy class `EfficientZeroV2Policy` within `lzero/policy`. This involves migrating the `ValuePrefix` loss logic and the improved MCTS search policies from the `Shengjiewang-Jason` repository. $$ \mathcal{L} _{prefix} = \sum_ {k=0}^K |

| r*{t+k} - \hat{r}(s*{t+k}) ||^2 $$ The implementation must extend these mechanisms to handle joint actions **a**=**(**a**1**,**…**,**a**N**)** and potentially factorized value functions supported by LightZero's `ma_muzero`.

**Evaluation Strategy:** Benchmarks: **GoBigger** (a complex multi-agent game supported by LightZero) or **SMAC** (StarCraft Multi-Agent Challenge). The metric of interest is the **Win Rate vs. Sample Count** , comparing standard Multi-Agent MuZero against the newly implemented Multi-Agent EfficientZero V2.

### Assignment 11: Transformer-Based World Models with State-Space Duality

**Selected Papers:**

1. **Transformer-based World Models Are Happy With 100k Interactions** (ICLR 2023) ^^ \*\* \*\*
   - _GitHub:_ [https://github.com/jrobine/twm](https://github.com/jrobine/twm)
2. **Mamba-2: State Space Duality** ^^ \*\* \*\*

**Novel Research Question:** Transformers are data-efficient world models but suffer during inference (planning) due to the KV-cache bottleneck. Mamba-2 establishes a theoretical duality between Linear Attention and State Space Models (SSMs). The research question is: Can we replace the dense attention layers in the Transformer World Model (TWM) with the **State Space Duality (SSD)** layers from Mamba-2? This would allow the model to be trained in parallel (Transformer mode) for efficiency but executed recurrently (SSM mode) during MCTS planning, offering **O**(**1**) step complexity.

**Implementation Plan:** Using `jrobine/twm` as the base, the researcher must replace the `SelfAttention` modules with `Mamba2Block` from the `mamba-ssm` library. The critical engineering challenge is implementing the dual forward passes: one for training that takes the full sequence **x**1**:**T\***\* and computes gradients in parallel, and one for the MCTS planner that maintains a recurrent state **h**t and steps forward **x**t**+**1\*\***=**step**(**x**t,**h**t).

**Evaluation Strategy:** Benchmarks: **Atari 100k** . The evaluation will compare **Planning Speed (nodes/second)** and **Training Throughput** between the TWM-Attention and TWM-SSD models. We expect comparable sample efficiency with significantly faster wall-clock planning times.

### Assignment 12: UniZero vs. EfficientZero V2: A Rigorous Comparative Study

**Selected Papers:**

1. **UniZero: Unified Tree Search** (arXiv 2024) ^^ \*\* \*\*
2. **EfficientZero V2** (ICML 2024) ^^ \*\* \*\*

**Novel Research Question:** Both UniZero and EfficientZero V2 claim state-of-the-art performance on the Atari 100k benchmark using MCTS-based approaches. UniZero emphasizes a unified architecture, while EZ-V2 focuses on algorithmic refinements. No direct, controlled comparison exists. The assignment is to implement both algorithms within the **same** codebase (LightZero) to decouple architectural benefits from implementation details. Which approach yields better sample efficiency when hyperparameters are normalized?

**Implementation Plan:** This is a benchmarking and integration assignment. The researcher must ensure LightZero has fully compliant implementations of both algorithms (UniZero is likely present, EZ-V2 must be added as per Assignment 10). A unified configuration file must be created to ensure both agents use identical network sizes, replay buffer configurations, and pre-processing steps.

**Evaluation Strategy:** Benchmarks: A subset of 10 diverse **Atari** games and 5 **DeepMind Control Suite** tasks. Metrics: Median Human Normalized Score and Interquartile Mean (IQM) with stratified bootstrap confidence intervals (using the `rliable` library).

### Assignment 13: SimGolf for Online Planning with Latent World Models

**Selected Papers:**

1. **SimGolf: Simulation-Based Global Search** (NeurIPS 2024) ^^ \*\* \*\*
2. **DreamerV3** ^^ \*\* \*\*

**Novel Research Question:** SimGolf introduces a method for "local simulator access" to reset to previously visited states and replan, enhancing sample efficiency. DreamerV3 learns a latent world model. Can we implement the SimGolf planning logic _inside_ the latent space of DreamerV3? Instead of resetting the real environment (which is often impossible), the agent resets its _latent_ state to a critical decision point and performs SimGolf-style search to refine its policy.

**Implementation Plan:** Integrate SimGolf's search strategy into the `DreamerV3` planning loop. When the agent encounters high Bellman error or uncertainty, it triggers a "SimGolf phase":

1. Reset latent state **z**t to a stored checkpoint **z**s**a**v**e**d.
2. Roll out multiple trajectories from **z**s**a**v**e**d using the world model.
3. Update the value function estimates based on these imagined rollouts.
4. Update the policy.

**Evaluation Strategy:** Benchmarks: **Complex Navigation Tasks** (e.g., D4RL AntMaze or custom mazes) where backtracking and broad search are necessary. Metric: Success rate in finding the goal compared to standard DreamerV3.

### Assignment 14: Model-Based Meta-RL (MAMBA) with Cross-Embodiment

**Selected Papers:**

1. **MAMBA: MetA-RL Model-Based Algorithm** (NeurIPS/ICLR 2024 Rebuttal context) ^^ \*\* \*\*
2. **PEAC: Unsupervised Pre-training for Cross-Embodiment RL** (NeurIPS 2024) ^^ \*\* \*\*

**Novel Research Question:** MAMBA applies Dreamer-style world models to Meta-RL. PEAC addresses cross-embodiment (agents with different physical structures). The question: Can a MAMBA agent meta-learn a "Universal World Model" that adapts not just to different tasks, but to different _bodies_ (embodiments) by inferring a latent "morphology vector" from interaction history?

**Implementation Plan:** Modify the MAMBA architecture to include a "Morphology Encoder" that takes the history of observations and actions **(**o**1**:**t\*\***,**a**1**:**t\***\*)** and outputs a latent **z**m**or**p**h**. This latent conditions the transition dynamics **p**(**s**′**∣**s**,**a**,**z**m**or**p**h). Train this on a suite of MuJoCo agents (HalfCheetah, Ant, Walker) simultaneously.

**Evaluation Strategy:** Benchmarks: **Cross-Embodiment MuJoCo** (train on Walker/Hopper, test on Ant). Metric: Zero-shot or Few-shot adaptation performance on the unseen embodiment.

### Assignment 15: Economic Dispatch with Multi-Agent MuZero

**Selected Papers:**

1. **Economic Dispatch with PPO** (2024) ^^ \*\* \*\*
2. **LightZero** ^^ \*\* \*\*

**Novel Research Question:** The Economic Dispatch (ED) problem in power grids involves balancing generation and demand while respecting constraints. Paper ^^ solves this with PPO. Can **Multi-Agent MuZero** , which can plan ahead and anticipate grid dynamics (load changes), outperform the reactive PPO agent in minimizing cost and constraint violations in a dynamic ED scenario? \*\* \*\*

**Implementation Plan:** Define the Economic Dispatch problem as a `Gymnasium` environment compatible with LightZero. The environment must simulate a 15-generator system with dynamic load profiles. Configure `ma_muzero` in LightZero to control the generators.

**Evaluation Strategy:** Benchmark: IEEE 30-bus or 118-bus systems. Metrics: Total Operational Cost ($) and Frequency of Constraint Violations. Compare PPO vs. MA-MuZero.

### Assignment 16: TD7 with Model-Based Rollouts (Hybrid RL)

**Selected Papers:**

1. **TD7** (2023/2024 context) ^^ \*\* \*\*
2. **Diffusion World Model** ^^ \*\* \*\*

**Novel Research Question:** TD7 is a model-free algorithm that improves over TD3. Model-Based Policy Optimization (MBPO) showed that augmenting replay buffers with model-generated data improves efficiency. Can we combine TD7 with a **Diffusion World Model** to generate high-fidelity synthetic rollouts, creating a "Model-Based TD7"?

**Implementation Plan:** Use `seungju-k1m/sac-td3-td7` as the base. Train a Diffusion World Model (from ^^) concurrently. In the update loop, sample real transitions, generate **k**-step synthetic rollouts using DWM, and add them to the TD7 training batch. \*\* \*\*

**Evaluation Strategy:** Benchmarks: **MuJoCo** (Humanoid, Dog). Metric: Sample efficiency (performance at 100k, 300k steps).

### Assignment 17: Stochastic MuZero with Sinkhorn Divergence

**Selected Papers:**

1. **MuZero** ^^ \*\* \*\*
2. **Sinkhorn Distributional RL** ^^ \*\* \*\*

**Novel Research Question:** Stochastic MuZero handles chance by learning afterstates. Can we improve the representation of the stochastic value function in the search tree by using **Sinkhorn Distributional Value Estimation** instead of scalar values? This would allow the MCTS to make risk-sensitive decisions based on the full distribution of returns.

**Implementation Plan:** Modify a MuZero implementation (e.g., `LightZero`) to predict a categorical distribution of returns at each node. Use Sinkhorn divergence loss to train the value prediction network. During MCTS backpropagation, aggregate distributions using convolution (for addition) and mixture (for branching).

**Evaluation Strategy:** Benchmarks: **Stochastic Atari** (e.g., Backgammon or Poker-like environments if available, or noisy Atari). Metric: Performance in high-variance scenarios.

---

## Theme III: Offline and Data-Efficient RL

This theme addresses the challenge of learning policies from fixed datasets without interaction, a prerequisite for real-world deployment where trial-and-error is costly.

### Assignment 18: Offline RL with Consistency Models (Generative QDQ)

**Selected Papers:**

1. **Q-Distribution Guided Q-Learning (QDQ)** (NeurIPS 2024) ^^ \*\* \*\*
2. **Consistency Models** (Contextual)

**Novel Research Question:** QDQ uses a consistency model to estimate the uncertainty of Q-values for pessimistic updates in offline RL. We propose extending this: Can the Consistency Model be used to _generate_ synthetic, high-value transitions (data augmentation) that are "stitched" from the dataset distribution but lie in optimistic regions? This "Generative QDQ" would actively expand the support of the dataset towards higher rewards before the RL agent trains on it.

**Implementation Plan:** Using a standard Offline RL repo (e.g., `CORL`), implement a Consistency Model trained on the trajectory distribution **p**(**s**′**,**r**∣**s**,**a**)**. Construct a "Generation Phase" where the model generates transitions **(**s**^**′**,**r**^**) conditioned on state-action pairs in the dataset that have high Q-values. Augment the replay buffer **D**a**ug\*\***=**D**∪**D**g**e**n\*\* and train a CQL or IQL agent on this enriched dataset.

**Evaluation Strategy:** Benchmarks: **D4RL** (Medium-Replay datasets where data is scarce and stitching is required). Metric: Normalized Average Return improvement over vanilla QDQ.

### Assignment 19: Resetting Neural Networks for Continual Offline-to-Online RL

**Selected Papers:**

1. **Online non-stationary learning with automatic soft parameter reset** (NeurIPS 2024) ^^ \*\* \*\*
2. **Resetting Neural Networks** (NeurIPS 2023) ^^ \*\* \*\*

**Novel Research Question:** When fine-tuning an offline RL agent (e.g., CQL) online, the agent often suffers from "plasticity loss"—it cannot adapt to the new online data because its weights have converged to a sharp minimum. Can the "Automatic Soft Parameter Reset" (ASPR) method, governed by an Ornstein-Uhlenbeck drift process, outperform standard "Hard Resets" (ReDo) in preserving the useful offline priors while regaining plasticity during the online phase?

**Implementation Plan:** The researcher will implement the ASPR optimizer wrapper. This wrapper maintains a reference to **θ**ini**t** (the weights at the end of offline training). The update rule is modified:

**θ**t=**θ**t**−**1\***\*−**α**t∇**L**(**θ**t**−**1\*\***)**+**γ**(**θ**ini**t−**θ**t**−**1\*\*\*\*)

This drift term pulls weights softly towards the prior (offline knowledge) while allowing adaptation. Apply this to the Critic networks of a CQL agent during the online fine-tuning phase on D4RL datasets.

**Evaluation Strategy:** Benchmarks: **D4RL to Online** (e.g., train offline on `antmaze-medium`, fine-tune online on `antmaze-large`). Metric: Learning curve slope immediately after the online phase starts and final asymptotic performance.

### Assignment 20: Q-Value Regularized Mamba (QM) for Long-Horizon Offline RL

**Selected Papers:**

1. **Q-value Regularized Transformer (QT)** (ICML 2024) ^^ \*\* \*\*
2. **Mamba** ^^ \*\* \*\*

**Novel Research Question:** Similar to Assignment 4 but focused specifically on the _regularization_ aspect. QT relies on the Transformer's attention map to propagate Q-values. Mamba's hidden state **h**t compresses the history. The question: Can we formulate a Q-value regularization term directly on the Mamba hidden state **h**t that is mathematically equivalent to the attention-based regularization in QT, thereby enabling "Stitching via State Space Models"?

**Implementation Plan:** Derive the gradient of the Q-value with respect to the Mamba hidden state **∇**h**t**Q**(**s**t,**a**t)**. Add a loss term that maximizes this Q-value by adjusting the _past_ hidden states through the recurrent dynamics. Implement in `charleshsc/QT` by swapping the backbone.

**Evaluation Strategy:** Benchmarks: **D4RL Kitchen** (long horizon, sequential subtasks). Metric: Success rate.

### Assignment 21: Top-ERL vs. HarmoDT on Multi-Task Benchmarks

**Selected Papers:**

1. **TOP-ERL: Transformer-based Off-Policy Episodic RL** (ICLR 2025) ^^ \*\* \*\*
2. **HarmoDT: Harmony Multi-Task Decision Transformer** (ICML 2024) ^^ \*\* \*\*

**Novel Research Question:** Top-ERL and HarmoDT represent two competing approaches to multi-task offline RL: one uses episodic memory with Transformers (Top-ERL), the other uses gradient modulation (HarmoDT). A direct comparison is missing. The assignment is to benchmark them on the **Meta-World MT50** task set to determine which architectural bias (episodic memory vs. gradient harmony) is superior for generalization.

**Implementation Plan:** Set up the `MT50` benchmark. Run official implementations of both papers. Create a hybrid: **Harmo-Top-ERL** , which uses HarmoDT's gradient masking _inside_ the Top-ERL update loop.

**Evaluation Strategy:** Benchmark: **Meta-World MT50** . Metric: Success rate on held-out tasks.

### Assignment 22: ADEPT Data Exploitation for Offline RL

**Selected Papers:**

1. **ADEPT: Adaptive Data Exploitation** (arXiv 2025) ^^ \*\* \*\*
2. **QDQ** ^^ \*\* \*\*

**Novel Research Question:** ADEPT uses Multi-Armed Bandits to schedule data usage. In Offline RL, we often have "Conservative" (in-distribution) data and "Optimistic" (generated or OOD) data. Can ADEPT be applied to dynamically select the ratio of Conservative vs. Optimistic data batches during training to maximize learning speed without divergence?

**Implementation Plan:** Implement the ADEPT bandit controller. The "arms" of the bandit are different data samplers (Real Data, Generated Data from QDQ). The reward for the bandit is the validation TD-error (lower is better).

**Evaluation Strategy:** Benchmarks: **D4RL** . Metric: Convergence stability and final score.

### Assignment 23: LCPO with Domain Randomization for Sim-to-Real

**Selected Papers:**

1. **Online Reinforcement Learning in Non-Stationary Context-Driven Environments (LCPO)** (ICLR 2025) ^^ \*\* \*\*
2. **Robustness Reprogramming** ^^ \*\* \*\*

**Novel Research Question:** LCPO is designed for non-stationary environments. Sim-to-Real transfer is a form of non-stationarity (simulation physics ****= real physics). Can LCPO, trained on a domain-randomized simulation (varying friction, mass), adapt _online_ to the "real" physics (held-out parameters) faster than standard Domain Randomization baselines?

**Implementation Plan:** Use `pybullet` to simulate a robot. Train LCPO while varying physics parameters sinusoidally. Test on a setting with fixed but _unseen_ physics parameters.

**Evaluation Strategy:** Benchmark: **PyBullet Hopper** and **Walker** . Metric: Adaptation time (steps to recover 80% performance) after physics shift.

### Assignment 24: Offline RL with Hyperparameter Tuning Networks (HyQ)

**Selected Papers:**

1. **QT** ^^ \*\* \*\*
2. **Generalization Analysis** ^^ \*\* \*\*

**Novel Research Question:** Offline RL algorithms like QT are sensitive to hyperparameters (e.g., the weight of the Q-regularization **β**). Can we train a **Hypernetwork** that takes dataset statistics (sparsity, horizon length, return variance) as input and predicts the optimal **β** for the QT loss function?

**Implementation Plan:** Extract meta-features from D4RL datasets. Train a small MLP (Hypernet) to predict **β**. The training signal for the Hypernet comes from the validation performance of the QT agent (meta-gradient).

**Evaluation Strategy:** Benchmarks: **D4RL** suite. Metric: Performance across diverse datasets without manual hyperparameter tuning.

### Assignment 25: Decision Transformer with RIME for Preference-Based Offline RL

**Selected Papers:**

1. **RIME: Robust Preference-based RL** (ICML 2024) ^^ \*\* \*\*
2. **Decision Transformer**

**Novel Research Question:** RIME handles noisy preferences in standard RL. Can we integrate RIME's robust loss function into a **Decision Transformer** (DT) framework? instead of conditioning the DT on returns (which we might not know), we condition it on _preference tokens_ , trained via the RIME objective to be robust to labeling errors in the offline dataset.

**Implementation Plan:** Modify `DecisionTransformer` to accept preference tokens. Train the sequence model using RIME's robust loss instead of standard cross-entropy on returns.

**Evaluation Strategy:** Benchmarks: **D4RL** with synthetic noisy preferences (simulating human error). Metric: Correlation between conditioned preference and actual return.

---

## Theme IV: Exploration and Intrinsic Motivation

This theme addresses the sparse reward problem by endowing agents with curiosity and the ability to leverage external knowledge.

### Assignment 26: INSIGHT with Semantic RND

**Selected Papers:**

1. **INSIGHT: Neuro-Symbolic RL** (ICML 2024) ^^ \*\* \*\*
2. **Intrinsic Motivation with Foundation Models** ^^ \*\* \*\*

**Novel Research Question:** Standard Random Network Distillation (RND) calculates novelty on pixels, which is sensitive to noise (e.g., moving leaves). INSIGHT generates structured, semantic embeddings of the state using a distilled Vision Foundation Model. The research question: Does applying RND on the **Semantic Embeddings** of INSIGHT (Semantic RND) lead to more meaningful exploration in visually complex, sparse-reward games?

**Implementation Plan:** Fork `ins-rl/insight`. Extract the `PerceptionModule` output. Feed this into an RND predictor/target pair. Add the prediction error as an intrinsic reward. $$ r\_{int} = |

| \hat{f}(\psi*{INSIGHT}(s)) - f(\psi*{INSIGHT}(s)) ||^2 $$

**Evaluation Strategy:** Benchmarks: **Montezuma's Revenge** and **Pitfall** (Atari). Metric: Number of unique rooms visited compared to Pixel-RND.

### Assignment 27: Joint Intrinsic Motivation (JIM) with Language-Guided Goals

**Selected Papers:**

1. **Joint Intrinsic Motivation (JIM)** (AAMAS 2024) ^^ \*\* \*\*
2. **ReLara** (ICML 2024) ^^ \*\* \*\*

**Novel Research Question:** JIM rewards joint novelty in multi-agent systems. ReLara uses an assistant to shape rewards. We propose a hybrid: A "Language Assistant" (LLM) observes the multi-agent team and suggests **joint goals** in text (e.g., "Pass the ball to Agent 2"). These goals are embedded and used to shape the intrinsic reward, guiding the JIM exploration toward semantically meaningful coordination patterns.

**Implementation Plan:** Adapt `ReLara` to a multi-agent setting (e.g., Google Research Football). Use a local LLM (Llama-3-8B) to generate textual goals based on game snapshots. Compute cosine similarity between the current state embedding and the goal embedding. Add this to the JIM intrinsic reward.

**Evaluation Strategy:** Benchmark: **Google Research Football** . Metric: Goals scored and pass completion rate.

### Assignment 28: Neural MMO with Foundation Model Intrinsic Motivation

**Selected Papers:**

1. **Neural MMO 2.0** ^^ \*\* \*\*
2. **Fostering Intrinsic Motivation with Foundation Models** ^^ \*\* \*\*

**Novel Research Question:** Neural MMO is a massive multi-agent environment with open-ended tasks. Can **CLIP-based Intrinsic Motivation** (using pre-trained CLIP embeddings to measure novelty) drive more diverse emergent behaviors (trading, fighting, exploration) in Neural MMO than standard count-based mechanisms?

**Implementation Plan:** Integrate `pufferlib` (for Neural MMO) with a CLIP embedding extraction loop. Calculate the entropy of the visited CLIP embedding clusters as a population-level intrinsic reward.

**Evaluation Strategy:** Benchmark: **Neural MMO** . Metric: Diversity of learned skills/professions in the population.

### Assignment 29: AuxDistill with Auto-Generated Auxiliary Tasks

**Selected Papers:**

1. **AuxDistill** (2024) ^^ \*\* \*\*
2. **Intrinsic Motivation** ^^ \*\* \*\*

**Novel Research Question:** AuxDistill relies on hand-defined auxiliary tasks to accelerate learning. Can we use an **Intrinsic Motivation** module (like RND or ICM) to _automatically_ identify salient events (e.g., opening a door, picking up a key) and treat them as auxiliary tasks for distillation, removing the need for manual task engineering?

**Implementation Plan:** Modify `AuxDistill`. Run an RND module. When RND error spikes (surprise), create a temporary auxiliary task: "Return to this state". Train the policy to solve this aux task and distill it into the main policy.

**Evaluation Strategy:** Benchmark: **Habitat Object Rearrangement** . Metric: Success rate without hand-crafted aux rewards.

### Assignment 30: Lion Optimizer with Adaptive Parameter Space Noise

**Selected Papers:**

1. **Lion Optimizer** ^^ \*\* \*\*
2. **Parameter Space Noise** ^^ \*\* \*\*

**Novel Research Question:** Lion uses sign-based updates: **θ**←**θ**−**η**⋅**sign**(**m**). This binary nature might interact uniquely with Parameter Space Noise (PSN). We propose "Noisy Lion": injecting discrete noise (flipping signs) into the update step based on an exploration schedule. Does this induce better exploration in parameter space than adding Gaussian noise to the weights?

**Implementation Plan:** Modify `lion-pytorch`. Inside the `step` function, add:

**u**t=**sign**(**m**t+**N**(**0**,**σ**t))

This probabilistically flips the update direction for parameters with low momentum (uncertain directions). Integrate this into a **TD3** agent.

**Evaluation Strategy:** Benchmark: **PyBullet Robotics** . Metric: Exploration efficiency (steps to first reward).

### Assignment 31: DiCuRL with Mamba Curriculum Generator

**Selected Papers:**

1. **DiCuRL: Diffusion Curriculum RL** (NeurIPS 2024) ^^ \*\* \*\*
2. **Mamba** ^^ \*\* \*\*

**Novel Research Question:** DiCuRL uses diffusion to generate curriculum goals. Diffusion generation is slow. Can we train a **Mamba** model to autoregressively generate a sequence of curriculum goals (**g**1→**g**2→**g**f**ina**l) based on the agent's current capability? Mamba's efficiency allows generating long chains of sub-goals rapidly during training.

**Implementation Plan:** Train a Mamba model on successful trajectories to predict the sequence of "achieved goals". Use this model to propose intermediate goals for the RL agent during training.

**Evaluation Strategy:** Benchmark: **PointMaze** and **AntMaze** . Metric: Wall-clock training time vs. DiCuRL.

### Assignment 32: Rational Exploration in Sparse Rewards

**Selected Papers:**

1. **Rational Activation Functions** ^^ \*\* \*\*
2. **Exploration Papers** ^^ \*\* \*\*

**Novel Research Question:** Rational Activation Functions (RAFs) are highly flexible. Can we exploit their instability? In sparse reward environments, we can _increase_ the flexibility (degree) of the RAFs in the Actor network to induce high-variance exploratory actions ("Wild Exploration"), and then anneal the degree down to stabilize the policy once rewards are found.

**Implementation Plan:** Implement a dynamic RAF module where the polynomial degree **d** is a hyperparameter. Schedule **d** from high (e.g., 5) to low (e.g., 2) based on the replay buffer's average reward.

**Evaluation Strategy:** Benchmark: **Sparse MuJoCo** . Metric: Time to solve.

### Assignment 33: Count-Based Exploration with VQ-VAE States

**Selected Papers:**

1. **LightZero** (includes VQ-VAE models) ^^ \*\* \*\*
2. **Exploration Papers** ^^ \*\* \*\*

**Novel Research Question:** Count-based exploration is powerful but intractable in continuous spaces. Can we use the **VQ-VAE** discrete codes from a MuZero-style model (LightZero) to perform exact count-based exploration in the _latent_ space?

**Implementation Plan:** Use `LightZero`'s VQ-VAE model. Maintain a hash table of visited latent codes **z**q**u**an**t**i**ze**d. Add intrinsic reward **r**∝**1/**N**(**z**)**![]().

**Evaluation Strategy:** Benchmark: **Montezuma's Revenge** . Metric: Exploration bonus consistency.

---

## Theme V: Optimization and Stability

This theme focuses on the "engine room" of RL—the optimizers and regularization techniques that ensure learning does not collapse.

### Assignment 34: Spectral Normalization with Normalize-and-Project (NaP)

**Selected Papers:**

1. **Normalization and effective learning rates (NaP)** (arXiv 2024) ^^ \*\* \*\*
2. **Spectral Regularization** (ICLR 2025) ^^ \*\* \*\*

**Novel Research Question:** NaP controls the effective learning rate by fixing parameter norms. Spectral Regularization maintains the spectral radius of weight matrices. The synthesis: Does combining NaP (fixing the norm) with Spectral Regularization (shaping the singular values) provide the ultimate defense against **Loss of Plasticity** in continual RL?

**Implementation Plan:** Implement a custom Optimizer wrapper.

1. Apply Spectral Regularization loss: **L**s**p**ec=**(**σ**1\*\***(**W**)**−**1**)**2\*\*.
2. Apply NaP projection after the gradient update: **W**←**W**⋅**∣∣**W**∣**∣**F\*\***R\*\*. Apply this to a PPO agent in a Continual Learning setting.

**Evaluation Strategy:** Benchmark: **Continual-MuJoCo** (HalfCheetah with changing gravity/friction every 1M steps). Metric: Ability to adapt to new physics without performance degradation (forgetting or plasticity loss).

### Assignment 35: Distributed Lion for Multi-Agent RL

**Selected Papers:**

1. **Distributed Lion** (NeurIPS 2024) ^^ \*\* \*\*
2. **MAPPO** (Multi-Agent PPO context)

**Novel Research Question:** Distributed Lion reduces communication overhead by transmitting only the _sign_ of updates (1 bit per parameter). Can we apply this to **Multi-Agent PPO (MAPPO)** where centralized critics require massive bandwidth to aggregate gradients from many agents? This could enable scaling MAPPO to thousands of agents on modest hardware.

**Implementation Plan:** Implement the `DistributedLion` algorithm using `torch.distributed`. Apply it to the gradient synchronization step of a MAPPO implementation (e.g., `cleanrl` or `marl-benchmark`).

**Evaluation Strategy:** Benchmark: **StarCraft II (SMAC)** with massive agent counts (if available) or **Neural MMO** . Metric: Bandwidth usage vs. Performance.

### Assignment 36: Sophia-Optimized CrossQ

_(See Assignment 2 - this is a key optimization assignment placed in Theme I for impact, but belongs conceptually here)._

### Assignment 37: Proximal Feature Optimization (PFO) with ReDo

**Selected Papers:**

1. **Proximal Feature Optimization (PFO)** (NeurIPS 2024) ^^ \*\* \*\*
2. **ReDo: Resetting Dormant Neurons** ^^ \*\* \*\*

**Novel Research Question:** PFO regularizes pre-activations to stay close to a prior. ReDo resets dormant neurons. Are they complementary? We hypothesize that PFO might _prevent_ neurons from becoming dormant in the first place, making ReDo unnecessary. Alternatively, ReDo might clean up the neurons that PFO fails to save.

**Implementation Plan:** Train PPO agents with four configs: Baseline, PFO, ReDo, PFO+ReDo. Monitor the "Dormant Neuron Ratio".

**Evaluation Strategy:** Benchmark: **Procgen** (high diversity, prone to dormancy). Metric: Generalization gap and dormant ratio.

### Assignment 38: Soft Parameter Reset for PPO Plasticity

**Selected Papers:**

1. **Automatic Soft Parameter Reset** ^^ \*\* \*\*
2. **PPO** ^^ \*\* \*\*

**Novel Research Question:** Apply the SDE-based Soft Parameter Reset (from ^^) to the **Policy Network** of PPO in a non-stationary environment. Does soft resetting allow the policy to explore new behaviors (regain plasticity) without catastrophic forgetting of the locomotion primitives? \*\* \*\*

**Implementation Plan:** Wrap the PPO Actor optimizer with the Soft Reset logic. Trigger resets when entropy drops below a threshold.

**Evaluation Strategy:** Benchmark: **Non-Stationary Ant** (Ant's legs change length during training). Metric: Recovery speed.

### Assignment 39: Context-Encoder-Specific Learning Rates (RESeL) in World Models

**Selected Papers:**

1. **RESeL** (NeurIPS 2024) ^^ \*\* \*\*
2. **DreamerV3** ^^ \*\* \*\*

**Novel Research Question:** RESeL shows that recurrent encoders need higher learning rates. Does applying a distinct, higher learning rate to the **RSSM Encoder** of DreamerV3 (relative to the dynamics/decoder models) improve its ability to capture fast-changing details in the environment?

**Implementation Plan:** Modify `dreamerv3` optimizer config to use parameter groups. Set `lr_encoder = 5 * lr_dynamics`.

**Evaluation Strategy:** Benchmark: **Crafter** (requires memory and detail). Metric: Score and reconstruction error.

### Assignment 40: Constraint-Based PPO with Economic Dispatch Logic

**Selected Papers:**

1. **PPO for Economic Dispatch** ^^ \*\* \*\*
2. **Safe RL**

**Novel Research Question:** Paper ^^ uses penalty methods and action projection for power grid constraints. Can we generalize this "Action Projection" layer to generic **Safe RL** tasks (e.g., SafetyGym), where the projection maps unsafe actions to the nearest valid action on the constraint manifold? \*\* \*\*

**Implementation Plan:** Implement a differentiable projection layer `safety_projection(action)` that solves a QP (Quadratic Program) to satisfy safety constraints. Integrate into PPO Actor.

**Evaluation Strategy:** Benchmark: **SafetyGym** . Metric: Number of safety violations vs. Baseline PPO-Lagrangian.

### Assignment 41: Gradient Spectral Normalization (GSN) in Transformers

**Selected Papers:**

1. **Gradient Spectral Normalization** ^^ \*\* \*\*
2. **Decision Transformer**

**Novel Research Question:** GSN reshapes gradient distributions in the frequency domain. Transformers often suffer from gradient scaling issues. Can applying GSN to the gradients of a **Decision Transformer** stabilize training and allow for larger learning rates?

**Implementation Plan:** Implement the FFT-based GSN filter. Apply it to the gradients of the Attention layers in a DT.

**Evaluation Strategy:** Benchmark: **D4RL** . Metric: Max stable learning rate and convergence speed.

---

## Theme VI: Architectures (Transformers, Mamba, Neuro-Symbolic)

This theme explores the cutting-edge architectures that are replacing the standard MLP/CNN backbones in RL.

### Assignment 42: RL-GPT with Self-Correction

**Selected Papers:**

1. **RL-GPT** (NeurIPS 2024) ^^ \*\* \*\*
2. **LLM Reasoning Papers**

**Novel Research Question:** RL-GPT integrates RL with Code-as-Policy. Can we integrate a **Self-Correction (Reflection)** loop? The LLM generates code, a "Critic LLM" (or the same LLM prompted differently) critiques it, and the RL rewards the _improvement_ between the draft and the corrected code.

**Implementation Plan:** Modify the RL-GPT generation loop. State -> LLM -> Code_v1 -> Reflection -> Code_v2 -> Execute. Reward = **R**(**Code_v2**)**−**R**(**Code_v1**)**.

**Evaluation Strategy:** Benchmark: **Minecraft (MineDojo)** . Metric: Task completion rate.

### Assignment 43: Symbolic Regression Machine (RSRM) for Reward Discovery

**Selected Papers:**

1. **Reinforcement Symbolic Regression Machine (RSRM)** (ICLR 2024) ^^ \*\* \*\*
   - _GitHub:_ ([https://github.com/intell-sci-comput/RSRM](https://github.com/intell-sci-comput/RSRM))

**Novel Research Question:** RSRM uses RL to find mathematical formulas. Can we use RSRM to **discover the underlying reward function** of a black-box environment from observed transitions **(**s**,**a**,**s**′**) and preferences? Ideally, RSRM would output a symbolic equation (e.g., **R**=**−**0.5**⋅**v**2**) which is then used to train a PPO agent.

**Implementation Plan:** Collect a dataset of transitions and human preferences. Use RSRM to search for a symbolic formula **f**(**s**) that correlates with preferences. Train PPO using **f**(**s**).

**Evaluation Strategy:** Benchmark: **CartPole** (Hidden reward). Metric: Recovery of the ground-truth reward formula (interpretability) and agent performance.

### Assignment 44: Text-to-SVG RL with Diffusion Feedback

**Selected Papers:**

1. **Text-to-SVG RL** (ICLR 2026 sub) ^^ \*\* \*\*
2. **DIAMOND** ^^ \*\* \*\*

**Novel Research Question:** The Text-to-SVG paper uses a VLM as a reward model. This is sparse and noisy. Can we use a **Diffusion Model** trained on SVG-to-Image to provide a _dense, differentiable_ signal? The diffusion model acts as a "critic" that estimates how "real" the rendered SVG looks given the text prompt.

**Implementation Plan:** Train a diffusion model conditioned on text to denoise SVG renderings. Use the diffusion loss (SDS - Score Distillation Sampling) as a gradient for the RL agent generating the SVG code.

**Evaluation Strategy:** Benchmark: **SVG Generation** . Metric: CLIP Score of generated SVGs.

### Assignment 45: Neuro-Symbolic Graph RL

**Selected Papers:**

1. **INSIGHT** ^^ \*\* \*\*
2. **Graph Neural Networks Papers** ^^ \*\* \*\*

**Novel Research Question:** INSIGHT uses an MLP perception module. Many environments (Atari, Starcraft) are naturally graph-structured (objects and relations). Can we replace the perception module with a **Graph Neural Network (GNN)** that extracts a "Symbolic Graph State", allowing for even more interpretable explanations (e.g., "Agent jumped because Enemy A is close to Platform B")?

**Implementation Plan:** Modify `INSIGHT` to extract object lists from Atari RAM or Vision. Build a graph **G**=**(**V**,**E**)**. Feed **G** into a GNN. Feed GNN output to the symbolic policy.

**Evaluation Strategy:** Benchmark: **Atari** (Object-centric games like Skiing, Freeway). Metric: Explanation quality (user study or heuristic) and agent score.

### Assignment 46: Quantized LLMs in RLHF Loops

**Selected Papers:**

1. **DuQuant** (NeurIPS 2024) ^^ \*\* \*\*
2. **SRPPO** ^^ \*\* \*\*

**Novel Research Question:** RLHF (Reinforcement Learning from Human Feedback) is expensive. Can we train an RLHF policy using **4-bit Quantized LLMs** (via DuQuant) without significant performance degradation? This would democratize RLHF research.

**Implementation Plan:** Integrate `bitsandbytes` and `peft` (LoRA) into the SRPPO training loop. Apply DuQuant quantization to the Actor and Critic models.

**Evaluation Strategy:** Benchmark: **HH-RLHF** dataset. Metric: Win-rate against full-precision model vs. Memory savings.

### Assignment 47: Mamba-2 Recurrent PPO

**Selected Papers:**

1. **Mamba-2** ^^ \*\* \*\*
2. **Recurrent PPO**

**Novel Research Question:** Replace the LSTM in a standard Recurrent PPO agent with a **Mamba-2** block. Does the superior state tracking and linear complexity of Mamba-2 allow PPO to handle significantly longer effective horizons in POMDPs than LSTMs?

**Implementation Plan:** Swap `LSTM` for `Mamba2` in `cleanrl`'s `ppo_recurrent.py`. Manage the hidden state passing carefully.

**Evaluation Strategy:** Benchmark: **Memory Gym** (tasks requiring memorizing sequences of 1000+ steps). Metric: Success rate.

### Assignment 48: HarmoDT with Label-Specific Representation Learning

**Selected Papers:**

1. **HarmoDT** ^^ \*\* \*\*
2. **Generalization Analysis for Label-Specific Representation** ^^ \*\* \*\*

**Novel Research Question:** HarmoDT optimizes masks for tasks. Can we augment it with **Label-Specific Representation Learning** to explicitly disentangle the task ID embedding from the state embedding? We hypothesize this will prevent negative transfer in highly diverse multi-task datasets.

**Implementation Plan:** Add an orthogonality loss term to HarmoDT: **∣∣**Cov**(**z**t**a**s**k,**z**s**t**a**t**e)**∣**∣**F**2.

**Evaluation Strategy:** Benchmark: **Meta-World MT50** . Metric: Generalization to unseen tasks.

### Assignment 49: Boosting RL with Auxiliary Short Delays and Mamba

**Selected Papers:**

1. **Boosting RL with Auxiliary Short Delays (AD-RL)** (ICML 2024) ^^ \*\* \*\*
2. **Mamba** ^^ \*\* \*\*

**Novel Research Question:** AD-RL helps with delayed feedback. Mamba naturally handles long delays. Are they complementary? We propose combining them: Use Mamba as the backbone to capture the long-term trace, and use AD-RL's auxiliary heads to provide intermediate supervision.

**Implementation Plan:** Integrate AD-RL auxiliary losses into a Mamba-based Actor-Critic.

**Evaluation Strategy:** Benchmark: **Delayed MuJoCo** (rewards delayed by 100 steps). Metric: Convergence speed.

### Assignment 50: The "Grand Unified" Dreamer Agent

**Selected Papers:**

1. **DreamerV3** ^^ \*\* \*\*
2. **CrossQ** ^^ \*\* \*\*
3. **Mamba** ^^ \*\* \*\*

**Novel Research Question:** **The Capstone.** Can we architect a single agent that combines the three most significant recent advancements?

1. **World Model:** Uses **Mamba** dynamics (instead of RSSM/RNN).
2. **Critic:** Uses **CrossQ** (no target networks, batch norm) for value estimation.
3. **Optimization:** Uses **Sophia** (second-order). This "Frankenstein" agent tests the limits of integrating disparate state-of-the-art modules.

**Implementation Plan:** This requires a massive engineering effort. Start with `dreamerv3`. Replace RSSM with Mamba-SSM. Replace the Actor-Critic update logic with CrossQ's logic. Replace the optimizer with Sophia.

**Evaluation Strategy:** Benchmark: **Atari 100k** and **DMC** . Metric: Is it the SOTA of SOTAs?

---

## 3. Technical Infrastructure and Evaluation Standards

To ensure the scientific validity of these assignments, strict adherence to the following technical standards is mandatory:

1. **Containerization:** All deliverables must be encapsulated in Docker containers. Use `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` or `ghcr.io/nvidia/jax:2024-03` as base images. This ensures reproducibility of CUDA kernel compilations (essential for Mamba and Sinkhorn).
2. **Reproducibility Protocol:** All evaluation runs must use at least 5 random seeds **(**0**,**1**,**2**,**3**,**4**)**.
3. **Statistical Rigor:** Results must be reported using the **Interquartile Mean (IQM)** and **Stratified Bootstrap Confidence Intervals** as implemented in the `rliable` library. Point estimates (mean/median) are insufficient.
4. **Code Standards:** Python 3.10+ is required. All code must be fully typed (mypy strict mode) and formatted via Black.

This roadmap represents a strategic investment in solving the "Grand Challenges" of RL: sample efficiency, stability, and generalization. The convergence of these methods is not just a possibility; it is the inevitable direction of the field.

**Signed,**

**Director of Advanced Research** **Top-Tier AI Institute**

[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccDistributional Reinforcement Learning with Regularized Wasserstein Loss - NIPS papers**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/7371ee6a40da2951303ec7ebdb2150ce-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://danijar.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)danijar.comMastering Diverse Control Tasks through World Models - Danijar Hafner**Opens in a new window**](https://danijar.com/project/dreamerv3/)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2301.04104] Mastering Diverse Domains through World Models - arXiv**Opens in a new window**](https://arxiv.org/abs/2301.04104)[![](https://t1.gstatic.com/faviconV2?url=https://slogix.in/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)slogix.inTop 50 Research Papers in Distributional Reinforcement Learning ...**Opens in a new window**](https://slogix.in/machine-learning/latest-research-papers-in-distributional-reinforcement-learning/)[![](https://t2.gstatic.com/faviconV2?url=https://sb3-contrib.readthedocs.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)sb3-contrib.readthedocs.ioCrossQ — Stable Baselines3 - Contrib 2.8.0a1 documentation**Opens in a new window**](https://sb3-contrib.readthedocs.io/en/master/modules/crossq.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comOfficial code release for &#34;CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity&#34; - GitHub**Opens in a new window**](https://github.com/adityab/CrossQ)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgCrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity - arXiv**Opens in a new window**](https://arxiv.org/html/1902.05605v4)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[1902.05605] CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity - arXiv**Opens in a new window**](https://arxiv.org/abs/1902.05605)[![](https://t0.gstatic.com/faviconV2?url=https://nn.labml.ai/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)nn.labml.aiSophia Optimizer - labml.ai**Opens in a new window**](https://nn.labml.ai/optimizers/sophia.html)[![](https://t0.gstatic.com/faviconV2?url=https://proceedings.iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.iclr.ccSOPHIA: ASCALABLE STOCHASTIC SECOND-ORDER OPTIMIZER FOR LANGUAGE MODEL PRE-TRAINING - ICLR Proceedings**Opens in a new window**](https://proceedings.iclr.cc/paper_files/paper/2024/file/06960915ba8674c7a898ec0b472b80ff-Paper-Conference.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://papers.nips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)papers.nips.ccExclusively Penalized Q-learning for Offline Reinforcement Learning - NIPS papers**Opens in a new window**](https://papers.nips.cc/paper_files/paper/2024/file/cdc1d08ee82d4818758d229abb7f1ce8-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgBalancing Expressivity and Robustness: Constrained Rational Activations for Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/html/2507.14736v1)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2405.17098] Q-value Regularized Transformer for Offline Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/abs/2405.17098)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comcharleshsc/QT: ICML&#39;2024: Q-value Regularized Transformer for Offline Reinforcement Learning - GitHub**Opens in a new window**](https://github.com/charleshsc/QT)[![](https://t2.gstatic.com/faviconV2?url=https://huggingface.co/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)huggingface.coMamba - Hugging Face**Opens in a new window**](https://huggingface.co/docs/transformers/model_doc/mamba)[![](https://t0.gstatic.com/faviconV2?url=https://thegradient.pub/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)thegradient.pubMamba Explained - The Gradient**Opens in a new window**](https://thegradient.pub/mamba-explained/)[![](https://t3.gstatic.com/faviconV2?url=https://michielh.medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)michielh.medium.comMamba for Dummies: Efficient Linear-Time LLMs Explained | by Michiel Horstman - Medium**Opens in a new window**](https://michielh.medium.com/mamba-for-dummies-linear-time-llms-explained-0d4b51efcf9f)[![](https://t0.gstatic.com/faviconV2?url=https://towardsdatascience.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)towardsdatascience.comMamba: SSM, Theory, and Implementation in Keras and TensorFlow**Opens in a new window**](https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-keras-and-tensorflow-32d6d4b32546/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comyeshenpy/VEB-RL: (ICML 2024) The official code for Value-Evolutionary-Based Reinforcement Learning - GitHub**Opens in a new window**](https://github.com/yeshenpy/VEB-RL)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgEfficient Reinforcement Learning by Reducing Forgetting with Elephant Activation Functions**Opens in a new window**](https://arxiv.org/html/2509.19159v1)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)researchgate.netReinforcement Learning with Adaptive Discount Factor for Clutch Judder Suppression with Stable Learning in Two-Speed EV Transmission | Request PDF - ResearchGate**Opens in a new window**](https://www.researchgate.net/publication/396273607_Reinforcement_Learning_with_Adaptive_Discount_Factor_for_Clutch_Judder_Suppression_with_Stable_Learning_in_Two-Speed_EV_Transmission)[![](https://t3.gstatic.com/faviconV2?url=https://rlj.cs.umass.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)rlj.cs.umass.eduReinforcement Learning with Adaptive Temporal Discounting**Opens in a new window**](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_321.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)researchgate.netPPO Improvement in Different Environments - ResearchGate**Opens in a new window**](https://www.researchgate.net/publication/387925569_PPO_Improvement_in_Different_Environments)[![](https://t0.gstatic.com/faviconV2?url=https://spinningup.openai.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)spinningup.openai.comProximal Policy Optimization — Spinning Up documentation - OpenAI**Opens in a new window**](https://spinningup.openai.com/en/latest/algorithms/ppo.html)[![](https://t3.gstatic.com/faviconV2?url=https://rlj.cs.umass.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)rlj.cs.umass.eduDeep Reinforcement Learning with Gradient Eligibility Traces**Opens in a new window**](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)researchgate.net(PDF) Deep Reinforcement Learning with Gradient Eligibility Traces - ResearchGate**Opens in a new window**](https://www.researchgate.net/publication/393685696_Deep_Reinforcement_Learning_with_Gradient_Eligibility_Traces)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2507.09087] Deep Reinforcement Learning with Gradient Eligibility Traces - arXiv**Opens in a new window**](https://arxiv.org/abs/2507.09087)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comFanmingL/Recurrent-Offpolicy-RL: Implementation of SAC and TD3 based on various RNN and Transformer. - GitHub**Opens in a new window**](https://github.com/FanmingL/Recurrent-Offpolicy-RL)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comCode accompanying our ICML 2024 paper &#34;To the Max: Reinventing Reward in Reinforcement Learning&#34; - GitHub**Opens in a new window**](https://github.com/veviurko/To-the-Max)[![](https://t2.gstatic.com/faviconV2?url=https://diamond-wm.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)diamond-wm.github.ioDiamond - diffusion for world modeling**Opens in a new window**](https://diamond-wm.github.io/)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccDiffusion for World Modeling: Visual Details Matter in Atari - NIPS papers**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/6bdde0373d53d4a501249547084bed43-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2402.03570] Diffusion World Model: Future Modeling Beyond Step-by-Step Rollout for Offline Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/abs/2402.03570)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comShengjiewang-Jason/EfficientZeroV2: [ICML 2024, Spotlight] EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data - GitHub**Opens in a new window**](https://github.com/Shengjiewang-Jason/EfficientZeroV2)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2403.00564] EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data**Opens in a new window**](https://arxiv.org/abs/2403.00564)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.com[NeurIPS 2023 Spotlight] LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios (awesome MCTS) - GitHub**Opens in a new window**](https://github.com/opendilab/LightZero)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netLightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios | OpenReview**Opens in a new window**](https://openreview.net/forum?id=oIUXpBnyjv)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comjrobine/twm: Transformer-based World Models - GitHub**Opens in a new window**](https://github.com/jrobine/twm)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgUniZero: Generalized and Efficient Planning with Scalable Latent World Models - arXiv**Opens in a new window**](https://arxiv.org/html/2406.10667v2)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netThe Power of Resets in Online Reinforcement Learning - OpenReview**Opens in a new window**](https://openreview.net/forum?id=7sACcaOmGi&noteId=EkELBhEf7V)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netMAMBA: an Effective World Model Approach for Meta-Reinforcement Learning**Opens in a new window**](https://openreview.net/forum?id=1RE0H6mU7M)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comA curated list of awesome exploration RL resources (continually updated) - GitHub**Opens in a new window**](https://github.com/opendilab/awesome-exploration-rl)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)mdpi.comA Reinforcement Learning-Based Proximal Policy Optimization Approach to Solve the Economic Dispatch Problem - MDPI**Opens in a new window**](https://www.mdpi.com/2673-4591/97/1/24)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.compytorch implementation of SAC, TD3 and TD7 with Mujoco Benchmark results from 4 seeds.**Opens in a new window**](https://github.com/seungju-k1m/sac-td3-td7)[![](https://t3.gstatic.com/faviconV2?url=https://ar5iv.labs.arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)ar5iv.labs.arxiv.org[2102.12924] Visualizing MuZero Models - ar5iv**Opens in a new window**](https://ar5iv.labs.arxiv.org/html/2102.12924)[![](https://t2.gstatic.com/faviconV2?url=https://en.wikipedia.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)en.wikipedia.orgMuZero - Wikipedia**Opens in a new window**](https://en.wikipedia.org/wiki/MuZero)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccQ-Distribution guided Q-learning for offline reinforcement learning: Uncertainty penalized Q-value via consistency model - NIPS papers**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/61caa89f7a5366023db6f5736b2c579d-Paper-Conference.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://discovery.ucl.ac.uk/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)discovery.ucl.ac.ukNon-Stationary Learning of Neural Networks with Automatic Soft Parameter Reset - UCL Discovery**Opens in a new window**](https://discovery.ucl.ac.uk/10207117/1/NeurIPS-2024-non-stationary-learning-of-neural-networks-with-automatic-soft-parameter-reset-Paper-Conference.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccSample-Efficient and Safe Deep Reinforcement Learning via Reset Deep Ensemble Agents**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2023/file/a6f6a5c517b2b92f3d309786af64086c-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comyinizhilian/ICLR2025-Papers-with-Code: 历年 ICLR 论文和 ... - GitHub**Opens in a new window**](https://github.com/yinizhilian/ICLR2025-Papers-with-Code)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comICML&#39;2024: HarmoDT: Harmony Multi-Task Decision Transformer for Offline Reinforcement Learning - GitHub**Opens in a new window**](https://github.com/charleshsc/HarmoDT)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgAdaptive Data Exploitation in Deep Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/html/2501.12620v1)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netNeurIPS 2024 Conference - OpenReview**Opens in a new window**](https://openreview.net/group?id=NeurIPS.cc/2024/Conference)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comCJReinforce/RIME_ICML2024: Official code for ICML 2024 paper, &#34;RIME: Robust Preference-based Reinforcement Learning with Noisy Preferences&#34; (ICML 2024 Spotlight) - GitHub**Opens in a new window**](https://github.com/CJReinforce/RIME_ICML2024)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgEnd-to-End Neuro-Symbolic Reinforcement Learning with Textual Explanations - arXiv**Opens in a new window**](https://arxiv.org/abs/2403.12451)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comins-rl/insight: Official implementation of the paper &#34;End-to-End Neuro-Symbolic Reinforcement Learning with Textual Explanations&#34; (ICML 2024) - GitHub**Opens in a new window**](https://github.com/ins-rl/insight)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2410.07404] Fostering Intrinsic Motivation in Reinforcement Learning with Pretrained Foundation Models - arXiv**Opens in a new window**](https://arxiv.org/abs/2410.07404)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2402.03972] Joint Intrinsic Motivation for Coordinated Exploration in Multi-Agent Deep Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/abs/2402.03972)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.com[ICML 2024] The algorithm of Reinforcement Learning with an Assistant Reward Agent (ReLara) - GitHub**Opens in a new window**](https://github.com/mahaozhe/ReLara)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgSyllabus: Portable Curricula for Reinforcement Learning Agents - arXiv**Opens in a new window**](https://arxiv.org/html/2411.11318v2)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2406.17168] Reinforcement Learning via Auxiliary Task Distillation - arXiv**Opens in a new window**](https://arxiv.org/abs/2406.17168)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comabsdnd/aux_distill - GitHub**Opens in a new window**](https://github.com/absdnd/aux_distill)[![](https://t1.gstatic.com/faviconV2?url=https://wiki.cloudfactory.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)wiki.cloudfactory.comLion - Computer Vision Wiki - CloudFactory**Opens in a new window**](https://wiki.cloudfactory.com/docs/mp-wiki/solvers-optimizers/lion)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comLion, new optimizer discovered by Google Brain using genetic algorithms that is purportedly better than Adam(w), in Pytorch - GitHub**Opens in a new window**](https://github.com/lucidrains/lion-pytorch)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgLearning Agents With Prioritization and Parameter Noise in Continuous State and Action Space - arXiv**Opens in a new window**](https://arxiv.org/html/2410.11250v1)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[1706.01905] Parameter Space Noise for Exploration - arXiv**Opens in a new window**](https://arxiv.org/abs/1706.01905)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccDiffusion-based Curriculum Reinforcement Learning**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/b0e89a49af1fb2ebea69bfc39df0be4a-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2407.01800] Normalization and effective learning rates in reinforcement learning - arXiv**Opens in a new window**](https://arxiv.org/abs/2407.01800)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgNormalization and effective learning rates in reinforcement learning - arXiv**Opens in a new window**](https://arxiv.org/html/2407.01800v1)[![](https://t0.gstatic.com/faviconV2?url=https://proceedings.iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.iclr.ccLEARNING CONTINUALLY BY SPECTRAL REGULARIZATION - ICLR Proceedings**Opens in a new window**](https://proceedings.iclr.cc/paper_files/paper/2025/file/5565ab682d6c7f8d9da34ba0919974b0-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgLearning Continually by Spectral Regularization - arXiv**Opens in a new window**](https://arxiv.org/html/2406.06811v2)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccDistributed Lion for Communication Efficient Distributed Training**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/20cea6c1b36ae5f69c48427a68b67fbc-Paper-Conference.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccConnecting Representation, Collapse, and Trust Issues in PPO - NIPS papers**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/81166fbd9cc5adf14031cdb69d3fd6a8-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comtimoklein/redo: ReDo: The Dormant Neuron Phenomenon in Deep Reinforcement Learning (pytorch) - GitHub**Opens in a new window**](https://github.com/timoklein/redo)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)mdpi.comRestoring Spectral Symmetry in Gradients: A Normalization Approach for Efficient Neural Network Training - MDPI**Opens in a new window**](https://www.mdpi.com/2073-8994/17/10/1648)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comintell-sci-comput/RSRM: code of ICLR 2024 paper Reinforcement Symbolic Regression Machine - GitHub**Opens in a new window**](https://github.com/intell-sci-comput/RSRM)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netReinforcement Learning for Symbolic Graphics Code with Visual Feedback - OpenReview**Opens in a new window**](https://openreview.net/forum?id=HkpqT07shd)[![](https://t0.gstatic.com/faviconV2?url=https://nips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)nips.ccNeurIPS 2024 Papers**Opens in a new window**](https://nips.cc/virtual/2024/papers.html)[![](https://t1.gstatic.com/faviconV2?url=https://papercopilot.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)papercopilot.comNeurIPS 2024 Accepted Paper List - Paper Copilot**Opens in a new window**](https://papercopilot.com/paper-list/neurips-paper-list/neurips-2024-paper-list/)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgSelf-Rewarding PPO: Aligning Large Language Models with Demonstrations Only - arXiv**Opens in a new window**](https://arxiv.org/html/2510.21090v1)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2402.03141] Boosting Reinforcement Learning with Strongly Delayed Feedback Through Auxiliary Short Delays - arXiv**Opens in a new window**](https://arxiv.org/abs/2402.03141)

[![](https://t3.gstatic.com/faviconV2?url=https://www.paperdigest.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.paperdigest.org/2024/10/neurips-2024-highlights/)[![](https://t2.gstatic.com/faviconV2?url=https://www.inria.fr/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.inria.fr/en/neurips-2024-19-papers-selected-inria-saclay-center)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/mikelma/componet)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/DmitryRyumin/ICML-2025-Papers)[![](https://t0.gstatic.com/faviconV2?url=https://iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://iclr.cc/virtual/2025/events/spotlight-posters)[![](https://t1.gstatic.com/faviconV2?url=https://papercopilot.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://papercopilot.com/paper-list/iclr-paper-list/iclr-2025-paper-list/)[![](https://t3.gstatic.com/faviconV2?url=https://rl-conference.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://rl-conference.cc/2024/papers.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/LantaoYu/MARL-Papers)[![](https://t2.gstatic.com/faviconV2?url=https://huggingface.co/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://huggingface.co/papers/trending)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2401.16025v9)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2412.20367)[![](https://t0.gstatic.com/faviconV2?url=https://ojs.aaai.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ojs.aaai.org/index.php/AAAI/article/view/29115/30108)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/forum?id=vI5cjHMzP4)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2507.07451v1)[![](https://t2.gstatic.com/faviconV2?url=https://www.ijcai.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.ijcai.org/proceedings/2025/0788.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2410.20487v2)[![](https://t3.gstatic.com/faviconV2?url=https://search.proquest.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://search.proquest.com/openview/e48d6c2bde48303e653b567ec07f76d2/1.pdf?pq-origsite=gscholar&cbl=36790)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2410.20487v1)[![](https://t0.gstatic.com/faviconV2?url=https://www.jmlr.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.jmlr.org/papers/volume25/24-0087/24-0087.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://proceedings.neurips.cc/paper_files/paper/2024/file/c1f66abb52467443ba8fc70e0a32e061-Paper-Conference.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://ieeexplore.ieee.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ieeexplore.ieee.org/document/10933670/)[![](https://t3.gstatic.com/faviconV2?url=https://rlhfbook.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://rlhfbook.com/c/11-policy-gradients.html)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2509.25762v1)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/forum?id=uGp0mZk0ld)[![](https://t0.gstatic.com/faviconV2?url=https://tisl.cs.toronto.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://tisl.cs.toronto.edu/publication/202407-rlc-aux_tasks_in_rl/rlc2024-aux_tasks_in_rl.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/Allenpandas/Reinforcement-Learning-Papers)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/PWhiddy/dreamerv3-poke)[![](https://t2.gstatic.com/faviconV2?url=https://www.reddit.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.reddit.com/r/reinforcementlearning/comments/1j8sldl/solo_developed_natural_dreamer_simplest_and/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/danijar/dreamerv3)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/eloialonso/iris)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/volcengine/verl)[![](https://t3.gstatic.com/faviconV2?url=https://www.instaclustr.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.instaclustr.com/education/open-source-ai/top-10-open-source-llms-for-2025/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/opendilab/awesome-diffusion-model-in-rl)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2305.13301v4)[![](https://t3.gstatic.com/faviconV2?url=https://www.learndatasci.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/371175373_Bigger_Better_Faster_Human-level_Atari_with_human-level_efficiency)[![](https://t1.gstatic.com/faviconV2?url=https://proceedings.mlr.press/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://proceedings.mlr.press/v202/schwarzer23a.html)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/pdf/2305.19452)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/NoSavedDATA/PyTorch-BBF-Bigger-Better-Faster-Atari-100k)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/lezhang-thu/bigger-better-faster-SAC)[![](https://t1.gstatic.com/faviconV2?url=https://www.mdpi.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.mdpi.com/2079-3197/11/8/148)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/Nov05/udacity-deep-reinforcement-learning)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/naivoder/TD3)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/topics/td3?l=python&o=asc&s=updated)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/BY571/Soft-Actor-Critic-and-Extensions)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/yhisaki/average-reward-drl)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/topics/sac?o=asc&s=forks)[![](https://t0.gstatic.com/faviconV2?url=https://medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://medium.com/@kiranvutukuri/43-activation-functions-in-neural-networks-eb5f84b0d496)[![](https://t0.gstatic.com/faviconV2?url=https://www.superannotate.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.superannotate.com/blog/activation-functions-in-neural-networks)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/397200080_Normalization_and_effective_learning_rates_in_reinforcement_learning)[![](https://t1.gstatic.com/faviconV2?url=https://ideas.repec.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://ideas.repec.org/p/arx/papers/2508.03910.html)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://proceedings.neurips.cc/paper_files/paper/2024/file/c04d37be05ba74419d2d5705972a9d64-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/mhngu23/Intrinsic-Reward-Motivati-Reinforcement-Learning-Re-Implementation)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://openreview.net/forum?id=mlzh3jX6gW)[![](https://t1.gstatic.com/faviconV2?url=https://www.fransoliehoek.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.fransoliehoek.net/publications/htmlfiles/b2hd-He24ECAI.html)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/380218280_MiniZero_Comparative_Analysis_of_AlphaZero_and_MuZero_on_Go_Othello_and_Atari_Games)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/YeWR/EfficientZero)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/index-tts/index-tts)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/opendilab/LightZero/discussions/375)[![](https://t0.gstatic.com/faviconV2?url=https://opendilab.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://opendilab.github.io/LightZero/)[![](https://t0.gstatic.com/faviconV2?url=https://opendilab.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://opendilab.github.io/LightZero/tutorials/installation/installation_and_quickstart.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/opendilab/LightZero/blob/main/docs/source/tutorials/algos/customize_algos.md)[![](https://t0.gstatic.com/faviconV2?url=https://www.semanticscholar.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.semanticscholar.org/paper/Parameter-Space-Noise-for-Exploration-Plappert-Houthooft/142497432fe179ddb6ffe600c64a837ec6179550)[![](https://t3.gstatic.com/faviconV2?url=https://www.worldscientific.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.worldscientific.com/doi/full/10.1142/S0218001421520133)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/385679808_RLion_A_Refined_Lion_Optimizer_for_Deep_Learning)[![](https://t2.gstatic.com/faviconV2?url=https://par.nsf.gov/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://par.nsf.gov/servlets/purl/10539920)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2405.11432v2)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/363307581_Stability-certified_reinforcement_learning_control_via_spectral_normalization)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.researchgate.net/publication/387873043_Neuro-Symbolic_AI_in_2024_A_Systematic_Review)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/abs/2503.16799)[![](https://t3.gstatic.com/faviconV2?url=https://icml.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://icml.cc/virtual/2024/papers.html)[![](https://t2.gstatic.com/faviconV2?url=https://web.stanford.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/CaiaMaiCostelloJasonDanielLazar.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://arxiv.org/html/2305.14342v4)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/Liuhong99/Sophia)[![](https://t2.gstatic.com/faviconV2?url=https://iabac.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://iabac.org/blog/relu-activation-function)[![](https://t1.gstatic.com/faviconV2?url=https://www.reinforcementlearningpath.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.reinforcementlearningpath.com/relu-activation-function/)[![](https://t2.gstatic.com/faviconV2?url=https://developers.google.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://developers.google.com/machine-learning/crash-course/neural-networks/activation-functions?authuser=1)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://github.com/xbeat/Machine-Learning/blob/main/Activation%20Functions%20The%20Driving%20Force%20of%20Neural%20Networks.md)[![](https://t2.gstatic.com/faviconV2?url=https://www.analyticsvidhya.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)Opens in a new window](https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/)
