# Advanced Reinforcement Learning: From Foundations to Frontier

## A Comprehensive PhD Research Curriculum

### Executive Summary

The field of Reinforcement Learning (RL) has undergone a paradigmatic shift in the last half-decade. Moving beyond the foundational successes of tabular methods and early deep Q-networks, the discipline now grapples with the complexities of high-dimensional continuous control, sample efficiency in sparse-reward environments, and the grand unification of planning and learning through world models. Furthermore, the emergent synergy between RL and Generative AI—specifically Large Language Models (LLMs)—has redefined the boundaries of what constitutes an "agent," introducing concepts of verbal reasoning, intrinsic curiosity mediated by semantic understanding, and verifiable self-correction.

This curriculum is designed as a rigorous, 50-assignment research program for a doctoral candidate. It is not merely a collection of papers but a structured pathway through the "hard problems" of modern AI. It demands the synthesis of disparate mathematical frameworks—from optimal transport in distributional RL to control-theoretic bounds in safe RL—and requires the practical engineering of these systems using state-of-the-art codebases. The curriculum is divided into six advanced modules, each targeting a specific frontier of RL research as evidenced by the literature from top-tier venues (NeurIPS, ICML, ICLR) in 2024 and 2025.

---

## Module I: Advanced Policy Optimization and Distributional Dynamics

The first module revisits the fundamental objective of RL: policy optimization. However, it moves beyond simple gradient ascent to explore the geometry of the policy space and the statistical richness of value distributions. The prevailing view in 2024 is that treating value functions as scalar expectations is insufficient for robust performance; instead, capturing the full distribution of returns allows for risk-sensitive and theoretically convergent updates.

### Assignment 1: Manifold-Constrained Trust Region Methods

**Context & Motivation** : Standard Trust Region Policy Optimization (TRPO) utilizes the Kullback-Leibler (KL) divergence to ensure monotonic improvement. However, in safety-critical domains, remaining within a trust region is necessary but not sufficient; the agent must also respect safety constraints which define a safe manifold within the policy space. Recent work in 2024 has introduced methods to explicitly shape this trust region based on safety margins.^^ \*\* \*\*

**Core Readings** :

1. _Constrained Trust Region Policy Optimization (C-TRPO)_ (NeurIPS 2024).^^ \*\* \*\*
2. _Adaptive Trust Region Radius for Robust Policy Optimization_ (ICONIP 2024).^^ \*\* \*\*

**Mathematical Synthesis** : The candidate must derive a **Dynamic-Constraint Trust Region Objective** . While standard TRPO enforces a hard KL constraint $D*{KL}(\pi*{\theta} |

| \pi\_{old}) \le \delta$, C-TRPO modifies the geometry of the policy space to ensure safety constraints **C**(**π**)**≤**d are met within the trust region itself. The synthesis involves formulating a dual-update rule where the trust region radius **δ**k\***\* is adaptively scaled based on the safety margin **d**−**C**(**π**k) and the gradient conflict angle between the reward objective **∇**J**R** and the cost objective **∇**J**C\*\*.

**θ**max∇**J**R(**θ**)**T**(**θ**−**θ**o**l**d)**s.t.**2**1\*\***(**θ**−**θ**o**l**d)**T**H**(**θ**−**θ**o**l**d)**≤**δ**a**d**a**pt**i**v**e\***\*and**∇**J**C(**θ**)**T**(**θ**−**θ**o**l**d)**≤**d**−**C**(**π**o**l**d\*\***)\*\*

Here, **δ**a**d**a**pt**i**v**e is derived from the adaptive radius logic found in ATRPO ^^, allowing larger updates when far from safety boundaries. \*\* \*\*

**Implementation Plan** : The implementation should leverage the `adaptive-trpo` repository ^^ as a foundation. The candidate must extend the `AdaptiveTrustRegion` class to inherit from the standard TRPO agent but modify the conjugate gradient step. The key engineering challenge is to project gradients onto the intersection of the safety half-space and the KL-ball using a quadratic programming solver (like `cvxpy`) or a closed-form dual approximation if feasible. The environment of choice is `SafetyPointGoal1-v0` from `Safety-Gymnasium`, which provides explicit cost signals.^^ \*\* \*\*

**Evaluation Strategy** : The evaluation must characterize the **Safety-Performance Pareto Frontier** . By training with varying safety thresholds **d**, the candidate should plot the trade-off between Reward Regret (loss of reward due to safety) and Constraint Violation Regret. A successful implementation will show strictly fewer violations during the early phases of training compared to a standard Lagrangian-PPO baseline.

### Assignment 2: Sinkhorn Distributional Reinforcement Learning

**Context & Motivation** : Distributional RL, which models the distribution **Z**(**s**,**a**) of returns rather than the mean **Q**(**s**,**a**), has shown superior performance in stochastic environments. However, widely used metrics like the Wasserstein distance are computationally expensive to optimize directly. The application of Sinkhorn iterations offers a differentiable approximation, bridging the gap between optimal transport theory and practical deep RL.^^ \*\* \*\*

**Core Readings** :

1. _Distributional Reinforcement Learning with Regularized Wasserstein Loss_ (NeurIPS 2024).^^ \*\* \*\*
2. _Distributional RL Decision and Control_ (RA-L 2024).^^ \*\* \*\*

**Mathematical Synthesis** : The goal is to formulate the **Sinkhorn-Bellman Operator** . Standard Distributional RL (like QR-DQN) minimizes the quantile Huber loss. This assignment requires replacing that with the Entropic Wasserstein distance, computable via Sinkhorn-Knopp. The objective function becomes:

**L**(**θ**)**=**E**s**,**a**

where **W**ϵ is the entropic Wasserstein distance. The synthesis requires deriving the gradient of the Sinkhorn loss with respect to the quantile locations of the implicit quantile network (IQN), specifically showing how entropy regularization **ϵ** mitigates the gradient vanishing problem in the tails of the distribution.

**Implementation Plan** : Using the `SinkhornDistRL` repository ^^, the candidate will integrate the Sinkhorn loss function into an `AC-IQN` (Actor-Critic Implicit Quantile Network) architecture described in.^^ The challenge is implementing the Sinkhorn iteration efficiently in PyTorch to avoid stalling the training loop. The testbed will include chaotic Atari games like `Breakout` and `Asterix`, where distributional shifts are frequent. \*\* \*\*

**Evaluation Strategy** : The primary metric is the **Collapse of Support Variance** . In standard QR-DQN, the learned distribution often collapses to a Dirac delta in deterministic environments. The candidate must empirically demonstrate that Sinkhorn loss maintains a wider support, indicating better capturing of epistemic uncertainty, which correlates with improved asymptotic performance in the `VRX` surface vehicle navigation task.^^ \*\* \*\*

### Assignment 3: Variance-Reduced Policy Gradients

**Context & Motivation** : Policy gradient methods suffer from high variance, leading to poor sample efficiency. Recent theoretical advancements in 2024 have focused on applying momentum-based variance reduction techniques (borrowed from stochastic optimization like STORM/SPIDER) to the RL setting, specifically in decentralized or zeroth-order contexts.^^ \*\* \*\*

**Core Readings** :

1. _Decentralized Natural Policy Gradient with Variance Reduction_ (JMLR 2024).^^ \*\* \*\*
2. _Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models_ (ICML 2024).^^ \*\* \*\*
3. _Policy Gradient with Active Importance Sampling_ (RLC 2024).^^ \*\* \*\*

**Mathematical Synthesis** : The candidate will synthesize a **Recursive Momentum Estimator with Active Importance Sampling** . The standard gradient estimator is replaced by a recursive term **v**t that corrects the current gradient estimate using importance sampling weights **w**t=**π**t**−**1\***\*(**a**∣**s**)**π**t(**a**∣**s\*\*):

**v**t=**∇**J**(**π**t\*\***)**+**(**1**−**α**)**(**v**t**−**1\*\***−**w**t∇**J**(**π**t**−**1\*\*\*\*))

The novel synthesis involves proving that active selection of the behavioral policy **π**b (as discussed in ^^) minimizes the trace of the covariance matrix of **v**t, thereby accelerating convergence. \*\* \*\*

**Implementation Plan** : Starting with the `mezo_svrg` ^^ codebase for the variance reduction logic, the candidate will adapt this to a standard PPO agent from `Safe-RL-Baselines`.^^ This involves modifying the `update` method to maintain a running buffer of past gradients and applying the recursive correction. The target environments are high-dimensional MuJoCo tasks (`Humanoid-v4`, `Ant-v4`). \*\* \*\*

**Evaluation Strategy** : The evaluation must explicitly measure **Gradient Variance** during training (using multiple seeds to estimate empirical variance of the gradient estimator). The candidate should plot "Samples to Threshold" metrics, demonstrating that the increased computational cost per step is outweighed by the reduction in total environment interactions required.

### Assignment 4: Functional Critic Convergence in Off-Policy RL

**Context & Motivation** : The "Deadly Triad"—the instability caused by combining function approximation, bootstrapping, and off-policy learning—remains a central theoretical hurdle. New work in 2025 proposes treating the critic not as a fixed network but as a functional mapping, providing a path to provable convergence.^^ \*\* \*\*

**Core Readings** :

1. _Functional Critic Modeling for Provably Convergent Off-Policy Actor-Critic_ (ArXiv 2025/NeurIPS 2024).^^ \*\* \*\*
2. _Efficient Recurrent Off-Policy RL_ (NeurIPS 2024).^^ \*\* \*\*

**Mathematical Synthesis** : The assignment requires formalizing the **Functional Bellman Error** . Instead of a parametric critic **Q**ϕ, the critic is modeled as a functional **Q**:**Π**→**R**S**×**A. The synthesis requires defining an objective that accounts for the "moving target" problem where the actor **π** changes during the critic's convergence:

**L**f**u**n**c\*\***(**Q**,**π**)**=**∥**Q**(**π**)**−**(**r**+**γ**P**Q**(**π**))**∥**D**2\*\***

The candidate must derive the update rule that ensures the critic functional remains Lipschitz continuous with respect to the policy parameters, a condition necessary for stability.

**Implementation Plan** : Using `Soft-Actor-Critic-and-Extensions` ^^ as a base, the candidate will replace the standard Q-network with a Hypernetwork or a functional encoding architecture that takes policy embeddings as input alongside state-action pairs. This will be tested in the `DeepMind Control Suite`. \*\* \*\*

**Evaluation Strategy** : A **Deadly Triad Stress Test** is required. The agent should be trained on a fixed offline dataset with a behavior policy significantly different from the optimal policy. The candidate must measure the divergence of Q-values (soft divergence) compared to true Monte Carlo returns to verify stability.

### Assignment 5: Fisher-Information Guided Parameter Noise

**Context & Motivation** : Exploration in continuous control is often limited to additive Gaussian noise on actions. "Parameter Space Noise" perturbs the weights of the policy network directly, inducing state-dependent exploration. Recent work has revisited this with adaptive scaling based on information geometry.^^ \*\* \*\*

**Core Readings** :

1. _Parameter Space Noise for Exploration_ (Revisited context).^^ \*\* \*\*
2. _PDDPG with Parameter Noise_ (ArXiv 2024).^^ \*\* \*\*
3. _Parameter Space Exploration of Neural Network Inference_ (IEEE 2024).^^ \*\* \*\*

**Mathematical Synthesis** : The candidate will formulate **Adaptive-Scaling Parameter Noise** using the **Fisher Information Matrix (FIM)** . Instead of **θ**~**=**θ**+**N**(**0**,**σ**I**), the noise is shaped by **Σ**∝**(**I**+**λ**F**(**θ**)**)**−**1**. This ensures that perturbations in parameter space result in consistent and meaningful divergence in action space, preventing the agent from exploring useless high-frequency noise regions.

**Implementation Plan** : Implementing this in a TD3 agent (using `Deep-reinforcement-learning-with-pytorch` ^^), the candidate will use K-FAC (Kronecker-Factored Approximate Curvature) to approximate the inverse FIM efficiently. The method will be tested on sparse reward environments like `MountainCarContinuous-v0`. \*\* \*\*

**Evaluation Strategy** : **State Space Coverage Entropy** : By discretizing the state space, the candidate will calculate the entropy of the visitation distribution. Successful parameter noise should yield higher entropy (more diverse exploration) compared to action noise baselines.

### Assignment 6: Diffusion-Regularized On-Policy Learning

**Context & Motivation** : Diffusion models have revolutionized generative modeling and are now being applied to RL to capture complex, multi-modal distributions. Integrating diffusion priors into on-policy methods like PPO offers a way to improve sample efficiency by guiding exploration towards high-density regions of good trajectories.^^ \*\* \*\*

**Core Readings** :

1. _Enhancing Sample Efficiency... through Integration of Diffusion Models and PPO_ (ArXiv 2024).^^ \*\* \*\*
2. _Diffusion Trusted Q-Learning_ (NeurIPS 2024).^^ \*\* \*\*

**Mathematical Synthesis** : Develop a **Diffusion-Regularized PPO (DR-PPO)** . Standard PPO uses a KL constraint against the previous policy. Here, the constraint is against a **Diffusion Prior** **D**p**r**i**or** trained on a buffer of successful past trajectories. $$ L*{CLIP}(\theta) - \beta D*{KL}(\pi\_{\theta} |

| \mathcal{D} _{prior}) $$ The synthesis involves deriving the score-based gradient of this KL term, $\nabla_ \theta \log p\_{\text{diff}}(a|s)$, allowing the PPO agent to "query" the diffusion model for gradients that push it towards the safe manifold.

**Implementation Plan** : Combining `Diffusion_Trusted_Q_Learning` ^^ and a clean PPO implementation, the candidate will integrate a pre-trained diffusion model as a regularizer. The experiment will be conducted on `MuJoCo` tasks with strict interaction budgets (e.g., 100k steps). \*\* \*\*

**Evaluation Strategy** : **Area Under the Learning Curve (ALC)** : This metric penalizes slow learning. The candidate must demonstrate that the diffusion prior accelerates the initial phase of learning compared to vanilla PPO.

### Assignment 7: Decoupled Entropy for Discrete SAC

**Context & Motivation** : Soft Actor-Critic (SAC) is dominant in continuous control but has historically struggled in discrete domains due to the difficulty of defining "entropy" appropriately when optimal policies are often deterministic. New research in 2024 proposes decoupling the entropy terms in the actor and critic updates.^^ \*\* \*\*

**Core Readings** :

1. _Revisiting Discrete Soft Actor-Critic_ (TMLR 2024).^^ \*\* \*\*
2. _Discrete Action Off-Policy Actor-Critic_ (OpenReview 2024).^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize a **Tsallis-Entropy Discrete SAC** . Instead of Shannon entropy, use Tsallis entropy **H**q(**π**)**=**q**−**1**1\*\***(**1**−**∑**π**(**a**)**q**) which generalizes the sparsity-inducing properties. The synthesis requires modifying the soft Bellman backup to account for the Tsallis regularizer, proving that as **q**→**0**, the update resembles a "hard" max, while **q**→**1\*\* recovers Shannon entropy.

**Implementation Plan** : Using `SD-SAC` ^^ as the codebase, the candidate will implement the Gumbel-Softmax relaxation for the critic update to allow differentiable reparameterization in discrete space. The target environments are `Atari` games like `Qbert` and `Pong`. \*\* \*\*

**Evaluation Strategy** : **Entropy Collapse Monitoring** : The candidate must track the entropy of the policy throughout training. A successful implementation will show delayed entropy collapse, indicating sustained exploration, and higher final asymptotic scores.

### Assignment 8: Wasserstein Robust Soft Actor-Critic

**Context & Motivation** : Deploying RL agents in the real world requires robustness to parameter mismatches (sim-to-real gap). Distributionally Robust RL optimizes for the worst-case scenario within an uncertainty set. Recent work combines this with the maximum entropy framework of SAC.^^ \*\* \*\*

**Core Readings** :

1. _Robust Reinforcement Learning Papers (Foundation to 2024)_ .^^ \*\* \*\*
2. _Risk-Sensitive Soft Actor-Critic for Robust DRL_ (ArXiv 2024).^^ \*\* \*\*

**Mathematical Synthesis** : Formulate **Distributionally Robust SAC (DR-SAC)** using a Wasserstein Uncertainty Set. The objective is to maximize **min**P**′**∈**P\*\***E**τ**∼**P**′**,**π**. The synthesis involves defining the dual form of the Wasserstein constraint, converting the inner minimization into a tractable penalty term involving the Lipschitz constant of the value function: **λ**∥**∇**s\*\***V**(**s**)**∥.

**Implementation Plan** : The candidate will implement the dual gradient descent for the robustness Lagrangian multiplier within the `RiskSensitiveSAC` ^^ framework. Testing will involve `MuJoCo` environments with **Perturbed Physics** (e.g., gravity or friction varied by **±**50%). \*\* \*\*

**Evaluation Strategy** : **Robustness Profile** : The candidate will generate a "performance vs. perturbation" curve. The DR-SAC agent should exhibit a flatter curve (less degradation) compared to standard SAC.

---

## Module II: Exploration, Intrinsic Motivation & Sample Efficiency

This module addresses the central challenge of RL: learning effectively when rewards are sparse, delayed, or absent. The assignments leverage large language models and latent space dynamics to synthesize "curiosity" and "surprise."

### Assignment 9: Semantic Curiosity with Large Language Models

**Context & Motivation** : Traditional intrinsic motivation relies on pixel-level prediction error, which is sensitive to noise (the "noisy TV problem"). In 2024/2025, researchers are using LLMs to provide "semantic" curiosity—rewarding the agent for discovering states that are linguistically novel or surprising.^^ \*\* \*\*

**Core Readings** :

1. _LLM-Driven Intrinsic Motivation for Sparse Reward Reinforcement Learning_ (ArXiv 2025).^^ \*\* \*\*
2. _Curiosity-Driven Exploration (CDE) for RLVR_ (ArXiv 2025).^^ \*\* \*\*

**Mathematical Synthesis** : The candidate must develop a **Semantic Surprise Intrinsic Reward** . Given a state **s**t, use a quantized VLM to generate a description **d**t. The intrinsic reward is defined as the negative log-likelihood (perplexity) of **d**t**+**1\***\* given **d\*\*t under a pre-trained LLM:

**r**in**t\*\***=**−**lo**g**P**LL**M(**desc**(**s**t**+**1\***\*)**∣**desc**(**s**t\***\*)**,**a**t\***\*)**

This formalizes curiosity as "narrative surprise."

**Implementation Plan** : Using the `verl` ^^ framework, the candidate will integrate a Llama-3-8B model (quantized) to score transitions in a `MiniGrid` environment. The PPO agent will optimize the sum of extrinsic and intrinsic rewards. \*\* \*\*

**Evaluation Strategy** : **Semantic Coverage** : In a sparse-reward maze with distinct rooms (e.g., "Blue Room", "Kitchen"), measure how many semantically distinct regions the agent visits compared to a Random Network Distillation (RND) baseline.

### Assignment 10: Random Latent Exploration (RLE)

**Context & Motivation** : Random Network Distillation is a powerful exploration baseline, but recent work suggests simpler mechanisms can be effective. Random Latent Exploration (RLE) perturbs rewards with random projections in a latent space, inducing a "covering" behavior without complex density modeling.^^ \*\* \*\*

**Core Readings** :

1. _Random Latent Exploration for Deep Reinforcement Learning_ (ICML 2024).^^ \*\* \*\*
2. _Expansive Latent Planning for Sparse Reward Offline RL_ (CoRL 2023).^^ \*\* \*\*

**Mathematical Synthesis** : The synthesis involves unifying **RLE with Generalized Goal Conditioned RL** . Define a random goal **z**g in the latent space of the value function. The intrinsic reward is **r**in**t\*\***(**s**,**a**)**=**r**e**x**t\*\***(**s**,**a**)**+**λ**⟨**ϕ**(**s**)**,**z**g⟩. The candidate must prove that switching **z**g periodically is equivalent to maximizing the entropy of the state visitation distribution under linear value approximation assumptions.

**Implementation Plan** : The candidate will implement RLE on top of a standard DQN or SAC agent using the official `random-latent-exploration` logic. The implementation requires designing the latent projection **ϕ**(**s**) (e.g., a random but fixed neural network). Testbed: `Montezuma's Revenge`.

**Evaluation Strategy** : **Dormant Neuron Ratio** : Monitor the percentage of "dead" neurons in the policy network. Effective exploration should utilize the full capacity of the network, resulting in a lower dormant ratio compared to **ϵ**-greedy exploration.

### Assignment 11: Hierarchical Goal-Based Pre-Training (PTGM)

**Context & Motivation** : Sample efficiency is drastically improved by pre-training on offline data. PTGM (Pre-Training Goal-based Models) introduces a hierarchical approach where a high-level policy learns to set goals for a low-level policy, improving temporal abstraction.^^ \*\* \*\*

**Core Readings** :

1. _Pre-Training Goal-based Models for Sample-Efficient Reinforcement Learning_ (ICLR 2024).^^ \*\* \*\*
2. _Zero-Shot Reinforcement Learning from Low Quality Data_ (NeurIPS 2024).^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **Hierarchical Goal-Conditioned Pre-training** . Formulate a bi-level optimization where the high-level policy **π**hi(**g**∣**s**) maximizes the coverage of the pre-training dataset, while **π**l**o\*\***(**a**∣**s**,**g**)\*\* minimizes the goal-reaching error.

**J**(**θ**)**=**E**offline\*\***[**log**π**l**o(**a**∣**s**,**g**)]**+**λ**H**(**π**hi(**⋅**∣**s**))\*\*

The synthesis requires deriving a fine-tuning update where only **π**hi adapts to the online task reward, freezing **π**l**o** to preserve primitive skills.

**Implementation Plan** : Using the `PTGM` repository ^^, pre-train on the `MineDojo` (Minecraft) dataset. Then, fine-tune on a specific task like "Harvest Wood." \*\* \*\*

**Evaluation Strategy** : **Few-Shot Adaptation Rate** : Measure the number of online interaction steps required to reach 50% of expert performance. PTGM should be significantly faster than training a flat policy from scratch.

### Assignment 12: EfficientZero V2 and Consistent Reanalysis

**Context & Motivation** : Model-based methods like MuZero are sample efficient but computationally heavy. EfficientZero V2 optimizes this by ensuring consistency between the learned value function and the MCTS-improved value estimates, effectively "distilling" the search into the network.^^ \*\* \*\*

**Core Readings** :

1. _EfficientZero V2: General Framework for Sample-Efficient RL_ (ArXiv 2024).^^ \*\* \*\*
2. _Numerical Evidence for Sample Efficiency of Model-Based Over Model-Free..._ (IEEE 2024).^^ \*\* \*\*

**Mathematical Synthesis** : Analyze the **Lookahead Consistency Loss** . Define a regularization term between the online value network **v**θ(**s**) and the MCTS root value **ν**MCTS(**s**):

**L**=**∥**v**θ\*\***(**s**)**−**ν**MCTS\*\***(**s**)**∥**2**+**α**∥**h**ψ\*\***(**s**)**−**Proj**(**o**)**∥\*\*2

The candidate must derive the bias-variance trade-off of the value target as a function of the MCTS simulation count **k**.

**Implementation Plan** : Implement the consistency loss within an `EfficientZero` framework (often found in `ding` or `LightZero`). The target is the `Atari 100k` benchmark, specifically hard games like `Solaris`.

**Evaluation Strategy** : **Human Normalized Score (HNS)** at 100k steps. The implementation should surpass Rainbow DQN by at least 2x.

### Assignment 13: Plasticity and Forgetting in Continual RL

**Context & Motivation** : Deep RL agents suffer from "loss of plasticity"—the inability to learn new tasks after being trained on old ones. The "Forget and Grow" (FoG) strategy proposes mechanisms to periodically reset parts of the network to maintain learnability.^^ \*\* \*\*

**Core Readings** :

1. _Forget and Grow: A Forget-and-Grow Strategy for Deep Reinforcement Learning Scaling..._ (NeurIPS 2024).^^ \*\* \*\*
2. _Continual Knowledge Adaptation for Reinforcement Learning_ (NeurIPS 2025).^^ \*\* \*\*

**Mathematical Synthesis** : Formulate the **Plasticity-Stability Dynamics** . Introduce a **Network Expansion Operator** **E** and a **Selective Reset Mask** **M**.

**θ**t**+**1\***\*=**E**(**θ**t−**η**∇**J**)**⊙\*\*M

where **M** resets weights with low gradient magnitudes ("dormant" weights). Synthesize a criterion for expansion based on the saturation of the effective rank of the feature matrix.

**Implementation Plan** : Implement a PPO agent (using `cleanrl` structure) that monitors feature rank. If rank plateaus, expand the MLP width. Periodically reset the last layer of the critic. Test on `Procgen` environments, switching levels every 10M steps.

**Evaluation Strategy** : **Plasticity Retention** : Measure the learning speed (slope of reward curve) on the _N-th_ task compared to the _1st_ task. FoG should maintain a constant learning speed.

### Assignment 14: Vision-Language Model Feedback for RL

**Context & Motivation** : In the real world, reward functions are hard to code. Using Vision-Language Models (VLMs) as "judges" allows agents to learn from natural language instructions. However, VLMs are noisy. Recent work proposes robust ways to use this feedback.^^ \*\* \*\*

**Core Readings** :

1. _Real-World Offline Reinforcement Learning from Vision Language Model Feedback_ (ArXiv 2024).^^ \*\* \*\*
2. _Multi-objective Reinforcement learning from AI Feedback_ (ArXiv 2024).^^ \*\* \*\*

**Mathematical Synthesis** : Develop a **VLM-generated Reward Shaping** function with **Confidence Weighting** . Given image **o**t and goal **g**, reward **R**=**VLM**(**o**t,**g**). To handle noise, weight the Bellman update by the VLM's confidence (or entropy of the VLM output distribution).

**w**i=**1/**(**Entropy**(**VLM**(**o**i,**g**))**+**ϵ**)**

**Implementation Plan** : Use `CLIP` or `LLaVA` to label frames in a `MetaWorld` task (e.g., "Push Box"). Train an Offline RL agent (IQL) on these VLM-labeled rewards.

**Evaluation Strategy** : **Label Efficiency** : Compare policy success rate when trained on VLM rewards vs. sparse rewards.

### Assignment 15: Value-Conservative Forward-Backward Representations

**Context & Motivation** : Zero-shot transfer requires learning representations that disentangle dynamics from rewards. Forward-Backward (FB) representations learn successor measures for _all_ policies. Recent work adds conservatism to allow transfer from low-quality data.^^ \*\* \*\*

**Core Readings** :

1. _Zero-Shot Reinforcement Learning from Low Quality Data_ (NeurIPS 2024).^^ \*\* \*\*
2. _VC-FB and MC-FB algorithms_ .^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize the **Value-Conservative FB (VC-FB)** objective. The successor measure **M** is decomposed into **ψ**(**s**,**z**) and **ξ**(**s**′**,**z**)**. Add a penalty for out-of-distribution state-action pairs to the FB loss:

**L**FB+**α**E**s**,**a**∼**π**n**e**w

This bounds the transfer error to unseen tasks.

**Implementation Plan** : Using `zero-shot-rl` ^^, train VC-FB on `ExORL` datasets. Attempt zero-shot transfer from "Walker Walk" to "Walker Run". \*\* \*\*

**Evaluation Strategy** : **Transfer Regret** : Difference between zero-shot performance and fine-tuned performance.

### Assignment 16: Average Reward Soft Actor-Critic (RVI-SAC)

**Context & Motivation** : Most RL maximizes discounted return, but many real-world tasks are continuing (infinite horizon). RVI-SAC adapts maximum entropy RL to the average reward criterion, crucial for stable long-term operation.^^ \*\* \*\*

**Core Readings** :

1. _RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning_ (ICML 2024).^^ \*\* \*\*
2. _Risk-Sensitive Soft Actor-Critic_ .^^ \*\* \*\*

**Mathematical Synthesis** : Reformulate the **Differential Bellman Equation** for SAC.

**Q**(**s**,**a**)**=**r**(**s**,**a**)**−**r**~**+**γ**E**[**V**(**s**′**)]\*\*

where **r**~ is the learnable average reward. Derive the stability conditions for learning **r**~ concurrently with the soft Q-function.

**Implementation Plan** : Using `average-reward-drl` ^^, compare RVI-SAC against standard Discounted SAC on continuous `MuJoCo` tasks (modifying them to never terminate). \*\* \*\*

**Evaluation Strategy** : **Average Reward Stability** : Plot the estimated average reward **r**~ vs. the true rolling average. Check for divergence.

---

## Module III: Offline Reinforcement Learning & Generalization

This module tackles the "Static Dataset" problem: learning optimal policies without further environment interaction, and ensuring those policies generalize to new, unseen situations. This is critical for applying RL to healthcare, robotics, and industrial control.

### Assignment 17: Calibrated Conservative Q-Learning (Cal-QL)

**Context & Motivation** : Standard Conservative Q-Learning (CQL) suppresses Q-values of out-of-distribution actions to prevent overestimation. However, this often leads to "uncalibrated" values that are far lower than the true return, making online fine-tuning difficult. Cal-QL imposes a constraint that Q-values must be lower-bounded by the value of the behavioral policy, facilitating smoother adaptation.^^ \*\* \*\*

**Core Readings** :

1. _Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning_ (NeurIPS 2023/2024 context).^^ \*\* \*\*
2. _The Generalization Gap in Offline Reinforcement Learning_ (ICLR 2024).^^ \*\* \*\*

**Mathematical Synthesis** : The candidate must synthesize the **Calibrated Conservatism Objective** . Combining the CQL lower bound with the calibration upper bound:

**Q**minα**E**s**∼**D**,**a**∼**μ\***\*[**Q**(**s**,**a**)**−**V**π**(**s**)**]**++**2**1L**T**D(**Q\*\*)

Derive why this objective guarantees that the learned **Q** function acts as a conservative lower bound on **Q**∗ but an upper bound on **V**π, effectively "sandwiching" the true value.

**Implementation Plan** : Using the `d3rlpy` or `CORL` library, implement Cal-QL. Train on `D4RL` datasets (Medium-Expert). Then, perform online fine-tuning on `HalfCheetah-v2`.

**Evaluation Strategy** : **Calibration Error** : Measure **E**[**Q**(**s**,**a**)]**−**Return**(**π**)**. A lower error indicates better calibration and predicts faster online fine-tuning.

### Assignment 18: Robust Implicit Q-Learning (RIQL)

**Context & Motivation** : Offline datasets are often corrupted with noisy rewards or state observations. Standard IQL (Implicit Q-Learning), which relies on expectile regression, can be sensitive to these outliers. RIQL introduces robust loss functions to mitigate this.^^ \*\* \*\*

**Core Readings** :

1. _Towards Robust Offline Reinforcement Learning under Diverse Data Corruption_ (ICLR 2024).^^ \*\* \*\*
2. _Robust IQL (RIQL)_ .^^ \*\* \*\*

**Mathematical Synthesis** : Develop **Quantile-Robust IQL** . Replace the squared error in expectile regression with a **Huber-Quantile Loss** or a **Welsch Loss** that downweights samples with large TD errors (potential outliers).

**L**(**θ**)**=**E**(**s**,**a**,**r**)**∼**D\*\***[**ρ**robust(**r**+**γV**(**s**′**)**−**Q**(**s**,**a**))]\*\*

Synthesize the proof that this formulation is robust to **ϵ**-fraction reward corruption in the dataset.

**Implementation Plan** : Using the `RIQL` repository ^^, the candidate will artificially corrupt a D4RL dataset (e.g., flipping signs of 10% of rewards). Train standard IQL and RIQL and compare resilience. \*\* \*\*

**Evaluation Strategy** : **Performance Degradation Ratio** : (Score on Clean Data - Score on Corrupted Data) / Score on Clean Data. RIQL should minimize this ratio.

### Assignment 19: Trust-Region Diffusion for Offline RL (DTQL)

**Context & Motivation** : Diffusion policies are expressive but slow to sample. DTQL (Diffusion Trusted Q-Learning) proposes using the diffusion model not as the policy itself, but as a constraint (a "trust region") for a faster, feed-forward policy.^^ \*\* \*\*

**Core Readings** :

1. _Diffusion Trusted Q-Learning (DTQL)_ (NeurIPS 2024).^^ \*\* \*\*
2. _Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model_ (ICLR 2024).^^ \*\* \*\*

**Mathematical Synthesis** : Formulate the **Diffusion-Constraint Objective** . Maximize **Q**(**s**,**π**(**s**)) subject to **π**(**s**) lying in the high-likelihood region of a pre-trained diffusion behavior model **p**β.

**θ**maxE**s**∼**D\*\***[**Q**(**s**,**π**θ(**s**))**−**λ**L**d**i**ff(**π**θ(**s**))]\*\*

Derive the gradient of the diffusion loss **L**d**i**ff with respect to the policy actions, utilizing the diffusion score function.

**Implementation Plan** : Leverage the `Diffusion_Trusted_Q_Learning` repo.^^ Implement the dual-policy architecture (diffusion constraint + fast actor). Compare inference speed against pure Diffusion-QL. \*\* \*\*

**Evaluation Strategy** : **Inference Latency vs. Reward** : Plot a scatter plot of Wall-Clock Inference Time (ms) vs. Episode Reward. DTQL should be in the "fast and high-performing" quadrant.

### Assignment 20: Offline Trajectory Optimization (OTTO)

**Context & Motivation** : Offline RL struggles with "stitching"—combining parts of suboptimal trajectories to form an optimal one. OTTO uses a Transformer-based world model to "hallucinate" better trajectories and add them to the dataset, effectively performing offline data augmentation.^^ \*\* \*\*

**Core Readings** :

1. _Offline Trajectory Optimization for Offline Reinforcement Learning (OTTO)_ (ArXiv 2024).^^ \*\* \*\*
2. _Trajectory Optimization reinforcement learning_ .^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **Consistency-Constrained Trajectory Generation** . Train a World Transformer to predict dynamics. Generate synthetic trajectories **τ**sy**n** that maximize predicted reward. Apply a **Kinematic Consistency Filter** :

**Keep **τ**sy**n if **∥**s**t**+**1\*\***−**f**d**y**n(**s**t,**a**t)**∥**<\*\*δ

This prevents the "exploitation" of model errors.

**Implementation Plan** : Using `Decision Transformer` codebases as a starting point (or OTTO code if available), train a world model on `AntMaze`. Generate 100k synthetic "stitched" trajectories. Augment the D4RL dataset and train a Behavior Cloning agent.

**Evaluation Strategy** : **Stitching Success** : Evaluate on `AntMaze-Large`. The agent must navigate between start and goal using paths never fully seen in the source data.

### Assignment 21: Latent Action Policies (LAPO)

**Context & Motivation** : Learning from video (Observation-only) is a holy grail. LAPO infers "Latent Actions" from video transitions, allowing policies to be trained without explicit action labels.^^ \*\* \*\*

**Core Readings** :

1. _Learning to Act without Actions (LAPO)_ (ICLR 2024).^^ \*\* \*\*
2. _Latent Action Spaces_ .^^ \*\* \*\*

**Mathematical Synthesis** : Formulate **Latent Inverse Dynamics** . Learn a latent action **z**a such that it explains the transition **s**t→**s**t**+**1\*\*\*\*.

**ϕ**,**ψ**min∥**s**t**+**1\***\*−**D**ψ(**E**ϕ(**s**t)**,**z**a\***\*)**∥**2**

Synthesize a **Vector Quantization** constraint on **z**a to force the latent actions to correspond to discrete primitives (e.g., "Jump", "Duck"), making the latent space semantic.

**Implementation Plan** : Using the `LAPO` repo ^^, train on `Procgen` videos (CoinRun). Then, train a small "Action Decoder" using <1% labeled data to map latent actions **z**a to real actions **a**. \*\* \*\*

**Evaluation Strategy** : **Label Efficiency** : Plot Policy Performance vs. Number of Labeled Actions. LAPO should reach expert performance with orders of magnitude fewer labels than supervised learning.

### Assignment 22: Distributional Shift in Offline Generalization

**Context & Motivation** : Offline agents trained on one environment often fail when transferred to a slightly different one. New benchmarks in 2024 quantify this "Generalization Gap".^^ \*\* \*\*

**Core Readings** :

1. _The Generalization Gap in Offline Reinforcement Learning_ (ICLR 2024).^^ \*\* \*\*
2. _Identifiability/Generalization in Offline RL_ .^^ \*\* \*\*

**Mathematical Synthesis** : Define **Epistemic Robustness** . The test environment parameters **ξ** are drawn from **P**(**ξ**). Formulate a **Robust Bellman Operator** that penalizes Q-values based on the variance of the transition model ensemble over the parameter space.

**Q**(**s**,**a**)**←**r**+**γ**(**E**P**^[**V**(**s**′**)]**−**β**Var**P**^[**V**(**s**′**)])\*\*

**Implementation Plan** : Using the benchmark code from ^^, train CQL on `Procgen` (Easy distribution). Test on `Procgen` (Hard distribution). Implement the variance penalty in the CQL loss. \*\* \*\*

**Evaluation Strategy** : **Generalization Gap** : Quantify the percentage drop in return from Train to Test environments.

### Assignment 23: Feasibility-Guided Safe Offline RL

**Context & Motivation** : In offline Safe RL, we must avoid unsafe regions even if they have high reward. FISOR identifies the "Feasible Region" (safe states) and constrains the policy to stay within it using a classifier-guided diffusion model.^^ \*\* \*\*

**Core Readings** :

1. _Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model_ (ICLR 2024).^^ \*\* \*\*
2. _Feasibility Consistent Representation Learning (FCSRL)_ (ICML 2024).^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize the **Feasibility-Value Objective** . Train a safety classifier **C**(**s**,**a**)**=**P**(**safe**∣**s**,**a**)**. Modulate the value target:

**T**Q**(**s**,**a**)**=**C**(**s**,**a**)**⋅**(**r**+**γV**(**s**′**))**+**(**1**−**C**(**s**,**a**))**⋅**(**−**∞**)

This effectively "cuts off" unsafe paths in the Bellman backup.

**Implementation Plan** : Using `FISOR` ^^, train on a D4RL dataset augmented with "speed limits" (treating high velocity as unsafe). \*\* \*\*

**Evaluation Strategy** : **Safety Violation Rate** : Count constraint violations during evaluation. FISOR should approach zero violations.

### Assignment 24: Adaptive Offline-to-Online Fine-Tuning

**Context & Motivation** : Moving from offline pre-training to online fine-tuning is unstable due to distribution shift. "Hybrid RL" approaches mix offline and online data. Recent work optimizes the mixing ratio.^^ \*\* \*\*

**Core Readings** :

1. _Hybrid RL: Using both offline and online data_ .^^ \*\* \*\*
2. _Off-policy Reinforcement Learning with Model-based Exploration Augmentation_ (NeurIPS 2025).^^ \*\* \*\*

**Mathematical Synthesis** : Formulate **TD-Error Weighted Sampling** . The replay buffer mixes static offline data and growing online data. Derive an adaptive ratio **α**t where the probability of sampling online data is proportional to its relative TD-error magnitude (representing "surprise" or "learning opportunity").

**Implementation Plan** : Start with an IQL checkpoint (from Assignment 18). Fine-tune on `Ant-v4`. Implement the adaptive sampling loader.

**Evaluation Strategy** : **Online Sample Efficiency** : Measure accumulated reward in the first 100k online steps.

### Assignment 25: Offline-Boosted Residual Learning

**Context & Motivation** : Instead of fine-tuning the whole policy (which risks catastrophic forgetting), learn a "residual" or "boost" policy that adds to the offline base policy.

**Core Readings** :

1. _Offline-Boosted Actor-Critic_ (ICML 2024).^^ \*\* \*\*
2. _Real-World Offline Reinforcement Learning_ .^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **Residual Policy Gradient** . **π**t**o**t**a**l(**s**)**=**π**o**ff**l**in**e\*\***(**s**)**+**π**res**i**d**u**a**l(**s**)**. Constrain **∥**π**res**i**d**u**a**l\*\***∥**≤**ϵ. Derive the gradient update for **π**res**i**d**u**a**l** such that it only corrects the base policy where the value advantage is positive and significant.

**Implementation Plan** : Implement a residual agent in `d3rlpy`. Train the base on `HalfCheetah-Medium`. Train the residual on `HalfCheetah-Expert` (simulating online improvement).

**Evaluation Strategy** : **Stability vs. Plasticity** : Does the agent retain the base competence while improving peak performance?

---

## Module IV: Model-Based Reinforcement Learning & World Models

This module focuses on agents that learn an internal model of the world to plan, "dream," and generalize. This includes the prominent Dreamer architecture, MuZero, and emerging Video World Models.

### Assignment 26: DreamerV3 and Symlog Predictions

**Context & Motivation** : DreamerV3 achieves state-of-the-art performance across diverse domains by learning a world model in latent space. A key innovation is the use of "symlog" transformations to handle rewards of varying magnitudes without tuning.^^ \*\* \*\*

**Core Readings** :

1. _Mastering Diverse Domains through World Models (DreamerV3)_ (ICLR 2024).^^ \*\* \*\*
2. _Natural Dreamer_ .^^ \*\* \*\*

**Mathematical Synthesis** : Analyze the **Symlog Bellman Operator** . **sy**m**l**o**g**(**x**)**=**sign**(**x**)**ln**(**∣**x**∣**+**1**)**.

**sy**m**l**o**g**(**V**(**s**))**≈**E**[**sy**m**l**o**g**(**r**+**γ**⋅**symexp**(**V**(**s**′**)))]

The candidate must prove that this contraction mapping is stable and reduces the "popcorn instability" problem in deep RL where gradients explode due to large reward scales.

**Implementation Plan** : Using the `dreamerv3` repo ^^, train on `Crafter` (a Minecraft-like 2D survival game). Ablation: Remove the symlog transform and observe training instability. \*\* \*\*

**Evaluation Strategy** : **Multi-Domain Robustness** : Compare scores on `Crafter` (sparse, small rewards) vs. `Atari` (dense, large rewards) using a _single_ set of hyperparameters.

### Assignment 27: Harmonizing World Model Objectives

**Context & Motivation** : World models optimize multiple losses: image reconstruction, reward prediction, and dynamics consistency. Usually, image reconstruction dominates. "HarmonyDream" introduces a dynamic weighting scheme to balance these.^^ \*\* \*\*

**Core Readings** :

1. _HarmonyDream: Task Harmonization Inside World Models_ (ICML 2024).^^ \*\* \*\*
2. _DreamerPro_ .^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **Gradient Harmonization** . Define a dynamic weight **λ**k for task **k** (e.g., reconstruction) based on the _velocity_ of its loss reduction.

**λ**k(**t**)**∝**Loss**k\*\***(**t**)**Loss**k(**t**−**1**)\*\*

This ensures that tasks that are learning slowly get up-weighted (similar to GradNorm).

**Implementation Plan** : Modify `HarmonyDream` ^^ to include this dynamic weighting. Test on `DeepMind Control Suite` from pixels. \*\* \*\*

**Evaluation Strategy** : **Sample Efficiency** : Compare HarmonyDream vs. standard DreamerV2 on 100k step benchmarks.

### Assignment 28: UniZero: Transformers in MuZero

**Context & Motivation** : MuZero uses a recurrent dynamics model. UniZero replaces this with a Transformer, allowing the agent to attend to the entire history of past states during the tree search.^^ \*\* \*\*

**Core Readings** :

1. _UniZero: Generalized MuZero with Transformers_ (ArXiv 2024).^^ \*\* \*\*
2. _MuZero Extensions_ .^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **Latent Attention Search** . In MCTS, the value of a node is usually backed up along the tree path. In UniZero, the value **V**(**s**t) is computed by attending to the sequence of ancestors **s**0,**…**,**s**t**−**1\*\*\*\*.

**V**(**s**t)**=**Attention**(**Q**q**u**ery\*\***=**h**(**s**t)**,**K**=**H**hi**s**t**ory\***\*,**V**=**V**v**a**l**u**es\*\***)\*\*

Derive how this "non-Markovian" value estimation helps in partially observable environments.

**Implementation Plan** : Adapt `LightZero` or the methodology in.^^ Replace the LSTM dynamics in MuZero with a causal Transformer. \*\* \*\*

**Evaluation Strategy** : **Memory Horizon** : Test on `Solaris` (Atari), a game requiring long-term memory. Compare UniZero vs. MuZero.

### Assignment 29: Latent Planning with JEPA (PLDM)

**Context & Motivation** : Instead of reconstruction (like Dreamer), Joint Embedding Predictive Architectures (JEPA) learn a latent space by predicting future embeddings directly. PLDM uses this for planning.^^ \*\* \*\*

**Core Readings** :

1. _Planning with a Latent Dynamics Model (PLDM)_ (NeurIPS 2024).^^ \*\* \*\*
2. _Latent Action Spaces_ .^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **Latent Energy Minimization** . Define the planning cost not as reward maximization, but as minimizing the distance to a goal embedding **z**g.

**a**0**:**H\***\*min\*\***∥**Pred**(**s**0,**a**0**:**H\***\*)**−**z**g\***\*∥**2

Derive the condition under which this latent Euclidean distance corresponds to the geodesic distance in the true state space (using contrastive learning bounds).

**Implementation Plan** : Using the `latent-planning` repo ^^, train a JEPA model on offline maze data. Plan using Gradient-Based Planning (shooting method) in the latent space. \*\* \*\*

**Evaluation Strategy** : **Zero-Shot Goal Reaching** : Train on Maze A, specify a goal in Maze B (new layout). Check if the planner finds a path.

### Assignment 30: SafeDreamer: Lagrangian World Models

**Context & Motivation** : Safety constraints must be satisfied _during imagination_ . SafeDreamer integrates Lagrangian multipliers into the world model's rollout loop.^^ \*\* \*\*

**Core Readings** :

1. _SafeDreamer: Safe Reinforcement Learning with World Models_ (ICLR 2024).^^ \*\* \*\*
2. _Verified Safe RL_ .^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **Lagrangian-Dreaming** . The value function **V**(**z**) is augmented:

**V**λ(**z**)**=**r**(**z**)**−**λ**⋅**c**(**z**)**+**γ**V**λ(**z**′**)**

The multiplier **λ** is updated via dual ascent _inside the imagination loop_ . Derive the convergence of this inner-loop optimization.

**Implementation Plan** : Using `SafeDreamer` ^^, train on `SafetyPointGoal1-v0` (Vision). \*\* \*\*

**Evaluation Strategy** : **Zero-Cost Violation** : Achieve near-zero constraint violations while maximizing reward.

### Assignment 31: Causal World Models

**Context & Motivation** : Standard world models learn correlations. Causal world models learn the effect of interventions (**d**o**(**a**)**), which is crucial for robustness against spurious correlations.^^ \*\* \*\*

**Core Readings** :

1. _Causal Reinforcement Learning Survey_ .^^ \*\* \*\*
2. _ACE: Off-Policy Actor-Critic with Causality-Aware Entropy_ (ICML 2024).^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **Intervention-Invariant Dynamics** . The model should minimize prediction error across different "interventional distributions" (environments with different spurious features).

**θ**mine**∈**E**∑∥**s**e**′−**f**θ(**s**e,**a**e)**∥**2

where **f**θ is constrained to use only causal parents of **s**′.

**Implementation Plan** : Implement a "Causal Curiosity" module using `ACE` ^^ concepts. Reward the agent for taking actions that reduce the uncertainty of the causal graph structure. \*\* \*\*

**Evaluation Strategy** : **Confounder Robustness** : Train on a "Colored Key" task where color is spuriously correlated with the key. Test on a version where the correlation is inverted.

### Assignment 32: Generative Video World Models (Genie)

**Context & Motivation** : Genie learns a world model from unlabeled internet videos by inferring latent actions. This allows training agents in a simulated "video game" created from real footage.^^ \*\* \*\*

**Core Readings** :

1. _Genie: Generative Interactive Environments_ (ICML 2024).^^ \*\* \*\*
2. _Latent Diffusion Planning_ .^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **Video-to-Action Quantization** . Formulate a VQ-VAE that compresses the transition **x**t→**x**t**+**1\***\* into a discrete token **z\*\*a (latent action).

**L**=**∥**x**t**+**1\*\***−**D**(**E**(**x**t)**,**z**a\*\***)**∥**2**+**∥**sg**[**z**a]**−**e**∥**2

The synthesis involves mapping these unsupervised tokens to real actions via a small labeled dataset.

**Implementation Plan** : Train a Video World Model on `CoinRun` gameplay. Use it as a simulator. (Requires high compute; use smaller scale `Minigrid` if limited).

**Evaluation Strategy** : **Sim-to-Real** : Train a policy inside the video model. Transfer it to the real game environment.

### Assignment 33: Goal-Space Planning (GSP)

**Context & Motivation** : Planning in raw action space is hard for long horizons. GSP plans sequences of subgoals.^^ \*\* \*\*

**Core Readings** :

1. _Goal-Space Planning_ (ICAPS/RLC 2024).^^ \*\* \*\*
2. _Pre-training Goal-based Models_ .^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **Subgoal-Conditioned Value Propagation** . Learn a reachability network **R**(**g**i,**g**i**+**1\*\*\*\*). Update values via:

**V**(**g**i)**=**g**′**max\*\*\*\*

This acts as a "highway" for value propagation.

**Implementation Plan** : Implement GSP on top of `PTGM`.^^ Test on `AntMaze-UltraLarge`. \*\* \*\*

**Evaluation Strategy** : **Long-Horizon Success Rate** : Completion rate on tasks requiring >1000 steps.

### Assignment 34: Continuous-Time Model-Based RL

**Context & Motivation** : Many physical systems are continuous. Modeling them as discrete MDPs introduces discretization error. Neural ODEs allow learning continuous dynamics.^^ \*\* \*\*

**Core Readings** :

1. _Physics-Informed Continuous-Time RL_ (JMLR 2024).^^ \*\* \*\*
2. _Neural ODEs in RL_ .^^ \*\* \*\*

**Mathematical Synthesis** : Synthesize **HJB-Regularized Neural ODEs** . Learn **V**(**x**) by minimizing the Hamilton-Jacobi-Bellman residual:

**u**min=**0**

Use the adjoint method for planning gradients.

**Implementation Plan** : Use `torchdiffeq` to model `Pendulum-v1` dynamics. Train the controller via HJB minimization.

**Evaluation Strategy** : **Control Smoothness** : Measure the "jerk" (derivative of acceleration). Continuous-time policies should be smoother.

---

## Module V: Multi-Agent & Multi-Objective Optimization

### Assignment 35: Decentralized Consensus with Variance Reduction

**Context** : Cooperative MARL requires agents to agree on gradients. ^^ combines consensus algorithms with variance reduction. **Math** : **y**i**,**t\***\*=**∇**J**i\***\*(**θ**t)**+**∑**w**ij(**y**j**,**t**−**1\*\***−**∇**J**j\*\***(**θ**t**−**1\***\*)). **Repo** : `marlbenchmark`. **Eval** : Comm. rounds to convergence in `SMACv2`. ** \*\*

### Assignment 36: Multi-Agent MuZero (MAZero)

**Context** : Applying MuZero to MARL.^^ **Math** : Factorized Policy Prior for MCTS: **P**(**a**1…**a**n)**≈**∏**P**(**a**i). **Repo** : `MAZero`. **Eval** : Win rate in `Google Research Football`. \*\* \*\*

### Assignment 37: Multi-Objective RL Scalarization

**Context** : Optimizing vector rewards.^^ **Math** : Non-linear scalarization Jacobian **∇**θ\***\*f**(**J![]()). **Repo** : `mo-gymnasium`. **Eval** : Hypervolume of Pareto Front. ** \*\*

### Assignment 38: MORL from AI Feedback (MORLAIF)

**Context** : Using AI to score multiple objectives (Helpful vs Harmless).^^ **Math** : Decomposed Reward **R**=**w**1R**h**e**lp\*\***+**w**2R**ha**r**m\*\***. **Repo** : `verl`. **Eval** : Pareto curve on `HH-RLHF`. \*\* \*\*

### Assignment 39: Risk-Sensitive CVaR RL

**Context** : Optimizing tail risk.^^ **Math** : Quantile Bellman Update for CVaR. **Repo** : `RiskSensitiveSAC...`.^^ **Eval** : Avg return of worst 5% episodes. \*\* \*\*

### Assignment 40: Byzantine-Resilient MARL

**Context** : MARL with faulty agents.^^ **Math** : Trust-weighted aggregation **T**ij. **Repo** : `Byzantine-Federated-RL`. **Eval** : Resilience to 1 adversarial agent. \*\* \*\*

### Assignment 41: Co-Evolutionary MARL (CORY)

**Context** : Role-switching (Pioneer/Observer) to prevent collapse.^^ **Math** : Dual-objective **J**P(**R**)**+**J**O\*\***(**BC**)**. **Repo** : `CORY`. **Eval** : Diversity of strategies. ** \*\*

### Assignment 42: Diversity via Determinantal Point Processes

**Context** : Encouraging diverse team behavior.^^ **Math** : Intrinsic reward **r**=**det**(**KernelMatrix**). **Repo** : Custom MARL. **Eval** : State space coverage. \*\* \*\*

---

## Module VI: RL for Large Language Models & Generative AI

### Assignment 43: Margin-based DPO (SimPO)

**Context** : Improving DPO with margins.^^ **Math** : **L**S**im**PO=**−**log**σ**(**β**(**lo**g**π**lπ**w**−**γ**)). **Repo** : `SimPO`. **Eval** : `AlpacaEval 2`. \*\* \*\*

### Assignment 44: Stepwise Constrained Alignment (SACPO)

**Context** : Safety constraints at every token step.^^ **Math** : Lagrangian DPO with dynamic **β**. **Repo** : `sacpo`. **Eval** : Harmlessness rate. \*\* \*\*

### Assignment 45: Group Relative Policy Optimization (GRPO)

**Context** : DeepSeek-R1 style RL.^^ **Math** : Advantage **A**=**(**r**−**μ**g**ro**u**p)**/**σ**g**ro**u**p. **Repo** : `EasyR1`. **Eval** : `GSM8K` Pass@1. \*\* \*\*

### Assignment 46: Curiosity-Driven RLHF

**Context** : Improving diversity in RLHF.^^ **Math** : **R**=**R**h**u**man+**α**⋅**Perplexity**. **Repo** : `CD-RLHF`. **Eval** : Self-BLEU (Diversity). \*\* \*\*

### Assignment 47: Weak-to-Strong Generalization

**Context** : Supervising super-human models.^^ **Math** : Confidence-weighted loss. **Repo** : `weak-to-strong`. **Eval** : W2S generalization gap. \*\* \*\*

### Assignment 48: Generative World Models for Robotics

**Context** : Next-token prediction for robot physics.^^ **Math** : VLM as World Model **P**(**v**i**s**u**a**l**t**+**1\*\***∣**v**i**s**u**a**l**t\*\***,**a**t). **Repo** : `RLVR-World`. **Eval** : Simulation MSE. \*\* \*\*

### Assignment 49: Large Language Diffusion Models

**Context** : Non-autoregressive text gen.^^ **Math** : Diffusion loss with RL fine-tuning. **Repo** : `LLaDA`. **Eval** : Generation speed. \*\* \*\*

### Assignment 50: Chain-of-Thought RL

**Context** : Incentivizing reasoning traces.^^ **Math** : Value of CoT **V**(**s**re**a**so**nin**g). **Repo** : `EasyR1`. **Eval** : Correlation(Reasoning Length, Accuracy). \*\* \*\*

---

## Conclusion

This curriculum represents the bleeding edge of Reinforcement Learning research as of late 2024 and early 2025. By traversing these six modules, the researcher moves from the foundational mathematics of distributional and safe RL, through the sample-efficiency revolutions of offline and model-based methods, to the transformative integration of RL with Large Language Models. Each assignment is designed to be a publishable unit of work, providing a comprehensive toolkit for the modern AI scientist.

[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netEmbedding Safety into RL: A New Take on Trust Region Methods - OpenReview**Opens in a new window**](https://openreview.net/forum?id=wQkERVYqui)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comshiralab/adaptive-trpo: [ICONIP 2024] Official implementation of the paper &#34;Adaptive Trust Region Radius for Robust Policy Optimization&#34; - GitHub**Opens in a new window**](https://github.com/shiralab/adaptive-trpo)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgEmbedding Safety into RL: A New Take on Trust Region Methods - arXiv**Opens in a new window**](https://arxiv.org/html/2411.02957v1)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comjlwu002/VSRL: [NeurIPS 2024] Verified Safe Reinforcement Learning for Neural Network Dynamic Models - GitHub**Opens in a new window**](https://github.com/jlwu002/VSRL)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comdatake/SinkhornDistRL: Implementation of &#39;Distributional Reinforcement Learning with Regularized Wasserstein Loss&#39; (NeurIPS 2024) - GitHub**Opens in a new window**](https://github.com/datake/SinkhornDistRL)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comRobustFieldAutonomyLab/Distributional_RL_Decision_and_Control: [RA-L 2025] Distributional Reinforcement Learning Based Integrated Decision Making and Control for Autonomous Surface Vehicles - GitHub**Opens in a new window**](https://github.com/RobustFieldAutonomyLab/Distributional_RL_Decision_and_Control)[![](https://t0.gstatic.com/faviconV2?url=https://jmlr.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)jmlr.orgDecentralized Natural Policy Gradient with Variance Reduction for Collaborative Multi-Agent Reinforcement Learning**Opens in a new window**](https://jmlr.org/papers/volume25/22-1036/22-1036.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comamazon-science/mezo_svrg: Code the ICML 2024 paper: &#34;Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models&#34; - GitHub**Opens in a new window**](https://github.com/amazon-science/mezo_svrg)[![](https://t3.gstatic.com/faviconV2?url=https://rlj.cs.umass.edu/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)rlj.cs.umass.eduPolicy Gradient with Active Importance Sampling - Reinforcement Learning Journal (RLJ)**Opens in a new window**](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_90.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comSafeRL-Lab/Robust-RL-Baselines: Robust Reinforcement Learning Benchmark - GitHub**Opens in a new window**](https://github.com/SafeRL-Lab/Robust-RL-Baselines)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2509.22964] Functional Critic Modeling for Provably Convergent Off-Policy Actor-Critic - arXiv**Opens in a new window**](https://arxiv.org/abs/2509.22964)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgFunctional Critic Modeling for Provably Convergent Off-Policy Actor-Critic - arXiv**Opens in a new window**](https://arxiv.org/html/2509.22964v1)[![](https://t0.gstatic.com/faviconV2?url=https://nips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)nips.ccNeurIPS 2024 Papers**Opens in a new window**](https://nips.cc/virtual/2024/papers.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comBY571/Soft-Actor-Critic-and-Extensions - GitHub**Opens in a new window**](https://github.com/BY571/Soft-Actor-Critic-and-Extensions)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netParameter Space Noise for Exploration - OpenReview**Opens in a new window**](https://openreview.net/forum?id=ByBAl2eAZ)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgLearning Agents With Prioritization and Parameter Noise in Continuous State and Action Space - arXiv**Opens in a new window**](https://arxiv.org/html/2410.11250v1)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[1706.01905] Parameter Space Noise for Exploration - arXiv**Opens in a new window**](https://arxiv.org/abs/1706.01905)[![](https://t3.gstatic.com/faviconV2?url=https://ieeexplore.ieee.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)ieeexplore.ieee.orgParameter Space Exploration of Neural Network Inference Using Ferroelectric Tunnel Junctions for Processing-In-Memory - IEEE Xplore**Opens in a new window**](https://ieeexplore.ieee.org/document/10741680/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comdqn · GitHub Topics**Opens in a new window**](https://github.com/topics/dqn)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2409.01427] Enhancing Sample Efficiency and Exploration in Reinforcement Learning through the Integration of Diffusion Models and Proximal Policy Optimization - arXiv**Opens in a new window**](https://arxiv.org/abs/2409.01427)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccDiffusion Policies Creating a Trust Region for Offline Reinforcement Learning - NIPS papers**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/59a48c111f97f2174709ea9ed8e920d1-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comTianyuCodings/Diffusion_Trusted_Q_Learning: [NeuIPS2024 DTQL] Diffusion Trusted Q-Learning for Offline RL — Official PyTorch Implementation - GitHub**Opens in a new window**](https://github.com/TianyuCodings/Diffusion_Trusted_Q_Learning)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comcoldsummerday/SD-SAC: Revisiting Discrete Soft Actor-Critic Accepted by Transactions on Machine Learning Research (TMLR) - GitHub**Opens in a new window**](https://github.com/coldsummerday/SD-SAC)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netRevisiting Actor-Critic Methods in Discrete Action Off-Policy Reinforcement Learning**Opens in a new window**](https://openreview.net/forum?id=LOuMDqWxNN)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comRisk-Sensitive Soft Actor-Critic for Robust Deep Reinforcement Learning under Distribution Shifts - GitHub**Opens in a new window**](https://github.com/tumBAIS/RiskSensitiveSACforRobustDRLunderDistShifts)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comClementineY/Robust-RL-Papers: Must-read Papers on Robust Reinforcement Learning**Opens in a new window**](https://github.com/ClementineY/Robust-RL-Papers)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2508.18420] LLM-Driven Intrinsic Motivation for Sparse Reward Reinforcement Learning**Opens in a new window**](https://arxiv.org/abs/2508.18420)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgCDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models - arXiv**Opens in a new window**](https://arxiv.org/pdf/2509.09675?)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comvolcengine/verl: verl: Volcano Engine Reinforcement Learning for LLMs - GitHub**Opens in a new window**](https://github.com/volcengine/verl)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comA curated list of awesome exploration RL resources (continually updated) - GitHub**Opens in a new window**](https://github.com/opendilab/awesome-exploration-rl)[![](https://t1.gstatic.com/faviconV2?url=https://srinathm1359.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)srinathm1359.github.ioRandom Latent Exploration for Deep Reinforcement Learning - Srinath Mahankali**Opens in a new window**](https://srinathm1359.github.io/random-latent-exploration/)[![](https://t1.gstatic.com/faviconV2?url=https://proceedings.mlr.press/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.mlr.pressExpansive Latent Planning for Sparse Reward Offline Reinforcement Learning**Opens in a new window**](https://proceedings.mlr.press/v229/gieselmann23a.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comPKU-RL/PTGM: [ICLR 2024 oral] Pre-Training Goal-based Models for Sample-Efficient Reinforcement Learning - GitHub**Opens in a new window**](https://github.com/PKU-RL/PTGM)[![](https://t0.gstatic.com/faviconV2?url=https://iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)iclr.ccPre-Training Goal-based Models for Sample-Efficient Reinforcement Learning - ICLR 2026**Opens in a new window**](https://iclr.cc/virtual/2024/oral/19728)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comenjeeneer/zero-shot-rl: VC-FB and MC-FB algorithms from &#34;Zero-Shot Reinforcement Learning from Low Quality Data&#34; (NeurIPS 2024) - GitHub**Opens in a new window**](https://github.com/enjeeneer/zero-shot-rl)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2403.00564] EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data**Opens in a new window**](https://arxiv.org/abs/2403.00564)[![](https://t3.gstatic.com/faviconV2?url=https://ieeexplore.ieee.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)ieeexplore.ieee.orgNumerical Evidence for Sample Efficiency of Model-Based Over Model-Free Reinforcement Learning Control of Partial Differential Equations - IEEE Xplore**Opens in a new window**](https://ieeexplore.ieee.org/document/10590945/)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netA Forget-and-Grow Strategy for Deep Reinforcement Learning Scaling in Continuous Control | OpenReview**Opens in a new window**](https://openreview.net/forum?id=VhmTXbsdtx)[![](https://t0.gstatic.com/faviconV2?url=https://www.researchgate.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)researchgate.netA Forget-and-Grow Strategy for Deep Reinforcement Learning Scaling in Continuous Control | Request PDF - ResearchGate**Opens in a new window**](https://www.researchgate.net/publication/393379233_A_Forget-and-Grow_Strategy_for_Deep_Reinforcement_Learning_Scaling_in_Continuous_Control)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comdatake/Papers-Of-Continual-RL - GitHub**Opens in a new window**](https://github.com/datake/Papers-Of-Continual-RL)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2411.05273] Real-World Offline Reinforcement Learning from Vision Language Model Feedback - arXiv**Opens in a new window**](https://arxiv.org/abs/2411.05273)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2406.07295] Multi-objective Reinforcement learning from AI Feedback - arXiv**Opens in a new window**](https://arxiv.org/abs/2406.07295)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comyhisaki/average-reward-drl: [ICML 2024] Author&#39;s Implementation of RVI-SAC - GitHub**Opens in a new window**](https://github.com/yhisaki/average-reward-drl)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comlinhlpv/awesome-offline-to-online-RL-papers - GitHub**Opens in a new window**](https://github.com/linhlpv/awesome-offline-to-online-RL-papers)[![](https://t0.gstatic.com/faviconV2?url=https://proceedings.iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.iclr.ccTHE GENERALIZATION GAP IN OFFLINE REINFORCEMENT LEARNING - ICLR Proceedings**Opens in a new window**](https://proceedings.iclr.cc/paper_files/paper/2024/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comYangRui2015/RIQL: [ICLR 2024 Spotlight] Code for ICLR 2024 paper &#34;Towards Robust Offline Reinforcement Learning under Diverse Data Corruption&#34; - GitHub**Opens in a new window**](https://github.com/YangRui2015/RIQL)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comZhengYinan-AIR/FISOR: [ICLR 2024] The official implementation of &#34;Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model&#34; - GitHub**Opens in a new window**](https://github.com/ZhengYinan-AIR/FISOR)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2404.10393] Offline Trajectory Optimization for Offline Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/abs/2404.10393)[![](https://t3.gstatic.com/faviconV2?url=https://www.worldscientific.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)worldscientific.comAI-Driven Real-Time UAV Autonomous Trajectory Optimization Using Deep Reinforcement Learning in Dynamic and Partially Observable Environments | Vietnam Journal of Computer Science - World Scientific Publishing**Opens in a new window**](https://www.worldscientific.com/doi/10.1142/S219688882550023X)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comschmidtdominik/LAPO: Code for the ICLR 2024 spotlight paper: &#34;Learning to Act without Actions&#34; (introducing Latent Action Policies) - GitHub**Opens in a new window**](https://github.com/schmidtdominik/LAPO)[![](https://t0.gstatic.com/faviconV2?url=https://proceedings.iclr.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.iclr.ccEFFICIENT PLANNING WITH LATENT DIFFUSION - ICLR Proceedings**Opens in a new window**](https://proceedings.iclr.cc/paper_files/paper/2024/file/b2ac1112e14fac8d07275a7f482e0c11-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comczp16/FCSRL: Feasibility Consistent Representation Learning for Safe Reinforcement Learning (ICML 2024). Current SOTA model-free safe RL algorithm on safety-gymnasium - GitHub**Opens in a new window**](https://github.com/czp16/FCSRL)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comReleases · danijar/dreamerv3 - GitHub**Opens in a new window**](https://github.com/danijar/dreamerv3/releases)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comdanijar/dreamerv3: Mastering Diverse Domains through World Models - GitHub**Opens in a new window**](https://github.com/danijar/dreamerv3)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comInexperiencedMe/NaturalDreamer: Simplest and Cleanest DreamerV3 implementation out there - GitHub**Opens in a new window**](https://github.com/InexperiencedMe/NaturalDreamer)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comHarmonyDream: Task Harmonization Inside World Models (ICML 2024) - GitHub**Opens in a new window**](https://github.com/thuml/HarmonyDream)[![](https://t3.gstatic.com/faviconV2?url=https://ise.thss.tsinghua.edu.cn/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)ise.thss.tsinghua.edu.cnHarmonyDream: Task Harmonization Inside World Models**Opens in a new window**](https://ise.thss.tsinghua.edu.cn/~mlong/doc/HarmonyDream-icml24.pdf)[![](https://t3.gstatic.com/faviconV2?url=https://proceedings.neurips.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.neurips.ccPolicy-shaped prediction: avoiding distractions in model-based reinforcement learning - NIPS papers**Opens in a new window**](https://proceedings.neurips.cc/paper_files/paper/2024/file/17af43527227c5c96db0f8d4c6aadc4e-Paper-Conference.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgUniZero: Generalized and Efficient Planning with Scalable Latent World Models - arXiv**Opens in a new window**](https://arxiv.org/html/2406.10667v2)[![](https://t0.gstatic.com/faviconV2?url=https://medium.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)medium.comPaper Review 2024 - MuZero: Mastering Go, chess, shogi and Atari without rules (01/50)**Opens in a new window**](https://medium.com/@phchen715/paper-review-2024-muzero-mastering-go-chess-shogi-and-atari-without-rules-01-50-2720b42b692e)[![](https://t3.gstatic.com/faviconV2?url=https://latent-planning.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)latent-planning.github.ioLearning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models**Opens in a new window**](https://latent-planning.github.io/)[![](https://t3.gstatic.com/faviconV2?url=https://latent-planning.github.io/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)latent-planning.github.ioarXiv:2502.14819v1 [cs.LG] 20 Feb 2025 - Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models**Opens in a new window**](https://latent-planning.github.io/static/paper.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comICLR 2024: SafeDreamer: Safe Reinforcement Learning with World Models - GitHub**Opens in a new window**](https://github.com/PKU-Alignment/SafeDreamer)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comlibo-huang/Awesome-Causal-Reinforcement-Learning - GitHub**Opens in a new window**](https://github.com/libo-huang/Awesome-Causal-Reinforcement-Learning)[![](https://t3.gstatic.com/faviconV2?url=https://icml.cc/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)icml.ccICML 2024 Orals**Opens in a new window**](https://icml.cc/virtual/2024/events/oral)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgLatent Diffusion Planning for Imitation Learning - arXiv**Opens in a new window**](https://arxiv.org/html/2504.16925v1)[![](https://t2.gstatic.com/faviconV2?url=https://icaps24.icaps-conference.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)icaps24.icaps-conference.orgA New View on Planning in Online Reinforcement Learning - ICAPS 2024**Opens in a new window**](https://icaps24.icaps-conference.org/program/workshops/prl-papers/7.pdf)[![](https://t0.gstatic.com/faviconV2?url=https://www.jmlr.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)jmlr.orgA New, Physics-Informed Continuous-Time Reinforcement Learning Algorithm with Performance Guarantees**Opens in a new window**](https://www.jmlr.org/papers/v25/24-0017.html)[![](https://t2.gstatic.com/faviconV2?url=https://openreview.net/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)openreview.netEfficient Multi-agent Reinforcement Learning by Planning - OpenReview**Opens in a new window**](https://openreview.net/forum?id=CpnKq3UJwp)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.org[2410.11221] Multi-objective Reinforcement Learning: A Tool for Pluralistic Alignment - arXiv**Opens in a new window**](https://arxiv.org/abs/2410.11221)[![](https://t1.gstatic.com/faviconV2?url=https://proceedings.mlr.press/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)proceedings.mlr.pressRisk-Sensitive Reward-Free Reinforcement Learning with CVaR**Opens in a new window**](https://proceedings.mlr.press/v235/ni24c.html)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comsample-efficient-rl · GitHub Topics**Opens in a new window**](https://github.com/topics/sample-efficient-rl)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comHarry67Hu/CORY: Official implementation of the NeurIPS 2024 paper CORY - GitHub**Opens in a new window**](https://github.com/Harry67Hu/CORY)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comLantaoYu/MARL-Papers: Paper list of multi-agent reinforcement learning (MARL) - GitHub**Opens in a new window**](https://github.com/LantaoYu/MARL-Papers)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.com[NeurIPS 2024] SimPO: Simple Preference Optimization with a Reference-Free Reward - GitHub**Opens in a new window**](https://github.com/princeton-nlp/SimPO)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.com[NeurIPS 2024] SACPO (Stepwise Alignment for Constrained Policy Optimization) - GitHub**Opens in a new window**](https://github.com/line/sacpo)[![](https://t1.gstatic.com/faviconV2?url=https://arxiv.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)arxiv.orgRLVR-World: Training World Models with Reinforcement Learning - arXiv**Opens in a new window**](https://arxiv.org/html/2505.13934v2)[![](https://t2.gstatic.com/faviconV2?url=https://aclanthology.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)aclanthology.orgCuriosity-Driven Reinforcement Learning from Human Feedback - ACL Anthology**Opens in a new window**](https://aclanthology.org/2025.acl-long.1146.pdf)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comRLVR-World: Training World Models with Reinforcement Learning (NeurIPS 2025) - GitHub**Opens in a new window**](https://github.com/thuml/RLVR-World)[![](https://t3.gstatic.com/faviconV2?url=https://www.paperdigest.org/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)paperdigest.orgNeurIPS 2025 Papers with Code &amp; Data**Opens in a new window**](https://www.paperdigest.org/2025/11/neurips-2025-papers-with-code-data/)[![](https://t1.gstatic.com/faviconV2?url=https://github.com/&client=BARD&type=FAVICON&size=256&fallback_opts=TYPE,SIZE,URL)github.comoperator22th/awesome-rl-algorithms - GitHub](https://github.com/operator22th/awesome-rl-algorithms)
