# Advanced Research Curriculum: Assignments 1-50

## Assignment 1: Bellman Bootstrapping Meets Replay-Stabilized DQN
**Selected Papers:**  
- Dopamine Rainbow DQN (Bellemare et al., JMLR 2022) — GitHub: https://github.com/google/dopamine  
- TD-MPC2: Scalable Model-Predictive Control (Hansen et al., ICLR 2024) — GitHub: https://github.com/nicklashansen/tdmpc2  

**Novel Synthesis:** Inject TD-MPC2’s target-network soft updates and consistency regularizer into Dopamine’s Rainbow DQN to stabilize Bellman bootstrapping on pixel control.

**Implementation & Improvement Plan:**  
- Refactor Dopamine’s `rainbow_agent.py` to expose target update rate; import TD-MPC2 EMA utilities for target network smoothing.  
- Add consistency loss from TD-MPC2 to Rainbow’s loss stack; gate via config flag.  
- Container: build on `pytorch/pytorch:2.2.1-cuda12.1-cudnn8` with Poetry/conda env; include headless ALE deps.

**Evaluation Strategy:**  
- Benchmark: Atari 100k (ALE).  
- Ablation: disable consistency loss; vary EMA tau.

## Assignment 2: Policy Evaluation via Batched Dynamic Programming
**Selected Papers:**  
- rlax / JAX Bellman ops (DeepMind, 2022) — GitHub: https://github.com/deepmind/rlax  
- CleanRL PPO-JAX (Huang et al., 2023) — GitHub: https://github.com/vwxyzjn/cleanrl  

**Novel Synthesis:** Use rlax vectorized Bellman evaluators inside CleanRL’s PPO rollout buffer to compute exact policy evaluation targets for small MDPs, contrasting with GAE.

**Implementation & Improvement Plan:**  
- Add rlax Bellman evaluation module; expose switch in PPO config to choose GAE vs exact DP.  
- Ensure JIT-compiled batched policy evaluation over enumerated states.  
- Container: `nvcr.io/nvidia/jax:23.05-py3` with CUDA if available; fallback CPU.

**Evaluation Strategy:**  
- Benchmark: FrozenLake-v1 (small tabular), Gridworld-8x8.  
- Ablation: GAE vs exact DP returns.

## Assignment 3: Policy Iteration with Soft Greedy Improvements
**Selected Papers:**  
- Soft Actor-Critic (Haarnoja et al., revised JMLR 2022) — GitHub: https://github.com/denisyarats/pytorch_sac  
- ReBRAC (Brandfonbrener et al., ICLR 2023) — GitHub: https://github.com/Farama-Foundation/ReBRAC  

**Novel Synthesis:** Interpret SAC as soft policy iteration; plug ReBRAC’s conservative regularizer into SAC’s policy improvement step for stabler soft-greedy updates.

**Implementation & Improvement Plan:**  
- In `pytorch_sac/agent/sac.py`, wrap policy loss with ReBRAC’s log-sum-exp regularizer; add config for behavior-cloning weight.  
- Share replay/data loader from ReBRAC when offline; keep SAC alpha auto-tune.  
- Container: `pytorch/pytorch:2.1.2-cuda12.1` with MuJoCo 2.3.7, mujoco-py.

**Evaluation Strategy:**  
- Benchmark: D4RL HalfCheetah/Walker2d medium-expert.  
- Ablation: remove conservative term; compare online vs offline.

## Assignment 4: Value Iteration with Quantile Targets
**Selected Papers:**  
- Quantile Regression DQN (Dabney et al., AAAI 2018; maintained 2023 in Dopamine) — GitHub: https://github.com/google/dopamine  
- XQL: X-Entropy RL (Ma et al., ICLR 2023) — GitHub: https://github.com/coh250/xql  

**Novel Synthesis:** Perform value iteration using quantile backups but enforce XQL’s reverse KL objective to regularize policy toward behavior.

**Implementation & Improvement Plan:**  
- Extend Dopamine QR-DQN agent to output action logits; add XQL reverse-KL term against behavior policy inferred from replay.  
- Quantile targets unchanged; policy update uses KL-weighted advantage over quantile means.  
- Container: `python:3.10-slim` + JAX/TF for Dopamine; or PyTorch port with torchrl.

**Evaluation Strategy:**  
- Benchmark: Atari 26 games (200M frames).  
- Ablation: KL weight sweep; quantile vs mean backups.

## Assignment 5: First-Visit MC with Modern Replay
**Selected Papers:**  
- CleanRL Monte-Carlo PG (Huang et al., 2023) — GitHub: https://github.com/vwxyzjn/cleanrl  
- Sample Factory v2 (Petrenko et al., 2023) — GitHub: https://github.com/alex-petrenko/sample-factory  

**Novel Synthesis:** Add episodic first-visit MC returns as a target head inside Sample Factory’s IMPALA-like learner to reduce variance in sparse-reward VizDoom tasks.

**Implementation & Improvement Plan:**  
- Implement episodic return buffer keyed by (env_id, timestep); expose MC head in model.  
- Blend MC loss with V-trace critic loss via coefficient.  
- Container: `nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04`; install VizDoom deps.

**Evaluation Strategy:**  
- Benchmark: VizDoom MyWayHome, DMLab Sparse Reward.  
- Ablation: V-trace only vs V-trace+MC head.

## Assignment 6: MC Control with ε-Scheduled Advantage Clipping
**Selected Papers:**  
- PPO (Schulman et al., arXiv 2017; maintained in CleanRL 2023) — GitHub: https://github.com/vwxyzjn/cleanrl  
- Epsilon Decay Schedules in Atari (Machado et al., JMLR 2022 via Dopamine) — GitHub: https://github.com/google/dopamine  

**Novel Synthesis:** Drive PPO policy improvement using MC returns and scheduled ε-greedy exploration blended into action sampling to control variance in discrete control.

**Implementation & Improvement Plan:**  
- Add MC return computation in CleanRL PPO for discrete envs; modify action sampler to mix ε-greedy with categorical sampling.  
- Clip advantages with PPO ratio; log effect of ε on entropy.  
- Container: CleanRL Docker (`ghcr.io/vwxyzjn/cleanrl`) extended with Atari ROMs.

**Evaluation Strategy:**  
- Benchmark: Atari Breakout/Pong 10M frames.  
- Ablation: ε=0 (vanilla PPO) vs cosine-decay ε.

## Assignment 7: TD(0) Prediction with Equivariant Function Approximation
**Selected Papers:**  
- TD Learning with Neural Approx (rlax TD ops, 2022) — GitHub: https://github.com/deepmind/rlax  
- Equivariant Networks for RL (Finzi et al., NeurIPS 2020; maintained 2023) — GitHub: https://github.com/QUVA-Lab/e2cnn  

**Novel Synthesis:** Apply group-equivariant CNNs to TD(0) value prediction on symmetric gridworlds to improve sample efficiency.

**Implementation & Improvement Plan:**  
- Replace value network with e2cnn equivariant layers; keep rlax td_learning update.  
- Add group definition (C4 rotations) configurable per env.  
- Container: `pytorch/pytorch:2.2.1-cuda12.1` + `e2cnn`.

**Evaluation Strategy:**  
- Benchmark: MiniGrid (symmetric mazes).  
- Ablation: equivariant vs standard CNN.

## Assignment 8: n-Step Returns inside Distributional Control
**Selected Papers:**  
- Rainbow n-step Distributional DQN (Hessel et al.; Dopamine 2022) — GitHub: https://github.com/google/dopamine  
- TD-MPC2 short-horizon rollouts (Hansen et al., ICLR 2024) — GitHub: https://github.com/nicklashansen/tdmpc2  

**Novel Synthesis:** Use learned short-horizon rollouts from TD-MPC2 to generate n-step targets for Rainbow, replacing raw environment n-steps to reduce variance.

**Implementation & Improvement Plan:**  
- Train TD-MPC2 world model alongside Rainbow; for each transition, roll out model for n steps to compute model-based n-step return; mix with real n-step via trust weight.  
- Add config flag for model-based targets; reuse Rainbow’s prioritized replay.  
- Container: joint image (PyTorch + MuJoCo for TD-MPC2 + ALE for Atari).

**Evaluation Strategy:**  
- Benchmark: Atari (Seaquest, Q*bert) and DMControl (Cartpole Swingup).  
- Ablation: model-based n-step on/off.

## Assignment 9: Eligibility Traces with Transformer Critics
**Selected Papers:**  
- TD(λ) in rlax (2022) — GitHub: https://github.com/deepmind/rlax  
- Decision Transformer (Chen et al., NeurIPS 2021) — GitHub: https://github.com/kzl/decision-transformer  

**Novel Synthesis:** Implement TD(λ) targets for a transformer critic within Decision Transformer, enabling online fine-tuning with bootstrapped returns.

**Implementation & Improvement Plan:**  
- Add TD(λ) return computation module; feed as supervised targets for critic head predicting value-to-go.  
- Keep DT’s causal transformer; switch optimizer to handle online streaming batches.  
- Container: `pytorch/pytorch:2.1-cuda11.8`; enable FlashAttention optional.

**Evaluation Strategy:**  
- Benchmark: Gymnasium HalfCheetah medium-replay (offline-to-online).  
- Ablation: λ=0 (TD0) vs λ=0.95 vs MC.

## Assignment 10: SARSA with Self-Predictive Representations
**Selected Papers:**  
- SARSA reference in Dopamine (2022) — GitHub: https://github.com/google/dopamine  
- SPR (Schwarzer et al., ICLR 2021; maintained 2023) — GitHub: https://github.com/facebookresearch/spr  

**Novel Synthesis:** Combine on-policy SARSA updates with SPR representation loss to stabilize learning from pixels under high exploration.

**Implementation & Improvement Plan:**  
- Add SPR encoder and prediction head to Dopamine SARSA agent; train jointly with TD error.  
- Use momentum encoder and augmentations from SPR; keep on-policy buffer small and prioritized.  
- Container: `pytorch/pytorch:2.2-cuda12.1` + torchvision.

**Evaluation Strategy:**  
- Benchmark: Atari Freeway/UpNDown with ε=0.1.  
- Ablation: SPR loss off vs on.

## Assignment 11: Double Q-Learning with Behavior Cloning Warm-Start
**Selected Papers:**  
- Double DQN (van Hasselt et al.; Dopamine 2022 impl) — GitHub: https://github.com/google/dopamine  
- TD3+BC (Fujimoto & Gu, NeurIPS 2021) — GitHub: https://github.com/sfujim/TD3_BC  

**Novel Synthesis:** Use behavior cloning pretrain (from TD3+BC) to initialize Double DQN on image tasks, reducing overestimation and speeding convergence.

**Implementation & Improvement Plan:**  
- Pretrain policy logits with BC loss from an offline dataset; initialize Double DQN networks with those weights.  
- During online phase, keep BC loss as auxiliary with decay schedule.  
- Container: `pytorch/pytorch:2.0-cuda11.8`; include dataset mount.

**Evaluation Strategy:**  
- Benchmark: Atari (MsPacman), D4RL Adroit (image).  
- Ablation: BC warm-start vs random init.

## Assignment 12: Dueling Networks with CVaR Risk Heads
**Selected Papers:**  
- Dueling DQN (Wang et al.; Dopamine 2022) — GitHub: https://github.com/google/dopamine  
- Iterated CVaR RL (Du et al., ICLR 2023) — GitHub: https://github.com/duyihan/Iterated-CVaR-RL  

**Novel Synthesis:** Add CVaR head to dueling architecture to optimize tail-risk-sensitive Q-values.

**Implementation & Improvement Plan:**  
- Extend dueling head to output quantiles; compute CVaR over tail quantiles; integrate Iterated CVaR loss.  
- Risk level α configurable; add risk-aware policy selection.  
- Container: `pytorch/pytorch:2.1-cuda12`; add CVaR utils.

**Evaluation Strategy:**  
- Benchmark: Financial RL (FinRL stock trading), Atari (Risky Road).  
- Ablation: α = 0.1 vs 0.25 vs mean-value.

## Assignment 13: Prioritized Replay with Contrastive State Alignment
**Selected Papers:**  
- PER (Schaul et al.; Dopamine 2022) — GitHub: https://github.com/google/dopamine  
- CURL (Laskin et al., ICML 2020; maintained 2023) — GitHub: https://github.com/MishaLaskin/curl  

**Novel Synthesis:** Use contrastive similarity scores as priority signals in PER to focus replay on representation-hard transitions.

**Implementation & Improvement Plan:**  
- Compute CURL loss per sample; set PER priority = |TD error| + β * contrastive loss.  
- Add reservoir replay for low-priority items to avoid starvation.  
- Container: `pytorch/pytorch:2.2-cuda12.1`.

**Evaluation Strategy:**  
- Benchmark: DMControl Cheetah/Walker pixel.  
- Ablation: TD-only priority vs TD+contrastive.

## Assignment 14: Distributional RL with Noisy Exploration
**Selected Papers:**  
- C51 (Bellemare et al.; Dopamine 2022) — GitHub: https://github.com/google/dopamine  
- Noisy Networks (Fortunato et al.; baselines maintained 2023) — GitHub: https://github.com/openai/baselines  

**Novel Synthesis:** Integrate NoisyLinear layers into C51 to unify exploration and distributional targets.

**Implementation & Improvement Plan:**  
- Replace linear layers in C51 head with NoisyLinear; anneal σ parameters.  
- Ensure categorical projection stable with noise; tune atom support accordingly.  
- Container: Baselines Docker + dopamine deps.

**Evaluation Strategy:**  
- Benchmark: Atari full suite.  
- Ablation: noisy layers on/off.

## Assignment 15: QR-DQN with Entropy-Regularized Greedy Policies
**Selected Papers:**  
- QR-DQN (Dabney et al.; Dopamine 2022) — GitHub: https://github.com/google/dopamine  
- Maximum Entropy RL (SAC; Haarnoja et al., JMLR 2022) — GitHub: https://github.com/denisyarats/pytorch_sac  

**Novel Synthesis:** Use SAC-style entropy regularization for action selection in QR-DQN to reduce overestimation and encourage exploration.

**Implementation & Improvement Plan:**  
- Compute log-sum-exp over quantile means for policy logits; add entropy bonus to Bellman target.  
- Temperature α auto-tuned via dual optimization.  
- Container: PyTorch + Dopamine.

**Evaluation Strategy:**  
- Benchmark: Atari 57; ablate α fixed vs learned.

## Assignment 16: Rainbow-lite for Low-Resource Devices
**Selected Papers:**  
- Rainbow DQN (Hessel et al.; Dopamine 2022) — GitHub: https://github.com/google/dopamine  
- TinyRL (edge RL profiling; Qiu et al., 2023) — GitHub: https://github.com/mit-han-lab/tinyml/tree/main/tinyrl  

**Novel Synthesis:** Compress Rainbow (prune/quantize) for edge deployment guided by TinyRL profiling.

**Implementation & Improvement Plan:**  
- Apply post-training quantization + structured pruning; retrain with distillation to full Rainbow.  
- Replace PER heap with ring-buffer + top-k sampling to cut memory.  
- Container: ONNX Runtime + PyTorch quantization toolchain.

**Evaluation Strategy:**  
- Benchmark: Atari Pong/Breakout on Jetson Nano.  
- Ablation: quantized vs full precision; pruning ratios.

## Assignment 17: Multi-Step TD with Advantage Clipping in Continuous Control
**Selected Papers:**  
- TD3 (Fujimoto et al.; maintained 2023) — GitHub: https://github.com/sfujim/TD3  
- GAE (Schulman et al., arXiv 2016; CleanRL 2023 impl) — GitHub: https://github.com/vwxyzjn/cleanrl  

**Novel Synthesis:** Add multi-step (k) TD targets with clipped advantages (GAE-style) into TD3 critic updates to stabilize early training.

**Implementation & Improvement Plan:**  
- Modify TD3 replay sampler to return n-step returns; compute GAE on policy rollouts; clip advantages in actor loss.  
- Tune k, λ; ensure target policy smoothing still applied.  
- Container: PyTorch 2.1 + Mujoco.

**Evaluation Strategy:**  
- Benchmark: MuJoCo Hopper/Walker2d.  
- Ablation: k=1 vs 3 vs 5; advantage clipping on/off.

## Assignment 18: TD(λ) with Eligibility Traces in Offline Setting
**Selected Papers:**  
- ReBRAC (ICLR 2023) — GitHub: https://github.com/Farama-Foundation/ReBRAC  
- Eligibility Traces in JAX (rlax 2022) — GitHub: https://github.com/deepmind/rlax  

**Novel Synthesis:** Add offline TD(λ) critic targets into ReBRAC to better balance bias/variance with fixed datasets.

**Implementation & Improvement Plan:**  
- Implement λ-return calculator over replay sequences; integrate into ReBRAC critic loss.  
- Clip λ to avoid divergence; keep conservative Q penalty.  
- Container: JAX/Flax in ReBRAC Docker.

**Evaluation Strategy:**  
- Benchmark: D4RL kitchen/antmaze.  
- Ablation: λ sweep (0, 0.5, 0.95).

## Assignment 19: Actor-Critic with Dueling Critics
**Selected Papers:**  
- A2C (Mnih et al.; PyTorch impl 2023) — GitHub: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail  
- Dueling Architecture (Wang et al.; Dopamine 2022) — GitHub: https://github.com/google/dopamine  

**Novel Synthesis:** Replace single critic in A2C with dueling critic (value + advantage) to reduce variance in policy gradient estimates.

**Implementation & Improvement Plan:**  
- Modify critic network to output V and A; aggregate Q = V + A - mean(A); use log-prob * advantage for actor loss.  
- Keep entropy bonus; adjust learning rate for bigger critic.  
- Container: PyTorch 2.2 + atari-py.

**Evaluation Strategy:**  
- Benchmark: Atari (Asterix, SpaceInvaders).  
- Ablation: dueling critic vs standard critic.

## Assignment 20: GAE with Discount Annealing
**Selected Papers:**  
- GAE (Schulman et al.; CleanRL 2023 PPO) — GitHub: https://github.com/vwxyzjn/cleanrl  
- Discount Annealing for Long Horizon (Zhang et al., NeurIPS 2023) — GitHub: https://github.com/zhanglonghao1992/discount-annealing-rl  

**Novel Synthesis:** Anneal γ during training inside GAE to gradually extend effective horizon while keeping early stability.

**Implementation & Improvement Plan:**  
- Add schedule for γ (start low -> target); recompute GAE per rollout with current γ.  
- Log impact on advantage variance.  
- Container: CleanRL docker.

**Evaluation Strategy:**  
- Benchmark: Procgen (CoinRun, CaveFlyer).  
- Ablation: fixed γ vs annealed.

## Assignment 21: PPO with Trust-Region Clipping and Fisher Preconditioning
**Selected Papers:**  
- PPO (Schulman et al.; CleanRL 2023) — GitHub: https://github.com/vwxyzjn/cleanrl  
- TRPO (Schulman et al.; PyTorch impl 2023) — GitHub: https://github.com/ikostrikov/pytorch-trpo  

**Novel Synthesis:** Fuse PPO clipping with a lightweight Fisher preconditioner from TRPO to approximate trust-region steps without conjugate gradients.

**Implementation & Improvement Plan:**  
- Compute Fisher-vector product via empirical KL; use as preconditioner on PPO gradient step (one CG iteration).  
- Keep clip ratio; add KL penalty fallback.  
- Container: PyTorch 2.1 + mpi4py optional.

**Evaluation Strategy:**  
- Benchmark: MuJoCo Ant/HalfCheetah.  
- Ablation: PPO baseline vs PPO+Fisher.

## Assignment 22: PPO with Entropy-Normalized Objectives
**Selected Papers:**  
- PPO (CleanRL 2023) — GitHub: https://github.com/vwxyzjn/cleanrl  
- Entropy-Regularized PG Theory (Neu et al., 2022) — GitHub: https://github.com/facebookresearch/entropy-rl  

**Novel Synthesis:** Normalize entropy bonus by state-dependent visitation counts to prevent collapse on sparse rewards.

**Implementation & Improvement Plan:**  
- Track visitation counts per state hash; scale entropy coefficient inversely.  
- Add hashing for images (SimHash).  
- Container: PyTorch + CleanRL.

**Evaluation Strategy:**  
- Benchmark: MiniGrid, Sparse MountainCar.  
- Ablation: uniform entropy vs normalized.

## Assignment 23: Deterministic Policy Gradient with Distributional Critics
**Selected Papers:**  
- DDPG/TD3 (Fujimoto; TD3 repo 2023) — GitHub: https://github.com/sfujim/TD3  
- Implicit Quantile Networks (Dabney et al.; PyTorch impl 2023) — GitHub: https://github.com/ku2482/iqn  

**Novel Synthesis:** Replace scalar critic in TD3 with IQN distributional critic; actor optimized over mean/quantile risk.

**Implementation & Improvement Plan:**  
- Integrate IQN critic; use twin critics with quantiles; target smoothing unchanged.  
- Actor loss uses expected Q or CVaR; config risk_level.  
- Container: PyTorch + Mujoco.

**Evaluation Strategy:**  
- Benchmark: MuJoCo Humanoid/Ant.  
- Ablation: scalar critic vs IQN; CVaR vs mean.

## Assignment 24: TD3 with State Augmentation for Safety
**Selected Papers:**  
- TD3 (Fujimoto; repo 2023) — GitHub: https://github.com/sfujim/TD3  
- Sauté RL (Sootla et al., ICML 2022) — GitHub: https://github.com/saferl/saute-rl  

**Novel Synthesis:** Apply Sauté state augmentation to TD3 for constraint satisfaction in continuous control.

**Implementation & Improvement Plan:**  
- Augment state with budget variable; modify reward/termination per Sauté formulation.  
- Add Lagrange multiplier for constraint violation to actor-critic losses.  
- Container: PyTorch 2.1 + Mujoco.

**Evaluation Strategy:**  
- Benchmark: Safety Gymnasium (PointGoal).  
- Ablation: with/without Sauté augmentation.

## Assignment 25: TD3+BC with Representation Learning
**Selected Papers:**  
- TD3+BC (NeurIPS 2021) — GitHub: https://github.com/sfujim/TD3_BC  
- DrQ-v2 (Yarats et al., 2021; maintained 2023) — GitHub: https://github.com/facebookresearch/drqv2  

**Novel Synthesis:** Use DrQ-v2 encoder in TD3+BC to improve pixel-based offline control.

**Implementation & Improvement Plan:**  
- Swap MLP encoder with DrQ augmentations + CNN; freeze after warm-up or train jointly.  
- Keep BC regularizer; tune augmentation strength.  
- Container: PyTorch + torchvision; D4RL pixel datasets.

**Evaluation Strategy:**  
- Benchmark: D4RL pixel Adroit/DMControl offline.  
- Ablation: with/without DrQ augmentations.

## Assignment 26: Implicit Q-Learning with KL-Regularized Targets
**Selected Papers:**  
- IQL (Kostrikov et al., NeurIPS 2021; maintained 2023) — GitHub: https://github.com/ikostrikov/implicit_q_learning  
- XQL (Ma et al., ICLR 2023) — GitHub: https://github.com/coh250/xql  

**Novel Synthesis:** Add XQL reverse-KL constraint to IQL advantage-weighted policy extraction to reduce extrapolation error.

**Implementation & Improvement Plan:**  
- Modify IQL policy loss to include reverse-KL to behavior logits; reuse XQL importance weights.  
- Keep expectile regression critic; tune KL weight.  
- Container: PyTorch + D4RL.

**Evaluation Strategy:**  
- Benchmark: D4RL antmaze/hand-manipulation.  
- Ablation: KL weight 0 vs tuned.

## Assignment 27: ReBRAC with Diffuser Rollout Augmentation
**Selected Papers:**  
- ReBRAC (ICLR 2023) — GitHub: https://github.com/Farama-Foundation/ReBRAC  
- Diffuser (Janner et al., ICML 2022) — GitHub: https://github.com/ethanluoyc/diffuser  

**Novel Synthesis:** Use Diffuser to generate synthetic trajectories; feed to ReBRAC with conservative penalty to expand coverage.

**Implementation & Improvement Plan:**  
- Train Diffuser on offline data; sample trajectories respecting dynamics; add to replay with lower priority weight.  
- Ensure reward normalization matches ReBRAC.  
- Container: PyTorch 2.1 + diffusion libs.

**Evaluation Strategy:**  
- Benchmark: D4RL kitchen/maze2d.  
- Ablation: real-only vs real+diffusion rollouts.

## Assignment 28: Conservative Q-Learning with Quantile Critics
**Selected Papers:**  
- CQL (Kumar et al., NeurIPS 2020; maintained 2023) — GitHub: https://github.com/aviralkumar2907/CQL  
- QR-DQN (Dabney et al.; Dopamine 2022) — GitHub: https://github.com/google/dopamine  

**Novel Synthesis:** Replace scalar Q in CQL with quantile critic to better estimate risk-sensitive penalties.

**Implementation & Improvement Plan:**  
- Extend CQL loss to sum over quantiles; compute conservative penalty on distribution tails.  
- Actor trained on expected Q or CVaR.  
- Container: PyTorch + D4RL.

**Evaluation Strategy:**  
- Benchmark: D4RL medium-expert tasks.  
- Ablation: scalar vs quantile CQL.

## Assignment 29: Offline Advantage-Weighted Actor-Critic with Language Goals
**Selected Papers:**  
- AWAC (Nair et al., NeurIPS 2020; maintained 2023) — GitHub: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/tree/master/awac  
- SayCan (Brohan et al., 2022) — GitHub: https://github.com/google-research/google-research/tree/master/saycan  

**Novel Synthesis:** Condition AWAC policies on language goals using SayCan grounding; advantage weights computed per grounded goal.

**Implementation & Improvement Plan:**  
- Add text encoder (T5/DistilBERT) to AWAC actor; concatenate with state.  
- Use SayCan grounding scores to mask unsafe actions during policy extraction.  
- Container: PyTorch + transformers; use Docker with CUDA.

**Evaluation Strategy:**  
- Benchmark: RLBench language-conditioned tasks.  
- Ablation: grounding on/off.

## Assignment 30: Decision Transformer with Behavior Cloning Pretraining
**Selected Papers:**  
- Decision Transformer (Chen et al., NeurIPS 2021) — GitHub: https://github.com/kzl/decision-transformer  
- BC pretraining for offline RL (RLPD, ICLR 2023) — GitHub: https://github.com/soroushmehraban/rlpd  

**Novel Synthesis:** Pretrain DT on BC loss (RLPD style) before return-conditioning to improve stability on sparse datasets.

**Implementation & Improvement Plan:**  
- Add BC warm-up phase using RLPD dataloader; freeze embeddings or fine-tune during RL phase.  
- Implement curriculum of return-to-go targets after BC plateau.  
- Container: PyTorch + HuggingFace tokenizers.

**Evaluation Strategy:**  
- Benchmark: D4RL medium-replay; Atari trajectories.  
- Ablation: BC warm-up length; none vs full.

## Assignment 31: Trajectory Transformer with TD Online Fine-Tuning
**Selected Papers:**  
- Trajectory Transformer (Janner et al., NeurIPS 2021) — GitHub: https://github.com/jannerm/trajectory-transformer  
- TD-MPC2 online planner (ICLR 2024) — GitHub: https://github.com/nicklashansen/tdmpc2  

**Novel Synthesis:** Use TD-MPC2 rollouts to generate fresh transitions for online fine-tuning of Trajectory Transformer policy head.

**Implementation & Improvement Plan:**  
- Add online buffer fed by TD-MPC2 planner; fine-tune transformer with KL regularization to offline model.  
- Keep autoregressive decoding; adjust reward scaling to match planner.  
- Container: JAX/Flax or PyTorch; include MuJoCo.

**Evaluation Strategy:**  
- Benchmark: DMControl locomotion.  
- Ablation: online data on/off; planner depth.

## Assignment 32: Diffuser for Multi-Step Model-Based Policy Improvement
**Selected Papers:**  
- Diffuser (ICML 2022) — GitHub: https://github.com/ethanluoyc/diffuser  
- DreamerV3 (Hafner et al., ICLR 2023) — GitHub: https://github.com/danijar/dreamerv3  

**Novel Synthesis:** Use DreamerV3 latent rollouts as conditioning context for Diffuser to generate action sequences, enabling sample-efficient policy improvement.

**Implementation & Improvement Plan:**  
- Train Diffuser on imagined trajectories from DreamerV3; feed generated actions back to Dreamer policy optimization as candidates.  
- Add likelihood weighting to keep within model trust region.  
- Container: PyTorch + TensorFlow (if needed) in unified image.

**Evaluation Strategy:**  
- Benchmark: DMControl Humanoid/Cheetah.  
- Ablation: real-only conditioning vs latent-conditioned.

## Assignment 33: Model-Based Planning with EfficientZero-style Value Reanalysis
**Selected Papers:**  
- EfficientZero (Ye et al., ICML 2021; maintained 2023) — GitHub: https://github.com/YeWR/EfficientZero  
- MuZero Reanalyze (Schrittwieser et al., 2021; cleanrl_muzero 2023) — GitHub: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/atari/muzero_atari.py  

**Novel Synthesis:** Add reanalysis of stored trajectories with updated value head to EfficientZero, improving target quality without new environment steps.

**Implementation & Improvement Plan:**  
- Implement reanalyze worker that runs updated value/policy over stored roots; replace targets in replay.  
- Keep EfficientZero’s consistency loss; tune reanalyze ratio.  
- Container: PyTorch 2.1 + C++ MCTS deps.

**Evaluation Strategy:**  
- Benchmark: Atari 200M; board games (Gomoku).  
- Ablation: reanalysis ratio 0 vs >0.

## Assignment 34: Planning-Guided Q-Learning with Short-Horizon MPC
**Selected Papers:**  
- TD-MPC2 (ICLR 2024) — GitHub: https://github.com/nicklashansen/tdmpc2  
- CQL (NeurIPS 2020; maintained 2023) — GitHub: https://github.com/aviralkumar2907/CQL  

**Novel Synthesis:** Use TD-MPC2 to generate target Q-values for CQL updates, anchoring conservative estimates to MPC rollouts.

**Implementation & Improvement Plan:**  
- For each batch, run TD-MPC2 imagined rollouts to compute return targets; plug into CQL Bellman loss.  
- Keep conservative penalty; schedule rollout horizon.  
- Container: PyTorch + MuJoCo.

**Evaluation Strategy:**  
- Benchmark: DMControl & D4RL locomotion.  
- Ablation: MPC-guided targets on/off.

## Assignment 35: Eligibility Traces in Model-Based Actor-Critic
**Selected Papers:**  
- DreamerV3 (ICLR 2023) — GitHub: https://github.com/danijar/dreamerv3  
- rlax TD(λ) (2022) — GitHub: https://github.com/deepmind/rlax  

**Novel Synthesis:** Compute TD(λ) targets over imagined trajectories in Dreamer to reduce bias in value learning.

**Implementation & Improvement Plan:**  
- Add λ-return calculator inside Dreamer’s imagination rollout; adjust actor gradients accordingly.  
- Log λ impact on policy entropy.  
- Container: DreamerV3 stack.

**Evaluation Strategy:**  
- Benchmark: Atari 100k, DMControl.  
- Ablation: λ sweep.

## Assignment 36: MAPPO with Graph Neural Communication
**Selected Papers:**  
- MAPPO (Yu et al., NeurIPS 2022) — GitHub: https://github.com/marlbenchmark/on-policy  
- Graph Policy Networks (ICLR 2020; maintained 2023) — GitHub: https://github.com/semitable/graph-policy-network  

**Novel Synthesis:** Replace MAPPO shared encoder with graph neural message passing to model agent interactions.

**Implementation & Improvement Plan:**  
- Build GNN encoder over agent observations; integrate into MAPPO actor/critic; maintain centralized critic.  
- Support variable team sizes.  
- Container: PyTorch Geometric + on-policy repo.

**Evaluation Strategy:**  
- Benchmark: SMAC (3s5z, MMM2).  
- Ablation: GNN vs MLP encoder.

## Assignment 37: MARL with Emergent Communication and Partial Observability
**Selected Papers:**  
- JaxMARL (Iqbal et al., ICLR 2023) — GitHub: https://github.com/instadeepai/jaxmarl  
- EGG Emergent Communication (Lazaridou et al., 2020; maintained 2023) — GitHub: https://github.com/facebookresearch/EGG  

**Novel Synthesis:** Add differentiable communication channel from EGG into JaxMARL PPO learners to handle partial observability.

**Implementation & Improvement Plan:**  
- Implement comm head producing messages; aggregate via attention; share gradients across agents.  
- Enforce bandwidth constraints; optionally discrete Gumbel-softmax.  
- Container: JAX/Flax + EGG.

**Evaluation Strategy:**  
- Benchmark: Multi-Agent Particle Env, Hanabi.  
- Ablation: communication on/off; bandwidth limits.

## Assignment 38: Safe RL with Probabilistic Shields in OmniSafe
**Selected Papers:**  
- Probabilistic Shields (Jansen et al., CONCUR 2020) — GitHub: https://github.com/probabilistic-shields/safe-rl  
- OmniSafe (Ji et al., NeurIPS 2023) — GitHub: https://github.com/OmniSafeAI/OmniSafe  

**Novel Synthesis:** Integrate shield synthesis into OmniSafe training loop to filter unsafe actions before execution.

**Implementation & Improvement Plan:**  
- Build shield from environment MDP; wrap OmniSafe policy to project actions through shield.  
- Log shield interventions; adapt Lagrange multiplier when shield triggers.  
- Container: OmniSafe Docker; add PRISM/Storm deps if needed.

**Evaluation Strategy:**  
- Benchmark: Safety Gymnasium tasks.  
- Ablation: shield on/off.

## Assignment 39: Risk-Sensitive PPO with CVaR Critics
**Selected Papers:**  
- PPO (CleanRL 2023) — GitHub: https://github.com/vwxyzjn/cleanrl  
- Iterated CVaR RL (Du et al., ICLR 2023) — GitHub: https://github.com/duyihan/Iterated-CVaR-RL  

**Novel Synthesis:** Train PPO critic to predict CVaR values; optimize policy on risk-aware advantages.

**Implementation & Improvement Plan:**  
- Extend value head to quantiles; compute CVaR at α; replace baseline with CVaR.  
- Keep PPO clip; adjust entropy bonus.  
- Container: CleanRL base.

**Evaluation Strategy:**  
- Benchmark: MuJoCo with stochastic perturbations; finance RL.  
- Ablation: mean-value vs CVaR critic.

## Assignment 40: Distributional Actor-Critic with Noisy Exploration
**Selected Papers:**  
- Implicit Quantile Networks (PyTorch 2023) — GitHub: https://github.com/ku2482/iqn  
- NoisyNets (Fortunato; PyTorch 2023) — GitHub: https://github.com/Kaixhin/NoisyNet-DQN  

**Novel Synthesis:** Build an IQN-based actor-critic where both actor and critic use noisy layers for exploration in continuous control.

**Implementation & Improvement Plan:**  
- Actor outputs mean action with noisy linear layers; critic IQN returns distribution; policy optimized on expected Q.  
- Tune σ decay schedule.  
- Container: PyTorch + Mujoco.

**Evaluation Strategy:**  
- Benchmark: DMControl Quadruped/Acrobot.  
- Ablation: noisy on critic only vs both.

## Assignment 41: Exploration via Information Gain in World Models
**Selected Papers:**  
- Plan2Explore (Sekar et al., ICLR 2020; maintained 2023) — GitHub: https://github.com/Danijar/plan2explore  
- DreamerV3 (ICLR 2023) — GitHub: https://github.com/danijar/dreamerv3  

**Novel Synthesis:** Plug Plan2Explore information-gain intrinsic reward into DreamerV3 to drive exploration in sparse 3D tasks.

**Implementation & Improvement Plan:**  
- Add disagreement-based intrinsic reward head; mix with extrinsic via coefficient.  
- Ensure latent rollout uses intrinsic reward for actor gradients.  
- Container: DreamerV3 stack.

**Evaluation Strategy:**  
- Benchmark: DMLab sparse tasks, Crafter.  
- Ablation: intrinsic off vs on.

## Assignment 42: Representation Learning with Self-Predictive Control
**Selected Papers:**  
- SPR (ICLR 2021; maintained 2023) — GitHub: https://github.com/facebookresearch/spr  
- DrQ-v2 (2021; maintained 2023) — GitHub: https://github.com/facebookresearch/drqv2  

**Novel Synthesis:** Replace SPR’s RL head with DrQ-v2 SAC head, keeping SPR representation loss to improve pixel control.

**Implementation & Improvement Plan:**  
- Share encoder with augmentations; compute SPR loss + SAC losses; stop-grad into target.  
- Tune loss weights; keep EMA targets.  
- Container: PyTorch + CUDA.

**Evaluation Strategy:**  
- Benchmark: DMControl pixels.  
- Ablation: SAC-only vs SAC+SPR.

## Assignment 43: Meta-RL with MAML over Multi-Task Control
**Selected Papers:**  
- MAML (Finn et al.; repo maintained 2023) — GitHub: https://github.com/cbfinn/maml  
- Meta-World (Yu et al., CoRL 2019; maintained 2023) — GitHub: https://github.com/Farama-Foundation/Metaworld  

**Novel Synthesis:** Apply MAML to actor-critic on Meta-World; use RL gradient as inner loop, meta-update outer loop.

**Implementation & Improvement Plan:**  
- Implement RL-specific MAML step (policy gradient) per task; outer update aggregates.  
- Add task-specific normalization; shared encoder.  
- Container: PyTorch + Mujoco.

**Evaluation Strategy:**  
- Benchmark: Meta-World MT10/MT50.  
- Ablation: meta step size; no-meta baseline.

## Assignment 44: Hierarchical RL with Options in Continuous Control
**Selected Papers:**  
- HIRO (Nachum et al.; PyTorch impl 2023) — GitHub: https://github.com/tensorflow/agents/tree/master/tf_agents/agents/hiro  
- TD3 (Fujimoto; 2023) — GitHub: https://github.com/sfujim/TD3  

**Novel Synthesis:** Use TD3 as low-level option policy within HIRO high-level manager; options learned via off-policy data.

**Implementation & Improvement Plan:**  
- Replace HIRO low-level SAC with TD3; adjust hindsight relabeling for deterministic actions.  
- Sync replay buffers for both levels; shared encoder.  
- Container: PyTorch + Mujoco.

**Evaluation Strategy:**  
- Benchmark: Ant Maze, Kitchen tasks.  
- Ablation: SAC vs TD3 low-level.

## Assignment 45: Offline-to-Online Fine-Tuning with KL Safety
**Selected Papers:**  
- RLPD (Shao et al., ICLR 2023) — GitHub: https://github.com/soroushmehraban/rlpd  
- SAC (Haarnoja; pytorch_sac 2022) — GitHub: https://github.com/denisyarats/pytorch_sac  

**Novel Synthesis:** Initialize SAC with RLPD offline pretraining; during online, add KL penalty to stay near behavior policy until performance surpasses threshold.

**Implementation & Improvement Plan:**  
- Load RLPD weights; estimate behavior policy via fitted model; include KL(π||β) in actor loss with annealing.  
- Monitor return threshold to relax KL.  
- Container: PyTorch + D4RL.

**Evaluation Strategy:**  
- Benchmark: D4RL -> online MuJoCo.  
- Ablation: KL penalty on/off.

## Assignment 46: Policy Gradient with Generalized Advantage Estimation in Language-Conditioned Envs
**Selected Papers:**  
- GAE PPO (CleanRL 2023) — GitHub: https://github.com/vwxyzjn/cleanrl  
- RLBench Language Tasks (James et al., 2020; maintained 2024) — GitHub: https://github.com/stepjam/RLBench  

**Novel Synthesis:** Extend PPO with text encoder; compute GAE on language-conditioned episodes for robotic manipulation.

**Implementation & Improvement Plan:**  
- Add transformer encoder for text; fuse with visual embeddings; GAE unchanged.  
- Curriculum over task descriptions; entropy bonus tuned.  
- Container: PyTorch + transformers + RLBench deps.

**Evaluation Strategy:**  
- Benchmark: RLBench pick-place variants with language.  
- Ablation: text conditioning on/off.

## Assignment 47: Trust-Region Policy Optimization with KL-Probing Schedules
**Selected Papers:**  
- TRPO (PyTorch 2023) — GitHub: https://github.com/ikostrikov/pytorch-trpo  
- KL-Probing (Huang et al., 2024, arXiv) — GitHub: https://github.com/huangleiBuaa/KL-Probing-RL  

**Novel Synthesis:** Dynamically adjust TRPO KL constraint via probing episodes to detect policy drift.

**Implementation & Improvement Plan:**  
- Run periodic probe rollouts; if KL spike detected, tighten constraint; else relax.  
- Implement scheduler around conjugate gradient step.  
- Container: PyTorch + mpi4py.

**Evaluation Strategy:**  
- Benchmark: MuJoCo Humanoid/Ant.  
- Ablation: fixed KL vs probing.

## Assignment 48: Deterministic Actor-Critic with Target Policy Smoothing Noise Schedules
**Selected Papers:**  
- TD3 (2023 repo) — GitHub: https://github.com/sfujim/TD3  
- Scheduled Exploration Noise (Gaussian/OU anneal; OpenAI Baselines 2023) — GitHub: https://github.com/openai/baselines  

**Novel Synthesis:** Schedule both action noise and target policy smoothing noise separately to balance bias/variance.

**Implementation & Improvement Plan:**  
- Add dual schedulers (cosine/linear) for exploration noise and target smoothing stddev.  
- Log Q overestimation vs noise; auto-tune if divergence.  
- Container: PyTorch.

**Evaluation Strategy:**  
- Benchmark: MuJoCo Swimmer/HalfCheetah.  
- Ablation: fixed vs scheduled noises.

## Assignment 49: TD(λ) for Distributional Control in Stochastic Gridworlds
**Selected Papers:**  
- rlax td_lambda (2022) — GitHub: https://github.com/deepmind/rlax  
- Categorical Distributional RL (Bellemare; Dopamine 2022) — GitHub: https://github.com/google/dopamine  

**Novel Synthesis:** Use TD(λ) to train categorical distributional Q in stochastic tabular/grid tasks.

**Implementation & Improvement Plan:**  
- Implement λ-returns per atom; update categorical logits accordingly.  
- Ensure projection handles multi-step returns; tune λ.  
- Container: Python slim + JAX or PyTorch.

**Evaluation Strategy:**  
- Benchmark: Stochastic Windy Gridworld, FrozenLake slippery.  
- Ablation: λ sweep.

## Assignment 50: Model-Based Planning with Risk-Sensitive Distributional Value Heads
**Selected Papers:**  
- DreamerV3 (ICLR 2023) — GitHub: https://github.com/danijar/dreamerv3  
- QR-DQN (Dabney; Dopamine 2022) — GitHub: https://github.com/google/dopamine  

**Novel Synthesis:** Attach quantile value head to Dreamer’s latent critic; plan with CVaR over imagined rollouts for risk-aware control.

**Implementation & Improvement Plan:**  
- Replace scalar value with quantile head; compute CVaR for actor objective; keep world model unchanged.  
- Add risk level hyperparam; log return distribution.  
- Container: DreamerV3 setup.

**Evaluation Strategy:**  
- Benchmark: autonomous driving simulator (CARLA) with stochastic traffic.  
- Ablation: mean vs CVaR planning; quantile count.

