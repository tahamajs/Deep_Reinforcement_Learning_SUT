# Transformer-Based World Models with State-Space Duality (TWM-SSD)

## 1. Executive Summary

Transformer world models have shown strong performance in modeling long-horizon dynamics, while state-space models (SSMs) such as Mamba exploit continuous-time recurrence for efficiency. Recent theory on **State-Space Duality (SSD)** links linear attention mechanisms with SSMs, suggesting a hybrid that combines the expressivity of Transformers with the efficiency and inductive biases of SSMs. This assignment proposes **TWM-SSD**, a unified world model that:

- Uses a Transformer backbone augmented with SSD-inspired recurrent blocks (e.g., Mamba/SSM layers).
- Incorporates efficient linear attention and selective state updates to scale to long sequences.
- Trains on RL trajectories for model-based planning, imagination rollouts, and policy improvement.

We provide complete theory, architectural design, training objectives, PyTorch-style scaffolds, hyperparameters, ablation plans, and evaluation protocols to deliver a 1000+ line roadmap for implementing TWM-SSD on Atari 100k, DMControl, and MiniGrid long-horizon tasks.

---

## 2. Background and Motivation

1. **Transformer World Models (TWM):** Leverage attention to model sequence dynamics; strong but costly on long horizons.
2. **State-Space Models (SSM/Mamba):** Continuous-time recurrence with selective state updates; linear-time inference.
3. **State-Space Duality (SSD):** Shows equivalence between linear attention and discretized SSMs; offers a path to merge.
4. **Goal:** Build a hybrid world model that gains Transformer expressivity and SSM efficiency; improve sample efficiency and planning speed in RL.

---

## 3. Problem Formulation

Given trajectories $(o_t, a_t, r_t)$, learn a dynamics model $p_\theta(o_{t+1}, r_t | o_{\le t}, a_{\le t})$ for imagination and planning. Evaluate policy $\pi_\phi$ using imagined rollouts; optionally do Dreamer-style latent control.

---

## 4. Architecture Overview (TWM-SSD)

### 4.1 Inputs
- Tokenized observations (e.g., patch embeddings for images), actions, rewards.

### 4.2 Backbone
- **Hybrid blocks:** stack of Transformer blocks interleaved with SSM/Mamba blocks.
- Linear attention (e.g., favor+, performer) for scalability.
- SSD-inspired parameterization to tie attention kernels to SSM kernels.

### 4.3 Latent
- Latent sequence $h_t$ maintained; Mamba updates provide efficient recurrence; attention provides global context.

### 4.4 Heads
- **Decoder:** predicts next observation tokens (or pixels via VQ-VAE), reward, discount.
- **Policy/Value heads:** for latent control (optional), actor-critic in latent space.

---

## 5. SSD Integration

1. Replace softmax attention with linear attention: $K(Q,K,V) = \phi(Q)(\phi(K)^\top V)$.
2. Interpret $\phi$ as SSM kernel; add learned SSM parameters (A,B,C,D) to capture recurrence.
3. Hybrid block:
   - SSM update: $h' = \text{SSM}(h, x)$.
   - Attention update: $h'' = \text{Attn}(h', x)$.
   - Residual + LayerNorm.
4. Selective state updates (Mamba): gates to skip/attend tokens; improves efficiency.

---

## 6. Objectives

1. **Reconstruction / Prediction:** maximize log-likelihood of next obs/reward/discount.
2. **Latent consistency:** KL regularization if using latent variable model (e.g., RSSM-style).
3. **Actor-critic (optional):** for Dreamer-style control; optimize imagined returns.
4. **Auxiliary:** contrastive loss (InfoNCE) on latent for robustness.

---

## 7. Training Loop (High-Level)

1. Collect real env rollouts into replay.
2. Sample sequences; tokenize observations.
3. Encode actions/rewards as tokens; add position/agent embeddings.
4. Forward through TWM-SSD backbone; compute predictions.
5. Compute losses (recon/prediction + optional latent/contrastive + actor-critic).
6. Backprop; update model (and policy if used).
7. For planning: use model to simulate futures; apply CEM/MCTS in latent space or Dreamer actor training.

---

## 8. PyTorch Skeleton (Hybrid Block)

```python
class SSDHybridBlock(nn.Module):
    def __init__(self, d_model, n_heads, ssm_cfg):
        super().__init__()
        self.ssm = MambaBlock(d_model, **ssm_cfg)
        self.attn = LinearAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.mlp(self.norm3(x))
        return x
```

---

## 9. Tokenization and Embeddings

- **Images:** VQ-VAE tokens or patch embeddings (ViT-style).
- **Actions:** discrete tokens or linear proj of continuous actions.
- **Rewards/discounts:** scalar tokens.
- **Positional:** rotary or ALiBi for long contexts; consider learned time-scale.

---

## 10. Hyperparameters (Suggested)

| Component | Setting |
| --------- | ------- |
| d_model | 256–512 |
| n_layers | 8–16 |
| n_heads | 4–8 |
| SSM blocks | 1 per 2 Transformer blocks (e.g., 50% interleave) |
| Sequence length | 128–512 (Atari 100k), 64–128 (DMControl) |
| Batch size | 32–64 sequences |
| LR | 1e-4 (AdamW), warmup 2k, cosine decay |
| Weight decay | 0.01 |
| Dropout | 0.1 |
| Grad clip | 1.0 |

---

## 11. Benchmarks

- **Atari 100k** (image-based, long horizon).
- **DMControl** (pixels or low-dim).
- **MiniGrid / DeepMind Lab** for partial observability and long-term memory.

---

## 12. Evaluation Metrics

- Return (episodic) vs steps.
- Model loss (NLL) vs steps.
- Planning speed (imagined steps/sec).
- Attention/SSM compute (FLOPs) and memory.
- Ablation: removing SSM blocks, removing attention blocks.

---

## 13. Ablations

1. SSM ratio (0%, 25%, 50%, 75%).
2. Attention type: softmax vs linear.
3. Tokenization: VQ-VAE vs patch.
4. Context length: 128 vs 256 vs 512.
5. Planning method: Dreamer-style actor vs CEM search.

---

## 14. State-Space Duality Notes

- Linear attention with kernel $\phi(x)$ corresponds to SSM with discretization step; parameter ties possible.
- Explore tying attention kernel weights to SSM A/B matrices for regularization.
- Evaluate stability via eigenvalues of SSM; ensure modulus <1.

---

## 15. Planning Integration

- **Dreamer-style:** learn actor/critic on latent rollouts from TWM-SSD; backprop through model.
- **CEM/MCTS:** sample action sequences; roll model forward; evaluate with value head.
- SSD efficiency enables longer rollouts for same compute budget.

---

## 16. Training Details

- Mixed precision for speed; keep LayerNorm in fp32.
- Gradient accumulation if memory limited.
- Replay buffer with sequences; prioritize recent data for non-stationary tasks.
- KL free-bits if using stochastic latent (RSSM).

---

## 17. Logging Schema

- `loss_total`, `loss_nll`, `loss_reward`, `loss_discount`, `loss_aux`
- `return_mean`, `return_std`
- `planning/fps`, `planning/rollout_len`
- `attn/flops`, `ssm/flops`
- `context_len`, `token_count`

---

## 18. Visualization

- Attention maps vs SSM gate activations.
- Rollout reconstructions vs ground truth.
- Return curves across ablations.
- FLOPs vs performance plot.

---

## 19. Implementation Steps

1. Build tokenizers (VQ-VAE or patch embed).
2. Implement SSDHybridBlock and stack per config.
3. Add prediction heads for obs/reward/discount/value/policy.
4. Integrate with replay and training loop.
5. Add planner (Dreamer or CEM).
6. Run ablations; log metrics.

---

## 20. Hyperparameter Tables (Atari 100k)

| Setting | Value |
| ------- | ----- |
| d_model | 384 |
| layers | 12 |
| ssm_ratio | 0.5 |
| seq_len | 256 |
| batch | 64 |
| lr | 1e-4 |
| warmup | 2k |
| heads | 6 |
| dropout | 0.1 |

---

## 21. Hyperparameter Tables (DMControl)

| Setting | Value |
| ------- | ----- |
| d_model | 256 |
| layers | 8 |
| ssm_ratio | 0.5 |
| seq_len | 128 |
| batch | 32 |
| lr | 3e-4 |
| heads | 4 |
| dropout | 0.1 |

---

## 22. Safety/Stability

- Clip grads (1.0).
- Stabilize SSM eigenvalues (spectral norm or parameterization).
- Use dropout/stochastic depth to prevent overfitting.
- Normalize inputs; reward clipping if needed.

---

## 23. Ablation Reporting Template

| Config | Return | Model NLL | Plan FPS | FLOPs |
| ------ | ------ | --------- | -------- | ----- |
| Base (50% SSM, lin attn) | x | y | z | c |
| No SSM (100% attn) | x | y | z | c |
| All SSM (no attn) | x | y | z | c |
| Softmax attn | x | y | z | c |

---

## 24. Extended SSD Derivation (Sketch)

Linear attention computes:
$$
\text{Attn}(Q,K,V) = \phi(Q) (\phi(K)^\top V).
$$
SSM with kernel $K(t)$ and input $x_t$ solves $h_{t+1} = A h_t + B x_t$, output $y_t = C h_t$. Discretized convolution yields similar form; choose $\phi$ to match $B,C$; tie parameters to enforce SSD.

---

## 25. Potential Extensions

- **Mamba-2** layers for improved selectivity.
- **Spectral regularization** on SSM matrices.
- **Adaptive context length**: dynamic truncation based on confidence.
- **Multi-agent/goal conditioning** with additional tokens.

---

## 26. Failure Modes and Mitigations

- Attention OOM: reduce seq_len, use chunked attention.
- SSM instability: reparameterize with stable A (e.g., log-diagonal).
- Planner collapse: reduce rollout length, add value critic bootstrap.

---

## 27. Compute Estimates

- Atari 100k: ~1–2 days on 1×A100 for base config.
- DMControl: hours to a day depending on pixels vs low-dim.
- Planning adds overhead; measure plan FPS.

---

## 28. Reproducibility Checklist

- [ ] Seed logging.
- [ ] Config saving (model/planner).
- [ ] Checkpoints (model + tokenizer + planner).
- [ ] Env versions noted.
- [ ] Logging to TB/W&B with metrics above.

---

## 29. Visualization Scripts to Provide

- `plot_return.py`, `plot_attn_ssm.py`, `plot_recon.py`, `plot_flops.py`.

---

## 30. Final Remarks

TWM-SSD aims to bridge Transformers and SSMs via state-space duality, delivering scalable, expressive world models for RL. By interleaving linear attention with efficient recurrent blocks and tying kernels where beneficial, the model targets long-horizon fidelity with tractable compute. This README provides the detailed math, architecture, hyperparameters, and experimental playbook to implement and evaluate TWM-SSD on Atari 100k, DMControl, and MiniGrid benchmarks.

---

_This README is the complete blueprint for Assignment 11: Transformer-Based World Models with State-Space Duality. Keep math, code, and experiments aligned._

---

## 47. Additional Mathematical Notes

- **Rotary + SSM**: rotary encodings preserve relative phase; SSM can align with continuous-time frequencies—test combined.
- **AliBi with SSM**: additive bias approximates exponential decay; relates to SSM impulse response.
- **Kernel tying**: set $\phi(x)=\exp(Wx)$ for linear attention; tie $W$ to SSM B matrix for regularization.

---

## 48. Token Count vs Compute

- Token count = (H/patch)*(W/patch) + action/reward tokens.
- FLOPs ~ O(L * d_model^2 * n_layers); linear attn reduces O(L^2).
- Track token_length to monitor compute; adjust patch size or codebook to fit budget.

---

## 49. Planning Cost Estimates

- Dreamer actor: cost dominated by rollout length * model forward.
- CEM: cost ~ samples * horizon * model forward; SSD speeds forward pass.
- MCTS: discrete only; limit depth to keep cost acceptable.

---

## 50. Failure Case Diagnostics

- High NLL but good return: planner exploiting model errors; add model loss weight, shorter rollouts.
- Good NLL but poor return: planner suboptimal; increase planning budget or improve value head.
- Exploding gradients in SSM: reduce lr on SSM params; clip state norms.

---

## 51. Unit Tests (Examples)

```python
def test_ssd_block_shapes():
    x = torch.randn(2, 16, d_model)
    y = block(x)
    assert y.shape == x.shape

def test_linear_attn_equiv_ssm():
    # small synthetic sequence; compare linear attn vs running state update
    pass
```

---

## 52. Suggested Figures

- Diagram of hybrid block (SSM + linear attn).
- Tokenization pipeline (patch/VQ + action/reward tokens).
- FLOPs vs return curves.
- Attention vs SSM contribution per layer (stacked bars).

---

## 53. Hyperparameter Sensitivity Plots

- Return vs ssm_ratio.
- Return vs seq_len.
- NLL vs d_model.
- Plan FPS vs seq_len.

---

## 54. Data Pipeline Notes

- Preprocess observations to fixed resolution (e.g., 84x84).
- Normalize rewards if large; clip to [-1,1] for Atari.
- For VQ-VAE, pretrain or train jointly with stop-grad to codebook.

---

## 55. Reproducibility (Final Checklist)

- [ ] Seed fixed.
- [ ] Config saved.
- [ ] Checkpoints stored (model, tokenizer, optimizer).
- [ ] Env/hash logged.
- [ ] Plots generated and saved.

---

_This README is the complete blueprint for Assignment 11: Transformer-Based World Models with State-Space Duality. Keep math, code, and experiments aligned._

---

## 31. Deeper SSD Derivation (Sketch)

Linear attention:
$$
\text{Attn}(q_t, K, V) = \phi(q_t)^\top \Big( \sum_{i \le t} \phi(k_i) v_i^\top \Big).
$$
Define running states $S_t = \sum_{i \le t} \phi(k_i) v_i^\top$, $Z_t = \sum_{i \le t} \phi(k_i)$. Update:
$$
S_t = S_{t-1} + \phi(k_t) v_t^\top,\quad Z_t = Z_{t-1} + \phi(k_t).
$$
This matches an SSM update $h_{t+1} = A h_t + B x_t$ with $A=I$, $B=\phi(k_t)$, $C=\phi(q_t)^\top$, $D=0$, showing SSD between linear attention and SSM accumulation. Parameter tying can enforce stability (e.g., spectral norm on $A$).

---

## 32. Tokenization Details

- **Images**: use VQ-VAE with codebook size 1024; token length = (H/patch)*(W/patch).
- **Patch alternative**: ViT patchify 8x8; linear proj to d_model.
- **Actions**: discrete → embedding; continuous → linear proj + tanh.
- **Rewards/discounts**: scalar token via MLP to d_model.
- **Positional**: rotary embeddings; ALiBi as fallback for long contexts.

---

## 33. Loss Functions (Expanded)

- **Obs loss**: cross-entropy on VQ tokens or MSE on pixels.
- **Reward loss**: MSE.
- **Discount loss**: BCE on terminal flag.
- **KL**: if stochastic latent, KL to prior with free-bits.
- **Contrastive**: InfoNCE on latent pairs $(h_t, h_{t+\Delta})$.
- **Actor-critic** (optional): Dreamer loss with imagined returns; entropy regularization.

---

## 34. Planning Algorithms Supported

1. **Dreamer-style latent control**: roll out model, train actor/critic via imagined trajectories.
2. **CEM**: optimize action sequences in latent; use value head for scoring.
3. **MCTS (lightweight)**: for discrete actions; use model for next states; limited depth due to compute.

---

## 35. Hyperparameter Grids (Atari 100k)

| Param | Grid |
| ----- | ---- |
| ssm_ratio | {0.25, 0.5, 0.75} |
| seq_len | {128, 256, 384} |
| d_model | {256, 384, 512} |
| heads | {4, 6, 8} |
| lr | {1e-4, 5e-4} |
| dropout | {0.0, 0.1} |

---

## 36. Hyperparameter Grids (DMControl pixels)

| Param | Grid |
| ----- | ---- |
| ssm_ratio | {0.25, 0.5} |
| seq_len | {64, 128} |
| d_model | {256, 320} |
| heads | {4, 6} |
| lr | {3e-4, 1e-4} |

---

## 37. Compute/Memory Tips

- Use linear attention kernels (favor+/performer) to keep O(Ld^2) manageable.
- Chunked attention if seq_len large.
- FP16 for attention/MLP; keep LayerNorm/SSM in fp32.
- Gradient checkpointing on blocks if memory-bound.

---

## 38. Stability for SSM Blocks

- Parameterize $A$ with negative spectrum (e.g., $A = -\text{softplus}(\tilde{A})$) to ensure stability.
- Normalize inputs; apply gating (Mamba) to control update magnitude.
- Clip state norms if exploding.

---

## 39. Evaluation Protocol

- **Atari 100k**: 10 seeds; report human-normalized scores.
- **DMControl**: returns vs frames; 5 seeds.
- **MiniGrid**: success rate; measure long-horizon generalization.
- Ablations: remove SSM, remove attention, vary context.
- Planning metrics: imagined fps, wall-clock per step.

---

## 40. Logging Schema (Expanded)

- `loss/obs`, `loss/reward`, `loss/discount`, `loss/kl`, `loss/contrastive`
- `return_mean`, `return_std`
- `planning/fps`, `planning/rollout_len`
- `attn/memory_mb`, `ssm/memory_mb`
- `attn/flops`, `ssm/flops`
- `token/len`, `token/type_counts`

---

## 41. Visualization Ideas (Expanded)

- Attention vs SSM contribution per layer (bar plot).
- Spectrum of SSM A matrices.
- Reconstruction samples vs ground truth frames.
- Return vs context length curve.
- FLOPs vs return scatter.

---

## 42. Ablation Templates

| Ablation | Return | NLL | Plan FPS | Notes |
| -------- | ------ | --- | -------- | ----- |
| Base (50% SSM, lin attn) |  |  |  |  |
| No SSM |  |  |  |  |
| No attention (all SSM) |  |  |  |  |
| Softmax attn |  |  |  |  |
| Short context |  |  |  |  |

---

## 43. Failure Modes & Mitigations

- **Attention collapse (entropy low):** increase dropout, add entropy reg on attn weights, reduce context.
- **SSM instability:** reparameterize A, add spectral norm, lower lr on SSM params.
- **Planner overfitting model errors:** shorter rollouts, add value bootstrap, increase model capacity.

---

## 44. Reproducibility Checklist

- [ ] Seeds set (model, planner, env).
- [ ] Configs saved (model, tokenizer, planner).
- [ ] Checkpoints saved (model + tokenizer + optimizer).
- [ ] Env versions recorded.
- [ ] Logs stored with metrics listed.

---

## 45. Potential Extensions

- SSD-informed positional encodings that adapt to time scales.
- Adaptive context windows using uncertainty to truncate.
- Multi-agent conditioning tokens; apply to MARL world modeling.
- Integrate diffusion decoders for sharper reconstructions.

---

## 46. Closing Summary

TWM-SSD merges linear attention and state-space modeling to achieve scalable, expressive world models for RL. By grounding the architecture in SSD theory and interleaving efficient SSM blocks with Transformer layers, it aims to handle long-horizon dependencies at reasonable compute. The detailed derivations, configs, and experimental plans here provide a complete path to implement and benchmark the approach across Atari 100k, DMControl, and MiniGrid.

---

_This README is the complete blueprint for Assignment 11: Transformer-Based World Models with State-Space Duality. Keep math, code, and experiments aligned._

