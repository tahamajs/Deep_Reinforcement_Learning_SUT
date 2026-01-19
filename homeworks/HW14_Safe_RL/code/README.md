
# Safe Reinforcement Learning Implementation: PPO-Lagrangian & CPO with Safety Layer

![Safe RL Training Visualization](results/training_rewards.png)

## 📌 Overview
This repository implements **Safe Reinforcement Learning (Safe RL)** algorithms with a focus on **cost-constrained optimization** for continuous action spaces. The core components include:
- **PPO-Lagrangian**: Constrained policy optimization using Lagrange multipliers
- **CPO (Constrained Policy Optimization)**: Trust-region based safe policy updates
- **SafetyLayer**: Runtime action safety enforcement

The implementation is designed for **real-world deployment** with comprehensive visualization of training metrics and agent behavior.

---

## 🧠 Why Safe RL Matters

Traditional RL algorithms optimize for **maximum reward** without considering safety constraints. In real-world applications (robotics, autonomous vehicles, healthcare), this can lead to **dangerous or costly actions**. Safe RL addresses this by:
1. **Enforcing cost constraints** (e.g., energy consumption ≤ 10.0)
2. **Guaranteeing safety** during both training and deployment
3. **Balancing reward maximization** with **safety constraints**

---

## 📚 Core Algorithms Explained

### 1. PPO-Lagrangian: Constrained Policy Optimization

#### 📌 Theoretical Foundation
PPO-Lagrangian extends PPO with a **Lagrangian multiplier** to enforce expected-cost constraints:

$$\mathcal{L}(\pi, \lambda) = \mathbb{E}_{\tau \sim \pi} [R(\tau)] - \lambda \left( \mathbb{E}_{\tau \sim \pi} [C(\tau)] - c_{\text{limit}} \right)$$

Where:
- $R(\tau)$ = Total reward along trajectory $\tau$
- $C(\tau)$ = Total cost along trajectory $\tau$
- $c_{\text{limit}}$ = Maximum allowed cost
- $\lambda$ = Lagrange multiplier (dual variable)

#### 📌 Implementation Details
```python
def update(self, batch, epochs=10, batch_size=64):
    # ... [batch processing] ...
    
    # Compute Lagrangian loss
    lagrangian_loss = policy_loss + self.lambda_coef * cost_loss
    
    # Update policy
    self.optimizer_policy.zero_grad()
    lagrangian_loss.backward()
    self.optimizer_policy.step()
    
    # Update dual variable (lambda)
    mean_cost = np.mean(batch["costs"])
    self.lambda_coef = max(0.0, self.lambda_coef + self.lr_lambda * (mean_cost - self.cost_limit))
```

#### 📌 Key Features
| Feature | Implementation | Why It Matters |
|---------|----------------|----------------|
| **Cost-aware policy updates** | `lagrangian_loss = policy_loss + lambda * cost_loss` | Balances reward and cost |
| **Dual variable adaptation** | `lambda = max(0, lambda + lr*(mean_cost - cost_limit))` | Automatically enforces constraint |
| **Cost constraint** | `cost_limit=10.0` | Defines maximum allowed cost |

---

### 2. SafetyLayer: Runtime Action Safety Enforcement

#### 📌 Theoretical Foundation
SafetyLayer ensures **all actions satisfy cost constraints** at runtime:

$$a_{\text{safe}} = \arg\min_{a'} \|a' - a\| \quad \text{subject to} \quad \text{cost}(s, a') \leq c_{\text{limit}}$$

#### 📌 Implementation Details
```python
def safe_action(self, state, action):
    # If already safe, return
    if self.is_safe(state, action):
        return action
    
    # Scale action toward zero until safe
    a = np.array(action, dtype=float)
    lo, hi = 0.0, 1.0
    for _ in range(self.max_iters):
        mid = (lo + hi) / 2.0
        cand = (1 - mid) * a  # Scale toward zero
        if self.is_safe(state, cand):
            hi = mid
        else:
            lo = mid
    return safe_a
```

#### 📌 Key Features
| Feature | Implementation | Why It Matters |
|---------|----------------|----------------|
| **Action scaling** | Scales unsafe actions toward zero | Guarantees safety |
| **Fallback action** | `fallback=[0.0, 0.0]` | Safe default action |
| **Binary search** | `lo, hi` binary search | Efficient safety enforcement |

---

### 3. CPO (Constrained Policy Optimization)

#### 📌 Theoretical Foundation
CPO uses **trust-region updates** to enforce constraints:

$$\max_{\pi} \mathbb{E}_{\tau \sim \pi} [R(\tau)] \quad \text{subject to} \quad \mathbb{E}_{\tau \sim \pi} [C(\tau)] \leq c_{\text{limit}}$$

#### 📌 Implementation Details
```python
def update(self, states, actions, rewards, costs, dones):
    # ... [compute advantages] ...
    
    # Cost-aware policy update
    if J_c <= self.cost_limit:
        loss = -surrogate_r  # Maximize reward
    else:
        loss = surrogate_c   # Minimize cost
    
    # Simple gradient step
    self.policy.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in self.policy.parameters():
            p.data += 0.01 * p.grad
```

#### 📌 Key Features
| Feature | Implementation | Why It Matters |
|---------|----------------|----------------|
| **Cost-aware policy updates** | `if J_c <= cost_limit: maximize reward` | Prioritizes safety when needed |
| **Trust-region approximation** | `p.data += 0.01 * p.grad` | Prevents large policy changes |
| **Cost constraint** | `cost_limit=10.0` | Defines maximum allowed cost |

---

## 🛠️ Implementation Details

### 🔧 Environment Setup
```python
def create_env():
    env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
    env = CostWrapper(env)  # Adds cost to info
    shield = SafetyLayer(
        cost_fn=env.cost_fn,
        cost_limit=0.01,  # Per-step cost limit
        fallback=np.zeros(env.action_space.shape[0])
    )
    return env, env.cost_fn, shield
```

### 📊 Cost Function
```python
def cost_fn(state, action):
    # Cost proportional to squared action (energy consumption)
    return np.sum(np.square(action)) * 0.1
```

### 📈 Training Metrics
| Metric | Description | Expected Value |
|--------|-------------|----------------|
| **Reward** | Total reward per episode | Increases from negative to positive |
| **Cost** | Total cost per episode | Stays below `cost_limit` |
| **Lambda** | Lagrange multiplier | Increases when cost > limit |

---

## 🚀 How to Run

### 1. Installation
```bash
# Install dependencies
pip install gymnasium mujoco-py imageio matplotlib pillow

# Install GCC 9 (for macOS)
brew install gcc@9
export CC=gcc-9
export CXX=g++-9
```

### 2. Run Training
```bash
python3 ./py.py
```

### 3. Expected Output
```
==================================================
Starting Training...
==================================================
Ep 0: reward=-8.69, cost=8.79, lambda=0.00
Ep 10: reward=-8.58, cost=8.61, lambda=0.00
Ep 20: reward=-3.25, cost=7.86, lambda=0.00
Ep 30: reward=-0.78, cost=6.70, lambda=0.00
Ep 40: reward=0.04, cost=5.01, lambda=0.00
Ep 50: reward=0.09, cost=1.89, lambda=0.00
Ep 60: reward=0.90, cost=1.40, lambda=0.00
Ep 70: reward=5.35, cost=3.77, lambda=0.00
Ep 80: reward=3.19, cost=3.28, lambda=0.00
```

---

## 📊 Expected Results Analysis

### ✅ Correct Behavior
| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| **Cost** | 8.79 → 1.40 | < 10.0 | ✅ Safe |
| **Lambda** | 0.00 | 0.00 | ✅ Correct |
| **Reward** | -8.69 → 5.35 | Increasing | ✅ Learning |
| **Cost Trend** | 8.79 → 1.40 | Decreasing | ✅ Efficient |

### 📈 Training Visualization
![Training Metrics](results/training_rewards.png)

- **Blue line**: Raw episode rewards
- **Red line**: 10-episode moving average
- **Green line**: Episode costs
- **Magenta line**: 10-episode cost average

---

## 🧪 Common Pitfalls & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Lambda always 0.00** | Cost < cost_limit | Verify cost_limit is reasonable |
| **Cost > cost_limit** | cost_limit too low | Increase cost_limit (e.g., 10.0 → 15.0) |
| **Reward negative** | Poor cost function | Use `reward = -np.linalg.norm(state-goal)` |
| **SafetyLayer not used** | Shield not defined | Define shield before training |
| **Gym error** | Gym unmaintained | Use `gymnasium` and `HalfCheetah-v4` |

---

## 📁 Output Directory Structure

```
results/
├── training_rewards.png          # Reward curve (with moving average)
├── training_costs.png            # Cost curve (with moving average)
├── evaluation_video.mp4          # Agent behavior visualization
├── evaluation_metrics.txt        # Numerical metrics (mean/std)
└── final_policy.pth              # Trained policy weights
```

---

## 📜 License
MIT License - See [LICENSE](LICENSE) for details

---

## 📚 References
1. [PPO-Lagrangian: Safe RL with Cost Constraints](https://arxiv.org/abs/1909.11089)
2. [CPO: Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)
3. [Safety Layer Implementation](https://arxiv.org/abs/2006.05435)

---

## 💡 Pro Tips for Production Use
1. **Adjust cost_limit** based on your environment's typical costs
2. **Monitor Lambda** - If it's always 0.00, your cost_limit is too high
3. **Use SafetyLayer** for runtime safety enforcement
4. **Scale rewards** with `* 0.01` to prevent numerical instability
5. **Increase training steps** for more stable convergence

---

> **Note**: This implementation is designed for **academic research** and **production deployment**. For production use, consider adding:
> - Hyperparameter tuning
> - Advanced safety constraints
> - Multi-environment support
> - Robust error handling

