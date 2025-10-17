# HW14: Safe Reinforcement Learning

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Safety](https://img.shields.io/badge/Type-Safe--RL-red.svg)](.)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

Safe Reinforcement Learning focuses on training agents that not only maximize rewards but also satisfy safety constraints during both training and deployment. This is critical for real-world applications where failures can be costly or dangerous. This assignment explores constrained optimization, risk-sensitive methods, and safety verification.

## 🎯 Learning Objectives

1. **Safety Fundamentals**: Understand safety concepts and failure modes in RL
2. **Constrained RL**: Learn to optimize with hard constraints
3. **Risk-Sensitive RL**: Handle uncertainty and tail risks
4. **Safe Exploration**: Explore without violating constraints
5. **Verification**: Formally verify safety properties
6. **Real-World Safety**: Apply to robotics, autonomous vehicles, healthcare

## 📂 Directory Structure

```
HW14_Safe_RL/
├── code/
│   └── HW14_Notebook.ipynb        # Safe RL implementations
├── answers/                        # (No solutions provided yet)
├── reports/
│   └── HW14_Questions.pdf         # Assignment questions
└── README.md
```

## 📚 Core Concepts

### 1. What is Safe RL?

**Safety Requirements:**

- **Training Safety**: Don't violate constraints during learning
- **Deployment Safety**: Satisfy constraints after training
- **Robustness**: Handle distribution shift and adversarial inputs
- **Interpretability**: Understand agent's decision-making
- **Recoverability**: Return to safe states after perturbations

**Types of Safety:**

#### Hard Constraints

```
Cost c(s,a) ≤ threshold always
Example: Robot must not collide with humans
```

#### Soft Constraints

```
Expected cost 𝔼[∑ c(s,a)] ≤ threshold
Example: Average energy consumption limit
```

#### Probabilistic Constraints

```
P(accident) ≤ δ
Example: 99.9% safety guarantee
```

**Challenges:**

- Exploration may violate constraints
- Constraints may be unknown initially
- Trade-off between performance and safety
- Verifying safety properties is hard

### 2. Constrained Markov Decision Processes (CMDPs)

**Formulation:**

```
maximize  𝔼[∑t γᵗ rt]  (reward objective)
   π

subject to: 𝔼[∑t γᵗ ct] ≤ d  (cost constraint)

where:
- rt: reward at time t
- ct: cost at time t
- d: cost threshold
```

**Lagrangian Approach:**

```
L(π, λ) = J_r(π) - λ(J_c(π) - d)

where:
- J_r(π): expected reward
- J_c(π): expected cost
- λ ≥ 0: Lagrange multiplier
```

**Constrained Policy Optimization (CPO):**

```python
class CPO:
    def __init__(self, state_dim, action_dim, cost_limit):
        self.policy = Policy(state_dim, action_dim)
        self.value_reward = ValueNetwork(state_dim)
        self.value_cost = ValueNetwork(state_dim)
        self.cost_limit = cost_limit

    def trust_region_update(self, trajectories):
        """
        CPO: Trust region method with cost constraints
        """
        states, actions, rewards, costs = process(trajectories)

        # Compute advantages
        adv_reward = compute_advantages(rewards, self.value_reward)
        adv_cost = compute_advantages(costs, self.value_cost)

        # Policy gradient direction
        g = compute_policy_gradient(adv_reward)

        # Cost gradient
        b = compute_policy_gradient(adv_cost)

        # Current cost
        J_c = compute_cost_return(costs)

        # Fisher information matrix
        F = compute_fisher_information_matrix()

        # Solve: maximize improvement subject to KL and cost constraints
        # max_θ: gᵀ(θ - θ_old)
        # s.t.:  (θ - θ_old)ᵀF(θ - θ_old) ≤ δ_KL
        #        bᵀ(θ - θ_old) + J_c ≤ cost_limit

        # Solution involves projecting gradient
        if J_c < cost_limit:  # Feasible, maximize reward
            step = solve_trust_region(g, F, delta_KL)
        else:  # Infeasible, prioritize reducing cost
            step = solve_trust_region(-b, F, delta_KL)

        # Update policy
        self.policy.parameters += step

        return step
```

**Why CPO Works:**

- Monotonic cost improvement
- Stays within trust region
- Handles constraint violations during training

### 3. Safety Layer / Shield

**Key Idea:** Filter agent's actions through safety layer

```python
class SafetyLayer:
    def __init__(self, state_dim, action_dim):
        # Learn safe action set for each state
        self.safety_model = SafetyModel(state_dim, action_dim)

    def safe_action(self, state, proposed_action):
        """
        Project action to safe set
        """
        # Check if proposed action is safe
        if self.safety_model.is_safe(state, proposed_action):
            return proposed_action

        # Otherwise, find closest safe action
        safe_actions = self.safety_model.get_safe_actions(state)

        # Project to closest safe action
        closest = min(safe_actions,
                     key=lambda a: ||a - proposed_action||)

        return closest
```

**Applications:**

- **Runtime shielding**: Intervene only when necessary
- **Training time**: Use shield during exploration
- **Formal methods**: Synthesize shields from specifications

### 4. Risk-Sensitive RL

**Problem with Expectation:**

```
𝔼[Return] may hide tail risks

Example:
Policy A: 90% → $100, 10% → -$1000  (𝔼 = -$10)
Policy B: 100% → $0                  (𝔼 = $0)

Expected value says A is worse, but has high upside
```

**Risk Measures:**

#### Conditional Value at Risk (CVaR)

```
CVaR_α = 𝔼[X | X ≤ VaR_α]

where VaR_α is α-quantile

Interpretation: Average of worst α fraction of outcomes
```

#### Entropic Risk

```
ρ_β(X) = (1/β) log 𝔼[exp(βX)]

β → 0: Expected value
β → ∞: Worst case
```

**Implementation:**

```python
class RiskSensitiveActor:
    def __init__(self, risk_aversion=1.0):
        self.risk_aversion = risk_aversion

        # Distributional critic (learn return distribution)
        self.critic = DistributionalCritic()

    def compute_risk(self, return_distribution, alpha=0.1):
        """
        Compute CVaR_α
        """
        # Sort returns
        sorted_returns = return_distribution.sort()[0]

        # Take worst α fraction
        cutoff = int(alpha * len(sorted_returns))
        cvar = sorted_returns[:cutoff].mean()

        return cvar

    def select_action(self, state):
        """
        Select action minimizing risk
        """
        actions = sample_action_candidates()

        risks = []
        for action in actions:
            # Get return distribution
            return_dist = self.critic.get_distribution(state, action)

            # Compute risk
            risk = self.compute_risk(return_dist)

            risks.append(risk)

        # Select action with best risk-adjusted return
        best_idx = argmax(risks)
        return actions[best_idx]
```

### 5. Safe Exploration

**Challenge:** How to explore without violating safety constraints?

#### Safety Index (Control Barrier Functions)

```python
class SafetyIndex:
    def __init__(self, state_dim):
        # Learn function h(s) where h(s) > 0 → safe
        self.h = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def is_safe(self, state):
        return self.h(state) > 0

    def safe_action_set(self, state):
        """
        Actions that keep system safe

        Safety condition (Control Barrier Function):
        ḣ(s) = ∇h(s)·f(s,a) ≥ -α·h(s)

        where f(s,a) is dynamics
        """
        h_value = self.h(state)
        grad_h = torch.autograd.grad(h_value, state)[0]

        safe_actions = []
        for action in action_space:
            # Predicted next state
            next_state = dynamics_model(state, action)

            # Rate of change of h
            h_dot = torch.dot(grad_h, next_state - state)

            # Check barrier condition
            if h_dot >= -self.alpha * h_value:
                safe_actions.append(action)

        return safe_actions
```

#### LEARCH (Learning to SEARCH)

```python
def safe_exploration_with_learch():
    """
    Learn safe exploration policy from demonstrations
    """
    # 1. Collect safe demonstrations
    safe_trajectories = collect_expert_demonstrations()

    # 2. Learn cost function (IRL)
    cost_function = inverse_rl(safe_trajectories)

    # 3. Explore using learned cost as constraint
    policy = ConstrainedPolicy(cost_function)

    return policy
```

### 6. Robust RL

**Goal:** Perform well under model uncertainty and adversarial conditions

#### Domain Randomization

```python
class RobustTraining:
    def train_with_randomization(self, env_template):
        """
        Train on distribution of environments
        """
        for episode in range(N):
            # Randomize environment parameters
            env = env_template.randomize_parameters({
                'mass': uniform(0.5, 1.5),
                'friction': uniform(0.1, 0.9),
                'actuator_strength': uniform(0.8, 1.2)
            })

            # Collect trajectory
            trajectory = collect_episode(env)

            # Update policy
            update_policy(trajectory)
```

#### Adversarial Training

```python
class AdversarialEnv:
    def __init__(self, base_env, perturbation_budget):
        self.env = base_env
        self.budget = perturbation_budget
        self.adversary = AdversaryPolicy()

    def step(self, action):
        # Adversary perturbs state
        perturbation = self.adversary(self.state)
        perturbed_state = self.state + perturbation.clamp(-self.budget, self.budget)

        # Environment dynamics with perturbed state
        next_state, reward, done = self.env.step_from_state(perturbed_state, action)

        # Train adversary to minimize agent's reward
        adversary_reward = -reward
        self.adversary.update(adversary_reward)

        return next_state, reward, done
```

### 7. Verification and Interpretability

**Formal Verification:**

- Prove safety properties mathematically
- Check all possible state-action pairs
- Intractable for complex systems

**Neural Network Verification:**

```python
def verify_safety_property(policy, safety_property):
    """
    Verify policy satisfies safety property

    Methods:
    - Abstract interpretation
    - SMT solvers
    - Reachability analysis
    """
    # Example: Verify action always in safe range
    for state in discretize_state_space():
        action = policy(state)

        # Use interval arithmetic to bound output
        action_bounds = compute_output_bounds(policy, state)

        if not safety_property.satisfied(action_bounds):
            return False, state  # Counterexample

    return True, None
```

**Interpretable Policies:**

- Decision trees (extract from neural network)
- Linear models (interpret coefficients)
- Attention mechanisms (explain decisions)

### 8. Real-World Applications

**Autonomous Driving:**

- Must not collide with other vehicles
- Obey traffic laws
- Handle diverse scenarios

**Healthcare:**

- Patient safety is paramount
- Conservative treatment recommendations
- Explainable decisions

**Robotics:**

- Physical safety around humans
- Damage prevention to robot/environment
- Graceful degradation

**Finance:**

- Risk limits (VaR, CVaR)
- Regulatory compliance
- Worst-case analysis

## 📊 Topics Covered

1. **CMDPs**: Constrained optimization framework
2. **CPO**: Constrained policy optimization
3. **Safety Layers**: Runtime shielding
4. **Risk-Sensitive**: CVaR, entropic risk
5. **Safe Exploration**: Barrier functions
6. **Robustness**: Domain randomization, adversarial training
7. **Verification**: Formal methods for safety

## 📖 Key References

1. **Achiam, J., et al. (2017)** - "Constrained Policy Optimization" - ICML (CPO)

2. **García, J., & Fernández, F. (2015)** - "A Comprehensive Survey on Safe RL" - JMLR

3. **Tamar, A., et al. (2015)** - "Sequential Decision Making with Coherent Risk Measures" - arXiv

4. **Alshiekh, M., et al. (2018)** - "Safe RL via Shielding" - AAAI

5. **Shalev-Shwartz, S., et al. (2016)** - "Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving" - arXiv

## 💡 Discussion Questions

1. What are fundamental trade-offs between safety and performance in RL?
2. How can we ensure safety during exploration, not just exploitation?
3. When should we use hard constraints vs risk-sensitive objectives?
4. How can formal verification scale to complex neural policies?
5. What safety guarantees can we realistically achieve in practice?

## 🎓 Extensions

- Implement safe exploration with Gaussian Processes
- Try worst-case robust MDP formulation
- Explore safe transfer learning
- Study reward uncertainty and safe reward design
- Apply to sim-to-real robot transfer

---

**Course:** Deep Reinforcement Learning  
**Last Updated:** 2024

**⚠️ Note:** Safety is crucial for real-world RL deployment. Always validate safety properties thoroughly before deploying learned policies in critical applications.
