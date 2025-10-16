# HW11: Meta-Learning in Reinforcement Learning - Complete Solutions

**Course:** Deep Reinforcement Learning  
**Assignment:** Homework 11 - Meta-Learning  
**Date:** October 2024  
**Format:** IEEE Style

---

## Table of Contents

1. [Theoretical Questions](#theoretical-questions)
2. [MAML Implementation](#maml-implementation)
3. [Recurrent Meta-RL](#recurrent-meta-rl)
4. [Context-Based Meta-RL](#context-based-meta-rl)
5. [Experimental Analysis](#experimental-analysis)
6. [Discussion Questions](#discussion-questions)

---

## SECTION I: THEORETICAL QUESTIONS

### Question 1: Meta-Learning Problem Formulation

**Q1.1:** Define the meta-learning problem in reinforcement learning. What distinguishes it from standard RL and multi-task learning?

**Solution:**

Meta-learning in RL, also known as "learning to learn," is a paradigm where an agent learns across a distribution of tasks $p(\mathcal{T})$ to enable fast adaptation to new tasks from the same distribution.

**Key Components:**

- **Task Distribution:** $p(\mathcal{T})$, where each task $\mathcal{T}_i$ consists of:

  - State space $\mathcal{S}_i$
  - Action space $\mathcal{A}_i$
  - Reward function $R_i$
  - Transition dynamics $P_i$

- **Meta-Training:** Sample tasks from $p(\mathcal{T})$, learn meta-parameters $\theta$
- **Meta-Testing:** Quickly adapt to new task $\mathcal{T}_{new} \sim p(\mathcal{T})$ using few samples

**Differences from Standard RL:**

| Aspect             | Standard RL                  | Meta-RL                             |
| ------------------ | ---------------------------- | ----------------------------------- |
| Training           | Single task                  | Multiple related tasks              |
| Objective          | Maximize return on one task  | Enable fast adaptation across tasks |
| Evaluation         | Performance on training task | Few-shot adaptation to new tasks    |
| Knowledge Transfer | Limited                      | Explicit transfer mechanism         |

**Differences from Multi-Task Learning:**

| Aspect         | Multi-Task RL                       | Meta-RL                                |
| -------------- | ----------------------------------- | -------------------------------------- |
| Goal           | Solve multiple tasks simultaneously | Learn to quickly adapt to new tasks    |
| Test Time      | Use same policy for all tasks       | Adapt policy for each new task         |
| Generalization | To seen tasks                       | To unseen tasks from same distribution |

---

**Q1.2:** Describe the two-level optimization structure in meta-learning. What is the role of inner and outer loops?

**Solution:**

Meta-learning employs a nested optimization structure:

**Inner Loop (Task-Level Adaptation):**

$$\phi_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$$

- **Purpose:** Adapt meta-parameters $\theta$ to specific task $\mathcal{T}_i$
- **Learning Rate:** $\alpha$ (inner learning rate, typically larger)
- **Objective:** Task-specific loss $\mathcal{L}_{\mathcal{T}_i}$
- **Steps:** $K$ gradient steps (usually 1-5)

**Outer Loop (Meta-Level Optimization):**

$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{i=1}^{n} \mathcal{L}_{\mathcal{T}_i}(\phi_i)$$

- **Purpose:** Optimize initialization for fast adaptation
- **Learning Rate:** $\beta$ (meta learning rate, typically smaller)
- **Objective:** Meta-loss across tasks
- **Gradient:** Through adapted parameters $\phi_i$

**Key Insight:** The outer loop optimizes $\theta$ such that one or few gradient steps in the inner loop lead to good performance on new tasks.

---

### Question 2: Model-Agnostic Meta-Learning (MAML)

**Q2.1:** Explain the MAML algorithm. What is the key innovation that enables fast adaptation?

**Solution:**

MAML (Finn et al., 2017) is a gradient-based meta-learning algorithm that learns an initialization for model parameters that can be quickly fine-tuned to new tasks.

**Algorithm:**

```
Input: Task distribution p(T), learning rates α, β
Output: Meta-parameters θ

1. Initialize θ randomly
2. For meta-iteration = 1 to N:
   3. Sample batch of tasks {T_i} ~ p(T)
   4. For each task T_i:
      5. Sample K data points D_i^train from T_i
      6. Compute adapted parameters:
         φ_i = θ - α∇_θ L_{T_i}(θ, D_i^train)
      7. Sample data D_i^test from T_i
      8. Compute meta-loss: L_{T_i}(φ_i, D_i^test)
   9. Meta-update:
      θ ← θ - β∇_θ Σ_i L_{T_i}(φ_i, D_i^test)
```

**Key Innovation:**

The crucial innovation is **backpropagation through the optimization process**. MAML computes gradients with respect to the pre-update parameters $\theta$ by differentiating through the inner loop update:

$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\phi_i) = \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta))$$

This requires computing **second-order derivatives**, which is computationally expensive but enables finding an initialization that is in a region of parameter space where small gradient steps lead to large improvements.

---

**Q2.2:** Describe the computational challenges of MAML in RL settings and propose solutions.

**Solution:**

**Challenges:**

1. **High Variance:**

   - RL gradients have high variance due to policy gradient estimation
   - Compounded across inner and outer loops
   - Makes meta-optimization unstable

2. **Second-Order Derivatives:**

   - Computing Hessian-vector products is expensive
   - Memory intensive for large neural networks
   - Slows down training significantly

3. **Sample Inefficiency:**

   - Need multiple rollouts per task for both inner and outer loops
   - Each task requires fresh trajectories
   - Total sample requirement is $O(\text{tasks} \times \text{episodes per task})$

4. **Credit Assignment:**
   - Difficult to attribute success/failure to meta-parameters vs. adaptation
   - Long horizon RL makes this worse

**Solutions:**

1. **First-Order MAML (FOMAML):**

   - Ignore second-order derivatives
   - Approximate: $\nabla_\theta \mathcal{L}(\phi_i) \approx \nabla_{\phi_i} \mathcal{L}(\phi_i)$
   - Much faster, often comparable performance

2. **Reptile Algorithm:**

   - Even simpler first-order method
   - Update direction: $\theta \leftarrow \theta + \epsilon(\phi_i - \theta)$
   - No meta-gradient computation needed

3. **Use PPO Instead of Vanilla Policy Gradient:**

   - Clipped objective reduces variance
   - More stable optimization
   - Better for both inner and outer loops

4. **Increase Inner Learning Rate:**

   - Larger $\alpha$ means faster adaptation
   - Compensates for fewer inner steps
   - Reduces computational cost

5. **Task Batching:**
   - Process multiple tasks in parallel
   - Amortize computation cost
   - Better GPU utilization

---

### Question 3: Recurrent Meta-RL

**Q3.1:** Explain how RL² achieves meta-learning without explicit inner loop adaptation. What role does the recurrent hidden state play?

**Solution:**

RL² (Duan et al., 2016) takes a fundamentally different approach to meta-learning by treating the entire RL procedure as a "computation" performed by a recurrent network.

**Key Concept:**

Instead of explicitly optimizing for fast adaptation, RL² trains a recurrent policy that:

- Maintains a hidden state across episodes
- Uses this hidden state to encode task information
- Learns to adapt its behavior based on experience within the current task

**Hidden State as Task Representation:**

The LSTM hidden state $h_t$ serves as a sufficient statistic for the task:

$$h_t = \text{LSTM}([o_t, a_{t-1}, r_{t-1}, d_{t-1}], h_{t-1})$$

where:

- $o_t$: current observation
- $a_{t-1}$: previous action
- $r_{t-1}$: previous reward
- $d_{t-1}$: done flag
- $h_{t-1}$: previous hidden state

**How Adaptation Occurs:**

1. **Initially:** Policy behaves according to prior (encoded in initial weights)
2. **Early Episodes:** Exploration to discover task structure, encoded in $h_t$
3. **Later Episodes:** Exploitation using learned task representation
4. **Hidden State Updates:** Gradual refinement of task understanding

**Training Procedure:**

```python
# Training RL²
for meta_iteration in range(N):
    task = sample_task()
    h = initialize_hidden_state()

    trajectories = []
    # Collect multiple episodes from same task
    for episode in range(M):  # M typically 10-50
        trajectory, h = collect_episode(
            task, policy, h,
            reset_hidden=False  # Critical!
        )
        trajectories.append(trajectory)

    # Standard RL update (e.g., PPO) on all trajectories
    loss = compute_rl_loss(trajectories)
    optimize(loss)
```

**Key Insight:** The network learns to implement its own adaptation algorithm through its recurrent dynamics. The hidden state accumulates evidence about the task and adjusts behavior accordingly.

**Advantages:**

- No explicit inner loop (faster at test time)
- Single forward pass for action selection
- Can handle variable episode lengths
- Naturally implements exploration-exploitation trade-off

**Limitations:**

- Requires many episodes per task during training
- Limited by LSTM capacity (memory bottleneck)
- Black-box adaptation (harder to interpret)
- Less sample efficient than gradient-based methods

---

**Q3.2:** Design an RL² architecture for a continuous control task. Specify the input/output dimensions and network architecture.

**Solution:**

**Task:** Continuous control (e.g., Half-Cheetah with varying dynamics)

**Architecture:**

```python
import torch
import torch.nn as nn

class RL2Policy(nn.Module):
    def __init__(
        self,
        obs_dim=17,        # HalfCheetah observation
        action_dim=6,      # HalfCheetah action
        hidden_dim=256,
        num_lstm_layers=2
    ):
        super().__init__()

        # Input dimension: obs + prev_action + prev_reward + done
        input_dim = obs_dim + action_dim + 1 + 1

        # Recurrent encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Policy head (Gaussian policy for continuous actions)
        self.policy_mean = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.policy_logstd = nn.Parameter(
            torch.zeros(action_dim)
        )

        # Value head
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, prev_action, prev_reward, done, hidden):
        """
        Args:
            obs: (batch, obs_dim)
            prev_action: (batch, action_dim)
            prev_reward: (batch,)
            done: (batch,)
            hidden: tuple of (h, c) each (num_layers, batch, hidden_dim)

        Returns:
            action_mean: (batch, action_dim)
            action_std: (batch, action_dim)
            value: (batch, 1)
            hidden_new: updated hidden state
        """
        batch_size = obs.shape[0]

        # Concatenate inputs
        x = torch.cat([
            obs,
            prev_action,
            prev_reward.unsqueeze(-1),
            done.unsqueeze(-1).float()
        ], dim=-1)

        # LSTM forward
        x = x.unsqueeze(1)  # Add sequence dimension
        output, hidden_new = self.lstm(x, hidden)
        output = output.squeeze(1)

        # Policy outputs
        action_mean = self.policy_mean(output)
        action_std = torch.exp(self.policy_logstd).expand_as(action_mean)

        # Value output
        value = self.value(output)

        return action_mean, action_std, value, hidden_new

    def init_hidden(self, batch_size=1, device='cpu'):
        """Initialize hidden state for new task"""
        return (
            torch.zeros(self.lstm.num_layers, batch_size,
                       self.lstm.hidden_size).to(device),
            torch.zeros(self.lstm.num_layers, batch_size,
                       self.lstm.hidden_size).to(device)
        )

    def sample_action(self, obs, prev_action, prev_reward, done, hidden):
        """Sample action from policy"""
        mean, std, value, hidden_new = self.forward(
            obs, prev_action, prev_reward, done, hidden
        )

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob, value, hidden_new
```

**Input/Output Specifications:**

| Component        | Dimension       | Description                      |
| ---------------- | --------------- | -------------------------------- |
| Observation      | 17              | Joint angles, velocities         |
| Previous Action  | 6               | Torques applied in previous step |
| Previous Reward  | 1               | Scalar reward                    |
| Done Flag        | 1               | Episode termination              |
| **Total Input**  | **25**          | Concatenated                     |
| Hidden State (h) | (2, batch, 256) | LSTM hidden                      |
| Cell State (c)   | (2, batch, 256) | LSTM cell                        |
| Action Mean      | 6               | Policy mean                      |
| Action Std       | 6               | Policy standard deviation        |
| Value            | 1               | State value estimate             |

**Training Details:**

- **Episodes per Task:** 20-50
- **Reset Hidden State:** Only between tasks, not between episodes
- **Optimization:** PPO with 10-20 epochs per meta-batch
- **Meta-Batch Size:** 20-40 tasks
- **Horizon:** 200 steps per episode

---

## SECTION II: MAML IMPLEMENTATION

### Question 4: MAML for RL

**Q4.1:** Implement the MAML algorithm for a simple RL task (e.g., navigation with varying goals).

**Solution:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

class PolicyNetwork(nn.Module):
    """Simple MLP policy for MAML"""
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class MAML_RL:
    def __init__(
        self,
        obs_dim,
        action_dim,
        inner_lr=0.1,
        meta_lr=0.001,
        inner_steps=1,
        gamma=0.99
    ):
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.meta_optimizer = optim.Adam(
            self.policy.parameters(), lr=meta_lr
        )

        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.gamma = gamma

    def compute_returns(self, rewards, gamma):
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def collect_trajectory(self, env, policy, max_steps=200):
        """Collect one trajectory"""
        states, actions, rewards, log_probs = [], [], [], []

        state = env.reset()
        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Get action logits
            logits = policy(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state
            if done:
                break

        returns = self.compute_returns(rewards, self.gamma)

        return {
            'states': torch.FloatTensor(states),
            'actions': torch.stack(actions),
            'returns': returns,
            'log_probs': torch.stack(log_probs)
        }

    def compute_policy_loss(self, trajectory, policy):
        """Compute policy gradient loss"""
        states = trajectory['states']
        actions = trajectory['actions']
        returns = trajectory['returns']

        logits = policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        loss = -(log_probs * returns).mean()
        return loss

    def inner_loop_update(self, task_env, policy):
        """Perform inner loop adaptation"""
        # Clone current parameters
        adapted_policy = PolicyNetwork(
            policy.network[0].in_features,
            policy.network[-1].out_features
        )
        adapted_policy.load_state_dict(policy.state_dict())

        for step in range(self.inner_steps):
            # Collect trajectories
            trajectories = [
                self.collect_trajectory(task_env, adapted_policy)
                for _ in range(5)  # 5 trajectories per inner step
            ]

            # Compute loss
            total_loss = sum(
                self.compute_policy_loss(traj, adapted_policy)
                for traj in trajectories
            ) / len(trajectories)

            # Compute gradients
            grads = torch.autograd.grad(
                total_loss, adapted_policy.parameters(),
                create_graph=True  # Enable second-order derivatives
            )

            # Manual SGD update
            with torch.no_grad():
                for param, grad in zip(adapted_policy.parameters(), grads):
                    param.data = param.data - self.inner_lr * grad.data

        return adapted_policy

    def meta_train_step(self, task_envs):
        """One meta-training step"""
        meta_loss = 0

        for task_env in task_envs:
            # Inner loop: adapt to task
            adapted_policy = self.inner_loop_update(task_env, self.policy)

            # Collect test trajectories with adapted policy
            test_trajectories = [
                self.collect_trajectory(task_env, adapted_policy)
                for _ in range(10)  # More trajectories for meta-loss
            ]

            # Compute meta-loss
            task_loss = sum(
                self.compute_policy_loss(traj, adapted_policy)
                for traj in test_trajectories
            ) / len(test_trajectories)

            meta_loss += task_loss

        meta_loss = meta_loss / len(task_envs)

        # Meta-optimization step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt_to_new_task(self, task_env, num_adapt_steps=5):
        """Adapt to new task at test time"""
        # Clone current policy
        adapted_policy = PolicyNetwork(
            self.policy.network[0].in_features,
            self.policy.network[-1].out_features
        )
        adapted_policy.load_state_dict(self.policy.state_dict())

        optimizer = optim.SGD(
            adapted_policy.parameters(), lr=self.inner_lr
        )

        for _ in range(num_adapt_steps):
            trajectories = [
                self.collect_trajectory(task_env, adapted_policy)
                for _ in range(3)
            ]

            loss = sum(
                self.compute_policy_loss(traj, adapted_policy)
                for traj in trajectories
            ) / len(trajectories)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_policy

# Example usage
if __name__ == "__main__":
    # Create task distribution (e.g., navigation with different goals)
    def create_task_env(goal_position):
        env = gym.make('CartPole-v1')  # Placeholder
        env.goal = goal_position
        return env

    # Training
    maml = MAML_RL(
        obs_dim=4,
        action_dim=2,
        inner_lr=0.1,
        meta_lr=0.001,
        inner_steps=1
    )

    for meta_iter in range(100):
        # Sample batch of tasks
        task_envs = [
            create_task_env(np.random.uniform(-1, 1))
            for _ in range(20)
        ]

        meta_loss = maml.meta_train_step(task_envs)

        if meta_iter % 10 == 0:
            print(f"Meta-iteration {meta_iter}, Loss: {meta_loss:.4f}")
```

---

**Q4.2:** Compare MAML with first-order MAML (FOMAML). What are the trade-offs?

**Solution:**

**First-Order MAML (FOMAML)** simplifies MAML by ignoring second-order derivatives.

**Key Modification:**

In standard MAML:
$$\nabla_\theta \mathcal{L}(\phi_i) = \nabla_\theta \mathcal{L}(\theta - \alpha \nabla_\theta \mathcal{L}_{\text{train}}(\theta))$$

In FOMAML:
$$\nabla_\theta \mathcal{L}(\phi_i) \approx \nabla_{\phi_i} \mathcal{L}(\phi_i)$$

**Implementation Difference:**

```python
# MAML
grads = torch.autograd.grad(
    loss, policy.parameters(),
    create_graph=True  # Enable second-order derivatives
)

# FOMAML
grads = torch.autograd.grad(
    loss, policy.parameters(),
    create_graph=False  # Disable second-order derivatives
)
```

**Comparative Analysis:**

| Aspect                | MAML                       | FOMAML                   |
| --------------------- | -------------------------- | ------------------------ |
| **Computation Time**  | High (2nd order)           | Low (1st order only)     |
| **Memory Usage**      | High (computational graph) | Low                      |
| **Gradient Accuracy** | Exact                      | Approximation            |
| **Convergence**       | Potentially faster         | May need more iterations |
| **Implementation**    | Complex                    | Simple                   |
| **Performance**       | Slightly better            | Often comparable         |

**Trade-Offs:**

1. **Computational Cost:**

   - MAML: $O(N \cdot M \cdot K)$ where K is inner steps
   - FOMAML: $O(N \cdot M)$
   - FOMAML is typically 2-3x faster

2. **Performance:**

   - In many tasks, FOMAML achieves 80-90% of MAML's performance
   - Gap narrows with more inner steps
   - Task-dependent: some tasks benefit more from exact gradients

3. **Scalability:**
   - FOMAML scales better to larger networks
   - Can use deeper networks with FOMAML
   - MAML limited by memory for deep networks

**Experimental Results:**

```python
# Pseudo-results from typical experiments
results = {
    'MAML': {
        'accuracy': 0.82,
        'time_per_iter': 45.3,  # seconds
        'memory': 8.2  # GB
    },
    'FOMAML': {
        'accuracy': 0.78,  # 95% of MAML
        'time_per_iter': 16.7,  # 3x faster
        'memory': 2.1  # GB
    }
}
```

**Recommendation:**

- Start with FOMAML for rapid prototyping
- Use MAML if:
  - Peak performance is critical
  - Task requires precise gradients
  - Sufficient computational resources available

---

## SECTION III: CONTEXT-BASED META-RL

### Question 5: PEARL Algorithm

**Q5.1:** Explain the PEARL algorithm. How does it achieve probabilistic task inference?

**Solution:**

PEARL (Probabilistic Embeddings for Actor-critic RL, Rakelly et al., 2019) is an off-policy meta-RL algorithm that learns task embeddings through a variational approach.

**Key Components:**

1. **Context Encoder:** Maps transitions to task embedding distribution
2. **Policy:** Conditioned on task embedding
3. **Q-Function:** Conditioned on task embedding
4. **Variational Inference:** Probabilistic task representation

**Architecture:**

**1. Context Encoder (Variational):**

$$q_\psi(z|\mathcal{C}) = \mathcal{N}(\mu_\psi(\mathcal{C}), \sigma_\psi^2(\mathcal{C}))$$

where $\mathcal{C} = \{(s, a, r, s')\}$ is the context (set of transitions)

```python
class ContextEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, context_dim):
        super().__init__()

        input_dim = obs_dim + action_dim + 1  # s, a, r

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.mean = nn.Linear(128, context_dim)
        self.logstd = nn.Linear(128, context_dim)

    def forward(self, context):
        """
        Args:
            context: (batch, context_size, obs_dim + action_dim + 1)
        Returns:
            mean: (batch, context_dim)
            std: (batch, context_dim)
        """
        # Encode each transition
        encoded = self.encoder(context)

        # Aggregate (permutation invariant)
        aggregated = encoded.mean(dim=1)

        mean = self.mean(aggregated)
        std = torch.exp(self.logstd(aggregated))

        return mean, std
```

**2. Context-Conditioned Policy:**

$$\pi_\theta(a|s, z)$$

```python
class ContextPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, context_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs, context):
        x = torch.cat([obs, context], dim=-1)
        return self.network(x)
```

**3. Context-Conditioned Q-Function:**

$$Q_\phi(s, a, z)$$

**Training Procedure:**

```python
class PEARL:
    def __init__(self, obs_dim, action_dim, context_dim):
        self.context_encoder = ContextEncoder(obs_dim, action_dim, context_dim)
        self.policy = ContextPolicy(obs_dim, action_dim, context_dim)
        self.q_func = ContextQFunction(obs_dim, action_dim, context_dim)
        self.target_q = ContextQFunction(obs_dim, action_dim, context_dim)

        # Replay buffer per task
        self.replay_buffers = {}

    def meta_train_step(self, task):
        # 1. Sample context from task
        context_batch = self.replay_buffers[task].sample(K)

        # 2. Encode task
        mean, std = self.context_encoder(context_batch)
        z = mean + std * torch.randn_like(std)  # Sample z ~ q(z|C)

        # 3. Sample transitions for SAC update
        batch = self.replay_buffers[task].sample(N)

        # 4. SAC update with context
        q_loss = self.compute_q_loss(batch, z)
        policy_loss = self.compute_policy_loss(batch, z)

        # 5. Context encoder loss (KL divergence)
        kl_loss = 0.5 * (mean**2 + std**2 - torch.log(std**2) - 1).sum(-1).mean()

        # 6. Total loss
        total_loss = q_loss + policy_loss + self.beta * kl_loss

        return total_loss

    def adapt(self, new_task, num_context=10):
        """Few-shot adaptation to new task"""
        # Collect minimal context
        context = collect_transitions(new_task, num=num_context)

        # Infer task embedding (use mean at test time)
        mean, _ = self.context_encoder(context)

        # Return adapted policy
        return lambda obs: self.policy(obs, mean)
```

**Probabilistic Task Inference:**

PEARL uses variational inference to learn a distribution over task embeddings:

1. **Encoder learns:** $q_\psi(z|\mathcal{C})$ - approximate posterior
2. **Prior:** $p(z) = \mathcal{N}(0, I)$
3. **KL Regularization:** $\text{KL}(q_\psi(z|\mathcal{C}) \| p(z))$

This ensures:

- Task embeddings are structured (regularized by KL)
- Uncertainty quantification (via variance)
- Smooth interpolation between tasks
- Prevents overfitting to specific tasks

**Advantages over MAML:**

| Aspect            | MAML            | PEARL               |
| ----------------- | --------------- | ------------------- |
| Adaptation        | Gradient-based  | Context encoding    |
| Speed             | Slow (backprop) | Fast (forward pass) |
| Sample Efficiency | On-policy       | Off-policy (better) |
| Probabilistic     | No              | Yes (uncertainty)   |
| Scalability       | Limited         | Better              |

---

### Question 6: Task Embeddings

**Q6.1:** Design an experiment to visualize learned task embeddings. What structure would you expect to observe?

**Solution:**

**Experimental Setup:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class TaskEmbeddingVisualizer:
    def __init__(self, pearl_model, task_distribution):
        self.model = pearl_model
        self.tasks = task_distribution

    def collect_task_embeddings(self, num_tasks=100, context_size=10):
        """
        Collect embeddings for multiple tasks
        """
        embeddings = []
        task_labels = []
        task_params = []

        for i in range(num_tasks):
            # Sample task
            task = self.tasks.sample()
            task_params.append(task.get_parameters())

            # Collect context
            context = self.collect_context(task, context_size)

            # Encode task (use mean of distribution)
            with torch.no_grad():
                mean, std = self.model.context_encoder(context)

            embeddings.append(mean.cpu().numpy())
            task_labels.append(self.tasks.get_task_type(task))

        return np.array(embeddings), task_labels, task_params

    def visualize_embeddings(self, embeddings, labels, method='tsne'):
        """
        Visualize high-dimensional embeddings in 2D
        """
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # PCA
            reducer = PCA(n_components=2)

        embeddings_2d = reducer.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, label='Task Type')
        plt.title(f'Task Embeddings ({method.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'task_embeddings_{method}.png', dpi=300)
        plt.show()

    def analyze_embedding_structure(self, embeddings, task_params):
        """
        Analyze relationship between embeddings and task parameters
        """
        results = {}

        # 1. Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(embeddings, metric='euclidean'))
        results['distances'] = distances

        # 2. Correlation with task parameters
        for param_name in task_params[0].keys():
            param_values = np.array([p[param_name] for p in task_params])

            # Compute correlation between embedding distances
            # and parameter differences
            param_diffs = squareform(pdist(param_values.reshape(-1, 1)))

            correlation = np.corrcoef(
                distances.flatten(),
                param_diffs.flatten()
            )[0, 1]

            results[f'correlation_{param_name}'] = correlation

        # 3. Cluster analysis
        from sklearn.cluster import KMeans
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        results['clusters'] = cluster_labels

        return results
```

**Expected Structure:**

1. **Clustering by Task Type:**

   - Tasks with similar dynamics should cluster together
   - Clear separation between distinct task families
   - Example: Different goal positions in navigation tasks

2. **Smooth Manifold:**

   - Continuous variation along task parameters
   - Interpolation between similar tasks
   - No discontinuous jumps in embedding space

3. **Hierarchical Organization:**

   - Coarse structure: Major task categories
   - Fine structure: Within-category variations
   - Example: {Push, Pick, Place} at coarse level, goal positions at fine level

4. **Parameter Correlation:**
   - High correlation between embedding distance and:
     - Goal position differences
     - Dynamics parameter differences
     - Reward function variations

**Visualization Example:**

```python
# Example experiment
visualizer = TaskEmbeddingVisualizer(pearl_model, task_dist)

# Collect embeddings
embeddings, labels, params = visualizer.collect_task_embeddings(
    num_tasks=200,
    context_size=20
)

# Visualize with t-SNE
visualizer.visualize_embeddings(embeddings, labels, method='tsne')

# Analyze structure
analysis = visualizer.analyze_embedding_structure(embeddings, params)

print(f"Correlation with goal_x: {analysis['correlation_goal_x']:.3f}")
print(f"Correlation with goal_y: {analysis['correlation_goal_y']:.3f}")
print(f"Number of natural clusters: {len(np.unique(analysis['clusters']))}")
```

**Expected Results:**

For well-trained PEARL on navigation tasks:

- Goal position correlation: 0.7-0.9 (high)
- Clear clusters for different regions
- Smooth transitions between nearby goals
- Low-dimensional manifold structure (even with high-dim embedding)

---

## SECTION IV: EXPERIMENTAL ANALYSIS

### Question 7: Meta-World Benchmark

**Q7.1:** Describe the Meta-World benchmark. What makes it suitable for evaluating meta-RL algorithms?

**Solution:**

**Meta-World** (Yu et al., 2020) is a standardized benchmark for meta-RL consisting of 50 robotic manipulation tasks built on the MuJoCo physics simulator.

**Key Characteristics:**

1. **Task Diversity:**

   - 50 distinct tasks (10 task families × 5 variations)
   - Manipulation primitives: Reach, Push, Pick-Place, Door, Drawer, Button, etc.
   - Varying goals, object positions, and interaction modes

2. **Shared Structure:**

   - Common observation space (39-dim)
   - Common action space (4-dim: 3D position + gripper)
   - Shared state representation across tasks
   - Enables transfer learning

3. **Evaluation Protocols:**

   **MT10 (Multi-Task 10):**

   - Train on 10 tasks simultaneously
   - Evaluate on same 10 tasks
   - Tests multi-task learning

   **ML10 (Meta-Learning 10):**

   - Meta-train on 10 tasks
   - Meta-test on new instances of same task types
   - Tests few-shot adaptation

   **ML45:**

   - Meta-train on 45 tasks
   - Meta-test on 5 held-out task types
   - Tests generalization to new task types

4. **Success Metrics:**
   - Binary success rate (task-specific criteria)
   - Adaptation curves (success vs. number of samples)
   - Sample efficiency during adaptation

**Why It's Suitable:**

| Property             | Benefit for Meta-RL Evaluation                   |
| -------------------- | ------------------------------------------------ |
| **Standardized**     | Fair comparison across algorithms                |
| **Realistic**        | Complex manipulation (not toy problems)          |
| **Diverse**          | Tests generalization across task distribution    |
| **Shared Structure** | Enables transfer (key for meta-RL)               |
| **Difficult**        | Challenges current methods, room for improvement |
| **Reproducible**     | Open-source, deterministic environments          |

**Example Tasks:**

```python
import metaworld

# Load benchmark
ml10 = metaworld.ML10()

# Training tasks
train_tasks = ml10.train_classes
print(f"Training tasks: {list(train_tasks.keys())}")
# ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2',
#  'drawer-open-v2', 'drawer-close-v2', 'button-press-topdown-v2',
#  'window-open-v2', 'window-close-v2', 'peg-insert-side-v2']

# Sample task
env = ml10.train_classes['reach-v2']()
env.set_task(ml10.train_tasks[0])

obs = env.reset()
print(f"Observation space: {obs.shape}")  # (39,)
print(f"Action space: {env.action_space.shape}")  # (4,)
```

**Evaluation Procedure:**

```python
def evaluate_meta_rl(algorithm, ml10, num_adapt_steps=[0, 10, 50, 100]):
    """
    Evaluate meta-RL algorithm on ML10
    """
    results = {k: [] for k in num_adapt_steps}

    for task_idx, task in enumerate(ml10.test_tasks):
        # Create environment
        env_class = ml10.test_classes[task.env_name]
        env = env_class()
        env.set_task(task)

        for num_steps in num_adapt_steps:
            # Adapt to task
            if num_steps > 0:
                algorithm.adapt(env, num_steps=num_steps)

            # Evaluate
            success_rate = evaluate_policy(
                algorithm.policy, env, num_episodes=10
            )

            results[num_steps].append(success_rate)

    # Average across tasks
    for num_steps in num_adapt_steps:
        avg_success = np.mean(results[num_steps])
        print(f"Steps: {num_steps:3d}, Success: {avg_success:.3f}")

    return results
```

---

**Q7.2:** Implement an evaluation protocol for few-shot adaptation on Meta-World. Plot adaptation curves.

**Solution:**

```python
import metaworld
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class MetaRLEvaluator:
    def __init__(self, algorithm, benchmark='ML10'):
        self.algorithm = algorithm

        if benchmark == 'ML10':
            self.meta_world = metaworld.ML10()
        elif benchmark == 'ML45':
            self.meta_world = metaworld.ML45()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    def evaluate_adaptation(
        self,
        task,
        adaptation_steps=[0, 1, 5, 10, 25, 50, 100],
        num_eval_episodes=20,
        seed=42
    ):
        """
        Evaluate adaptation curve for single task

        Returns:
            dict: {num_steps: (mean_success, std_success, mean_return, std_return)}
        """
        # Create environment
        env_class = self.meta_world.test_classes[task.env_name]
        env = env_class()
        env.set_task(task)
        env.seed(seed)

        results = {}

        for num_steps in adaptation_steps:
            # Reset algorithm to meta-initialization
            self.algorithm.reset()

            # Adaptation phase
            if num_steps > 0:
                adapt_successes, adapt_returns = self.adaptation_phase(
                    env, num_steps
                )
            else:
                adapt_successes, adapt_returns = [], []

            # Evaluation phase
            eval_successes, eval_returns = self.evaluation_phase(
                env, num_eval_episodes
            )

            results[num_steps] = {
                'eval_success_mean': np.mean(eval_successes),
                'eval_success_std': np.std(eval_successes),
                'eval_return_mean': np.mean(eval_returns),
                'eval_return_std': np.std(eval_returns),
                'adapt_successes': adapt_successes,
                'adapt_returns': adapt_returns
            }

        return results

    def adaptation_phase(self, env, num_steps):
        """
        Collect data and adapt to task
        """
        successes = []
        returns = []

        for step in range(num_steps):
            obs = env.reset()
            done = False
            episode_return = 0

            while not done:
                # Select action
                action = self.algorithm.select_action(obs, explore=True)
                next_obs, reward, done, info = env.step(action)

                # Store transition
                self.algorithm.store_transition(
                    obs, action, reward, next_obs, done
                )

                episode_return += reward
                obs = next_obs

            successes.append(float(info['success']))
            returns.append(episode_return)

            # Update policy (online adaptation)
            if step % 5 == 0:  # Update every 5 episodes
                self.algorithm.update()

        return successes, returns

    def evaluation_phase(self, env, num_episodes):
        """
        Evaluate adapted policy
        """
        successes = []
        returns = []

        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            episode_return = 0

            while not done:
                # Select action (no exploration)
                action = self.algorithm.select_action(obs, explore=False)
                obs, reward, done, info = env.step(action)
                episode_return += reward

            successes.append(float(info['success']))
            returns.append(episode_return)

        return successes, returns

    def evaluate_all_tasks(
        self,
        adaptation_steps=[0, 1, 5, 10, 25, 50],
        num_tasks=10,
        seeds=[42, 43, 44]
    ):
        """
        Evaluate on multiple tasks and seeds
        """
        all_results = defaultdict(lambda: defaultdict(list))

        tasks = self.meta_world.test_tasks[:num_tasks]

        for task_idx, task in enumerate(tasks):
            print(f"Evaluating task {task_idx+1}/{num_tasks}: {task.env_name}")

            for seed in seeds:
                results = self.evaluate_adaptation(
                    task, adaptation_steps, seed=seed
                )

                for num_steps, metrics in results.items():
                    all_results[num_steps]['success'].append(
                        metrics['eval_success_mean']
                    )
                    all_results[num_steps]['return'].append(
                        metrics['eval_return_mean']
                    )

        # Aggregate results
        aggregated = {}
        for num_steps in adaptation_steps:
            aggregated[num_steps] = {
                'success_mean': np.mean(all_results[num_steps]['success']),
                'success_std': np.std(all_results[num_steps]['success']),
                'return_mean': np.mean(all_results[num_steps]['return']),
                'return_std': np.std(all_results[num_steps]['return'])
            }

        return aggregated

    def plot_adaptation_curves(self, results, save_path='adaptation_curves.png'):
        """
        Plot success rate and return vs. adaptation steps
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        steps = sorted(results.keys())

        # Success rate curve
        success_means = [results[s]['success_mean'] for s in steps]
        success_stds = [results[s]['success_std'] for s in steps]

        ax1.plot(steps, success_means, 'o-', linewidth=2, markersize=8)
        ax1.fill_between(
            steps,
            np.array(success_means) - np.array(success_stds),
            np.array(success_means) + np.array(success_stds),
            alpha=0.3
        )
        ax1.set_xlabel('Adaptation Steps', fontsize=12)
        ax1.set_ylabel('Success Rate', fontsize=12)
        ax1.set_title('Few-Shot Adaptation: Success Rate', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # Return curve
        return_means = [results[s]['return_mean'] for s in steps]
        return_stds = [results[s]['return_std'] for s in steps]

        ax2.plot(steps, return_means, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.fill_between(
            steps,
            np.array(return_means) - np.array(return_stds),
            np.array(return_means) + np.array(return_stds),
            alpha=0.3,
            color='orange'
        )
        ax2.set_xlabel('Adaptation Steps', fontsize=12)
        ax2.set_ylabel('Average Return', fontsize=12)
        ax2.set_title('Few-Shot Adaptation: Return', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Adaptation curves saved to {save_path}")

# Example usage
if __name__ == "__main__":
    # Assume we have a trained meta-RL algorithm
    from my_algorithm import PEARL  # or MAML, RL2, etc.

    algorithm = PEARL.load('trained_model.pt')
    evaluator = MetaRLEvaluator(algorithm, benchmark='ML10')

    # Evaluate adaptation
    results = evaluator.evaluate_all_tasks(
        adaptation_steps=[0, 1, 5, 10, 25, 50, 100],
        num_tasks=10,
        seeds=[42, 43, 44, 45, 46]
    )

    # Plot results
    evaluator.plot_adaptation_curves(results)

    # Print summary
    print("\n=== Adaptation Summary ===")
    for num_steps in [0, 10, 50, 100]:
        print(f"\nSteps: {num_steps}")
        print(f"  Success: {results[num_steps]['success_mean']:.3f} ± "
              f"{results[num_steps]['success_std']:.3f}")
        print(f"  Return:  {results[num_steps]['return_mean']:.1f} ± "
              f"{results[num_steps]['return_std']:.1f}")
```

**Expected Adaptation Curves:**

For a well-trained meta-RL algorithm on ML10:

```
Adaptation Steps    Success Rate    Average Return
      0                0.15 ± 0.08       120 ± 45
      1                0.32 ± 0.12       245 ± 62
      5                0.58 ± 0.15       412 ± 71
     10                0.72 ± 0.13       534 ± 68
     25                0.84 ± 0.10       625 ± 55
     50                0.89 ± 0.08       681 ± 48
    100                0.92 ± 0.06       712 ± 42
```

Key observations:

- Rapid improvement in first 10 steps (meta-learning effect)
- Diminishing returns after 50 steps
- Higher variance at 0 steps (no task information)
- Convergence to near-optimal performance

---

## SECTION V: DISCUSSION QUESTIONS

### Question 8: Meta-Learning vs. Transfer Learning

**Q8.1:** Compare and contrast meta-learning with traditional transfer learning. When is each approach more appropriate?

**Solution:**

**Transfer Learning:**

Transfer learning involves training a model on a source task/domain and adapting it to a target task/domain, typically through fine-tuning.

**Process:**

1. Pre-train on source task(s)
2. Fine-tune on target task
3. Use adapted model

**Meta-Learning:**

Meta-learning explicitly optimizes for the ability to quickly learn new tasks from a distribution.

**Process:**

1. Meta-train across task distribution
2. For each new task: adapt with few samples
3. Evaluate adapted model

**Comparative Analysis:**

| Aspect                 | Transfer Learning                       | Meta-Learning                |
| ---------------------- | --------------------------------------- | ---------------------------- |
| **Training Objective** | Optimize for source task                | Optimize for adaptability    |
| **Adaptation**         | Often slow (many gradient steps)        | Fast (few gradient steps)    |
| **Task Distribution**  | Typically single source → single target | Explicit distribution p(T)   |
| **Evaluation**         | Final performance on target             | Few-shot adaptation ability  |
| **Sample Efficiency**  | Needs substantial target data           | Minimal target data          |
| **Computational Cost** | Standard training                       | Higher (nested optimization) |
| **Guarantees**         | Task-specific                           | Distribution-level           |

**When to Use Transfer Learning:**

1. **Single Target Task:**

   - You have one specific target task
   - No need to adapt to multiple related tasks
   - Example: ImageNet → specific classification task

2. **Abundant Target Data:**

   - Sufficient data available for fine-tuning
   - Can afford longer adaptation
   - Example: Language model fine-tuning with 10K+ examples

3. **Large Domain Shift:**

   - Source and target tasks are quite different
   - Shared low-level features but different high-level semantics
   - Example: Natural images → Medical images

4. **Computational Constraints:**
   - Limited resources for meta-training
   - Can only afford single training run
   - Pre-trained models readily available

**When to Use Meta-Learning:**

1. **Few-Shot Scenarios:**

   - Very limited data for each new task (1-50 examples)
   - Need to adapt quickly
   - Example: Personalization with minimal user interaction

2. **Task Distribution:**

   - Clear distribution of related tasks
   - Expectation of encountering new tasks from same family
   - Example: Different MuJoCo environments with similar dynamics

3. **Lifelong Learning:**

   - Agent must continually adapt to new tasks
   - Sequential task arrival
   - Example: Robot learning in changing environments

4. **Explicit Transfer:**
   - Want to explicitly optimize for transfer
   - Have access to multiple training tasks
   - Example: Meta-World, multi-agent scenarios

**Hybrid Approaches:**

Modern methods often combine both:

```python
# Pseudo-code for hybrid approach
def hybrid_learning(source_data, meta_train_tasks, target_task):
    # Stage 1: Transfer learning (broad capabilities)
    model = pretrain_on_source(source_data)

    # Stage 2: Meta-learning (quick adaptation)
    meta_model = meta_train_MAML(model, meta_train_tasks)

    # Stage 3: Few-shot adaptation to target
    adapted_model = meta_model.adapt(target_task, num_shots=10)

    return adapted_model
```

**Example:**

- Pre-train vision encoder on ImageNet (transfer learning)
- Meta-train policy on Meta-World tasks (meta-learning)
- Adapt to new manipulation task with 10 demonstrations

**Recommendation:**

| Scenario                                     | Approach                                   |
| -------------------------------------------- | ------------------------------------------ |
| Single task, lots of data                    | Transfer Learning                          |
| Multiple related tasks, little data per task | Meta-Learning                              |
| One target, minimal data                     | Transfer Learning (with data augmentation) |
| Distribution of tasks, need fast adaptation  | Meta-Learning                              |
| Unclear task distribution                    | Transfer Learning (more robust)            |
| Well-defined task family                     | Meta-Learning (more sample efficient)      |

---

### Question 9: Challenges in Meta-RL

**Q9.1:** Identify three major challenges in meta-RL and propose potential solutions for each.

**Solution:**

**Challenge 1: Sample Inefficiency During Meta-Training**

**Problem:**

- Meta-RL requires many tasks for meta-training
- Each task requires multiple episodes
- Total sample cost: Tasks × Episodes × Steps
- Can easily reach millions of samples
- Impractical for real-world applications (robots, etc.)

**Impact:**

```
Typical meta-training:
- 100 tasks
- 100 episodes per task per meta-iteration
- 200 steps per episode
- 1000 meta-iterations
Total: 100 × 100 × 200 × 1000 = 2 billion steps
```

**Proposed Solutions:**

1. **Off-Policy Meta-RL (e.g., PEARL):**

```python
# Use replay buffers to reuse data
class OffPolicyMetaRL:
    def __init__(self):
        self.replay_buffers = {}  # One per task

    def meta_train_step(self):
        for task in sample_tasks():
            # Reuse old data from buffer
            batch = self.replay_buffers[task].sample(256)

            # Off-policy update (e.g., SAC)
            loss = compute_off_policy_loss(batch)
            optimize(loss)
```

Benefits:

- 10-100x sample efficiency vs. on-policy methods
- Can reuse data across meta-iterations
- Better for real-world applications

2. **Model-Based Meta-RL:**

```python
# Learn dynamics model, use for planning
class ModelBasedMetaRL:
    def __init__(self):
        self.dynamics_model = LearnedDynamicsModel()

    def meta_train_step(self, task):
        # Collect real data (expensive)
        real_data = collect_episodes(task, num=10)

        # Learn/adapt dynamics model
        self.dynamics_model.update(real_data)

        # Generate synthetic rollouts (cheap)
        synthetic_data = self.dynamics_model.simulate(num=100)

        # Train policy on synthetic data
        self.policy.update(synthetic_data)
```

Benefits:

- Amortize real samples through simulation
- Especially effective when dynamics transfer across tasks
- Example: Different goals but same physics

3. **Data Augmentation for Meta-RL:**

```python
# Augment observations to increase effective data
class AugmentedMetaRL:
    def augment_observation(self, obs, task):
        # Geometric: rotation, translation, cropping
        obs_aug = random_crop(random_rotate(obs))

        # Task-specific: goal position shifts
        if task.type == 'reach':
            obs_aug = shift_goal(obs_aug)

        return obs_aug

    def meta_train_step(self, task):
        batch = sample_batch(task)

        # Create augmented versions
        batch_aug = [self.augment_observation(obs, task)
                     for obs in batch]

        # Train on both original and augmented
        loss = compute_loss(batch + batch_aug)
```

Benefits:

- 2-5x improvement in sample efficiency
- Especially effective for vision-based tasks
- No additional environment interactions needed

---

**Challenge 2: Distribution Shift at Test Time**

**Problem:**

- Meta-training assumes tasks come from distribution $p(\mathcal{T})$
- Test tasks may be out-of-distribution
- Performance degrades gracefully for interpolation, fails for extrapolation
- Difficult to know boundaries of learned distribution

**Example:**

```python
# Meta-trained on goals in region [0, 1] × [0, 1]
meta_trained_region = (0, 1, 0, 1)

# Test scenarios:
test_goal_interpolation = (0.5, 0.5)  # Works well
test_goal_edge = (0.9, 0.9)           # Okay
test_goal_extrapolation = (1.5, 1.5)  # Fails!
```

**Proposed Solutions:**

1. **Domain Randomization During Meta-Training:**

```python
class DomainRandomizedMetaRL:
    def sample_task(self):
        # Randomize task parameters beyond observed range
        goal_range = (-0.5, 1.5)  # Wider than test range
        dynamics_noise = Uniform(0.8, 1.2)  # Friction, mass, etc.

        task = create_task(
            goal=sample_uniform(goal_range),
            friction=base_friction * sample(dynamics_noise),
            mass=base_mass * sample(dynamics_noise)
        )
        return task
```

Benefits:

- Broader support for learned distribution
- More robust to test-time variations
- Commonly used in sim-to-real transfer

2. **Uncertainty-Aware Adaptation:**

```python
class UncertaintyAwareMetaRL:
    def adapt(self, task, context):
        # Encode task with uncertainty
        mean, std = self.context_encoder(context)

        # Measure epistemic uncertainty
        uncertainty = std.mean()

        if uncertainty > threshold:
            # High uncertainty → explore more
            num_adapt_steps = 50
            exploration_bonus = 0.5
        else:
            # Low uncertainty → exploit
            num_adapt_steps = 10
            exploration_bonus = 0.0

        # Adapt with appropriate strategy
        self.adapt_policy(task, num_adapt_steps, exploration_bonus)
```

Benefits:

- Detect out-of-distribution tasks
- Adapt exploration/adaptation strategy accordingly
- More graceful degradation

3. **Continual Meta-Learning:**

```python
class ContinualMetaRL:
    def deploy_and_learn(self):
        while True:
            # Encounter new task
            task = environment.get_current_task()

            # Adapt using meta-knowledge
            self.adapt(task)

            # Collect experience
            data = self.interact(task)

            # Update meta-knowledge if task is novel
            if self.is_novel(task):
                self.meta_update(data)
                self.task_distribution.expand(task)
```

Benefits:

- Continuously expand task distribution
- No strict boundary between meta-train and meta-test
- More realistic for deployed systems

---

**Challenge 3: Credit Assignment in Nested Optimization**

**Problem:**

- Two-level optimization makes credit assignment difficult
- Is poor performance due to:
  - Bad meta-initialization?
  - Insufficient adaptation?
  - Bad task sampling?
  - High variance in gradients?
- Meta-gradients have very high variance
- Difficult to diagnose failures

**Proposed Solutions:**

1. **Separate Value Functions for Meta-Learning:**

```python
class HierarchicalMetaRL:
    def __init__(self):
        self.task_value = TaskValueFunction()  # Meta-level
        self.state_value = StateValueFunction()  # Task-level

    def meta_train_step(self, tasks):
        for task in tasks:
            # Task-level returns
            task_return = self.adapt_and_evaluate(task)

            # Meta-level value: expected return after adaptation
            meta_value = self.task_value(task_embedding)

            # Separate credit assignment
            task_level_error = task_return - self.state_value(states)
            meta_level_error = task_return - meta_value

            # Update both
            self.update_policy(task_level_error)
            self.update_meta_policy(meta_level_error)
```

Benefits:

- Explicit attribution to different levels
- Reduced variance in meta-gradients
- Better debugging and interpretation

2. **Variance Reduction Techniques:**

```python
class LowVarianceMetaRL:
    def compute_meta_gradient(self, tasks):
        meta_grads = []

        for task in tasks:
            # Baseline: average return across tasks
            baseline = self.compute_baseline(tasks)

            # Compute advantage
            task_return = self.evaluate_task(task)
            advantage = task_return - baseline

            # Gradient with advantage
            grad = self.compute_gradient(task) * advantage
            meta_grads.append(grad)

        # Additional variance reduction: clip gradients
        meta_grad = torch.stack(meta_grads).mean(0)
        meta_grad = torch.clip(meta_grad, -1.0, 1.0)

        return meta_grad
```

Benefits:

- Lower variance → faster convergence
- More stable meta-optimization
- Better sample efficiency

3. **Diagnostic Tools and Visualization:**

```python
class MetaRLDiagnostics:
    def diagnose_failure(self, task):
        # 1. Check adaptation progress
        returns_during_adaptation = []
        for step in range(num_adapt_steps):
            ret = evaluate_policy(step)
            returns_during_adaptation.append(ret)

        # Is adaptation working?
        adaptation_slope = np.polyfit(
            range(len(returns_during_adaptation)),
            returns_during_adaptation,
            deg=1
        )[0]

        if adaptation_slope < 0.01:
            print("Warning: Adaptation not improving performance")

        # 2. Check gradient magnitudes
        inner_grad_norm = compute_gradient_norm(inner_loop)
        meta_grad_norm = compute_gradient_norm(outer_loop)

        print(f"Inner grad norm: {inner_grad_norm:.4f}")
        print(f"Meta grad norm: {meta_grad_norm:.4f}")

        if meta_grad_norm < 1e-5:
            print("Warning: Vanishing meta-gradients")

        # 3. Visualize task embedding
        embedding = self.encode_task(task)
        plot_embedding_with_training_tasks(embedding)

        # 4. Compare with similar training tasks
        similar_tasks = find_similar_tasks(task, self.training_tasks)
        print(f"Similar training tasks: {similar_tasks}")

        # 5. Check policy entropy (exploration)
        entropy = compute_policy_entropy(self.policy, task)
        print(f"Policy entropy: {entropy:.4f}")
```

Benefits:

- Identify specific failure modes
- Guide hyperparameter tuning
- Better understanding of learned representations

---

**Summary Table:**

| Challenge           | Impact                       | Best Solution              | Expected Improvement    |
| ------------------- | ---------------------------- | -------------------------- | ----------------------- |
| Sample Inefficiency | 2B+ samples needed           | Off-policy methods (PEARL) | 10-100x                 |
| Distribution Shift  | Failure on new tasks         | Domain randomization       | +30% success on OOD     |
| Credit Assignment   | High variance, slow learning | Variance reduction         | 2-5x faster convergence |

---

### Question 10: Future Directions

**Q10.1:** Propose a novel research direction in meta-RL. Describe the motivation, approach, and potential impact.

**Solution:**

**Proposed Direction: Compositional Meta-Reinforcement Learning**

**Motivation:**

Current meta-RL algorithms learn monolithic policies that must be re-adapted for each new task. However, many complex tasks are compositions of simpler skills. For example:

- "Pick and place" = "Reach" + "Grasp" + "Move" + "Release"
- "Open door" = "Reach handle" + "Turn handle" + "Pull door"

**Key Insight:** If we can learn a library of composable primitive skills and a meta-policy for composing them, we can achieve:

1. Better generalization to novel task compositions
2. More interpretable policies
3. Faster adaptation (compose existing skills vs. learn from scratch)
4. Systematic generalization (combine skills in new ways)

**Approach:**

**1. Hierarchical Architecture:**

```python
class CompositionalMetaRL(nn.Module):
    def __init__(self, num_primitives=20):
        super().__init__()

        # Library of primitive skills
        self.primitives = nn.ModuleList([
            PrimitiveSkill(name=f"skill_{i}")
            for i in range(num_primitives)
        ])

        # Meta-controller: selects and sequences primitives
        self.meta_controller = nn.LSTM(
            input_size=obs_dim + num_primitives,
            hidden_size=256
        )

        # Task encoder: infers task structure
        self.task_encoder = TaskEncoder()

        # Composition module: combines primitives
        self.composer = AttentionBasedComposer(num_primitives)

    def forward(self, obs, task_context):
        # Encode task structure
        task_embedding = self.task_encoder(task_context)

        # Meta-controller selects primitive(s)
        primitive_weights = self.meta_controller(obs, task_embedding)

        # Compose primitives (soft attention)
        actions = []
        for i, primitive in enumerate(self.primitives):
            action_i = primitive(obs)
            actions.append(primitive_weights[i] * action_i)

        # Final action is weighted combination
        action = sum(actions)

        return action, primitive_weights
```

**2. Training Procedure:**

```python
def train_compositional_meta_rl(task_distribution):
    model = CompositionalMetaRL()

    # Phase 1: Discover primitive skills
    primitives = discover_primitives(task_distribution)
    model.primitives = primitives

    # Phase 2: Meta-train composition
    for meta_iter in range(N):
        # Sample compositional tasks
        task = task_distribution.sample_compositional_task()

        # Inner loop: adapt composition weights
        for adapt_step in range(K):
            trajectory = collect_trajectory(task, model)

            # Update only meta-controller and composer
            loss = compute_loss(trajectory)
            optimize(model.meta_controller, model.composer)

        # Outer loop: optimize primitives and meta-policy
        test_trajectory = collect_trajectory(task, model)
        meta_loss = compute_meta_loss(test_trajectory)

        optimize(model, meta_loss)
```

**3. Primitive Discovery:**

Use unsupervised learning to discover useful primitives:

```python
def discover_primitives(task_distribution, num_primitives=20):
    """
    Discover primitives using mutual information maximization
    """
    # Option-critic or similar
    primitives = []

    for i in range(num_primitives):
        # Learn option that maximizes:
        # I(option; outcome) - minimizes entropy over outcomes
        primitive = learn_option(
            objective=lambda option: mutual_information(
                option.behavior,
                environment.outcomes
            )
        )
        primitives.append(primitive)

    return primitives
```

**4. Compositional Structure Learning:**

```python
class CompositionGraph(nn.Module):
    """
    Learn graph structure of skill compositions
    """
    def __init__(self, num_skills):
        super().__init__()

        # Adjacency matrix (which skills can follow which)
        self.adjacency = nn.Parameter(
            torch.randn(num_skills, num_skills)
        )

        # Termination predictor for each skill
        self.terminators = nn.ModuleList([
            TerminationFunction() for _ in range(num_skills)
        ])

    def forward(self, current_skill, obs):
        # Predict if current skill should terminate
        should_terminate = self.terminators[current_skill](obs)

        if should_terminate:
            # Transition to next skill based on adjacency
            next_skill_logits = self.adjacency[current_skill]
            next_skill = Categorical(logits=next_skill_logits).sample()
            return next_skill
        else:
            return current_skill
```

**Potential Impact:**

1. **Systematic Generalization:**

   - Train on {"Reach A", "Pick B", "Place C"}
   - Generalize to {"Reach B", "Pick A", "Place at C"}
   - Current methods: ~30% success
   - Compositional method: ~70% success (predicted)

2. **Sample Efficiency:**

   - Learn primitives once, reuse for many tasks
   - Expected 5-10x improvement in sample efficiency
   - Especially beneficial for long-horizon tasks

3. **Interpretability:**

   - Can visualize which primitives are activated
   - Understand policy decisions
   - Debug failures more easily

4. **Transfer to Unseen Task Structures:**
   - Current meta-RL: interpolates within training tasks
   - Compositional: combines primitives in novel ways
   - Enables extrapolation to new task compositions

**Expected Results:**

```
Benchmark: Compositional Meta-World (Proposed)
- 10 primitive tasks (reach, grasp, push, etc.)
- 100 compositional tasks (combinations)
- Train on 50, test on 50 held-out compositions

Method                  Success Rate (0-shot)  Adaptation Steps to 80%
-----------------       ---------------------  ------------------------
MAML                           15%                    200+
PEARL                          22%                    150
RL²                            18%                    180
Compositional (Ours)           55%                     50
```

**Open Questions:**

1. How to automatically discover optimal set of primitives?
2. How to handle partial observability in composition?
3. Can we learn hierarchical compositions (primitives of primitives)?
4. How to balance exploration in primitive space vs. composition space?

**Related Work:**

- Hierarchical RL (Options, HAM)
- Modular networks
- Neural module networks (NMN)
- Program synthesis for RL

**Next Steps:**

1. Implement on simple gridworld with compositional structure
2. Evaluate on Meta-World subset with known compositions
3. Develop primitive discovery algorithm
4. Scale to full Meta-World and real robot

---

## SECTION VI: CONCLUSIONS

### Summary

This assignment covered fundamental concepts and algorithms in meta-reinforcement learning:

1. **Meta-Learning Problem:** Two-level optimization for fast adaptation
2. **MAML:** Gradient-based meta-learning through optimization
3. **RL²:** Recurrent meta-RL with implicit adaptation
4. **PEARL:** Context-based meta-RL with probabilistic embeddings
5. **Meta-World:** Standard benchmark for evaluation

### Key Takeaways

- Meta-RL enables few-shot adaptation to new tasks
- Different approaches (gradient, recurrent, context) have distinct trade-offs
- Sample efficiency and generalization remain major challenges
- Promising direction: compositional structure and hierarchical meta-learning

### References

[1] Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. _ICML_.

[2] Duan, Y., Schulman, J., Chen, X., Bartlett, P. L., Sutskever, I., & Abbeel, P. (2016). RL²: Fast reinforcement learning via slow reinforcement learning. _arXiv preprint arXiv:1611.02779_.

[3] Rakelly, K., Zhou, A., Quillen, D., Finn, C., & Levine, S. (2019). Efficient off-policy meta-reinforcement learning via probabilistic context variables. _ICML_.

[4] Yu, T., Quillen, D., He, Z., Julian, R., Hausman, K., Finn, C., & Levine, S. (2020). Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. _CoRL_.

[5] Nichol, A., Achiam, J., & Schulman, J. (2018). On first-order meta-learning algorithms. _arXiv preprint arXiv:1803.02999_.

---

**End of Solutions**

