# HW12: Hierarchical Reinforcement Learning

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Hierarchical](https://img.shields.io/badge/Type-Hierarchical-orange.svg)](.)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

Hierarchical Reinforcement Learning (HRL) structures policies across multiple levels of abstraction, enabling agents to solve complex, long-horizon tasks by decomposing them into simpler subtasks. This assignment explores temporal abstraction, options framework, feudal architectures, and goal-conditioned policies.

## 🎯 Learning Objectives

1. **Temporal Abstraction**: Understand multi-scale decision making
2. **Options Framework**: Master semi-Markov decision processes
3. **Feudal Hierarchies**: Learn manager-worker architectures
4. **Goal-Conditioned RL**: Train policies with diverse goals
5. **Skill Discovery**: Learn reusable primitives automatically
6. **Credit Assignment**: Address challenges across temporal scales

## 📂 Directory Structure

```
HW12_Hierarchical_RL/
├── code/
│   └── HW12_Notebook.ipynb        # HRL implementations
├── answers/                        # (No solutions provided yet)
├── reports/
│   └── HW12_Questions.pdf         # Assignment questions
└── README.md
```

## 📚 Core Concepts

### 1. Motivation for Hierarchy

**Challenges in Flat RL:**

```
Long Horizons: Credit assignment difficult over 1000+ steps
Sparse Rewards: Random exploration ineffective
Complex Tasks: Atomic actions insufficient
Transfer: Hard to reuse learned behaviors
```

**Human Example:**

```
Task: Make dinner
├─ Shop for ingredients
│  ├─ Drive to store
│  ├─ Find items
│  └─ Checkout
├─ Prepare food
│  ├─ Chop vegetables
│  ├─ Cook proteins
│  └─ Mix ingredients
└─ Serve meal
```

**Benefits of Hierarchy:**

- Temporal abstraction (plan at multiple scales)
- Reusable skills/subpolicies
- Exploration structure
- Transfer learning
- Compositional generalization

### 2. Options Framework

**Option:** Temporally extended action

**Formal Definition:**

```
Option ω = (I_ω, π_ω, β_ω)

where:
- I_ω ⊆ S: Initiation set (where option can start)
- π_ω: S × A → [0,1]: Option policy
- β_ω: S → [0,1]: Termination function
```

**Semi-Markov Decision Process (SMDP):**

```
Instead of choosing action at each step,
choose option, execute until termination

Q(s, ω) = Expected return from executing option ω in state s
```

**Option-Value Functions:**

```python
def intra_option_learning(s, omega, r, s_next):
    """
    Q-learning for options (intra-option)
    Can update Q while executing option
    """
    if not beta(s_next):  # Option continues
        target = r + gamma * Q(s_next, omega)
    else:  # Option terminates
        target = r + gamma * max_omega_prime Q(s_next, omega_prime)

    Q(s, omega) += alpha * (target - Q(s, omega))
```

**Option Discovery:**

#### 1. Handcrafted Options

```python
class NavigateOption:
    def __init__(self, goal_location):
        self.goal = goal_location
        self.policy = train_policy_to_reach(goal)

    def initiation_set(self, state):
        return True  # Can start anywhere

    def termination(self, state):
        return distance(state, self.goal) < threshold
```

#### 2. Learned Options

**Eigenoptions:** Use graph Laplacian eigenvectors

```
Eigenvectors of state transition graph
→ Diverse exploration directions
```

**Option-Critic:**

```python
class OptionCritic(nn.Module):
    def __init__(self, state_dim, num_options, action_dim):
        super().__init__()

        # Shared representation
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        # Option policies
        self.option_policies = nn.ModuleList([
            nn.Linear(128, action_dim)
            for _ in range(num_options)
        ])

        # Termination functions
        self.terminations = nn.Sequential(
            nn.Linear(128, num_options),
            nn.Sigmoid()
        )

        # Q-value over options
        self.q_omega = nn.Linear(128, num_options)

    def forward(self, state, current_option=None):
        features = self.encoder(state)

        if current_option is not None:
            # Get action from current option
            action_logits = self.option_policies[current_option](features)

            # Termination probability
            beta = self.terminations(features)[:, current_option]

            return action_logits, beta
        else:
            # Select option
            q_omega = self.q_omega(features)
            return q_omega
```

**Training:**

```python
def train_option_critic(env, model, episodes):
    for episode in range(episodes):
        state = env.reset()
        option = select_option(model, state)

        while not done:
            # Get action and termination
            action, beta = model(state, option)

            # Execute
            next_state, reward, done = env.step(action)

            # Update Q over options
            q_loss = compute_q_loss(state, option, reward, next_state, beta)

            # Update intra-option policy
            policy_loss = compute_policy_loss(state, option, action, advantage)

            # Update termination function
            term_loss = compute_termination_loss(state, option, next_state)

            # Optimize
            total_loss = q_loss + policy_loss + term_loss
            optimize(total_loss)

            # Termination
            if sample() < beta or done:
                option = select_option(model, next_state)

            state = next_state
```

### 3. Feudal Hierarchies

**Key Idea:** Manager sets goals, Worker achieves them

**FeudalNet (Feudal Networks):**

```python
class FeudalNet(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, c=10):
        super().__init__()

        # Perception module (shared)
        self.perception = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )

        # Manager (sets goals)
        self.manager = nn.LSTM(256, goal_dim)

        # Worker (achieves goals)
        self.worker = nn.LSTM(256 + goal_dim, 256)
        self.worker_policy = nn.Linear(256, action_dim)

        self.c = c  # Manager horizon

    def forward(self, state, manager_hidden, worker_hidden, t):
        # Shared perception
        z = self.perception(state)

        # Manager operates every c timesteps
        if t % self.c == 0:
            # Manager sets goal
            g, manager_hidden = self.manager(z.unsqueeze(0), manager_hidden)
            g = F.normalize(g, dim=-1)  # Normalize goal

        # Worker receives goal and state
        w_input = torch.cat([z, g.squeeze(0)], dim=-1)
        w, worker_hidden = self.worker(w_input.unsqueeze(0), worker_hidden)

        # Worker action
        action_logits = self.worker_policy(w.squeeze(0))

        return action_logits, g, manager_hidden, worker_hidden
```

**Training:**

```python
def train_feudal(trajectories):
    for trajectory in trajectories:
        states, actions, rewards = trajectory

        # Manager reward: transition embedding cosine similarity
        # Encourages setting goals in direction of state change
        z_t = perception(states[:-1])
        z_t_plus_c = perception(states[c:])

        manager_reward = cosine_similarity(
            goals[:-1],
            z_t_plus_c - z_t
        )

        # Worker reward: intrinsic + extrinsic
        # Intrinsic: progress toward goal
        worker_intrinsic = cosine_similarity(
            goals[:-1],
            z_t_plus_1 - z_t
        )

        worker_reward = rewards + alpha * worker_intrinsic

        # Update manager with manager_reward
        manager_loss = compute_policy_gradient(manager_reward)

        # Update worker with worker_reward
        worker_loss = compute_policy_gradient(worker_reward)
```

### 4. Goal-Conditioned RL

**Key Idea:** Train policy to reach any goal state

**Universal Value Function Approximators (UVFA):**

```python
class GoalConditionedPolicy(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state, goal):
        """
        Policy conditioned on current state and goal
        """
        x = torch.cat([state, goal], dim=-1)
        return self.policy(x)
```

**Hindsight Experience Replay (HER):**

```python
def hindsight_experience_replay(trajectory, strategy='future'):
    """
    Augment failed trajectories with alternative goals
    """
    states, actions, goals, rewards = trajectory

    # Original experience
    buffer.add(states, actions, goals, rewards)

    # Hindsight: "what if goal was different?"
    for t in range(len(states)):
        if strategy == 'future':
            # Sample achieved state as goal
            future_t = random.randint(t, len(states)-1)
            new_goal = states[future_t]
        elif strategy == 'final':
            new_goal = states[-1]
        elif strategy == 'random':
            new_goal = sample_random_goal()

        # Recompute rewards with new goal
        new_rewards = [compute_reward(s, new_goal)
                      for s in states]

        # Store modified experience
        buffer.add(states, actions, [new_goal]*len(states), new_rewards)
```

**Why HER Works:**

```
Even failed episode teaches agent something:
"How to reach the state I accidentally ended up at"

Dramatically improves sample efficiency in sparse reward settings
```

### 5. Skill Discovery

**Diversity is All You Need (DIAYN):**

**Objective:** Learn diverse skills without rewards

```python
class DIAYN:
    def __init__(self, num_skills, state_dim, action_dim):
        # Skill-conditioned policy
        self.policy = SkillConditionedPolicy(state_dim, num_skills, action_dim)

        # Discriminator: predict skill from state
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_skills)
        )

    def intrinsic_reward(self, state, skill):
        """
        Reward for making states predictive of skill
        """
        logits = self.discriminator(state)
        log_p_skill = F.log_softmax(logits, dim=-1)[skill]

        # Information-theoretic reward
        return log_p_skill - log(1/num_skills)

    def train(self, env):
        for episode in range(N):
            # Sample random skill
            skill = random.randint(0, num_skills)

            state = env.reset()
            trajectory = []

            while not done:
                # Policy conditioned on skill
                action = self.policy(state, skill)
                next_state, _, done = env.step(action)

                # Intrinsic reward
                reward = self.intrinsic_reward(next_state, skill)

                trajectory.append((state, skill, action, reward, next_state))
                state = next_state

            # Update policy with intrinsic rewards
            update_policy(trajectory)

            # Update discriminator
            update_discriminator(trajectory)
```

**What Emerges:**

- Diverse behaviors without reward engineering
- Reusable skills for downstream tasks
- Exploration through skill diversity

### 6. HAM (Hierarchical Abstract Machines)

**Formalism:** Finite state machines with abstract actions

```python
class AbstractMachine:
    def __init__(self):
        self.states = []
        self.transitions = {}
        self.choice_points = {}

    def add_state(self, name, actions):
        """
        Add state with possible actions
        """
        self.states.append(name)
        self.choice_points[name] = actions

    def add_transition(self, from_state, action, to_state):
        self.transitions[(from_state, action)] = to_state
```

**Example: Room Navigation**

```
Root
├─ Navigate(room1)
│  ├─ Move(door1)
│  └─ Enter
├─ Navigate(room2)
│  ├─ Move(door2)
│  └─ Enter
└─ done
```

## 📊 Topics Covered

1. **Options**: Framework, discovery, learning
2. **Feudal RL**: Manager-worker hierarchies
3. **Goal-Conditioned**: UVFA, HER
4. **Skill Discovery**: DIAYN, unsupervised learning
5. **HAM/MAXQ**: Formal hierarchical frameworks

## 📖 Key References

1. **Sutton, R. S., et al. (1999)** - "Between MDPs and Semi-MDPs" - Artificial Intelligence (Options)

2. **Bacon, P. L., et al. (2017)** - "The Option-Critic Architecture" - AAAI

3. **Vezhnevets, A. S., et al. (2017)** - "FeUdal Networks for Hierarchical RL" - ICML

4. **Andrychowicz, M., et al. (2017)** - "Hindsight Experience Replay" - NIPS

5. **Eysenbach, B., et al. (2018)** - "Diversity is All You Need" - ICLR (DIAYN)

## 💡 Discussion Questions

1. How does temporal abstraction help with long-horizon tasks?
2. Why is HER so effective for sparse reward problems?
3. What are trade-offs between learned vs handcrafted options?
4. How does DIAYN discover skills without external rewards?
5. When might feudal hierarchies be preferred over flat policies?

## 🎓 Extensions

- Implement MAXQ value decomposition
- Try hierarchical actor-critic
- Explore automatic subgoal generation
- Combine with meta-learning
- Apply to robotics manipulation

---

**Course:** Deep Reinforcement Learning  
**Last Updated:** 2024
