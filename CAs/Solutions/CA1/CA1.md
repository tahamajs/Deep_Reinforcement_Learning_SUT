# Ca1: Introduction to Deep Reinforcement Learning
# Table of Contents

- [CA1: Introduction to Deep Reinforcement Learning](#ca1-introduction-to-deep-reinforcement-learning)
- [Fundamental Concepts and Markov Decision Processes](#fundamental-concepts-and-markov-decision-processes)
- [Learning Objectives](#learning-objectives)
- [Prerequisites](#prerequisites)
- [Roadmap](#roadmap)
- [Part 2: Deep Q-Learning (DQN)](#part-2-deep-q-learning-dqn)
- [2.1 Q-Learning Algorithm](#21-q-learning-algorithm)
- [2.2 Deep Q-Network (DQN) Enhancements](#22-deep-q-network-dqn-enhancements)
- [Import necessary libraries](#import-necessary-libraries)
- [Set reproducible seed](#set-reproducible-seed)
- [Check if GPU is available and select device](#check-if-gpu-is-available-and-select-device)
- [Configure plotting style (nice for notebooks)](#configure-plotting-style-nice-for-notebooks)
- [Small utility: moving average for plotting](#small-utility-moving-average-for-plotting)
- [Quick smoke-test (keeps outputs small in notebooks)](#quick-smoke-test-keeps-outputs-small-in-notebooks)
- [Test the replay buffer](#test-the-replay-buffer)
- [Initialize agent for smoke test](#initialize-agent-for-smoke-test)
- [Part 3: Policy Gradient Methods](#part-3-policy-gradient-methods)
- [3.1 Policy Gradient Theorem](#31-policy-gradient-theorem)
- [3.2 REINFORCE Algorithm](#32-reinforce-algorithm)
- [3.3 Actor-Critic Methods](#33-actor-critic-methods)
- [Initialize agents for smoke tests](#initialize-agents-for-smoke-tests)
- [Initialize Actor-Critic agent](#initialize-actor-critic-agent)
- [Part 4: Practical Implementation and Comparison](#part-4-practical-implementation-and-comparison)
- [4.1 Training Environment Setup](#41-training-environment-setup)
- [Create environment (use gym if available; otherwise fallback to a simple MockEnv)](#create-environment-use-gym-if-available-otherwise-fallback-to-a-simple-mockenv)
- [Create environment (use gymnasium if available; otherwise fallback to a simple MockEnv)](#create-environment-use-gymnasium-if-available-otherwise-fallback-to-a-simple-mockenv)
- [Demonstration of training (short episodes for notebook demonstration)](#demonstration-of-training-short-episodes-for-notebook-demonstration)
- [Initialize fresh agents for fair comparison](#initialize-fresh-agents-for-fair-comparison)
- [Keep models small for quick demos in notebooks](#keep-models-small-for-quick-demos-in-notebooks)
- [Store results](#store-results)
- [Visualize training results (if results are available)](#visualize-training-results-if-results-are-available)
- [Part 5: Exercises and Questions](#part-5-exercises-and-questions)
- [Exercise 1: Theoretical Understanding](#exercise-1-theoretical-understanding)
- [Exercise 2: Implementation Analysis](#exercise-2-implementation-analysis)
- [Exercise 3: Experimental Design](#exercise-3-experimental-design)
- [Exercise 3: Sample Efficiency Experiment](#exercise-3-sample-efficiency-experiment)
- [Exercise 4: Hyperparameter Analysis (simple grid search wrapper)](#exercise-4-hyperparameter-analysis-simple-grid-search-wrapper)
- [Exercise 5: Advanced DQN Variants](#exercise-5-advanced-dqn-variants)
- [Part 6: Conclusions and Analysis](#part-6-conclusions-and-analysis)
- [6.1 Algorithm Comparison Summary](#61-algorithm-comparison-summary)
- [6.2 Key Insights](#62-key-insights)
- [6.3 When to Use Each Algorithm](#63-when-to-use-each-algorithm)
- [6.4 Advanced Topics for Further Study](#64-advanced-topics-for-further-study)
- [6.5 Further Reading](#65-further-reading)
- [Sample Efficiency Experiment Results (Updated)](#sample-efficiency-experiment-results-updated)
- [Run the sample efficiency experiment](#run-the-sample-efficiency-experiment)
- [Create a fresh environment factory](#create-a-fresh-environment-factory)
- [Define agent factories](#define-agent-factories)
- [Run the experiment (increase runs and max*episodes for more robust results, but keep small for demo)](#run-the-experiment-increase-runs-and-max*episodes-for-more-robust-results-but-keep-small-for-demo)
- [Assignment Submission Requirements](#assignment-submission-requirements)
- [What to Submit:](#what-to-submit)
- [Evaluation Criteria:](#evaluation-criteria)
- [Submission Deadline: [Insert Date]](#submission-deadline-insert-date)
- [Additional Notes:](#additional-notes)



## Fundamental Concepts and Markov Decision Processes

Welcome to Computer Assignment 1, where we explore the foundational concepts of Deep Reinforcement Learning. This assignment establishes the theoretical groundwork and practical implementations that form the basis for all subsequent advanced RL techniques.

### Learning Objectives
By the end of this assignment, you will understand and implement:

1. **Markov Decision Processes (MDPs)** - The mathematical framework for sequential decision making
2. **Value Functions** - State-value and action-value functions with Bellman equations
3. **Dynamic Programming** - Policy evaluation and policy improvement algorithms
4. **Monte Carlo Methods** - Learning from complete episodes without environment models
5. **Temporal Difference Learning** - Bootstrapping methods combining DP and MC approaches
6. **Deep Q-Networks (DQN)** - Combining Q-learning with deep neural networks

### Prerequisites
- Basic understanding of probability and statistics
- Familiarity with Python programming
- Knowledge of calculus and linear algebra fundamentals
- Introduction to machine learning concepts

### Roadmap
This comprehensive assignment is structured as follows:
- **Section 1**: Theoretical Foundations (MDPs, Value Functions, Bellman Equations)
- **Section 2**: Dynamic Programming Methods (Policy Evaluation, Value Iteration)
- **Section 3**: Monte Carlo Methods (First-Visit and Every-Visit MC)
- **Section 4**: Temporal Difference Learning (SARSA, Q-Learning)
- **Section 5**: Deep Reinforcement Learning (DQN Implementation)
- **Section 6**: Comparative Analysis and Advanced Topics

Let's begin our journey into the fundamentals of reinforcement learning!

## Part 2: Deep Q-learning (dqn)

### 2.1 Q-learning Algorithm

Q-Learning is a model-free, off-policy algorithm that learns the optimal action-value function:

**Q-Learning Update Rule:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

### 2.2 Deep Q-network (dqn) Enhancements

**Key Innovations:**
1. **Experience Replay**: Store transitions $(s,a,r,s')$ in replay buffer
2. **Target Network**: Use separate network for target values to improve stability
3. **Double DQN**: Mitigate overestimation bias
4. **Dueling DQN**: Separate value and advantage streams

**DQN Loss Function:**
$$L(\theta) = \mathbb{E}*{(s,a,r,s') \sim D} \left[ \left( r + \gamma \max*{a'} Q(s',a';\theta^-) - Q(s,a;\theta) \right)^2 \right]$$

Where $\theta^-$ represents the target network parameters.


```python
# Import Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import random
from collections import deque, namedtuple
import seaborn as sns
from typing import Tuple, List, Optional, Any


def set_seed(seed: int = 42) -> None:
    """Set seeds for numpy, torch and python random for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


# Set Reproducible Seed
set_seed(42)

# Check If Gpu Is Available and Select Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure Plotting Style (nice for Notebooks)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# Small Utility: Moving Average for Plotting
def moving_average(x: List[float], window: int = 10) -> np.ndarray:
    """Compute moving average using a convolution. Returns same-length array."""
    if len(x) < 1:
        return np.array([])
    if window <= 1:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode='valid')

```

    Using device: cpu



```python
class DQN(nn.Module):
    """
    Standard feed-forward Deep Q-Network.

    Simple MLP with two hidden layers and ReLU activations.
    """
    def **init**(self, state*size: int, action*size: int, hidden_size: int = 64) -> None:
        super(DQN, self).**init**()
        self.fc1 = nn.Linear(state*size, hidden*size)
        self.fc2 = nn.Linear(hidden*size, hidden*size)
        self.fc3 = nn.Linear(hidden*size, action*size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture: separate streams for state-value and advantage.

    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
    """
    def **init**(self, state*size: int, action*size: int, hidden_size: int = 64) -> None:
        super(DuelingDQN, self).**init**()
        self.feature_layer = nn.Sequential(
            nn.Linear(state*size, hidden*size),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden*size, hidden*size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden*size, hidden*size),
            nn.ReLU(),
            nn.Linear(hidden*size, action*size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage with mean subtraction for identifiability
        q_value = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_value


# Quick Smoke-test (keeps Outputs Small in Notebooks)
state_size = 4  # Example: CartPole observation size
action_size = 2  # Example: CartPole action size

*dqn = DQN(state*size, action_size)
*dueling = DuelingDQN(state*size, action_size)

print("DQN and DuelingDQN classes defined.")

```

    DQN and DuelingDQN classes defined.



```python
class ReplayBuffer:
    """
    Experience Replay Buffer for DQN. Stores experiences as namedtuples and
    provides a method to sample batches converted to torch tensors on the
    selected device.
    """
    def **init**(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Add an experience to the buffer."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences and return tensors.

        Returns:
            states: FloatTensor (batch*size, state*dim)
            actions: LongTensor (batch_size, 1)
            rewards: FloatTensor (batch_size, 1)
            next*states: FloatTensor (batch*size, state_dim)
            dones: FloatTensor (batch_size, 1)
        """
        experiences = random.sample(self.buffer, k=batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next*states = np.vstack([e.next*state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next*states = torch.from*numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        return states, actions, rewards, next_states, dones

    def **len**(self) -> int:
        return len(self.buffer)


# Test the Replay Buffer
buffer = ReplayBuffer(10000)
print(f"Replay buffer initialized with capacity: {10000}")
print(f"Current buffer size: {len(buffer)}")

```

    Replay buffer initialized with capacity: 10000
    Current buffer size: 0



```python
class DQNAgent:
    """
    DQN Agent with Experience Replay and Target Network. Supports optional
    Double DQN and Dueling architectures.
    """
    def **init**(self,
                 state_size: int,
                 action_size: int,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 update_every: int = 4,
                 tau: float = 1e-3,
                 use*double*dqn: bool = False,
                 use_dueling: bool = False) -> None:

        self.state*size = state*size
        self.action*size = action*size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon*decay = epsilon*decay
        self.epsilon*min = epsilon*min
        self.batch*size = batch*size
        self.update*every = update*every
        self.tau = tau
        self.use*double*dqn = use*double*dqn

        # Neural networks
        if use_dueling:
            self.q*network = DuelingDQN(state*size, action_size).to(device)
            self.target*network = DuelingDQN(state*size, action_size).to(device)
        else:
            self.q*network = DQN(state*size, action_size).to(device)
            self.target*network = DQN(state*size, action_size).to(device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer and time step counter
        self.memory = ReplayBuffer(buffer_size)
        self.t_step = 0

        # Initialize target network weights
        self.hard*update(self.target*network, self.q_network)

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Save experience and, every `update_every` steps, sample a batch and learn."""
        self.memory.add(state, action, reward, next_state, done)

        self.t*step = (self.t*step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def act(self, state: np.ndarray, eps: Optional[float] = None) -> int:
        """Return action for given state following epsilon-greedy policy."""
        if eps is None:
            eps = self.epsilon

        if random.random() > eps:
            state*t = torch.from*numpy(state).float().unsqueeze(0).to(device)
            self.q_network.eval()
            with torch.no_grad():
                action*values = self.q*network(state_t)
            self.q_network.train()
            return int(action_values.argmax(dim=1).item())
        else:
            return int(random.choice(np.arange(self.action_size)))

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """Update Q-network using a batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        # Double DQN: select actions using local network, evaluate with target network
        if self.use*double*dqn:
            next*actions = self.q*network(next_states).detach().argmax(1).unsqueeze(1)
            Q*targets*next = self.target*network(next*states).detach().gather(1, next_actions)
        else:
            Q*targets*next = self.target*network(next*states).detach().max(1)[0].unsqueeze(1)

        # Compute target Q values
        Q*targets = rewards + (self.gamma * Q*targets_next * (1 - dones))

        # Get expected Q values from local model
        Q*expected = self.q*network(states).gather(1, actions)

        # Compute loss
        loss = F.mse*loss(Q*expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft-update target network parameters
        self.soft*update(self.q*network, self.target_network, self.tau)

        # Decay epsilon (ensure it doesn't go below epsilon_min)
        self.epsilon = max(self.epsilon*min, self.epsilon * self.epsilon*decay)

    def soft*update(self, local*model: nn.Module, target_model: nn.Module, tau: float) -> None:
        """Soft update model parameters: target = tau*local + (1-tau)*target"""
        for target*param, local*param in zip(target*model.parameters(), local*model.parameters()):
            target*param.data.copy*(tau * local*param.data + (1.0 - tau) * target*param.data)

    def hard_update(self, target: nn.Module, source: nn.Module) -> None:
        """Copy weights from source to target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target*param.data.copy*(param.data)


# Initialize Agent for Smoke Test
agent = DQNAgent(state*size=4, action*size=2, use*dueling=True, use*double_dqn=True)
print("DQN Agent initialized successfully!")
print(f"Network type: {'Dueling' if isinstance(agent.q_network, DuelingDQN) else 'Standard'}")
print(f"Double DQN enabled: {agent.use*double*dqn}")

```

    DQN Agent initialized successfully!
    Network type: Dueling
    Double DQN enabled: True


## Part 3: Policy Gradient Methods

### 3.1 Policy Gradient Theorem

Instead of learning value functions, policy gradient methods directly optimize the policy parameters $\theta$.

**Policy Gradient Theorem:**
$$\nabla*\theta J(\theta) = \mathbb{E}*{\pi*\theta} \left[ \nabla*\theta \log \pi*\theta(a|s) \cdot Q^{\pi*\theta}(s,a) \right]$$

Where $J(\theta)$ is the expected return under policy $\pi_\theta$.

### 3.2 Reinforce Algorithm

REINFORCE uses Monte Carlo sampling to estimate the policy gradient:

**REINFORCE Update:**
$$\theta*{t+1} = \theta*t + \alpha \nabla*\theta \log \pi*\theta(a*t|s*t) G_t$$

Where $G_t$ is the return from time step $t$.

### 3.3 Actor-critic Methods

Actor-Critic combines policy gradient (actor) with value function approximation (critic):

- **Actor**: Updates policy parameters using policy gradient
- **Critic**: Updates value function parameters using TD learning

**Actor Update:**
$$\theta*{t+1} = \theta*t + \alpha*\theta \nabla*\theta \log \pi*\theta(a*t|s*t) \delta*t$$

**Critic Update:**
$$w*{t+1} = w*t + \alpha*w \delta*t \nabla*w V*w(s_t)$$

Where $\delta*t = r*t + \gamma V*w(s*{t+1}) - V*w(s*t)$ is the TD error.


```python
class PolicyNetwork(nn.Module):
    """
    Simple policy network that outputs action probabilities for discrete action spaces.
    """
    def **init**(self, state*size: int, action*size: int, hidden_size: int = 64) -> None:
        super(PolicyNetwork, self).**init**()
        self.fc1 = nn.Linear(state*size, hidden*size)
        self.fc2 = nn.Linear(hidden*size, hidden*size)
        self.fc3 = nn.Linear(hidden*size, action*size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class REINFORCEAgent:
    """
    REINFORCE policy gradient agent using Monte-Carlo returns.
    Stores log-probabilities to compute the policy gradient at episode end.
    """
    def **init**(self, state*size: int, action*size: int, lr: float = 1e-3, gamma: float = 0.99) -> None:
        self.state*size = state*size
        self.action*size = action*size
        self.gamma = gamma

        # Policy network and optimizer
        self.policy = PolicyNetwork(state*size, action*size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode storage
        self.reset_episode()

    def reset_episode(self) -> None:
        """Clear episode buffers."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []

    def act(self, state: np.ndarray) -> int:
        """Sample an action from the policy and store log-probability."""
        state*t = torch.from*numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy(state_t)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log*probs.append(m.log*prob(action))
        return int(action.item())

    def step(self, state: np.ndarray, action: int, reward: float) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self) -> float:
        """Compute discounted returns, update policy, and return loss value."""
        # Compute discounted returns
        returns: List[float] = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns).float().to(device)
        returns*t = (returns*t - returns*t.mean()) / (returns*t.std() + 1e-8)

        # Policy loss
        policy_loss = []
        for log*prob, Gt in zip(self.log*probs, returns_t):
            policy*loss.append(-log*prob * Gt)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # Clear episode buffers
        self.reset_episode()
        return float(loss.item())


class ValueNetwork(nn.Module):
    """Value network used by Actor-Critic (returns scalar state-values)."""
    def **init**(self, state*size: int, hidden*size: int = 64) -> None:
        super(ValueNetwork, self).**init**()
        self.fc1 = nn.Linear(state*size, hidden*size)
        self.fc2 = nn.Linear(hidden*size, hidden*size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorCriticAgent:
    """A simple one-step Actor-Critic agent using a separate actor and critic."""
    def **init**(self, state*size: int, action*size: int, lr*actor: float = 1e-3, lr*critic: float = 1e-3, gamma: float = 0.99) -> None:
        self.state*size = state*size
        self.action*size = action*size
        self.gamma = gamma

        # Actor and critic
        self.actor = PolicyNetwork(state*size, action*size).to(device)
        self.actor*optimizer = optim.Adam(self.actor.parameters(), lr=lr*actor)

        self.critic = ValueNetwork(state_size).to(device)
        self.critic*optimizer = optim.Adam(self.critic.parameters(), lr=lr*critic)

    def act(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state*t = torch.from*numpy(state).float().unsqueeze(0).to(device)
        probs = self.actor(state_t)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return int(action.item()), m.log_prob(action)

    def learn(self, state: np.ndarray, action: int, reward: float, next*state: np.ndarray, done: bool, log*prob: torch.Tensor) -> Tuple[float, float]:
        state*t = torch.from*numpy(state).float().unsqueeze(0).to(device)
        next*state*t = torch.from*numpy(next*state).float().unsqueeze(0).to(device)
        reward_t = torch.tensor([reward]).float().to(device)
        done_t = torch.tensor([done]).float().to(device)

        # Critic update: compute TD target & error
        current*value = self.critic(state*t)
        next*value = self.critic(next*state*t) if not done else torch.zeros*like(current_value).to(device)

        td*target = reward*t + self.gamma * next*value * (1 - done*t)
        td*error = td*target - current_value

        critic*loss = td*error.pow(2).mean()

        self.critic*optimizer.zero*grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update using the TD error as advantage estimate
        actor*loss = (-log*prob * td_error.detach()).mean()

        self.actor*optimizer.zero*grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return float(actor*loss.item()), float(critic*loss.item())


# Initialize Agents for Smoke Tests
reinforce*agent = REINFORCEAgent(state*size=4, action_size=2, lr=1e-3)
ac*agent = ActorCriticAgent(state*size=4, action_size=2)
print("REINFORCE and Actor-Critic agents initialized.")

```

    REINFORCE and Actor-Critic agents initialized.



```python
class ValueNetwork(nn.Module):
    """
    Value Network for Actor-Critic
    """
    def **init**(self, state*size, hidden*size=64):
        super(ValueNetwork, self).**init**()
        self.fc1 = nn.Linear(state*size, hidden*size)
        self.fc2 = nn.Linear(hidden*size, hidden*size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorCriticAgent:
    """
    Actor-Critic Agent implementation
    """
    def **init**(self, state*size, action*size, lr*actor=1e-3, lr*critic=1e-3, gamma=0.99):
        self.state*size = state*size
        self.action*size = action*size
        self.gamma = gamma

        # Actor network (policy)
        self.actor = PolicyNetwork(state*size, action*size).to(device)
        self.actor*optimizer = optim.Adam(self.actor.parameters(), lr=lr*actor)

        # Critic network (value function)
        self.critic = ValueNetwork(state_size).to(device)
        self.critic*optimizer = optim.Adam(self.critic.parameters(), lr=lr*critic)

    def act(self, state):
        """Choose action based on policy"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.actor(state)

        # Sample action from probability distribution
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)

    def learn(self, state, action, reward, next*state, done, log*prob):
        """Update actor and critic using one-step TD error"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next*state = torch.from*numpy(next_state).float().unsqueeze(0).to(device)
        reward = torch.tensor([reward]).float().to(device)
        done = torch.tensor([done]).float().to(device)

        # Critic update
        current_value = self.critic(state)
        next*value = self.critic(next*state) if not done else torch.zeros(1).to(device)

        # TD target and error
        td*target = reward + self.gamma * next*value * (1 - done)
        td*error = td*target - current_value

        # Critic loss (squared TD error)
        critic*loss = td*error.pow(2)

        # Actor loss (policy gradient with baseline)
        actor*loss = -log*prob * td_error.detach()

        # Update critic
        self.critic*optimizer.zero*grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        self.actor*optimizer.zero*grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor*loss.item(), critic*loss.item()

# Initialize Actor-critic Agent
ac*agent = ActorCriticAgent(state*size=4, action*size=2, lr*actor=1e-3, lr_critic=1e-3)
print("Actor-Critic Agent initialized successfully!")
print("Actor (Policy) network:")
print(ac_agent.actor)
print("\nCritic (Value) network:")
print(ac_agent.critic)
```

    Actor-Critic Agent initialized successfully!
    Actor (Policy) network:
    PolicyNetwork(
      (fc1): Linear(in*features=4, out*features=64, bias=True)
      (fc2): Linear(in*features=64, out*features=64, bias=True)
      (fc3): Linear(in*features=64, out*features=2, bias=True)
    )
    
    Critic (Value) network:
    ValueNetwork(
      (fc1): Linear(in*features=4, out*features=64, bias=True)
      (fc2): Linear(in*features=64, out*features=64, bias=True)
      (fc3): Linear(in*features=64, out*features=1, bias=True)
    )


## Part 4: Practical Implementation and Comparison

### 4.1 Training Environment Setup

We'll use the CartPole environment from OpenAI Gym to demonstrate the algorithms:

- **State Space**: 4-dimensional continuous (position, velocity, angle, angular velocity)
- **Action Space**: 2 discrete actions (left, right)
- **Reward**: +1 for every step the pole stays upright
- **Episode Termination**: Pole angle > 15° or cart position > 2.4 units
- **Success Criteria**: Average reward > 195 over 100 consecutive episodes


```python
def gym_reset(env: gym.Env) -> np.ndarray:
    """Handle different gym reset return signatures (state or (state, info))."""
    result = env.reset()
    if isinstance(result, tuple):
        state, _ = result
    else:
        state = result
    return np.array(state, dtype=np.float32)


def gym_step(env: gym.Env, action: Any) -> Tuple[np.ndarray, float, bool, dict]:
    """Handle different gym step return signatures and agent action types.

    Args:
        env: The gymnasium environment.
        action: The action to take. Can be an integer (for discrete actions) or a tuple
                (e.g., from Actor-Critic act method).

    Returns: next_state, reward, done, info
    """
    # Handle case where agent returns a tuple (e.g., Actor-Critic returns action and log_prob)
    if isinstance(action, tuple):
        action*to*env = action[0] # Assume the first element is the action
    else:
        action*to*env = action

    result = env.step(action*to*env)
    if len(result) == 4:
        next_state, reward, done, info = result
    else:  # new API: (obs, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    return np.array(next_state, dtype=np.float32), float(reward), bool(done), info


def train*dqn*agent(agent: DQNAgent, env: gym.Env, n*episodes: int = 1000, max*t: int = 1000) -> List[float]:
    """Train DQN agent and return per-episode scores."""
    scores: List[float] = []
    scores_window = deque(maxlen=100)

    for i*episode in range(1, n*episodes + 1):
        state = gym_reset(env)
        score = 0.0

        for t in range(max_t):
            action = agent.act(state)
            next*state, reward, done, * = gym_step(env, action)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)

        if i_episode % 100 == 0:
            print(f'Episode {i*episode}\tAverage Score: {np.mean(scores*window):.2f}\tEpsilon: {agent.epsilon:.3f}')

        if len(scores*window) == scores*window.maxlen and np.mean(scores_window) >= 195.0:
            print(f'Environment solved in {i*episode} episodes!\tAverage Score: {np.mean(scores*window):.2f}')
            break

    return scores


def train*reinforce*agent(agent: REINFORCEAgent, env: gym.Env, n*episodes: int = 1000, max*t: int = 1000) -> List[float]:
    """Train REINFORCE agent (full-episode updates)."""
    scores: List[float] = []
    scores_window = deque(maxlen=100)

    for i*episode in range(1, n*episodes + 1):
        state = gym_reset(env)
        agent.reset_episode()
        score = 0.0

        for t in range(max_t):
            action = agent.act(state)
            next*state, reward, done, * = gym_step(env, action)
            agent.step(state, action, reward)
            state = next_state
            score += reward
            if done:
                break

        loss = agent.learn()
        scores_window.append(score)
        scores.append(score)

        if i_episode % 100 == 0:
            print(f'Episode {i*episode}\tAverage Score: {np.mean(scores*window):.2f}\tLoss: {loss:.3f}')

        if len(scores*window) == scores*window.maxlen and np.mean(scores_window) >= 195.0:
            print(f'Environment solved in {i*episode} episodes!\tAverage Score: {np.mean(scores*window):.2f}')
            break

    return scores


def train*actor*critic*agent(agent: ActorCriticAgent, env: gym.Env, n*episodes: int = 1000, max_t: int = 1000) -> List[float]:
    """Train Actor-Critic agent with online updates per step."""
    scores: List[float] = []
    scores_window = deque(maxlen=100)

    for i*episode in range(1, n*episodes + 1):
        state = gym_reset(env)
        score = 0.0

        for t in range(max_t):
            action, log_prob = agent.act(state)
            next*state, reward, done, * = gym*step(env, (action, log*prob)) # Pass the tuple to gym_step
            actor*loss, critic*loss = agent.learn(state, action, reward, next*state, done, log*prob)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)

        if i_episode % 100 == 0:
            print(f'Episode {i*episode}\tAverage Score: {np.mean(scores*window):.2f}\tActorLoss: {actor*loss:.3f}\tCriticLoss: {critic*loss:.3f}')

        if len(scores*window) == scores*window.maxlen and np.mean(scores_window) >= 195.0:
            print(f'Environment solved in {i*episode} episodes!\tAverage Score: {np.mean(scores*window):.2f}')
            break

    return scores

print("Training functions defined successfully!")
```

    Training functions defined successfully!



```python
# Create Environment (use Gym If Available; Otherwise Fallback to a Simple Mockenv)
try:
    env_name = 'CartPole-v1'
    try:
        env = gym.make(env_name)
        print(f"Environment '{env_name}' created successfully!")
        print(f"State space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Quick reset test
        state = gym_reset(env)
        print(f"Initial state shape: {np.shape(state)}")

    except Exception as e:
        print(f"Error creating gym environment: {e}")
        print("Creating mock environment for demonstration...")
        raise ImportError from e

except Exception:
    # Fallback simple mock environment for demonstrations and testing
    class MockEnv:
        def **init**(self):
            self.observation_space = type('', (), {'shape': (4,)})()
            self.action_space = type('', (), {'n': 2})()
            self.state = np.random.random(4).astype(np.float32)

        def reset(self):
            self.state = np.random.random(4).astype(np.float32)
            return self.state

        def step(self, action):
            self.state = np.random.random(4).astype(np.float32)
            reward = float(np.random.randint(0, 2))
            done = np.random.random() < 0.05
            return self.state, reward, done, {}

    env = MockEnv()
    print("Mock environment created for demonstration purposes.")

```

    Environment 'CartPole-v1' created successfully!
    State space: Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
    Action space: Discrete(2)
    Initial state shape: (4,)



```python
%pip install gymnasium
```

    Collecting gymnasium
      Downloading gymnasium-1.2.1-py3-none-any.whl.metadata (10.0 kB)
    Requirement already satisfied: numpy>=1.21.0 in /Users/tahamajs/Documents/uni/DRL/venv/lib/python3.13/site-packages (from gymnasium) (2.2.6)
    Requirement already satisfied: cloudpickle>=1.2.0 in /Users/tahamajs/Documents/uni/DRL/venv/lib/python3.13/site-packages (from gymnasium) (3.1.1)
    Requirement already satisfied: typing-extensions>=4.3.0 in /Users/tahamajs/Documents/uni/DRL/venv/lib/python3.13/site-packages (from gymnasium) (4.15.0)
    Collecting farama-notifications>=0.0.1 (from gymnasium)
      Downloading Farama_Notifications-0.0.4-py3-none-any.whl.metadata (558 bytes)
    Downloading gymnasium-1.2.1-py3-none-any.whl (951 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m951.1/951.1 kB[0m [31m1.4 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hDownloading Farama_Notifications-0.0.4-py3-none-any.whl (2.5 kB)
    Installing collected packages: farama-notifications, gymnasium
    Successfully installed farama-notifications-0.0.4 gymnasium-1.2.1
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.3.1[0m[39;49m -> [0m[32;49m25.2[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.



```python
# Create Environment (use Gymnasium If Available; Otherwise Fallback to a Simple Mockenv)
try:
    env_name = 'CartPole-v1'
    import gymnasium as gym # Import gymnasium instead of gym
    try:
        env = gym.make(env_name)
        print(f"Environment '{env_name}' created successfully!")
        print(f"State space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Quick reset test
        state, info = env.reset(seed=42) # Use new reset signature
        print(f"Initial state shape: {np.shape(state)}")

    except Exception as e:
        print(f"Error creating gymnasium environment: {e}")
        print("Creating mock environment for demonstration...")
        raise ImportError from e

except Exception:
    # Fallback simple mock environment for demonstrations and testing
    class MockEnv:
        def **init**(self):
            self.observation_space = type('', (), {'shape': (4,)})()
            self.action_space = type('', (), {'n': 2})()
            self.state = np.random.random(4).astype(np.float32)

        def reset(self, seed=None, options=None): # Mock reset should match new signature
            if seed is not None:
                 np.random.seed(seed)
            self.state = np.random.random(4).astype(np.float32)
            return self.state, {} # Return state and info tuple

        def step(self, action):
            self.state = np.random.random(4).astype(np.float32)
            reward = float(np.random.randint(0, 2))
            terminated = np.random.random() < 0.05
            truncated = False # Mock env doesn't truncate
            info = {}
            return self.state, reward, terminated, truncated, info # Return new step signature

    env = MockEnv()
    print("Mock environment created for demonstration purposes.")
```

    Environment 'CartPole-v1' created successfully!
    State space: Box([-4.8               -inf -0.41887903        -inf], [4.8               inf 0.41887903        inf], (4,), float32)
    Action space: Discrete(2)
    Initial state shape: (4,)



```python
# Demonstration of Training (short Episodes for Notebook Demonstration)
n*demo*episodes = 50

print("=" * 60)
print("TRAINING DEMONSTRATION (short runs)")
print("=" * 60)

# Initialize Fresh Agents for Fair Comparison
# Keep Models Small for Quick Demos in Notebooks
*demo*hidden = 64

dqn*agent*demo = DQNAgent(state*size=4, action*size=2, use*dueling=True, use*double_dqn=True)
reinforce*agent*demo = REINFORCEAgent(state*size=4, action*size=2)
ac*agent*demo = ActorCriticAgent(state*size=4, action*size=2)

# Store Results
results = {}

print("\n1. Training DQN Agent (demo)...")
dqn*scores = train*dqn*agent(dqn*agent*demo, env, n*episodes=n*demo*episodes, max_t=200)
results['DQN'] = dqn_scores

print("\n2. Training REINFORCE Agent (demo)...")
reinforce*scores = train*reinforce*agent(reinforce*agent*demo, env, n*episodes=n*demo*episodes, max_t=200)
results['REINFORCE'] = reinforce_scores

print("\n3. Training Actor-Critic Agent (demo)...")
ac*scores = train*actor*critic*agent(ac*agent*demo, env, n*episodes=n*demo*episodes, max*t=200)
results['Actor-Critic'] = ac_scores

print("\nTraining demonstration completed!")
```

    ============================================================
    TRAINING DEMONSTRATION (short runs)
    ============================================================
    
    1. Training DQN Agent (demo)...
    
    2. Training REINFORCE Agent (demo)...
    
    3. Training Actor-Critic Agent (demo)...
    
    Training demonstration completed!



```python
# Visualize Training Results (if Results Are Available)
if not results:
    print("No results to plot. Run the demo training cell first.")
else:
    plt.figure(figsize=(15, 10))

    # Plot 1: Learning curves with moving averages
    plt.subplot(2, 2, 1)
    for algorithm, scores in results.items():
        plt.plot(scores, label=algorithm, linewidth=2)
        if len(scores) >= 3:
            window_size = min(10, max(1, len(scores)))
            ma = moving*average(scores, window=window*size)
            # Adjust the x-axis range for the moving average plot
            plt.plot(range(window_size - 1, len(scores)), ma, '--', alpha=0.7, linewidth=1)

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Final performance comparison
    plt.subplot(2, 2, 2)
    final_scores = [np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores) if len(scores) > 0 else 0.0 for scores in results.values()]
    algorithms = list(results.keys())
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = plt.bar(algorithms, final_scores, color=colors[:len(algorithms)], alpha=0.8)
    plt.ylabel('Average Score (Last 10 Episodes)')
    plt.title('Final Performance Comparison')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, final_scores):
        plt.text(bar.get*x() + bar.get*width() / 2, bar.get_height() + 1, f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Score distribution
    plt.subplot(2, 2, 3)
    for i, (algorithm, scores) in enumerate(results.items()):
        plt.hist(scores, bins=15, alpha=0.6, label=algorithm, color=colors[i % len(colors)], density=True)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Statistical summary
    plt.subplot(2, 2, 4)
    stats_data = []
    for algorithm, scores in results.items():
        stats_data.append({
            'Algorithm': algorithm,
            'Mean': float(np.mean(scores)) if len(scores) > 0 else 0.0,
            'Std': float(np.std(scores)) if len(scores) > 0 else 0.0,
            'Max': float(np.max(scores)) if len(scores) > 0 else 0.0,
            'Min': float(np.min(scores)) if len(scores) > 0 else 0.0
        })

    import pandas as pd
    df*stats = pd.DataFrame(stats*data).set_index('Algorithm')
    df_stats.plot(kind='bar', ax=plt.gca())
    plt.title('Statistical Summary')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(bbox*to*anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save the visualization
    import os
    os.makedirs('visualization', exist_ok=True)
    plt.savefig('visualization/learning*curves.png', dpi=300, bbox*inches='tight')

    # Save results as JSON
    import json
    json.dump(results, open('results.json', 'w'))

    plt.show()

    # Print detailed comparison
    print("\n" + "=" * 60)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("=" * 60)

    for algorithm, scores in results.items():
        if len(scores) == 0:
            print(f"\n{algorithm}: No data")
            continue
        print(f"\n{algorithm}:")
        print(f"  Mean Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        print(f"  Max Score: {np.max(scores):.2f}")
        print(f"  Min Score: {np.min(scores):.2f}")
        print(f"  Final 10 Episodes: {np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores):.2f}")

        best_avg = max([np.mean(scores[i:i+10]) for i in range(len(scores)-9)] if len(scores) >= 10 else [np.mean(scores)])
        print(f"  Best 10-Episode Average: {best_avg:.2f}")
        print(f"  Environment Solved: {'Yes' if best_avg >= 195 else 'No'}")
```


    
![png](CA1*files/CA1*15_0.png)
    


    
    ============================================================
    DETAILED PERFORMANCE ANALYSIS
    ============================================================
    
    DQN:
      Mean Score: 15.42 ± 6.06
      Max Score: 33.00
      Min Score: 8.00
      Final 10 Episodes: 11.60
      Best 10-Episode Average: 20.50
      Environment Solved: No
    
    REINFORCE:
      Mean Score: 25.90 ± 11.76
      Max Score: 66.00
      Min Score: 9.00
      Final 10 Episodes: 29.10
      Best 10-Episode Average: 33.10
      Environment Solved: No
    
    Actor-Critic:
      Mean Score: 26.50 ± 19.56
      Max Score: 118.00
      Min Score: 10.00
      Final 10 Episodes: 48.00
      Best 10-Episode Average: 48.00
      Environment Solved: No


## Part 5: Exercises and Questions

### Exercise 1: Theoretical Understanding

**Question 1.1**: Explain the difference between on-policy and off-policy learning. Which algorithms implemented in this notebook are on-policy and which are off-policy?

**Answer**:
**On-policy vs. Off-policy Learning:**

The distinction lies in how data is used to update the policy.

-   **On-policy algorithms** update the policy based on actions taken by the *current* version of that same policy. The agent learns from the experience it generates while following its own strategy. It's like learning to cook by trying your own recipes and adjusting them based on how the food tastes. You learn from what you are currently doing.

-   **Off-policy algorithms** update the policy using data generated by a *different* policy. The agent can learn from past experiences (e.g., from a replay buffer) or from observing another agent. This separates data collection (exploration) from the learning of the optimal policy (exploitation). It's like learning to cook by watching a master chef's videos; you learn from their experience, not your own.

**Algorithms in this Notebook:**

-   **DQN (Deep Q-Network)** is **off-policy**. It uses a replay buffer to store past experiences, which may have been generated by older versions of the policy. The learning update samples from this buffer, so the data used for learning is not strictly from the current policy. This improves sample efficiency and stability.

-   **REINFORCE** is **on-policy**. It collects a full trajectory of states, actions, and rewards using its current policy. At the end of the episode, it uses this trajectory to update the policy. The data is then discarded, and a new trajectory is collected with the updated policy.

-   **Actor-Critic** (as implemented here) is **on-policy**. The actor (policy) generates an action, and the critic evaluates it. The updates are based on this immediate experience. The data is generated and used by the current policy, and then the process repeats.

---

**Question 1.2**: What is the exploration-exploitation dilemma in reinforcement learning? How do the three algorithms (DQN, REINFORCE, Actor-Critic) handle this dilemma?

**Answer**:
**The Exploration-Exploitation Dilemma:**

This is a fundamental challenge in reinforcement learning. The agent must make a trade-off between:
-   **Exploitation**: Taking the action it currently believes is the best to maximize immediate reward. This leverages known information.
-   **Exploration**: Taking a different, potentially suboptimal action to gather more information about the environment. This might lead to discovering a better long-term strategy.

The dilemma is that excessive exploration can lead to poor performance, while excessive exploitation can cause the agent to get stuck in a suboptimal strategy, never discovering better alternatives.

**How the Algorithms Handle It:**

-   **DQN**: Uses an **ε-greedy (epsilon-greedy) strategy**. With a probability `ε`, the agent takes a random action (exploration). With probability `1-ε`, it takes the action with the highest estimated Q-value (exploitation). Typically, `ε` starts high (e.g., 1.0) and is gradually decayed to a small value (e.g., 0.01), shifting the agent from exploration to exploitation as it learns more about the environment.

-   **REINFORCE**: Handles exploration through its **stochastic policy**. The policy network outputs a probability distribution over all possible actions. Actions are then sampled from this distribution. This means that even actions with lower probabilities have a non-zero chance of being selected, leading to natural exploration. As the policy improves, it will assign higher probabilities to better actions, but the inherent randomness ensures exploration continues.

-   **Actor-Critic**: Similar to REINFORCE, the **actor is a stochastic policy**. It outputs probabilities for each action, and actions are sampled accordingly. This inherent stochasticity ensures exploration. The critic's feedback helps refine these probabilities, but the agent will always have a chance to try different actions.

---

**Question 1.3**: Derive the policy gradient theorem starting from the performance measure $J(\theta) = \mathbb{E}_{s \sim \rho^\pi}[V^\pi(s)]$.

**Answer**:
The goal is to find the gradient of the performance measure $J(\theta)$ with respect to the policy parameters $\theta$. We start with the definition of the state-value function:
$V^\pi(s) = \mathbb{E}*\pi[G*t | S_t=s]$

The policy gradient theorem states:
$$\nabla*\theta J(\theta) \propto \sum*s d^\pi(s) \sum*a \nabla*\theta \pi_\theta(a|s) Q^\pi(s,a)$$

Here is a common derivation:
1.  Start with the gradient of the state-value function:
    $\nabla*\theta V^\pi(s) = \nabla*\theta \sum*a \pi*\theta(a|s) Q^\pi(s,a)$
    $= \sum*a [\nabla*\theta \pi*\theta(a|s) Q^\pi(s,a) + \pi*\theta(a|s) \nabla_\theta Q^\pi(s,a)]$ (Product Rule)

2.  Now expand the gradient of the Q-value function:
    $\nabla*\theta Q^\pi(s,a) = \nabla*\theta \sum_{s',r} p(s',r|s,a) [r + \gamma V^\pi(s')]$
    $= \gamma \sum*{s'} p(s'|s) \nabla*\theta V^\pi(s')$

3.  Substitute (2) back into (1):
    $\nabla*\theta V^\pi(s) = \sum*a [\nabla*\theta \pi*\theta(a|s) Q^\pi(s,a) + \pi*\theta(a|s) \gamma \sum*{s'} p(s'|s,a) \nabla_\theta V^\pi(s')]$

4.  This equation expresses a recursive relationship for the gradient. If we unroll it, we can see how the gradient at a state `s` depends on the gradients of future states. Let's define the discounted state distribution $d^\pi(s)$.
    The performance measure is $J(\theta) = V^\pi(s_0)$.
    $\nabla*\theta J(\theta) = \nabla*\theta V^\pi(s_0)$

5.  Unrolling the recursion from step 3 gives:
    $\nabla*\theta J(\theta) = \sum*{x \in S} d^\pi(x) \sum*a \nabla*\theta \pi_\theta(a|x) Q^\pi(x,a)$

6.  Now, use the **log-derivative trick**: $\nabla*\theta \pi*\theta(a|s) = \pi*\theta(a|s) \nabla*\theta \log \pi_\theta(a|s)$.
    Substitute this into the equation:
    $\nabla*\theta J(\theta) = \sum*{s \in S} d^\pi(s) \sum*a \pi*\theta(a|s) (\nabla*\theta \log \pi*\theta(a|s)) Q^\pi(s,a)$

7.  This can be expressed as an expectation:
    $\nabla*\theta J(\theta) = \mathbb{E}*{\pi*\theta} [\nabla*\theta \log \pi*\theta(A*t|S*t) Q^{\pi*\theta}(S*t, A*t)]$

This final form is the most common expression of the Policy Gradient Theorem. It tells us to increase the probability of actions that lead to higher-than-expected rewards.

---

### Exercise 2: Implementation Analysis

**Question 2.1**: Compare the memory requirements of DQN vs REINFORCE. Which algorithm requires more memory and why?

**Answer**:
**DQN requires significantly more memory than REINFORCE.**

The primary reason is the **Experience Replay Buffer** in DQN.
-   **DQN**: To improve stability and sample efficiency, DQN stores a large number of past transitions (`state`, `action`, `reward`, `next_state`, `done`) in a replay buffer. This buffer can be very large (e.g., holding 10,000 to 1,000,000 experiences). The agent then samples mini-batches from this buffer to perform learning updates. The memory footprint is dominated by this buffer.
-   **REINFORCE**: This algorithm is much more memory-efficient. It only needs to store the states, actions, and rewards for the *current episode*. Once the episode is finished, it uses this data to perform a single policy update, and then the data is discarded. The memory required is proportional to the length of one episode, which is typically much smaller than the capacity of a DQN replay buffer.

---

**Question 2.2**: Explain why we use a target network in DQN. What would happen if we removed it?

**Answer**:
**Why we use a target network:**

The target network is a crucial innovation for stabilizing the learning process in DQN. The Q-learning update involves calculating a target value: $y*t = r*t + \gamma \max*{a'} Q(s*{t+1}, a'; \theta)$.

If we use the *same* network for both estimating the current Q-value ($Q(s*t, a*t; \theta)$) and the target Q-value ($Q(s*{t+1}, a'; \theta)$), a problem arises. Every time we update the network weights $\theta$, the target value $y*t$ also changes. This is like trying to hit a moving target. The learning process can become unstable, leading to oscillations or divergence.

The **target network** solves this by providing a stable, fixed target for a period of time. It is a separate network whose weights ($\theta^-$) are a copy of the main Q-network's weights. These weights are held constant for several training steps and are only updated periodically (e.g., by copying the main network's weights every C steps, or through a slow "soft" update).

**What would happen if we removed it?**

Without the target network, the Q-learning target would be constantly shifting. This leads to several problems:
1.  **Instability**: The learning process is more likely to be unstable and may diverge. The loss can fluctuate wildly instead of smoothly converging.
2.  **Poor Performance**: The agent would have a much harder time learning an effective policy because it is chasing a non-stationary target.
3.  **Correlations**: The updates would be highly correlated with the current weights, which can lead to a feedback loop where incorrect Q-value estimates are reinforced.

---

**Question 2.3**: In the Dueling DQN architecture, why do we subtract the mean of the advantage values? What would happen if we didn't do this?

**Answer**:
**Why we subtract the mean of the advantage values:**

The core idea of Dueling DQN is to separately estimate the state-value function $V(s)$ and the action-advantage function $A(s,a)$. The Q-value is then reconstructed as:
$Q(s,a) = V(s) + A(s,a)$

However, this formula has an **identifiability problem**. Given a Q-value, we cannot uniquely determine the values of $V(s)$ and $A(s,a)$. For example, we could add a constant `c` to $V(s)$ and subtract it from all $A(s,a)$ values, and the resulting Q-value would be the same. This ambiguity can make training less stable.

To solve this, we enforce a constraint on the advantage function. By subtracting the mean of the advantages, we ensure that the average advantage for any state is zero:
$$Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a') \right)$$
This forces $V(s)$ to be a good estimate of the state value, as it becomes the central point around which the advantages fluctuate. It stabilizes learning by ensuring that the advantage of the chosen action is a relative measure compared to the other actions.

**What would happen if we didn't do this?**

Without subtracting the mean, the network could learn to produce the same Q-values in many different ways. For example, it could:
-   Set $V(s)$ to zero and have all the Q-value information in $A(s,a)$.
-   Set all $A(s,a)$ to zero and have all the Q-value information in $V(s)$.

This ambiguity makes it difficult for the optimizer to know how to attribute the TD-error during backpropagation. The network might learn to change $V(s)$ when it should be changing $A(s,a)$, or vice-versa. This leads to **poorer performance and less stable training**. Subtracting the mean provides a clear separation of concerns, improving learning efficiency.

---

### Exercise 3: Experimental Design

Design and implement an experiment to compare the sample efficiency of the three algorithms. Consider:
- How would you measure sample efficiency?
- What metrics would you use?
- How would you ensure a fair comparison?

**Answer**:
**Experimental Design for Sample Efficiency:**

Sample efficiency refers to how much data (i.e., how many interactions with the environment) an agent needs to achieve a certain level of performance. An algorithm is more sample-efficient if it learns faster from fewer interactions.

**1. How to Measure Sample Efficiency:**

We can measure this by tracking the total number of environment steps (timesteps) taken by the agent. This is a more direct measure of experience than the number of episodes, as episodes can have variable lengths.

**2. Metrics to Use:**

-   **Timesteps to Threshold**: The primary metric would be the number of total environment interactions (timesteps) required to reach a predefined performance threshold (e.g., an average score of 195 over 100 episodes for CartPole). The algorithm that reaches this threshold in fewer timesteps is more sample-efficient.
-   **Area Under the Learning Curve (AUC)**: Plot the average score against the number of timesteps. A higher area under the curve indicates that the agent achieved higher scores earlier, signifying better sample efficiency.
-   **Performance after a Fixed Number of Steps**: Compare the average score of each algorithm after a fixed number of timesteps (e.g., after 50,000 steps). The algorithm with the higher score is more sample-efficient up to that point.

**3. How to Ensure a Fair Comparison:**

To ensure the comparison is fair, we must control for confounding variables:
-   **Identical Environments**: All agents must be trained on the exact same environment, initialized with the same random seed for the environment itself.
-   **Consistent Hyperparameters**: Use equivalent network architectures (e.g., same number of layers and hidden units) for all agents. Hyperparameters like learning rate and discount factor ($\gamma$) should be kept consistent or tuned optimally for each algorithm to ensure each is performing at its best.
-   **Averaging over Multiple Runs**: RL training can have high variance. To get reliable results, each experiment should be run multiple times (e.g., 5-10 runs) with different random seeds (for agent initialization and action selection). The results (e.g., timesteps to threshold) should then be averaged, and standard deviations should be reported to show the variance.
-   **Consistent Evaluation**: Use the same evaluation protocol for all agents, such as measuring the average score over the last 100 episodes.
-   **Total Timesteps**: The x-axis of all plots should be the total number of environment steps, not episodes, to account for varying episode lengths.


```python
# Exercise 3: Sample Efficiency Experiment
from collections import defaultdict

def sample*efficiency*experiment(env_factory,
                                 agent_factories: dict,
                                 threshold: float = 195.0,
                                 max_episodes: int = 1000,
                                 max*steps*per_episode: int = 1000,
                                 runs: int = 3) -> dict:
    """Run a small sample-efficiency experiment.

    Args:
        env_factory: callable that returns a fresh environment when called.
        agent_factories: dict[str, callable] mapping name->callable that returns a fresh agent.
        threshold: score threshold to consider the environment "solved" (average over window).
        max_episodes: maximum episodes to run per trial.
        max*steps*per_episode: maximum steps per episode.
        runs: number of independent runs (different random seeds) per agent.

    Returns:
        results: dict with per-agent statistics including timesteps-to-threshold per run
                 and summary statistics (mean, std). If threshold not reached, value is None.
    """
    rng = np.random.RandomState(0)
    results = {}

    for name, make*agent in agent*factories.items():
        steps*to*threshold = []
        for run in range(runs):
            env = env_factory()
            agent = make_agent()
            total_steps = 0
            scores_window = deque(maxlen=100)
            solved_at = None

            for ep in range(1, max_episodes + 1):
                state = gym_reset(env)
                score = 0.0
                for t in range(max*steps*per_episode):
                    action = agent.act(state)
                    next*state, reward, done, * = gym_step(env, action)
                    # If agent has step method (DQN), call it to store experience
                    if hasattr(agent, 'step'):
                        try:
                            agent.step(state, action, reward, next_state, done)
                        except Exception:
                            pass # Handle cases where step might not match exact signature for non-DQN agents
                    # If agent is REINFORCE or Actor-Critic, their step/learn is called differently
                    # REINFORCE steps accumulate in the agent until learn() is called at episode end
                    # Actor-Critic learns at every step
                    elif isinstance(agent, REINFORCEAgent):
                        agent.step(state, action, reward) # Accumulate experience in REINFORCE

                    state = next_state
                    score += reward
                    total_steps += 1
                    if done:
                        # For REINFORCE, call learn() at the end of the episode
                        if isinstance(agent, REINFORCEAgent):
                             try:
                                 agent.learn()
                             except Exception:
                                 pass # Handle potential errors during REINFORCE learn
                        break


                scores_window.append(score)

                # Check solve condition
                if len(scores*window) == scores*window.maxlen and np.mean(scores_window) >= threshold:
                    solved*at = total*steps
                    break

            steps*to*threshold.append(solved_at)
            # cleanup env if possible
            try:
                env.close()
            except Exception:
                pass

        # summarise
        numeric = [s for s in steps*to*threshold if s is not None]
        summary = {
            'per*run': steps*to_threshold,
            'mean_steps': float(np.mean(numeric)) if numeric else None,
            'std_steps': float(np.std(numeric)) if numeric else None,
            'runs': runs
        }
        results[name] = summary

    return results


# Exercise 4: Hyperparameter Analysis (simple Grid Search Wrapper)
def hyperparameter*sensitivity*analysis(env_factory,
                                        agent*factory*fn,
                                        lr_values: list = [1e-4, 1e-3, 1e-2],
                                        gamma_values: list = [0.9, 0.95, 0.99],
                                        n_episodes: int = 50) -> dict:
    """Very small hyperparameter sweep which returns average scores for combinations.

    agent*factory*fn should accept kwargs (lr, gamma) and return an agent instance.
    This function runs short experiments to get a rough sensitivity picture.
    """
    results = {}
    for lr in lr_values:
        for gamma in gamma_values:
            env = env_factory()
            agent = agent*factory*fn(lr=lr, gamma=gamma)
            # Assuming agent has a train method similar to DQN for this analysis
            # This part might need adjustment based on agent types
            if isinstance(agent, DQNAgent):
                 scores = train*dqn*agent(agent, env, n*episodes=n*episodes, max_t=200)
            elif isinstance(agent, REINFORCEAgent):
                 scores = train*reinforce*agent(agent, env, n*episodes=n*episodes, max_t=200)
            elif isinstance(agent, ActorCriticAgent):
                 scores = train*actor*critic*agent(agent, env, n*episodes=n*episodes, max*t=200)
            else:
                 print(f"Warning: Hyperparameter analysis not implemented for agent type {type(agent)}")
                 scores = []


            results[(lr, gamma)] = {
                'mean_score': float(np.mean(scores)) if len(scores) > 0 else 0.0,
                'scores': scores
            }
            try:
                env.close()
            except Exception:
                pass
    return results


# Exercise 5: Advanced Dqn Variants
class PrioritizedReplayBuffer:
    """A simple proportional Prioritized Replay Buffer implementation.

    Note: This is a straightforward implementation for educational purposes.
    It stores priorities in a numpy array and samples with probability p_i^alpha.
    It also provides importance-sampling (IS) weights for unbiased updates.
    """
    def **init**(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, next_state, done, priority: Optional[float] = None) -> None:
        """Add experience with given priority (or max priority if None)."""
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        if priority is None:
            # set to max priority so new transitions are likely to be sampled
            max_prio = self.priorities.max() if self.pos > 0 else 1.0
            prio = max_prio
        else:
            prio = float(priority)

        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample a batch of experiences with importance-sampling weights.

        Returns: (states, actions, rewards, next*states, dones), indices, is*weights
        """
        if len(self.buffer) == 0:
            raise ValueError("The buffer is empty")
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]

        # importance-sampling weights
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        is*weights = torch.from*numpy(weights).float().unsqueeze(1).to(device)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next*states = torch.from*numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next*states, dones), indices, is*weights

    def update_priorities(self, indices, priorities) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio) + 1e-6

    def **len**(self) -> int:
        return len(self.buffer)


class NoisyLinear(nn.Module):
    """A simple Noisy linear layer (independent Gaussian parameter noise).

    This implements: weight = mu*weight + sigma*weight * epsilon_weight
    where epsilon ~ N(0,1). During evaluation we use only mu.
    """
    def **init**(self, in*features: int, out*features: int, sigma_init: float = 0.017):
        super(NoisyLinear, self).**init**()
        self.in*features = in*features
        self.out*features = out*features

        self.mu*weight = nn.Parameter(torch.empty(out*features, in_features))
        self.sigma*weight = nn.Parameter(torch.empty(out*features, in_features))

        self.mu*bias = nn.Parameter(torch.empty(out*features))
        self.sigma*bias = nn.Parameter(torch.empty(out*features))

        # initialize
        bound = 1 / np.sqrt(in_features)
        nn.init.uniform*(self.mu*weight, -bound, bound)
        nn.init.uniform*(self.mu*bias, -bound, bound)
        nn.init.constant*(self.sigma*weight, sigma_init)
        nn.init.constant*(self.sigma*bias, sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps*w = torch.randn*like(self.sigma_weight)
            eps*b = torch.randn*like(self.sigma_bias)
            weight = self.mu*weight + self.sigma*weight * eps_w
            bias = self.mu*bias + self.sigma*bias * eps_b
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)


class NoisyDQN(nn.Module):
    """DQN using NoisyLinear layers for exploration (simple two hidden layers)."""
    def **init**(self, state*size: int, action*size: int, hidden_size: int = 64):
        super(NoisyDQN, self).**init**()
        self.fc1 = NoisyLinear(state*size, hidden*size)
        self.fc2 = NoisyLinear(hidden*size, hidden*size)
        self.fc3 = NoisyLinear(hidden*size, action*size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


print("Exercise implementations completed: sample*efficiency*experiment, hyperparameter*sensitivity*analysis, PrioritizedReplayBuffer, NoisyDQN")
```

    Exercise implementations completed: sample*efficiency*experiment, hyperparameter*sensitivity*analysis, PrioritizedReplayBuffer, NoisyDQN


## Part 6: Conclusions and Analysis

### 6.1 Algorithm Comparison Summary

| Algorithm | Type | Memory | Stability | Sample Efficiency | Exploration |
|-----------|------|--------|-----------|-------------------|-------------|
| **DQN** | Value-based | High (replay buffer) | High (target network) | High | ε-greedy |
| **REINFORCE** | Policy-based | Low | Low (high variance) | Low | Stochastic policy |
| **Actor-Critic** | Hybrid | Medium | Medium | Medium | Stochastic policy |

### 6.2 Key Insights

1. **DQN Advantages**:
- Sample efficient due to experience replay
- Stable learning with target networks
- Good for discrete action spaces

2. **REINFORCE Advantages**:
- Simple implementation
- Works with continuous actions
- Direct policy optimization

3. **Actor-Critic Advantages**:
- Lower variance than REINFORCE
- Online learning capability
- Balances bias-variance tradeoff

### 6.3 When to Use Each Algorithm

- **Use DQN when**: Discrete actions, sample efficiency is important, you have memory constraints
- **Use REINFORCE when**: Simple problems, continuous actions, you need interpretable policies
- **Use Actor-Critic when**: You need balance between sample efficiency and stability

### 6.4 Advanced Topics for Further Study

1. **Advanced DQN Variants**:
- Rainbow DQN (combines multiple improvements)
- Distributional DQN
- Quantile Regression DQN

2. **Advanced Policy Methods**:
- Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)
- Soft Actor-Critic (SAC)

3. **Model-Based RL**:
- Model-Predictive Control
- Dyna-Q
- Model-based Policy Optimization

### 6.5 Further Reading

- **Books**:
- "Reinforcement Learning: An Introduction" by Sutton & Barto
- "Deep Reinforcement Learning Hands-On" by Maxim Lapan

- **Papers**:
- DQN: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- Actor-Critic: "Actor-Critic Algorithms" (Konda & Tsitsiklis, 2000)
- Policy Gradients: "Policy Gradient Methods" (Sutton et al., 1999)

## Sample Efficiency Experiment Results (updated)

The sample efficiency experiment compared DQN, REINFORCE, and Actor-Critic algorithms. Here are the latest results:

**DQN:**
- Steps to threshold (per run): [None, 30950, None]
- Mean steps to threshold: 30,950
- Std dev steps: 0

**REINFORCE:**
- Steps to threshold (per run): [None, None, None]
- Mean steps to threshold: Not solved
- Std dev steps: N/A

**Actor-Critic:**
- Steps to threshold (per run): [None, None, None]
- Mean steps to threshold: Not solved
- Std dev steps: N/A

> Only one DQN run solved the task, while REINFORCE and Actor-Critic did not solve it in any run. Consider tuning hyperparameters, increasing training steps, or reviewing the implementations for better results.


```python
# Run the Sample Efficiency Experiment
# Create a Fresh Environment Factory
def create*cartpole*env():
    import gymnasium as gym
    return gym.make('CartPole-v1')

# Define Agent Factories
agent_factories = {
    'DQN': lambda: DQNAgent(state*size=4, action*size=2, use*dueling=True, use*double_dqn=True),
    'REINFORCE': lambda: REINFORCEAgent(state*size=4, action*size=2),
    'Actor-Critic': lambda: ActorCriticAgent(state*size=4, action*size=2)
}

# Run the Experiment (increase Runs and Max_episodes for More Robust Results, but Keep Small for Demo)
print("Running sample efficiency experiment...")
sample*efficiency*results = sample*efficiency*experiment(
    env*factory=create*cartpole_env,
    agent*factories=agent*factories,
    threshold=195.0,
    max_episodes=1500, # Increased episodes for better chance of solving
    max*steps*per_episode=200,
    runs=3 # Increased runs for more reliable average
)

print("\nSample efficiency experiment completed.")
print("Results:")
for name, data in sample*efficiency*results.items():
    print(f"  {name}:")
    print(f"    Steps to threshold (per run): {data['per_run']}")
    print(f"    Mean steps to threshold: {data['mean*steps'] if data['mean*steps'] is not None else 'Not solved'}")
    print(f"    Std dev steps: {data['std*steps'] if data['std*steps'] is not None else 'N/A'}")
```

    Running sample efficiency experiment...


    /var/folders/cg/l2rdx46d6lv3b5xc17b420yc0000gn/T/ipykernel_38153/1754553734.py:67: UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/ReduceOps.cpp:1839.)
      returns*t = (returns*t - returns*t.mean()) / (returns*t.std() + 1e-8)


    
    Sample efficiency experiment completed.
    Results:
      DQN:
        Steps to threshold (per run): [50514, None, None]
        Mean steps to threshold: 50514.0
        Std dev steps: 0.0
      REINFORCE:
        Steps to threshold (per run): [None, None, None]
        Mean steps to threshold: Not solved
        Std dev steps: N/A
      Actor-Critic:
        Steps to threshold (per run): [None, None, None]
        Mean steps to threshold: Not solved
        Std dev steps: N/A


## Assignment Submission Requirements

### What to Submit:

1. **This completed notebook** with:
- All code cells executed
- All theoretical questions answered
- Experimental results and analysis

2. **Written Report** (2-3 pages) including:
- Comparison of the three algorithms
- Analysis of experimental results
- Discussion of hyperparameter sensitivity
- Recommendations for different scenarios

3. **Code Implementation** of at least one advanced feature:
- Prioritized Experience Replay
- Dueling DQN improvements
- Custom environment implementation
- Hyperparameter optimization

### Evaluation Criteria:

- **Theoretical Understanding (30%)**: Correct answers to theoretical questions
- **Implementation Quality (40%)**: Working code, proper documentation, clean structure
- **Experimental Analysis (20%)**: Thorough analysis of results, meaningful comparisons
- **Innovation/Extensions (10%)**: Creative improvements or additional implementations

### Submission Deadline: [insert Date]

### Additional Notes:

- Ensure all code runs without errors
- Include clear comments and documentation
- Use proper citation for any external sources
- Submit both .ipynb and .pdf versions of the notebook


