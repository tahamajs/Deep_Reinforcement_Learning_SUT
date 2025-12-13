# Deep Reinforcement Learning Fundamentals (CA1)

## Abstract

This project provides a comprehensive introduction to the foundational concepts and practical implementations of Deep Reinforcement Learning (DRL). We delve into the theoretical underpinnings of Markov Decision Processes (MDPs), various value functions, and the fundamental Bellman equations. Building upon this, we explore dynamic programming, Monte Carlo methods, and temporal difference learning. The core of this assignment focuses on the implementation of Deep Q-Networks (DQN), REINFORCE, and Actor-Critic algorithms using PyTorch and Gymnasium environments. The code is structured modularly for clarity and reusability, demonstrating best practices in DRL development. Detailed mathematical derivations and a thorough explanation of implementation choices are provided, along with experimental analysis on the CartPole-v1 environment to compare algorithm performance, stability, and sample efficiency.

**Index Terms** — Deep Reinforcement Learning, Markov Decision Process, Value Functions, Bellman Equations, Deep Q-Networks, REINFORCE, Actor-Critic, Experience Replay, Target Networks, Dueling DQN, Double DQN, Prioritized Experience Replay, Policy Gradient, Temporal Difference Learning.

## 1. Introduction

Deep Reinforcement Learning (DRL) is a powerful paradigm that integrates deep neural networks with reinforcement learning, enabling agents to learn optimal behaviors directly from interactions with complex environments. This field has achieved remarkable success in diverse domains, from mastering intricate games like Go and Atari to controlling robotic systems. This assignment aims to establish a solid theoretical and practical foundation in DRL, serving as a stepping stone for more advanced topics.

### 1.1 Learning Objectives

Upon completion of this assignment, you will be able to:

1.  **Understand Markov Decision Processes (MDPs)**: Grasp the mathematical framework for modeling sequential decision-making in stochastic environments.
2.  **Analyze Value Functions and Bellman Equations**: Compute and interpret state-value and action-value functions, and understand their recursive relationships.
3.  **Implement Dynamic Programming Methods**: Apply policy evaluation and value iteration for solving small, known MDPs.
4.  **Explore Monte Carlo Methods**: Learn from complete episodes without explicit knowledge of environment dynamics.
5.  **Master Temporal Difference (TD) Learning**: Implement bootstrapping methods like SARSA and Q-Learning.
6.  **Develop Deep Reinforcement Learning Algorithms**: Implement and understand the mechanics of DQN, REINFORCE, and Actor-Critic.
7.  **Apply Advanced DQN Techniques**: Integrate features like Experience Replay, Target Networks, Double DQN, Dueling DQN, and Prioritized Experience Replay.
8.  **Conduct Experimental Analysis**: Design and execute experiments to compare DRL algorithms based on performance, stability, and sample efficiency.
9.  **Adhere to Production-Grade Code Standards**: Write modular, type-hinted, and well-documented Python code.

### 1.2 Prerequisites

To effectively engage with this assignment, a foundational understanding in the following areas is beneficial:

-   **Python Programming**: Proficiency in Python, including object-oriented programming concepts.
-   **Linear Algebra and Calculus**: Basic knowledge of vector operations, matrices, and differentiation.
-   **Probability and Statistics**: Understanding of random variables, expectation, variance, and basic distributions.
-   **Machine Learning Fundamentals**: Familiarity with supervised learning, neural networks, and optimization algorithms.
-   **PyTorch**: Basic experience with PyTorch tensors, `nn.Module`, and `optim`.

### 1.3 Assignment Structure

This assignment is structured to guide you through the fundamentals of DRL, from theory to advanced implementation:

-   **Section 2**: Theoretical Foundations (MDPs, Value Functions, Bellman Equations)
-   **Section 3**: Dynamic Programming Methods (Policy Evaluation, Value Iteration)
-   **Section 4**: Monte Carlo Methods (First-Visit, Every-Visit, Control)
-   **Section 5**: Temporal Difference Learning (SARSA, Q-Learning, TD(λ))
-   **Section 6**: Deep Reinforcement Learning (DQN, REINFORCE, Actor-Critic)
-   **Section 7**: Modular Codebase (`src/`) Explained
-   **Section 8**: Experimental Design and Results
-   **Section 9**: Comparative Analysis and Advanced Topics
-   **Section 10**: Conclusion and Future Work

## 2. Theoretical Foundations

### 2.1 Markov Decision Processes (MDPs)

A Markov Decision Process provides a mathematical framework for modeling sequential decision-making in environments where outcomes are partly random and partly under the control of a decision maker. An MDP is formally defined by a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$, where:

-   $\mathcal{S}$: A finite set of states.
-   $\mathcal{A}$: A finite set of actions.
-   $P(s'|s,a)$: The state transition probability, representing the probability of transitioning to state $s'$ from state $s$ after taking action $a$.
-   $R(s,a,s')$: The reward function, specifying the immediate reward received after transitioning from state $s$ to $s'$ via action $a$.
-   $\gamma \in [0, 1]$: The discount factor, which determines the present value of future rewards. A value of $\gamma$ close to 0 makes the agent short-sighted, while a value close to 1 makes it far-sighted.

The **Markov Property** is central to MDPs: "The future is independent of the past given the present." Mathematically, this means:

$$P(S_{t+1} = s' | S_t = s, A_t = a, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = P(S_{t+1} = s' | S_t = s, A_t = a)$$

### 2.2 Policies

A **policy** $\pi$ is a mapping from states to probabilities of selecting each action. It defines the agent's behavior.

-   **Deterministic Policy**: $a = \pi(s)$, where a specific action is chosen for each state.
-   **Stochastic Policy**: $\pi(a|s) = P(A_t = a | S_t = s)$, where a probability distribution over actions is given for each state.

### 2.3 Return and Value Functions

The goal of an RL agent is to maximize the **expected return**, which is the total discounted reward accumulated over time.

**Return ($G_t$)**: The total discounted reward from time step $t$ onwards:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**State-Value Function ($V^\pi(s)$)**: The expected return when starting in state $s$ and following policy $\pi$.

$$V^\pi(s) = \mathbb{E}_\pi [G_t | S_t = s]$$

**Action-Value Function ($Q^\pi(s,a)$)**: The expected return when starting in state $s$, taking action $a$, and thereafter following policy $\pi$.

$$Q^\pi(s,a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a]$$

### 2.4 Bellman Equations

The Bellman equations are fundamental to RL, providing recursive relationships for value functions.

**Bellman Expectation Equation for $V^\pi(s)$**:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

**Bellman Expectation Equation for $Q^\pi(s,a)$**:

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

**Bellman Optimality Equation for $V^*(s)$** (Optimal State-Value Function):

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$$

**Bellman Optimality Equation for $Q^*(s,a)$** (Optimal Action-Value Function):

$$Q^*(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

## 3. Dynamic Programming Methods

Dynamic Programming (DP) methods are a collection of algorithms that can be used to compute optimal policies and value functions, assuming a perfect model of the MDP (i.e., known $P$ and $R$).

### 3.1 Policy Evaluation

Policy evaluation aims to compute the state-value function $V^\pi$ for a given policy $\pi$. It is an iterative process:

$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$$

This iterative update is guaranteed to converge to $V^\pi(s)$ as $k \to \infty$.

### 3.2 Policy Improvement

Given the value function for a policy $\pi$, we can improve the policy by making it greedy with respect to $Q^\pi$.

$$\pi'(s) = \arg\max_a Q^\pi(s,a)$$

This new policy $\pi'$ is guaranteed to be better than or equal to $\pi$.

### 3.3 Value Iteration

Value iteration directly computes the optimal value function $V^*$ by repeatedly applying the Bellman Optimality Equation as an update rule:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$$

Once $V^*(s)$ is found, the optimal policy can be derived by choosing actions greedily with respect to $Q^*(s,a)$.

## 4. Monte Carlo Methods

Monte Carlo (MC) methods learn directly from episodes of experience. They do not require a model of the environment's dynamics. Instead, they average returns observed from many episodes.

### 4.1 First-Visit Monte Carlo Prediction

To estimate $V^\pi(s)$, First-Visit MC averages the returns received after the *first* time state $s$ is visited in an episode.

### 4.2 Every-Visit Monte Carlo Prediction

Every-Visit MC averages the returns received after *each* time state $s$ is visited in an episode.

### 4.3 Monte Carlo Control (GLIE)

Monte Carlo control aims to find an optimal policy without a model. It typically involves an iterative process of policy evaluation and policy improvement, often employing a "greedy in the limit of infinite exploration" (GLIE) condition to ensure convergence. This involves using $\epsilon$-greedy policies and decaying $\epsilon$ over time.

## 5. Temporal Difference (TD) Learning

Temporal Difference (TD) learning combines ideas from Monte Carlo and Dynamic Programming. Like MC, TD methods learn directly from experience without a model. Like DP, TD methods update estimates based in part on other learned estimates (bootstrapping), without waiting for a final outcome.

### 5.1 TD(0) / SARSA

SARSA (State-Action-Reward-State-Action) is an **on-policy** TD control algorithm. The update rule for the action-value function $Q(s,a)$ is:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

Here, the next action $A_{t+1}$ is chosen using the *current* policy, which is why it's on-policy.

### 5.2 Q-Learning

Q-Learning is an **off-policy** TD control algorithm. The update rule is:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$$

In Q-learning, the agent learns the value of the optimal action $a'$ in the next state $S_{t+1}$ (using $\max_a$), regardless of the action actually taken by the current policy. This makes it off-policy.

### 5.3 TD(λ) Methods

TD(λ) methods extend TD(0) by using eligibility traces, which provide a way to combine single-step TD updates with multi-step returns. It bridges the gap between TD(0) and Monte Carlo methods, allowing for more efficient credit assignment over longer sequences of actions.

## 6. Deep Reinforcement Learning Algorithms

This section details the core DRL algorithms implemented in this assignment.

### 6.1 Deep Q-Networks (DQN)

DQN extends Q-Learning by using a deep neural network (the Q-network) to approximate the optimal action-value function $Q^*(s,a)$. Key innovations include:

1.  **Experience Replay**: Stores transitions $(s_t, a_t, r_{t+1}, s_{t+1}, done)$ in a replay buffer, allowing the agent to sample mini-batches of past experiences for training. This breaks correlations between successive samples and smooths the data distribution, improving stability.
2.  **Target Network**: Uses a separate, periodically updated network (the target Q-network $Q(\cdot; \theta^-)$) to compute the target values for the Bellman update. This stabilizes training by providing a fixed target for a number of training steps, preventing oscillations.

The **DQN Loss Function** is typically a Mean Squared Error (MSE) between the current Q-value and the target Q-value:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta) \right)^2 \right]$$

Where $\theta$ are the parameters of the main Q-network and $\theta^-$ are the parameters of the target network.

#### Advanced DQN Variants

-   **Double DQN**: Addresses the overestimation bias of Q-values inherent in standard DQN. It decouples the action selection from action evaluation. The next action $A_{t+1}^*$ is chosen using the *online* Q-network, but its value is estimated using the *target* Q-network:
    $$Q_{target}(s,a) = r + \gamma Q(s', \arg\max_{a'} Q(s',a'; \theta); \theta^-)$$
-   **Dueling DQN**: Modifies the network architecture to estimate the state-value function $V(s)$ and the advantage function $A(s,a)$ separately, then combines them to produce the Q-values. This allows the network to learn which states are valuable independently of the actions taken. The Q-value is computed as:
    $$Q(s,a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s,a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a'; \theta, \alpha) \right)$$
    The subtraction of the mean advantage addresses the identifiability problem between $V(s)$ and $A(s,a)$.
-   **Prioritized Experience Replay (PER)**: Instead of uniform sampling from the replay buffer, PER samples transitions with higher TD-error more frequently. This prioritizes learning from surprising or important experiences, leading to faster convergence.
    -   **TD-error**: $\delta = |Q_{target} - Q_{expected}|$
    -   **Sampling Probability**: $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$, where $p_i$ is the priority of transition $i$ (e.g., $|\delt-i| + \epsilon$).
    -   **Importance Sampling (IS) Weights**: To correct for the bias introduced by prioritized sampling, IS weights are used in the loss function: $w_i = (N \cdot P(i))^{-\beta} / \max_k w_k$. The loss becomes $L(\theta) = w_i \cdot (TD\_error)^2$.

### 6.2 Policy Gradient Methods (REINFORCE)

REINFORCE (Monte Carlo Policy Gradient) is an **on-policy** algorithm that directly optimizes the policy $\pi_\theta(a|s)$ with parameters $\theta$. It uses Monte Carlo estimates of the return to update the policy.

The **Policy Gradient Theorem** states that the gradient of the performance measure $J(\theta)$ (e.g., expected return) with respect to the policy parameters is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(A_t|S_t) G_t \right]$$

In REINFORCE, we approximate this expectation using a single episode:

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \log \pi_\theta(A_t|S_t) G_t$$

Where $G_t$ is the return from time step $t$.

### 6.3 Actor-Critic Methods

Actor-Critic methods combine policy-based (actor) and value-based (critic) approaches to leverage the strengths of both. The **actor** is a policy network that decides which action to take, while the **critic** is a value network that evaluates the action taken by the actor.

-   **Actor Update**: Uses the policy gradient, but instead of the full return $G_t$, it uses a TD-error or advantage estimate from the critic.
    $$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \log \pi_\theta(A_t|S_t) \delta_t$$
    Where $\delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$ is the TD-error (or advantage $A(s,a) = Q(s,a) - V(s)$).
-   **Critic Update**: Uses TD learning (e.g., MSE loss) to update its value function parameters $w$.
    $$w_{t+1} = w_t + \alpha_w \delta_t \nabla_w V_w(S_t)$$

Actor-Critic methods typically have lower variance than REINFORCE (due to bootstrapping from the critic) and can be more sample efficient than pure policy gradient methods.

## 7. Modular Codebase (`src/`) Explained

The project's Python codebase is structured modularly to enhance readability, maintainability, and reusability, following best practices for DRL implementations.

```
CAs/Solutions/CA01_Deep_RL_Fundamentals/
├── src/
│   ├── __init__.py          # Package initializer
│   ├── agents/              # RL agent implementations
│   │   ├── __init__.py
│   │   └── ca1_agents.py
│   ├── data/                # Data structures like replay buffers
│   │   ├── __init__.py
│   │   └── buffers.py
│   ├── models/              # Neural network architectures
│   │   ├── __init__.py
│   │   └── ca1_models.py
│   ├── utils/               # General utility functions
│   │   ├── __init__.py
│   │   └── ca1_utils.py
│   └── config.py            # Hyperparameter configurations
├── CA1.ipynb               # Main notebook for theory, implementation demos, and experiments
├── README.md               # Project overview and detailed lecture notes (this file)
├── requirements.txt        # Python dependencies
├── run.sh                  # Example script to run experiments
├── test_setup.py           # Basic setup tests
├── environments/           # Custom Gymnasium environments or wrappers
├── evaluation/             # Scripts for evaluating agent performance
├── experiments/            # Experiment runners and logging
├── visualization/          # Saved plots and figures
└── results.json            # JSON file for storing experiment results
```

### 7.1 `src/config.py`

This file centralizes all hyperparameters for the various DRL algorithms and experiments. Using dataclasses ensures type safety and immutability for configurations.

-   **`DQNConfig`**: Configuration for DQN and its variants (Double, Dueling, Noisy).
-   **`REINFORCEConfig`**: Configuration for the REINFORCE agent.
-   **`ActorCriticConfig`**: Configuration for the Actor-Critic agent.
-   **`ExperimentConfig`**: Overall configuration for running experiments, including environment details and references to agent-specific configs.

### 7.2 `src/models/ca1_models.py`

Contains the neural network architectures used by the agents.

-   **`DQN(nn.Module)`**: Standard feedforward Q-network.
-   **`DuelingDQN(nn.Module)`**: Implements the Dueling architecture with separate value and advantage streams.
-   **`PolicyNetwork(nn.Module)`**: Network for approximating policies in REINFORCE and Actor-Critic, outputting action probabilities.
-   **`ValueNetwork(nn.Module)`**: Network for approximating the state-value function in Actor-Critic.
-   **`NoisyLinear(nn.Module)`**: A linear layer with added parameter noise for exploration, used in `NoisyDQN`.
-   **`NoisyDQN(nn.Module)`**: DQN agent using `NoisyLinear` layers for intrinsic exploration.

### 7.3 `src/data/buffers.py`

Manages experience replay mechanisms critical for off-policy algorithms.

-   **`ReplayBuffer`**: A standard experience replay buffer that stores and uniformly samples transitions (`state`, `action`, `reward`, `next_state`, `done`). Implemented using `collections.deque`.
-   **`PrioritizedReplayBuffer`**: An advanced replay buffer that samples transitions based on their temporal difference (TD) error, giving higher priority to more surprising experiences. Includes support for Importance Sampling (IS) weights to correct for sampling bias.

### 7.4 `src/agents/ca1_agents.py`

Implements the core logic for each DRL agent, inheriting from a `BaseAgent` abstract class for a consistent interface.

-   **`BaseAgent(ABC)`**: Abstract base class defining common methods for RL agents (`act`, `learn`, `save`, `load`).
-   **`DQNAgent(BaseAgent)`**: Implements the standard DQN algorithm with support for Double DQN and Dueling DQN, using `ReplayBuffer`.
-   **`ImprovedDQNAgent(DQNAgent)`**: Extends `DQNAgent` to incorporate `PrioritizedReplayBuffer`, demonstrating an advanced DQN variant.
-   **`REINFORCEAgent(BaseAgent)`**: Implements the REINFORCE policy gradient algorithm. It collects full episode trajectories before performing a single update.
-   **`ActorCriticAgent(BaseAgent)`**: Implements the Actor-Critic algorithm, which combines a policy network (actor) and a value network (critic) for online learning and reduced variance.

### 7.5 `src/utils/ca1_utils.py`

Provides a collection of general utility functions to support the DRL agents and experiments.

-   **`set_seed(seed: int)`**: Ensures reproducibility by setting random seeds for NumPy, PyTorch, and Python's `random` module.
-   **`moving_average(x: List[float], window: int)`**: Computes the moving average of a list of values, useful for smoothing learning curves.
-   **`gym_reset(env: gym.Env)`**: Handles the `reset()` method for different versions of Gymnasium, returning the initial state.
-   **`gym_step(env: gym.Env, action: Any)`**: Handles the `step()` method for different versions of Gymnasium, returning the next state, reward, done flag, and info.

## 8. Experimental Design and Results

To effectively evaluate and compare the DRL algorithms, a systematic experimental approach is crucial. This section outlines the design and expected results of experiments on the CartPole-v1 environment.

### 8.1 Environment Setup: CartPole-v1

-   **Observation Space**: 4 continuous values representing cart position, cart velocity, pole angle, and pole angular velocity.
-   **Action Space**: 2 discrete actions: 0 (push cart left) and 1 (push cart right).
-   **Reward**: +1 for every timestep the pole remains upright.
-   **Episode Termination**: Occurs when the pole angle exceeds $\pm 12$ degrees, the cart position exceeds $\pm 2.4$ units, or the episode length reaches 500 (or `max_t` if set).
-   **Goal**: Achieve an average reward of 195.0 over 100 consecutive episodes.

### 8.2 Training Functions

The `CA1.ipynb` notebook (and implicitly `src/agents/ca1_agents.py` through training functions) includes dedicated training loops for each agent type:

-   **`train_dqn_agent(agent: DQNAgent, env: gym.Env, ...)`**: Orchestrates the training of DQN agents, including experience collection, learning updates, and epsilon decay.
-   **`train_reinforce_agent(agent: REINFORCEAgent, env: gym.Env, ...)`**: Manages episode generation and policy updates for REINFORCE.
-   **`train_actor_critic_agent(agent: ActorCriticAgent, env: gym.Env, ...)`**: Coordinates actor and critic updates within an episode for Actor-Critic agents.

### 8.3 Sample Efficiency Experiment

**Objective**: Compare how efficiently different algorithms learn from environmental interactions.

-   **Measurement**: Number of total environment `steps` (timesteps) required to reach a predefined performance threshold (e.g., average score of 195 over 100 episodes).
-   **Metrics**: `Timesteps to Threshold` (primary), `Area Under the Learning Curve (AUC)` (secondary), `Performance after a Fixed Number of Steps`.
-   **Fair Comparison**: All agents are trained on the same environment, with consistent network architectures, and results are averaged over multiple runs with different random seeds for robustness.

**Expected Results**:

-   **DQN (and its variants)** are generally expected to be more sample-efficient than REINFORCE and Actor-Critic due to the use of experience replay and stable target networks.
-   **Prioritized Experience Replay** should further boost DQN's sample efficiency.
-   **REINFORCE** typically requires more samples due to its Monte Carlo nature and high variance.
-   **Actor-Critic** should offer a balance, being more sample-efficient than REINFORCE but potentially less so than advanced DQN.

### 8.4 Hyperparameter Sensitivity Analysis

**Objective**: Understand the impact of key hyperparameters (e.g., learning rate, discount factor) on algorithm performance and stability.

-   **Methodology**: Run agents with various combinations of hyperparameters and observe changes in average score, convergence speed, and stability.
-   **Visualization**: Plot performance curves for different hyperparameter settings.

**Expected Findings**:

-   Each algorithm will exhibit sensitivity to its specific hyperparameters. Optimal learning rates are crucial for all methods. Higher discount factors (close to 1) often lead to more long-sighted behavior but can also increase instability.
-   DQN's performance is sensitive to `epsilon_decay` and `buffer_size`.
-   REINFORCE can be highly sensitive to the learning rate due to its high variance.
-   Actor-Critic requires careful tuning of both actor and critic learning rates.

## 9. Comparative Analysis and Advanced Topics

### 9.1 Algorithm Comparison Summary

| Feature           | DQN                          | REINFORCE                    | Actor-Critic                |
| :---------------- | :--------------------------- | :--------------------------- | :-------------------------- |
| **Type**          | Value-based (Off-policy)     | Policy-based (On-policy)     | Hybrid (On-policy)          |
| **Memory Req.**   | High (Replay Buffer)         | Low (per episode)            | Medium                      |
| **Stability**     | High (Target Networks)       | Low (High Variance)          | Medium (Bootstrapping)      |
| **Sample Eff.**   | High                         | Low                          | Medium                      |
| **Exploration**   | $\epsilon$-greedy, Noisy Nets | Stochastic Policy            | Stochastic Policy           |
| **Action Space**  | Discrete                     | Discrete/Continuous (with GMM) | Discrete/Continuous         |
| **Key Mechanisms**| Q-network, Target Net, Replay | Policy Network, Monte Carlo  | Actor Net, Critic Net, TD-Error |

### 9.2 Key Insights

-   **DQN and its variants** are powerful for environments with discrete action spaces, offering high sample efficiency and stable learning through techniques like experience replay and target networks. They excel in scenarios where data can be reused effectively.
-   **REINFORCE** is a fundamental policy gradient method, simple to implement, and can handle continuous action spaces. However, its high variance often makes it less sample-efficient and stable, requiring many episodes for convergence.
-   **Actor-Critic methods** strike a balance between policy-based and value-based approaches. By using a critic to estimate the value function, they reduce the variance of policy gradient estimates compared to REINFORCE, leading to more stable and faster learning. They are versatile for both discrete and continuous action spaces.

### 9.3 When to Use Each Algorithm

-   **Use DQN when**: The environment has a discrete action space, sample efficiency is critical, and you can manage a large replay buffer. Advanced DQN variants (Dueling, Double, PER) offer significant improvements.
-   **Use REINFORCE when**: You are starting with policy gradients, the problem is relatively simple, or you need direct policy optimization for interpretability. It can be a good baseline for more complex policy-based methods.
-   **Use Actor-Critic when**: You need a balance of sample efficiency and stability for continuous control problems or complex discrete action spaces. It's often a stepping stone to more advanced actor-critic algorithms like A2C, A3C, PPO, or SAC.

### 9.4 Advanced Topics for Further Study

This assignment provides a solid foundation, but the field of DRL is vast. Consider exploring:

1.  **More Advanced DQN Variants**: Rainbow DQN (combining multiple DQN improvements), Distributional DQN, Quantile Regression DQN.
2.  **Advanced Policy Gradient Methods**: Proximal Policy Optimization (PPO), Trust Region Policy Optimization (TRPO), Soft Actor-Critic (SAC).
3.  **Model-Based Reinforcement Learning**: Algorithms that learn a model of the environment dynamics to plan or improve policy learning (e.g., Model-Predictive Control, Dyna-Q).
4.  **Multi-Agent Reinforcement Learning**: Extending DRL to scenarios with multiple interacting agents.
5.  **Offline Reinforcement Learning**: Learning policies from fixed datasets of experience without further environmental interaction.

## 10. Conclusion and Future Work

This assignment successfully introduces the core concepts and fundamental algorithms of Deep Reinforcement Learning. We've established a modular codebase for DQN, REINFORCE, and Actor-Critic, supported by essential utilities and configuration management. The theoretical derivations and practical implementations provide a clear understanding of how these algorithms function and interact with environments. While the CartPole-v1 environment serves as an excellent starting point, future work could involve applying these refined algorithms to more complex environments (e.g., Atari games, robotic control tasks), conducting extensive hyperparameter tuning using automated tools, and exploring cutting-edge DRL research to synthesize novel methods. The modular structure of this project facilitates such extensions, allowing for continuous growth and exploration within the DRL landscape.

## References

1.  Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2.  Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
3.  Hasselt, H. V., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. *Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence*.
4.  Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Silver, D., & de Freitas, N. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *Proceedings of the 33rd International Conference on Machine Learning*.
5.  Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. *International Conference on Learning Representations (ICLR)*.
6.  Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
7.  Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-Critic Algorithms. *Advances in Neural Information Processing Systems (NIPS)*.
