# Computer Assignment 12: Meta-Communicative Actor-Critic (MCAC) for Dynamic Multi-Agent Coordination

## Abstract

This assignment introduces the **Meta-Communicative Actor-Critic (MCAC)** framework, a novel approach designed to address dynamic coordination challenges in multi-agent reinforcement learning (MARL). MCAC synthesizes Multi-Agent Actor-Critic (MAAC) with emergent communication and Model-Agnostic Meta-Learning (MAML) to enable agents to rapidly adapt their communication protocols and policies to new tasks or changing team compositions with few-shot experience. We explicitly identify the research gap in enabling on-the-fly adaptation of emergent communication in complex MARL settings. Through rigorous mathematical derivations and a modular architectural design, MCAC provides a robust solution for enhanced sample efficiency and robustness in highly dynamic multi-agent environments.

**Keywords:** Multi-agent reinforcement learning, Meta-Learning, Emergent Communication, Actor-Critic, Dynamic Coordination, Few-Shot Adaptation.

## 1. Introduction

Multi-agent reinforcement learning (MARL) presents significant challenges due to non-stationarity, partial observability, and complex credit assignment problems. While considerable progress has been made in developing algorithms for cooperative and competitive multi-agent settings, a critical gap remains in enabling agents to dynamically adapt their communication strategies and policies in rapidly changing environments. Existing emergent communication methods often rely on extensive training within fixed scenarios, limiting their generalization capabilities to novel tasks or unforeseen communicative partners.

This project proposes the **Meta-Communicative Actor-Critic (MCAC)** framework, a novel synthesis designed to bridge this gap. MCAC integrates the strengths of:

1.  **Multi-Agent Actor-Critic (MAAC)**: Providing a robust foundation for centralized training and decentralized execution, crucial for stabilizing learning in non-stationary multi-agent environments.
2.  **Emergent Communication**: Allowing agents to learn a communication protocol from scratch, which is essential for effective coordination in complex tasks where explicit communication rules are unknown or difficult to design.
3.  **Model-Agnostic Meta-Learning (MAML)**: Empowering agents with the ability to "learn to adapt" their communication and action policies rapidly to new tasks or altered team dynamics with minimal experience.

The core research gap addressed by MCAC is the lack of adaptive emergent communication in dynamic MARL settings. By meta-learning how to generate and interpret messages, agents can overcome the limitations of fixed communication protocols, leading to superior coordination, improved sample efficiency, and enhanced robustness when faced with unpredictable changes in the environment or agent composition.

### 1.1 Research Motivation and Gap

Multi-agent systems in real-world applications often operate in dynamic and unpredictable environments. For example, autonomous vehicle fleets need to adapt to sudden traffic changes, robotic swarms must respond to evolving task requirements, and smart grids need to adjust to fluctuating energy demands. In such scenarios, effective coordination is paramount, and communication among agents plays a vital role.

However, traditional MARL approaches, including those with emergent communication, often suffer from several limitations in dynamic settings:

*   **Fixed Communication Protocols**: Most emergent communication methods learn a static communication protocol that is optimal for the training environment but may perform poorly when environmental dynamics or team compositions change.
*   **Slow Adaptation**: When faced with novel tasks or new team members, agents may require extensive retraining to adapt their communication and policies, which is impractical in real-time applications.
*   **Generalization Issues**: Learned communication strategies may not generalize well to unseen communication partners or tasks that deviate significantly from the training distribution.

The **Meta-Communicative Actor-Critic (MCAC)** framework aims to address these limitations by enabling agents to meta-learn an adaptive communication strategy. This allows agents to quickly infer and adjust to the optimal communication protocols and action policies with minimal experience when encountering new dynamic tasks. Our research aims to demonstrate how meta-learning can facilitate the on-the-fly adaptation of emergent communication, thereby enhancing robustness, accelerating learning, and improving coordination in highly dynamic multi-agent environments.

### 1.2 Learning Objectives

By engaging with this project, you will achieve a deep understanding of:

1.  **Meta-Learning for MARL**: How to integrate meta-learning principles (specifically MAML) into multi-agent reinforcement learning to enable rapid adaptation.
2.  **Adaptive Emergent Communication**: Designing and implementing communication modules that can adapt their encoding and decoding strategies to dynamic environmental and team conditions.
3.  **Centralized Training, Decentralized Execution with Meta-Adaptation**: Extending the CTDE paradigm to support meta-learning for both policies and communication.
4.  **Rigorous Mathematical Formulation**: Deriving meta-gradients for complex, differentiable communication and policy networks.
5.  **Modular Software Design**: Building a clean, type-hinted, and extensible codebase for advanced MARL research.
6.  **Experimental Design and Analysis**: Setting up experiments to evaluate adaptive communication and meta-learning in multi-agent systems, including ablation studies.

## 2. Theoretical Framework and Mathematical Foundations

### 2.1 Multi-Agent Reinforcement Learning (MARL) Fundamentals

MARL extends single-agent RL to environments with multiple interacting agents. Key concepts include:

*   **Multi-Agent Markov Decision Process (MMDP)**: A tuple $(\mathcal{S}, \mathcal{A}_1, \dots, \mathcal{A}_N, P, R_1, \dots, R_N, \gamma)$, where $\mathcal{S}$ is the state space, $\mathcal{A}_i$ is the action space for agent $i$, $P$ is the joint transition function, $R_i$ is the reward function for agent $i$, and $\gamma$ is the discount factor.
*   **Joint Action Space**: $\mathcal{A} = \mathcal{A}_1 \times \dots \times \mathcal{A}_N$.
*   **Observation Space**: Each agent $i$ receives a local observation $o_i \in \mathcal{O}_i$, which may be a partial view of the global state $s$.
*   **Non-Stationarity**: From an individual agent's perspective, the environment is non-stationary because other agents are also learning and changing their policies. This is a primary challenge in MARL.
*   **Credit Assignment**: Attributing global reward signals to individual agent actions, especially in cooperative settings.

### 2.2 Centralized Training, Decentralized Execution (CTDE)

The CTDE paradigm is a widely adopted approach to mitigate non-stationarity in MARL:

*   **Centralized Training**: During training, a centralized component (e.g., a critic) has access to global information (all observations and actions). This provides a stable learning signal for policy updates, addressing non-stationarity.
*   **Decentralized Execution**: During deployment, each agent makes decisions based solely on its local observation, promoting scalability and robustness.

#### Multi-Agent Actor-Critic (MAAC)

MAAC is a prominent CTDE algorithm. Each agent $i$ has an actor $\pi_i(-i | o_i)$ and a centralized critic $Q(s, a_1, \dots, a_N)$.

*   **Critic Update**: The centralized critic $Q(s, a_1, \dots, a_N)$ is updated using a temporal difference (TD) loss, typically minimizing the squared error between the predicted Q-value and a target Q-value:
    $$L(\phi) = \mathbb{E}[(Q_{\phi}(s, a_1, \dots, a_N) - y)^2]$$
    where $y = r + \gamma Q_{\phi'}(s', \pi_{\theta'_1}(o'_1), \dots, \pi_{\theta'_N}(o'_N))$ and $Q_{\phi'}$ and $\pi_{\theta'_i}$ are target networks.

*   **Actor Update**: Each agent $i$'s actor $\pi_i$ is updated using the policy gradient theorem, leveraging the centralized critic to provide an accurate advantage signal:
    $$ \nabla_{\thet-i} J_i = \mathbb{E}[\nabla_{\thet-i} \log \pi_{\thet-i}(-i | o_i) \cdot Q^{\pi}(s, a_1, \dots, a_N)] $$
    where $Q^{\pi}$ is the centralized Q-function evaluating the actions taken by the current policies.

### 2.3 Emergent Communication Theory

Emergent communication in MARL focuses on learning communication protocols implicitly through the reinforcement learning process. Instead of pre-defining message structures, agents learn to generate and interpret messages that are useful for coordination.

*   **Communication Channel**: Agents send messages $m_i$ to other agents $M_{\neg i}$ (messages from all agents except $i$).
*   **Message Encoding**: Each agent $i$ processes its local observation $o_i$ and potentially its internal state $h_i$ to generate a message:
    $$ m_i = f_{\text{enc},i}(o_i, h_i; \text{Comm}_i^{\text{enc}}) $$
    where $\text{Comm}_i^{\text{enc}}$ are the parameters of agent $i$'s message encoder.
*   **Message Decoding**: Each agent $i$ receives and decodes messages $M_{\neg i}$ from other agents to influence its policy:
    $$ \hat{m}_i = f_{\text{dec},i}(M_{\neg i}; \text{Comm}_i^{\text{dec}}) $$
    where $\text{Comm}_i^{\text{dec}}$ are the parameters of agent $i$'s message decoder.
*   **Policy with Communication**: The actor policy $\pi_i$ then takes into account both its local observation and the decoded messages:
    $$ \pi_i(-i | o_i, \hat{m}_i; \thet-i) $$

### 2.4 Model-Agnostic Meta-Learning (MAML)

MAML is a meta-learning algorithm that aims to find an initialization of model parameters that can quickly adapt to new, unseen tasks with only a few gradient steps.

*   **Task Distribution**: MAML operates over a distribution of tasks $\mathcal{T}$. Each task $\tau \sim \mathcal{T}$ involves a dataset $D_\tau = \{D_\tau^{\text{support}}, D_\tau^{\text{query}}\}$.
*   **Inner Loop (Adaptation)**: For a given task $\tau$ and initial parameters $\theta$, a few gradient steps are taken on the support set $D_\tau^{\text{support}}$ to adapt the parameters to $\phi_\tau$:
    $$ \phi_\tau = \theta - \alpha \nabla_{\theta} \mathcal{L}_\tau(D_\tau^{\text{support}}; \theta) $$
    where $\mathcal{L}_\tau$ is the loss function for task $\tau$ and $\alpha$ is the inner-loop learning rate.
*   **Outer Loop (Meta-Optimization)**: The initial parameters $\theta$ are updated to maximize performance on the query set $D_\tau^{\text{query}}$ after adaptation. This involves taking a gradient step with respect to $\theta$ through the inner-loop update:
    $$ \theta \leftarrow \theta - \gamma \nabla_{\theta} \mathcal{L}_\tau(D_\tau^{\text{query}}; \phi_\tau) $$
    where $\gamma$ is the outer-loop meta-learning rate.

## 3. The Meta-Communicative Actor-Critic (MCAC) Framework

The MCAC framework integrates MAAC, emergent communication, and MAML to enable rapid adaptation of both communication protocols and action policies in dynamic multi-agent environments.

### 3.1 System Architecture

The MCAC system consists of $N$ agents. Each agent $i$ maintains:

*   **Local Actor Network**: $\pi_i(-i | o_i, \hat{m}_i; \thet-i)$ that outputs a policy based on its local observation and decoded messages.
*   **Communication Encoder**: $f_{\text{enc},i}(o_i, h_i; \text{Comm}_i^{\text{enc}})$ that generates a message.
*   **Communication Decoder**: $f_{\text{dec},i}(M_{\neg i}; \text{Comm}_i^{\text{dec}})$ that processes messages from other agents.
*   **Centralized Critic Network**: $Q(s, a_1, \dots, a_N, m_1, \dots, m_N; \phi)$ that evaluates the joint actions and messages, providing a global value signal.

The parameters $\thet-i$, $\text{Comm}_i^{\text{enc}}$, and $\text{Comm}_i^{\text{dec}}$ are the ones that will be meta-learned.

### 3.2 Meta-Learning for Adaptive Communication and Policy

We extend the MAML framework to meta-learn the initial parameters for both the policy networks and the communication modules. Let $\Theta = \{\theta_1, \dots, \theta_N\}$ be the collection of all actor policy parameters and $\mathcal{C} = \{\text{Comm}_1^{\text{enc}}, \text{Comm}_1^{\text{dec}}, \dots, \text{Comm}_N^{\text{enc}}, \text{Comm}_N^{\text{dec}}\}$ be the collection of all communication module parameters.

For each task $\tau \sim \mathcal{T}$, we perform inner-loop updates:

*   **Policy Adaptation**: For each agent $i$, the policy parameters $\thet-i$ are adapted using its policy loss on the support set:
    $$ \phi_{i,\tau} = \thet-i - \alpha_{\theta} \nabla_{\thet-i} \mathcal{L}^{\text{policy}}_{\tau,i}(D_\tau^{\text{support}}; \thet-i, \text{Comm}_i^{\text{enc}}, \text{Comm}_i^{\text{dec}}) $$
*   **Communication Adaptation**: For each agent $i$, its communication encoder and decoder parameters are adapted using a communication-specific loss (e.g., maximizing mutual information with rewards, or minimizing coordination error) on the support set:
    $$ \text{Comm}_{i,\tau}^{\text{enc}} = \text{Comm}_i^{\text{enc}} - \alpha_{\text{comm}} \nabla_{\text{Comm}_i^{\text{enc}}} \mathcal{L}^{\text{comm}}_{\tau,i}(D_\tau^{\text{support}}; \thet-i, \text{Comm}_i^{\text{enc}}, \text{Comm}_i^{\text{dec}}) $$
    $$ \text{Comm}_{i,\tau}^{\text{dec}} = \text{Comm}_i^{\text{dec}} - \alpha_{\text{comm}} \nabla_{\text{Comm}_i^{\text{dec}}} \mathcal{L}^{\text{comm}}_{\tau,i}(D_\tau^{\text{support}}; \thet-i, \text{Comm}_i^{\text{enc}}, \text{Comm}_i^{\text{dec}}) $$
    where $\alpha_{\theta}$ and $\alpha_{\text{comm}}$ are inner-loop learning rates.

The outer-loop meta-optimization then updates the initial parameters $\Theta$ and $\mathcal{C}$ by evaluating the adapted parameters on the query set:

*   **Meta-Policy Update**:
    $$ \Theta \leftarrow \Theta - \gamma_{\theta} \nabla_{\Theta} \sum_{\tau \sim \mathcal{T}} \sum_{i=1}^N \mathcal{L}^{\text{meta-policy}}_{\tau,i}(D_\tau^{\text{query}}; \phi_{i,\tau}, \text{Comm}_{i,\tau}^{\text{enc}}, \text{Comm}_{i,\tau}^{\text{dec}}) $$
*   **Meta-Communication Update**:
    $$ \mathcal{C} \leftarrow \mathcal{C} - \gamma_{\text{comm}} \nabla_{\mathcal{C}} \sum_{\tau \sim \mathcal{T}} \sum_{i=1}^N \mathcal{L}^{\text{meta-comm}}_{\tau,i}(D_\tau^{\text{query}}; \phi_{i,\tau}, \text{Comm}_{i,\tau}^{\text{enc}}, \text{Comm}_{i,\tau}^{\text{dec}}) $$
    where $\gamma_{\theta}$ and $\gamma_{\text{comm}}$ are outer-loop meta-learning rates. This bi-level optimization enables the agents to learn an initial set of parameters that facilitates rapid adaptation of both their action-taking policies and their communication strategies to new tasks.

### 3.3 Loss Functions

#### Centralized Critic Loss
The centralized critic $Q(s, a_1, \dots, a_N, m_1, \dots, m_N)$ is trained to minimize the TD error:
$$ L_Q = \mathbb{E}[(Q_{\phi}(s, a_1, \dots, a_N, m_1, \dots, m_N) - y)^2] $$
where $y = r + \gamma Q_{\phi'}(s', a'_1, \dots, a'_N, m'_1, \dots, m'_N)$, and $Q_{\phi'}$ is the target critic network. The actions $a'_i$ and messages $m'_i$ in the target are generated by the adapted target policies and communication modules, respectively.

#### Actor Policy Loss with Communication
Each actor $\pi_i$ is updated using a policy gradient that incorporates the influence of messages:
$$ \nabla_{\thet-i} J_i = \mathbb{E}[\nabla_{\thet-i} \log \pi_i(-i | o_i, f_{\text{dec},i}(M_{\neg i})) \cdot A_i] $$
where $A_i$ is the advantage for agent $i$, typically calculated as $Q(s, a_1, \dots, a_N, m_1, \dots, m_N) - V(s, m_1, \dots, m_N)$, where $V$ is a centralized value function or a learned baseline. The communication module is trained implicitly through this policy gradient, as effective communication leads to higher rewards and thus larger advantages.

#### Communication-Specific Loss (Optional but Recommended)
To explicitly guide the learning of communication, an additional loss term can be introduced. For example, a mutual information maximization objective between messages and critical state information or rewards, or a loss that penalizes messages that do not contribute to coordinated behavior. For simplicity in the initial implementation, we primarily rely on the policy gradient to drive communication learning, as effective communication will lead to better team rewards and thus higher policy gradients.

## 4. Experimental Design

To validate the MCAC framework, we propose a series of experiments across various multi-agent environments with dynamic elements.

### 4.1 Environments

We will utilize modified versions of existing multi-agent environments to introduce dynamic elements that necessitate adaptive communication:

1.  **Dynamic Cooperative Navigation**: Agents must navigate to target locations, but the optimal path or target locations can change mid-episode, requiring flexible communication to adapt to new goals or avoid dynamically appearing obstacles.
2.  **Adaptive Resource Allocation**: Agents must collectively allocate resources to dynamic demands. Resource availability or agent utility functions can change, requiring communication to re-negotiate allocations.
3.  **Evolving Predator-Prey**: Competitive environments where predator or prey behaviors can subtly shift (e.g., changes in speed, stealth, or target preference), requiring adaptive communication for coordination or counter-strategies.
4.  **Multi-Agent Particle Environment (MPE) with Dynamic Communication Constraints**: MPE environments modified such that communication bandwidth, range, or reliability dynamically changes, forcing agents to adapt their communication strategies.

### 4.2 Baselines

We will compare MCAC against the following strong baselines to highlight its benefits:

1.  **Independent Learning (IL)**: Each agent acts independently, treating other agents as part of the environment, without explicit coordination or communication. This serves as a lower bound for performance in cooperative tasks.
2.  **Multi-Agent Actor-Critic (MAAC)**: A standard MAAC framework without meta-learning or explicit emergent communication. This will demonstrate the impact of communication and meta-learning.
3.  **MAAC with Fixed Communication (MAAC-FC)**: MAAC augmented with a pre-trained (but non-adaptive) emergent communication module. This baseline will assess the value of adaptive communication over static communication.
4.  **MAML-MAAC**: MAAC with MAML applied to policies, but without meta-learning the communication module. This will isolate the benefits of meta-learning the communication aspect.
5.  **VDN/COMA**: For specific cooperative tasks, Value Decomposition Networks (VDN) or Counterfactual Multi-Agent Policy Gradients (COMA) could serve as strong cooperative baselines to show how MCAC compares in credit assignment.

### 4.3 Metrics

To comprehensively evaluate MCAC, we will use the following metrics:

1.  **Episode Return**: The average cumulative reward per episode, reflecting overall task performance.
2.  **Adaptation Speed**: The number of gradient steps or episodes required to achieve a certain performance threshold on new, unseen tasks, quantifying the meta-learning capability.
3.  **Communication Efficiency**: Analysis of message content (e.g., message entropy, correlation with task-relevant information) and its correlation with improved coordination outcomes. This can be qualitative and quantitative.
4.  **Robustness**: Performance degradation (or lack thereof) when faced with significantly different or adversarial task variations, or changes in the number/type of communicating agents.
5.  **Coordination Score**: A task-specific metric reflecting how well agents coordinate (e.g., minimum distance to target in navigation, fairness in resource allocation).

### 4.4 Ablation Studies

To understand the individual contributions of MCAC's components, we will conduct ablation studies:

1.  **No Meta-Learning on Communication**: Only policies are meta-learned, while communication parameters are fixed after initial training. This isolates the benefit of adaptive communication.
2.  **Fixed Policy, Meta-Learned Communication**: Only communication parameters are meta-learned, while policy parameters are fixed or non-meta-learned. This isolates the benefit of adaptive policies.
3.  **Varied Communication Bandwidth**: Test the impact of different message dimensions (e.g., low-dimensional, high-dimensional) on adaptation speed and performance.
4.  **Impact of Centralized Critic**: Evaluate performance if the critic is also meta-learned (more complex) vs. kept centralized and non-meta-learned.

## 5. Code Structure and Implementation Details

### 5.1 Project Organization

The project will adhere to a modular structure to ensure clarity, maintainability, and extensibility, aligning with the "Production Codebase" deliverable.

```
CA12/
├── src/
│   ├── config.py               # All hyperparameters and configuration settings
│   ├── model.py                # Neural network architectures (Actor, Critic, Communication modules)
│   ├── losses.py               # Custom loss functions for policy, value, and communication
│   ├── data.py                 # Replay buffers, dataset wrappers for environments
│   ├── utils.py                # General utilities (seeding, device management, logging)
│   └── agents/                 # (New directory for MCAC agent and possibly other refactored agents)
│       ├── mcac_agent.py       # Implementation of the Meta-Communicative Actor-Critic agent
│       └── ...                 # Other refactored agent implementations
├── environments/               # Gymnasium-compatible multi-agent environments
│   └── ...                     # Cooperative, competitive, dynamic environments
├── notebooks/
│   └── main.ipynb              # Main execution notebook for training, evaluation, and visualization
├── tests/                      # Unit and integration tests
├── report.tex                  # Formal IEEE research paper
├── README.md                   # Comprehensive lecture notes and project overview
└── requirements.txt            # Python dependencies
```

### 5.2 Key Code Features

*   **Strict Type Hinting**: All Python code will utilize strict type hints for improved readability and error detection.
*   **Comprehensive Docstrings**: Every function, class, and module will have detailed docstrings explaining its purpose, arguments, and return values.
*   **Modular Neural Networks**: Network architectures (actors, critics, communication encoders/decoders) will be defined in `src/model.py`, promoting reusability.
*   **Hyperparameter Management**: All hyperparameters will be externalized into `src/config.py` for easy modification and reproducibility.
*   **Visualization**: `notebooks/main.ipynb` will contain all plotting code for training curves, performance comparisons, and qualitative analysis.
*   **Deterministic Seeding**: Global random seeds will be managed in `src/utils.py` to ensure reproducibility.

## 6. Conclusion and Broader Impact

### 6.1 Conclusion

The Meta-Communicative Actor-Critic (MCAC) framework offers a promising direction for developing intelligent multi-agent systems capable of dynamic coordination in complex and rapidly changing environments. By synthesizing MAAC, emergent communication, and MAML, MCAC addresses the critical research gap of adaptive communication in MARL. Our proposed methodology enables agents to meta-learn flexible communication protocols and policies, leading to enhanced adaptability, sample efficiency, and robustness. The rigorous experimental design, including comparisons against strong baselines and comprehensive ablation studies, will validate the effectiveness of MCAC in various dynamic multi-agent tasks.

### 6.2 Broader Impact

The development of highly adaptive multi-agent systems has significant broader impacts:

1.  **Autonomous Systems**: Improved coordination in autonomous vehicle fleets, drone swarms, and robotic teams operating in dynamic real-world scenarios, leading to safer and more efficient operations.
2.  **Resource Management**: More efficient and adaptive resource allocation in smart grids, supply chains, and disaster response, optimizing resource utilization and societal benefit.
3.  **Human-AI Collaboration**: Enabling AI agents to adapt their communication and collaboration strategies when interacting with humans in complex tasks, fostering more natural and effective human-AI partnerships.
4.  **Fundamental AI Research**: Advancing our understanding of emergent intelligence, communication, and learning-to-learn paradigms in multi-agent settings, pushing the boundaries of artificial general intelligence.

While the potential benefits are substantial, it is crucial to consider ethical implications, such as ensuring fairness in resource allocation, preventing adversarial exploitation in competitive scenarios, and designing transparent communication protocols for human oversight. Future work will focus on scaling MCAC to larger agent populations and more complex, real-world environments, alongside a thorough investigation of its societal impact.

## 7. References

[1] Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *Advances in neural information processing systems*, 30.
[2] Sukhbaatar, S., Fergus, R., et al. (2016). Learning multiagent communication with backpropagation. *Advances in neural information processing systems*, 29.
[3] Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *International conference on machine learning* (pp. 1126-1135).
[4] Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V., Jaderberg, M., ... & Graepel, T. (2017). Value-decomposition networks for cooperative multi-agent learning. *arXiv preprint arXiv:1706.05296*.
[5] Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., & Whiteson, S. (2018). Counterfactual multi-agent policy gradients. *Proceedings of the AAAI conference on artificial intelligence*, 32(1).
[6] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
[7] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *International conference on machine learning* (pp. 1861-1870).
[8] Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. *International conference on machine learning* (pp. 1928-1937).
[9] Espeholt, L., Soyer, H., Munos, R., Simonyan, K., Mnih, V., Ward, T., ... & Kavukcuoglu, K. (2018). Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. *International conference on machine learning* (pp. 1407-1416).
[10] Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning. *International conference on machine learning* (pp. 4295-4304).
[11] Tan, M. (1993). Multi-agent reinforcement learning: Independent vs. cooperative agents. *Proceedings of the tenth international conference on machine learning* (pp. 330-337).
[12] Tampuu, A., Matiisen, T., Kodelja, D., Kuzovkin, I., Korjus, K., Aru, J., ... & Vicente, R. (2017). Multiagent deep reinforcement learning with extremely sparse rewards. *arXiv preprint arXiv:1707.01495*.
[13] Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). Multi-agent reinforcement learning in sequential social dilemmas. *Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems* (pp. 464-473).
[14] Das, A., Gervet, T., Romoff, J., Batra, D., Parikh, D., Rabbat, M., & Pineau, J. (2019). Tarmac: Targeted multi-agent communication. *International Conference on Machine Learning* (pp. 1538-1546).

## 8. Usage Instructions

### 8.1 Installation

```bash
# Clone the repository
git clone <repository-url>
cd CA12

# Install dependencies (ensure a virtual environment is activated)
pip install -r requirements.txt
```

### 8.2 Running Experiments

To run the main training and evaluation experiments for the MCAC framework and baselines, use the main Jupyter notebook:

```bash
jupyter notebook notebooks/main.ipynb
```

Inside the notebook, you will find sections for:

*   Environment setup and data loading.
*   Training various algorithms (MCAC, MAAC, baselines).
*   Evaluating performance metrics and generating visualizations.
*   Ablation studies and detailed analysis.

### 8.3 Basic Code Usage Example

After setting up your environment and creating your `src/` modules, you could interact with an MCAC agent conceptually as follows:

```python
# This is a conceptual example, actual usage will involve src/ imports
from src.agents.mcac_agent import MCACAgent
from environments.dynamic_env import DynamicCooperativeNavigation
from src.config import MCACConfig

# Initialize environment and configuration
env = DynamicCooperativeNavigation(num_agents=MCACConfig.num_agents, dynamic_targets=True)
agent = MCACAgent(config=MCACConfig, obs_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

# Example of meta-training loop (simplified)
for meta_episode in range(MCACConfig.num_meta_episodes):
    # Sample a new task
    task = env.sample_new_task()

    # Inner loop: adapt agent to task using support set
    adapted_agent = agent.adapt_to_task(task.support_set)

    # Outer loop: update meta-parameters using query set performance
    meta_loss = adapted_agent.evaluate_on_query_set(task.query_set)
    agent.meta_update(meta_loss)

    if meta_episode % 10 == 0:
        print(f"Meta-Episode {meta_episode}: Meta-Loss = {meta_loss:.4f}")

# After meta-training, evaluate on a new unseen task
test_task = env.sample_new_task(unseen=True)
final_adapted_agent = agent.adapt_to_task(test_task.support_set)
performance = final_adapted_agent.evaluate_on_query_set(test_task.query_set)
print(f"Performance on unseen task: {performance:.2f}")
```

## 9. Requirements

### 9.1 System Requirements

*   Python 3.8 or higher
*   8GB RAM minimum (16GB recommended)
*   GPU support optional but highly recommended for large-scale experiments (CUDA compatible GPU).

### 9.2 Python Dependencies

The required Python packages are listed in `requirements.txt`:

```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
gymnasium>=0.28.1 # Using Gymnasium for modern API
tensorboard>=2.7.0
```

## 10. License

This project is part of the Deep Reinforcement Learning course materials and is intended for educational and research purposes. It is released under the MIT License.

## 11. Contact

For questions or issues related to this assignment, please contact the course instructors or refer to the course documentation.
