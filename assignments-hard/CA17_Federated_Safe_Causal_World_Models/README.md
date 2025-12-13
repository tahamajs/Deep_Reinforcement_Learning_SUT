# CA17: Federated Safe Causal World Models for Robust and Private Reinforcement Learning

## Project Overview

This capstone research project introduces **Federated Safe Causal World Models (FSCWM)**, a novel deep reinforcement learning framework that synthesizes federated learning, advanced safety constraints, causal inference, and world modeling. The primary goal is to develop an agent that can learn robust and safe policies in complex, multi-client environments while preserving data privacy and understanding causal relationships.

The core innovation lies in the integration of:
1.  **World Models**: To enable agents to learn a compressed, predictive representation of their environment for efficient planning and data generation.
2.  **Causal Reasoning**: To allow agents to understand the underlying cause-effect relationships, improving generalization, interpretability, and robustness to distributional shifts.
3.  **Advanced Safety Constraints**: To ensure learned policies adhere to specified safety critical rules, preventing undesirable or harmful behaviors.
4.  **Federated Learning**: To enable collaborative learning across multiple decentralized clients without direct data sharing, addressing privacy and data locality concerns.

This synthesis addresses a critical gap in current RL research: the lack of a unified framework that simultaneously tackles data privacy, safety, causality, and sample efficiency in dynamic, real-world settings. FSCWM aims to provide a robust solution for deploying RL agents in sensitive domains such as autonomous driving, healthcare, and industrial control, where data privacy, safety guarantees, and transparent decision-making are paramount.

## Theoretical Framework

### World Models (WM)
Inspired by Hafner et al. (2019, 2020), World Models learn a compressed latent representation of the environment. The agent interacts with this learned model, generating imagined experiences to train its policy. This greatly improves sample efficiency and allows for planning in the latent space.

A World Model typically consists of three main components:
1.  **Recurrent State-Space Model (RSSM)**: Predicts the next latent state given the current state and action. It comprises a recurrent model for the hidden state and a dynamics model for predicting the next stochastic state.
2.  **Encoder**: Maps observations from the environment into a latent representation.
3.  **Decoder**: Reconstructs observations from latent states.
4.  **Reward Predictor**: Predicts rewards from latent states.

#### RSSM Dynamics
The RSSM maintains a belief state \(h_t\) that is updated recurrently, and a stochastic state \(z_t\) sampled from a posterior distribution.
$$
h_t = f_\mu(h_{t-1}, z_{t-1}, a_{t-1}, o_t) \\
z_t \sim q_\phi(z_t | h_t, o_t) \quad \text{(Encoder)} \\
\hat{z}_t \sim p_\theta(z_t | h_t) \quad \text{(Prior)}
$$
where \(f_\mu\) is the recurrent model, \(q_\phi\) is the posterior (encoder), and \(p_\theta\) is the prior (dynamics model). The loss for the RSSM involves reconstructing observations and rewards, and a KL divergence between the posterior and prior:
$$
\mathcal{L}_{WM} = \mathbb{E}[\log p(o_t | z_t, h_t)] + \mathbb{E}[\log p(r_t | z_t, h_t)] - D_{KL}[q_\phi(z_t | h_t, o_t) || p_\theta(z_t | h_t)]
$$

### Causal Reasoning (CR)
Causal inference allows the agent to distinguish between correlation and causation, leading to more robust and generalizable policies. In FSCWM, causal reasoning is integrated into the world model to learn a disentangled latent space where causal variables are explicitly represented. This is inspired by works like Kocaoglu et al. (2017) and Goudet et al. (2018) on learning causal graphs from observational data.

We augment the RSSM to explicitly model causal dependencies between latent variables. This involves:
1.  **Causal Graph Learning**: Inferring a directed acyclic graph (DAG) over the latent state variables. This can be done using methods like PC algorithm or GFN-based approaches in a federated manner.
2.  **Causal Intervention**: Simulating interventions in the learned world model to understand "what if" scenarios and evaluate counterfactual outcomes.

#### Causal Latent Space
Let the latent state \(z_t\) be composed of causally disentangled components \(z_t = \{z_{t,1}, \dots, z_{t,k}\}\). We aim to learn a causal graph \(G\) where edges \(z_{t,i} \to z_{t,j}\) represent causal influence. The dynamics model can then be structured to respect this graph.
$$
p_\theta(z_t | h_t) = \prod_{i=1}^k p_\theta(z_{t,i} | \text{Pa}(z_{t,i}), h_t)
$$
where \(\text{Pa}(z_{t,i})\) are the parents of \(z_{t,i}\) in the causal graph \(G\). The causal discovery mechanism would be integrated into the world model's learning process, potentially through an additional loss term or a separate module.

### Advanced Safety Constraints (ASC)
Safety in RL is paramount for real-world deployment. FSCWM incorporates advanced safety mechanisms, drawing inspiration from Constrained Policy Optimization (CPO) (Achiam et al., 2017) and Safety-Critical RL (Amodei et al., 2016). We aim to ensure policies satisfy predefined constraints during both training and deployment.

Key aspects include:
1.  **Constraint Definition**: Specifying safety requirements as cost functions or predicates.
2.  **Constrained Optimization**: Modifying the policy optimization to maximize reward subject to constraints on expected costs.
3.  **Safety Monitoring**: Real-time monitoring of safety violations and adaptive control mechanisms.

#### Constrained Policy Optimization
The objective becomes:
$$
\max_\pi \mathbb{E}_{\pi}[\sum_{t=0}^T \gamma^t R_t] \quad \text{s.t.} \quad \mathbb{E}_{\pi}[\sum_{t=0}^T \gamma^t C_t] \le D
$$
where \(R_t\) is the reward, \(C_t\) is the cost (safety violation), and \(D\) is the maximum allowed cumulative cost. This can be solved using Lagrange multipliers or primal-dual approaches. In FSCWM, the world model can predict future costs, allowing for proactive safety enforcement.

### Federated Learning (FL)
Federated Learning enables multiple clients to collaboratively train a shared model without exchanging their raw data. This is crucial for privacy-preserving RL. FSCWM adapts Federated Averaging (FedAvg) (McMahan et al., 2017) to the world model and policy learning.

The process involves:
1.  **Client-side Training**: Each client trains its local world model and policy using its private data.
2.  **Server Aggregation**: The central server aggregates the model updates (weights) from multiple clients to create a global model.
3.  **Global Model Distribution**: The aggregated global model is then sent back to the clients for further local training.

#### Federated Averaging for World Models
Each client \(k\) has a local world model \(\mathcal{M}_k\) and policy \(\pi_k\).
1.  **Client Update**: Each client \(k\) performs local updates for \(E\) epochs on its dataset \(\mathcal{D}_k\):
    $$
    \mathcal{M}_k^{t+1}, \pi_k^{t+1} = \text{LocalTrain}(\mathcal{M}_t, \pi_t, \mathcal{D}_k)
    $$
2.  **Server Aggregation**: The server aggregates the models from \(K\) selected clients:
    $$
    \mathcal{M}_{t+1}, \pi_{t+1} = \sum_{k=1}^K \frac{n_k}{N} (\mathcal{M}_k^{t+1}, \pi_k^{t+1})
    $$
where \(n_k\) is the number of samples on client \(k\), and \(N = \sum n_k\). Differential Privacy (DP) mechanisms can be added during aggregation to enhance privacy.

### Synthesis: Federated Safe Causal World Models (FSCWM)

FSCWM combines these four powerful paradigms. The overall architecture consists of:
1.  **Federated World Model Learning**: Clients collaboratively train a shared World Model (RSSM, Encoder, Decoder, Reward Predictor) using federated averaging. Each client trains its local WM, and the server aggregates the weights.
2.  **Causal Discovery within WM**: The world model is augmented to learn and leverage causal graphs over its latent states. This causal discovery can occur locally on each client's data and be aggregated, or a global causal graph can be learned. This causal graph informs the structure of the dynamics model, ensuring that predictions respect causal relationships.
3.  **Imagination with Causal and Safety Awareness**: The policy agent uses the federated and causally-aware world model to generate imagined trajectories. During this imagination process, it explicitly evaluates potential safety violations (using predicted costs from the WM) and causal interventions.
4.  **Constrained Policy Optimization**: The policy is trained using imagined experiences, but its updates are constrained by the safety monitor. This ensures that even in imagination, the agent learns to avoid unsafe actions. The cost function for safety can also be learned or predicted by the world model.

The training loop for FSCWM would involve an outer federated loop and an inner world model/policy training loop.

#### High-Level Algorithm
1.  **Initialize Global World Model and Policy**: Server initializes \(\mathcal{M}\) and \(\pi\).
2.  **For each Federated Round**:
    a.  **Client Selection**: Server selects a subset of clients.
    b.  **For each selected Client \(k\)**:
        i.  **Download Global Model**: Client \(k\) downloads \(\mathcal{M}\) and \(\pi\).
        ii. **Local Data Collection**: Client \(k\) interacts with its local environment to collect new data.
        iii. **Local World Model Training**:
            *   Train local World Model \(\mathcal{M}_k\) using collected data, including reconstructing observations, rewards, and minimizing KL divergence.
            *   Integrate causal discovery mechanisms to refine the latent causal graph.
        iv. **Local Policy Training with Safety Constraints**:
            *   Generate imagined trajectories using \(\mathcal{M}_k\).
            *   Predict rewards and costs for imagined trajectories.
            *   Train local policy \(\pi_k\) using CPO or similar constrained optimization, ensuring imagined costs are within bounds.
    c.  **Upload Local Updates**: Clients upload their updated model weights (\(\Delta \mathcal{M}_k, \Delta \pi_k\)) to the server.
    d.  **Server Aggregation**: Server aggregates updates using FedAvg (potentially with DP) to update global \(\mathcal{M}\) and \(\pi\).

## Dataset Specifications

For demonstrating FSCWM, we will use a modified version of the [Safe Reinforcement Learning Benchmark](https://github.com/Safe-AI/safe-rl-benchmarks) environments, specifically focusing on a **Federated Safe Mountain Car with Causal Interventions**.

### Environment: Federated Safe Causal Mountain Car
This environment is based on the standard Continuous Mountain Car but with added complexities:
1.  **Multiple Clients**: Multiple instances of the Mountain Car environment, each representing a client with slightly different dynamics or reward functions (simulating data heterogeneity).
2.  **Safety Constraint**: A penalty for exceeding a certain velocity or for staying in a "danger zone" (e.g., a specific valley) for too long. This will be an additional cost signal \(C_t\).
3.  **Causal Intervention Points**: Certain environmental parameters (e.g., gravity, engine power) can be causally intervened upon. The agent needs to understand these causal links to make robust decisions.
4.  **Partial Observability**: Clients might have slightly different observation spaces or noise levels.

### Data Schema
Each client will generate data tuples: \((o_t, a_t, r_t, c_t, o_{t+1}, \text{done})\), where \(c_t\) is the instantaneous cost associated with safety violations.

### Preprocessing
-   Observations will be normalized to \([0, 1]\).
-   Rewards and costs will be scaled.

### Source URLs
-   Original Mountain Car: Gymnasium documentation
-   Safe RL Benchmarks: GitHub repository linked above.

## Code Map

The `src/` directory will contain the modular implementation of FSCWM.

-   **`src/config.py`**:
    -   Defines all hyperparameters for the World Model, Causal components, Safety, and Federated Learning.
    -   Includes network dimensions, learning rates, batch sizes, number of federated rounds, client participation rates, safety thresholds, and causal graph parameters.

-   **`src/model.py`**:
    -   `RSSMCore`: Implements the recurrent state-space model for world modeling.
    -   `CausalWorldModel`: Extends `RSSMCore` to include causal graph learning and integration. This model will predict future states, rewards, and costs, while respecting the learned causal dependencies.
    -   `PolicyNetwork`: Implements the actor-critic policy network, trained on imagined experiences.
    -   `SafetyCritic`: Predicts the expected cumulative cost (safety violation) for constrained policy optimization.

-   **`src/losses.py`**:
    -   `world_model_loss`: Combines reconstruction loss (observation, reward), KL divergence for RSSM, and a causal regularization loss.
    -   `policy_loss`: Standard policy gradient loss.
    -   `safety_loss`: Loss for the `SafetyCritic` and the constraint violation term for CPO.

-   **`src/data.py`**:
    -   `FederatedReplayBuffer`: A replay buffer adapted for federated settings, potentially managing multiple client buffers or aggregating experiences.
    -   `FederatedSafeCausalMountainCar`: A custom Gymnasium environment wrapper that simulates the federated, safe, and causal Mountain Car environment. This will manage multiple client environments and expose appropriate APIs for federated training.
    -   `ClientDataset`: Represents the local dataset on each client.

-   **`src/utils.py`**:
    -   `set_seed`: Utility for reproducible random seeding.
    -   `Logger`: Handles logging of training metrics (rewards, costs, losses) for both local clients and the global model.
    -   `checkpointing`: Saves and loads model checkpoints.
    -   `FederatedAggregator`: Implements the FedAvg aggregation logic (and potential DP).
    -   `CausalGraphLearner`: A module for learning the causal graph, potentially based on a PC algorithm variant or a neural approach, integrated within the world model training.

This comprehensive `README.md` will serve as the primary guide for understanding, implementing, and extending the FSCWM framework.

