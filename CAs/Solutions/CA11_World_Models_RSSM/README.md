# CA11: Advanced Model-Based Reinforcement Learning - Lecture Notes

This project provides a comprehensive implementation and theoretical exposition of advanced model-based reinforcement learning (MBRL) algorithms. It synthesizes several state-of-the-art techniques, including Variational Autoencoders (VAEs) for robust latent state representation, Recurrent State Space Models (RSSMs) for learning complex temporal dynamics, and the Dreamer agent architecture for efficient policy learning through imagination.

Our work addresses the critical challenges of sample efficiency and interpretability in reinforcement learning, particularly for continuous control tasks. By integrating these powerful components, we develop a modular framework that allows agents to learn predictive models of their environments and leverage these models for planning and policy optimization in a learned latent space, significantly reducing the need for real-world interactions.

## 1. Theoretical Framework: Why and How

Model-Based Reinforcement Learning (MBRL) aims to improve sample efficiency by learning a model of the environment dynamics. Instead of directly learning a policy from trial and error (model-free RL), an MBRL agent first learns how the world works. This "world model" can then be used for various purposes:
- **Planning**: Simulating future trajectories to evaluate potential actions.
- **Data Augmentation**: Generating synthetic experiences to train model-free components.
- **Understanding**: Providing insights into the environment's internal mechanisms.

This project focuses on a powerful subclass of MBRL known as *latent space world models*, where the environment's dynamics are learned and simulated within a compressed, learned latent representation rather than the raw observation space. This approach is motivated by the ability to handle high-dimensional observations (like images) and to capture the underlying causal factors of the environment.

### 1.1 The Research Gap: Integrating Perception, Dynamics, and Control

While individual components like VAEs, RSSMs, and Dreamer agents have demonstrated significant capabilities, a common challenge lies in their seamless integration into a robust, end-to-end framework that is both theoretically sound and practically efficient. The gap addressed by this work is the synthesis of these elements into a modular, production-grade system that facilitates understanding and further research into advanced MBRL. Specifically, we aim to:
1.  **Enhance Representation Learning**: Leverage VAEs to encode complex observations into a disentangled and compact latent space, crucial for efficient dynamics modeling.
2.  **Improve Temporal Dynamics**: Employ RSSMs to capture both deterministic and stochastic transitions in the latent space, providing a richer predictive model than simpler recurrent networks.
3.  **Optimize Policy Learning**: Utilize the Dreamer architecture's imagination capabilities for highly sample-efficient policy and value learning, divorcing policy optimization from direct environment interaction.

### 1.2 Synthesis: Combining State-of-the-Art Papers

This project synthesizes core ideas from several foundational papers:

-   **"World Models" by Ha and Schmidhuber (2018)**: This paper introduced the concept of learning a generative model of the environment (VAE for perception, RNN for dynamics) to facilitate control. Our work extends this by using the more advanced RSSM for dynamics and integrating a full Dreamer agent.
-   **"Dream to Control: Learning Behaviors by Latent Imagination" by Hafner et al. (ICLR 2019)**: This paper introduced the Dreamer agent, which learns policies and value functions purely from imagined trajectories within a learned world model (specifically, an RSSM). We adopt the fundamental policy optimization strategy from Dreamer.
-   **"Mastering Atari with Discrete World Models" by Hafner et al. (ICLR 2020)**: While our current focus is on continuous control, DreamerV2's emphasis on discrete latent states and improved value estimation informs the robustness considerations for stochasticity in our RSSM.

Our novel synthesis lies in creating a highly modular framework where the VAE handles the initial perceptual compression, the RSSM provides a sophisticated and predictive latent dynamics model, and the Dreamer agent's actor-critic networks operate entirely within the RSSM's imagined sequences. This clear separation of concerns, combined with careful hyperparameter management via `config.py`, yields a flexible and powerful MBRL system.

## 2. Mathematical Derivations

The mathematical foundations of this project are rigorously detailed in the accompanying LaTeX report: `report.tex`. Here, we provide a high-level overview of the key loss functions and objective functions.

### 2.1 Variational Autoencoder (VAE) Loss

The VAE aims to learn an encoding of observations \( o_t \) into a latent space \( z_t \) and reconstruct \( o_t \) from \( z_t \). The total loss for the VAE is a sum of the reconstruction loss and the KL divergence between the learned latent distribution and a prior.

-   **Reconstruction Loss**: Typically Mean Squared Error (MSE) for continuous observations, measuring how well the decoder \( p_{\theta}(o_t | z_t) \) reconstructs the original observation.
    \\[ \mathcal{L}_{recon} = -\mathbb{E}_{q_{\phi}(z_t | o_t)}[\log p_{\theta}(o_t | z_t)] \\]
    In our implementation, this is often approximated as:
    \\[ \mathcal{L}_{recon} = \|o_t - \hat{o}_t\|^2 \\]
-   **KL Divergence Loss**: Encourages the latent distribution \( q_{\phi}(z_t | o_t) \) to be close to a prior distribution (e.g., a standard Gaussian \( \mathcal{N}(0, I) \)), promoting a well-structured latent space.
    \\[ \mathcal{L}_{KL} = D_{KL}(q_{\phi}(z_t | o_t) || \mathcal{N}(0, I)) \\]
    For Gaussian distributions, this has a closed-form solution:
    \\[ D_{KL} = -0.5 \sum (1 + \log \sigma^2 - \mu^2 - \sigma^2) \\]
The total VAE loss:
\\[ \mathcal{L}_{VAE} = \mathcal{L}_{recon} + \mathcal{L}_{KL} \\]
(Refer to `report.tex` for the full derivation in Section III.A)

### 2.2 Recurrent State Space Model (RSSM) Loss

The RSSM learns the environment dynamics in the latent space. Its loss function combines several terms to ensure accurate prediction of observations, rewards, episode continuation, and consistent latent dynamics.

-   **Observation Reconstruction Loss**: Measures how well the RSSM's decoder reconstructs the observation \( o_t \) from the full latent state \( (h_t, z_t) \).
    \\[ \mathcal{L}_{obs\_recon} = \|o_t - \hat{o}_t\|^2 \\]
-   **Reward Prediction Loss**: Measures the accuracy of the reward model \( p_{\theta}(r_t | h_t, z_t) \) in predicting rewards \( r_t \) from the latent state.
    \\[ \mathcal{L}_{reward\_pred} = \|r_t - \hat{r}_t\|^2 \\]
-   **Continue Prediction Loss**: Binary cross-entropy loss for predicting whether an episode continues or terminates, based on the latent state.
    \\[ \mathcal{L}_{continue\_pred} = - (c_t \log \hat{c}_t + (1-c_t) \log (1-\hat{c}_t)) \\]
-   **KL Divergence (Dynamics Consistency)**: A crucial term that regularizes the stochastic latent state. It penalizes the difference between the posterior distribution \( q_{\text{posterior}}(z_t | h_t, o_t) \) (inferred from the current observation) and the prior distribution \( q_{\text{prior}}(z'_t | h_t) \) (predicted by the dynamics model).
    \\[ \mathcal{L}_{KL\_dynamics} = D_{KL}(q_{\text{posterior}}(z_t | h_t, o_t) || q_{\text{prior}}(z'_t | h_t)) \\]
The total RSSM loss:
\\[ \mathcal{L}_{RSSM} = \mathcal{L}_{obs\_recon} + \mathcal{L}_{reward\_pred} + \mathcal{L}_{continue\_pred} + \beta \mathcal{L}_{KL\_dynamics} \\]
where \( \beta \) is a weighting factor.
(Refer to `report.tex` for the full derivation in Section III.B)

### 2.3 Dreamer Agent Losses (Actor-Critic)

The Dreamer agent trains its actor (policy) and critic (value function) within the imagined trajectories generated by the RSSM.

-   **Actor Loss**: The actor \( \pi_{\psi}(a_t | h_t, z_t) \) is trained to maximize the expected future return from imagined trajectories. This is often achieved by maximizing a value estimate, such as the lambda-return \( V_{\text{target}}(s_t) \).
    \\[ \mathcal{L}_{actor} = -\mathbb{E}[V_{\text{target}}(s_t)] \\]
-   **Critic Loss**: The critic \( V_{\omega}(h_t, z_t) \) is trained to accurately predict the expected future return. It uses an MSE loss against the value targets.
    \\[ \mathcal{L}_{critic} = \frac{1}{2} \mathbb{E}[(V(s_t) - V_{\text{target}}(s_t))^2] \\]
The lambda-return \( V_{\text{target}} \) provides a balanced estimate between Monte Carlo returns and bootstrapped value estimates.
(Refer to `report.tex` for the full derivation in Section III.C)

## 3. Dataset Specifications

This project primarily utilizes standard Gymnasium environments. For data collection and training, we consider observations as raw state vectors for simplicity, though the VAE is designed to generalize to higher-dimensional inputs.

-   **Continuous CartPole**:
    -   **Observation Space**: `Box(4,)` - Contains cart position, cart velocity, pole angle, pole angular velocity.
    -   **Action Space**: `Box(1,)` - Represents the force applied to the cart (e.g., in `[-1, 1]`).
    -   **Data Collection**: Initial data is collected by interacting with the environment using a random policy for a specified number of episodes (`VAE_CONFIG.num_episodes_data_collection`).
-   **Continuous Pendulum**:
    -   **Observation Space**: `Box(3,)` - Cosine and sine of the pole angle, and pole angular velocity.
    -   **Action Space**: `Box(1,)` - Represents the torque applied to the pendulum (e.g., in `[-2, 2]`).
    -   **Data Collection**: Similar to CartPole, random policy interactions are used for initial data.

All observations are normalized or assumed to be within a reasonable range for neural network processing. We do not use external datasets; all data is generated dynamically through environment interaction.

## 4. Code Map: File-by-File Explanation

The project is structured into modular Python files, making it organized and reusable.

```
CA11/
├── agents/               # RL agent implementations
│   ├── __init__.py         # Imports for agents
│   ├── latent_actor.py     # Actor network for policy in latent space
│   ├── latent_critic.py    # Critic network for value estimation in latent space
│   └── dreamer_agent.py    # Main Dreamer agent combining RSSM and Actor-Critic
├── environments/         # Custom Gymnasium environments and wrappers
│   ├── __init__.py
│   ├── continuous_cartpole.py  # Continuous action CartPole environment
│   ├── continuous_pendulum.py  # Continuous action Pendulum environment
│   └── sequence_environment.py # (Optional) Environment for sequence modeling tasks
├── experiments/          # Experiment configurations and runners
│   ├── __init__.py
│   ├── config.py           # **Centralized hyperparameters for all components**
│   ├── world_model_experiment.py # Script for training world models (VAE, RSSM)
│   ├── rssm_experiment.py  # Script for training RSSM specifically
│   └── dreamer_experiment.py # Script for training the full Dreamer agent
├── models/               # Core neural network models for world modeling
│   ├── __init__.py         # Imports for models
│   ├── vae.py              # Variational Autoencoder (Encoder, Decoder, VAE classes)
│   ├── dynamics.py         # (Optional) Separate dynamics model if not fully integrated into RSSM
│   ├── reward_model.py     # (Optional) Separate reward prediction model
│   ├── world_model.py      # (Optional) Wrapper for combined world model components
│   ├── rssm.py             # Recurrent State Space Model (core dynamics model)
│   └── trainers.py         # (Optional) General training utilities for models
├── utils/                # Helper functions and utilities
│   ├── __init__.py
│   ├── data_collection.py  # Utilities for collecting experience from environments
│   └── visualization.py    # Functions for plotting and visualizing results
├── CA11.ipynb            # Original notebook, adapted to use modular code
├── training_examples.py  # Simplified script for running various training/analysis examples
├── requirements.txt      # Python dependencies
├── run.sh                # Main script to execute all experiments and generate reports
└── README.md             # This comprehensive documentation file
└── report.tex            # The full IEEE-formatted research paper with derivations
```

-   **`agents/latent_actor.py`**: Defines the `LatentActor` class, which is the policy network responsible for outputting action distributions (mean and log standard deviation) in the latent space. It uses `torch.tanh` to squash actions to the `[-1, 1]` range.
-   **`agents/latent_critic.py`**: Defines the `LatentCritic` class, which is the value network. It takes a latent state as input and outputs a scalar value prediction.
-   **`agents/dreamer_agent.py`**: Contains the `DreamerAgent` class, which orchestrates the entire Dreamer algorithm. It integrates the `RSSM` for world modeling, `LatentActorCritic` for policy learning, and manages the experience replay buffer. It handles `select_action`, `store_transition`, `update_world_model`, and `update_actor_critic` functionalities.
-   **`environments/`**: These files define custom or adapted Gymnasium environments that provide continuous observation and action spaces, suitable for the Dreamer agent. `ContinuousCartPole` and `ContinuousPendulum` are key examples.
-   **`experiments/config.py`**: **Crucially**, this file centralizes all hyperparameters (`VAEConfig`, `RSSMConfig`, `AgentConfig`, `DreamerConfig`, `GlobalConfig`). This modular approach ensures consistency, simplifies tuning, and improves reproducibility. It also includes `update_config_with_env_dims` to dynamically set observation and action dimensions based on the chosen environment.
-   **`models/vae.py`**: Implements the `VAEEncoder`, `VAEDecoder`, and `VariationalAutoencoder` classes. These components are responsible for learning effective latent representations of observations and reconstructing them.
-   **`models/rssm.py`**: Implements the `RSSM` (Recurrent State Space Model) class, which is the core dynamics model of the world. It predicts future latent states, rewards, and episode continuation probabilities. It includes `imagine_step` for stepping through imagined trajectories and `observe_step` for updating the model with real observations.
-   **`training_examples.py`**: A simplified script demonstrating how to train the VAE world model and the Dreamer agent. It leverages the modular classes and the `config.py` for its operations. It also includes functions for analyzing world model representations and conducting a comprehensive analysis.
-   **`utils/data_collection.py`**: Provides helper functions for collecting data from environments, which is essential for training the world model.
-   **`utils/visualization.py`**: Contains functions to generate various plots and visualizations, aiding in the analysis of model performance and learned representations.
-   **`run.sh`**: The main execution script that orchestrates the training of world models and Dreamer agents across different environments, and then generates comprehensive visualizations and reports. It uses the Python scripts in `experiments/` and `training_examples.py`.
-   **`report.tex`**: The formal IEEE-formatted research paper detailing the theoretical background, mathematical derivations, experimental setup, and results.

## 5. Installation

To set up the environment and install the necessary dependencies, follow these steps:

1.  **Clone the repository (if not already done):**
    ```bash
    git clone <repository_url>
    cd CA11_World_Models_RSSM
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 6. Usage

### 6.1 Running Experiments (using `run.sh`)

The `run.sh` script provides a convenient way to execute all predefined experiments, including world model training, RSSM training, Dreamer agent training, and generating analysis reports.

To run all experiments:

```bash
bash run.sh
```

This script will:
-   Create necessary directories (`visualizations`, `results`, `logs`).
-   Set up `PYTHONPATH`.
-   Run `world_model_experiment.py` for `continuous_cartpole` and `continuous_pendulum`.
-   Run `rssm_experiment.py` for `continuous_cartpole` and `continuous_pendulum`.
-   Run `dreamer_experiment.py` for `continuous_cartpole` and `continuous_pendulum`.
-   Execute `training_examples.py` for VAE and Dreamer agent training, and generate representation analyses.
-   Generate a `summary_report.png` and `summary.json` in the `results/` and `visualizations/` directories, respectively.

### 6.2 Running Individual Experiments (using scripts in `experiments/`)

You can also run individual experiment scripts directly. Ensure you are in the `CA11_World_Models_RSSM` directory or have set your `PYTHONPATH` correctly.

1.  **World Model Training (VAE/RSSM combined)**:
    ```bash
    python experiments/world_model_experiment.py --env continuous_cartpole
    ```
    (Note: `world_model_experiment.py` should be implemented to combine VAE and RSSM training. Currently, `training_examples.py` handles the VAE training example.)

2.  **RSSM Training**:
    ```bash
    python experiments/rssm_experiment.py --env continuous_pendulum
    ```

3.  **Dreamer Agent Training**:
    ```bash
    python experiments/dreamer_experiment.py --env continuous_cartpole
    ```

These scripts are designed to load hyperparameters from `experiments/config.py`.

### 6.3 Using Individual Components Programmatically

You can import and use any component directly in your Python code:

```python
import torch
import gymnasium as gym
from agents.dreamer_agent import DreamerAgent
from models.vae import VariationalAutoencoder
from models.rssm import RSSM
from experiments.config import GLOBAL_CONFIG, VAE_CONFIG, update_config_with_env_dims
from training_examples import train_vae_world_model, train_dreamer_agent, analyze_world_model_representations

# Set global seed
# from utils.misc import set_seed # Assuming set_seed is in utils.misc
# set_seed(GLOBAL_CONFIG.seed)

# Example: Train a VAE
# update_config_with_env_dims("Pendulum-v1") # Call this to set obs/action dims based on env
# vae_results = train_vae_world_model(env_name="Pendulum-v1")
# print("VAE Training Complete:", vae_results['losses']['total'][-1])

# Example: Initialize Dreamer Agent
# env_name = "continuous_cartpole"
# update_config_with_env_dims(env_name) # Update config with dimensions for this env
# env = gym.make(env_name)
# obs_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# env.close()

# dreamer = DreamerAgent(obs_dim, action_dim, GLOBAL_CONFIG)
# print("Dreamer Agent initialized.")

# Example: Run analysis
# fig = analyze_world_model_representations(save_path=f'{GLOBAL_CONFIG.visualizations_dir}/custom_representations.png')
# print(f"Custom representations saved to {GLOBAL_CONFIG.visualizations_dir}/custom_representations.png")
```

## 7. Key Features

-   **Modular Design**: Clean separation of world models, agents, environments, and utilities into distinct Python files and packages. This enhances readability, maintainability, and reusability.
-   **Centralized Configuration**: All hyperparameters are managed in `experiments/config.py`, providing a single source of truth for experiment settings and simplifying tuning.
-   **Comprehensive Training Examples**: `training_examples.py` offers clear examples of how to train VAE-based world models and the full Dreamer agent.
-   **Extensible Architecture**: The modular nature makes it easy to add new environments, experiment configurations, world model components, or agent architectures.
-   **Visualization Tools**: Built-in plotting functions (in `utils/visualization.py` and `training_examples.py`) for analyzing model performance, learned representations, and imagined trajectories.
-   **Rigorous Documentation**: This `README.md` serves as comprehensive lecture notes, and `report.tex` provides a formal research paper with detailed mathematical derivations.
-   **GPU Acceleration**: Models are designed to leverage PyTorch's GPU acceleration when available, improving training efficiency.

## 8. Algorithms Implemented

This project focuses on the following advanced model-based RL algorithms:

1.  **Variational Autoencoders (VAE)**: Employed for learning compressed, disentangled latent representations of high-dimensional observations. This forms the perceptual component of the world model.
2.  **Recurrent State Space Models (RSSM)**: The core dynamics model, capable of learning both deterministic and stochastic transitions within the latent space. It allows for accurate multi-step prediction and imagination.
3.  **Dreamer Agent**: A complete model-based RL agent that utilizes the learned RSSM to generate imagined trajectories. Policies and value functions are learned directly from these imagined sequences, leading to high sample efficiency.

## 9. Dependencies

The project relies on standard Python libraries for deep learning, numerical computation, and reinforcement learning:

-   `torch>=2.0.0`: Primary deep learning framework (PyTorch).
-   `numpy>=1.21.0`: Fundamental package for numerical computation.
-   `matplotlib>=3.5.0`: For creating static, interactive, and animated visualizations in Python.
-   `seaborn>=0.11.0`: Statistical data visualization library based on matplotlib.
-   `tqdm>=4.64.0`: For fast, extensible progress bars.
-   `gymnasium>=0.29.0`: Toolkit for developing and comparing reinforcement learning algorithms. (Successor to OpenAI Gym).
-   `pandas>=1.3.0`: For data manipulation and analysis, particularly for handling experiment results.
-   `scikit-learn>=1.0.0`: For various machine learning utilities (e.g., data preprocessing, if needed).
-   `tensorboard>=2.8.0`: For visualizing training runs and debugging models.
-   `wandb>=0.12.0`: (Weights & Biases) for experiment tracking and visualization.
-   `jupyter>=1.0.0`: For running the `CA11.ipynb` notebook.
-   `ipykernel>=6.0.0`: IPython Kernel for Jupyter.

## 10. Notes

-   **Continuous Action Spaces**: All components, especially the agent and environment wrappers, are designed to work effectively with continuous action spaces, which are common in robotics and control tasks.
-   **Device Agnostic**: Models and training loops are configured to automatically utilize GPU acceleration (CUDA) if available, falling back to CPU otherwise (`GLOBAL_CONFIG.device`).
-   **Comprehensive Analysis**: The visualization functions and `training_examples.py` script are tailored to provide in-depth analysis of learned representations, model predictive capabilities, and agent performance.
-   **Experiment Configuration**: The `experiments/` directory contains template scripts for running various experiments, each loaded with hyperparameters from `config.py` and designed to log metrics and save results.

## 11. Future Directions

This work lays a strong foundation for future research in advanced MBRL. Potential areas for extension include:

-   **Multi-Modal Observations**: Extending the VAE and RSSM to handle diverse observation modalities (e.g., images, proprioception, text) for more complex environments.
-   **Hierarchical Reinforcement Learning**: Integrating hierarchical policies with the Dreamer agent to tackle long-horizon tasks by learning sub-goals and abstract actions.
-   **Meta-Learning for World Models**: Developing methods for the world model itself to adapt quickly to new environments or tasks with limited data.
-   **Continual Learning**: Investigating strategies for world models and agents to continuously learn and adapt without forgetting previously acquired knowledge.
-   **Uncertainty Quantification**: Enhancing the RSSM to provide more robust uncertainty estimates, which can be crucial for risk-aware planning and safe exploration.
-   **Real-world Robotics Integration**: Adapting the framework for deployment on physical robotic platforms, addressing challenges related to sim-to-real transfer.

This project serves as a robust starting point for exploring the exciting frontier of model-based reinforcement learning with latent world models.
