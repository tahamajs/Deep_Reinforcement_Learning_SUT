from dataclasses import dataclass
import torch

@dataclass
class Config:
    # General
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Environment
    env_name: str = "FederatedSafeCausalMountainCar-v0"
    obs_dim: int = 2  # Position, Velocity
    action_dim: int = 1 # Force
    action_space_high: float = 1.0
    action_space_low: float = -1.0
    max_episode_steps: int = 200
    n_clients: int = 3 # Number of federated clients
    client_data_heterogeneity: float = 0.1 # Degree of data heterogeneity among clients

    # World Model (RSSM) Parameters
    rssm_h_dim: int = 256 # Hidden state dimension for RSSM
    rssm_stochastic_dim: int = 32 # Stochastic latent state dimension
    rssm_deterministic_dim: int = 256 # Deterministic latent state dimension
    rssm_hidden_layers: int = 1 # Number of hidden layers in RSSM components
    rssm_activation: str = "ELU"

    # Encoder/Decoder Parameters
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256

    # Reward/Cost Predictor Parameters
    reward_predictor_hidden_dim: int = 256
    cost_predictor_hidden_dim: int = 256

    # Policy Network Parameters
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    actor_layers: int = 2
    critic_layers: int = 2
    learning_rate_actor: float = 1e-4
    learning_rate_critic: float = 1e-4

    # Safety Parameters
    safety_threshold: float = 25.0 # Maximum allowed cumulative cost
    cost_scale: float = 100.0 # Scaling factor for cost function
    safety_margin: float = 0.5 # Margin for safety monitoring

    # Causal Reasoning Parameters
    causal_graph_num_nodes: int = 5 # Number of nodes in the causal graph (latent variables)
    causal_discovery_lr: float = 1e-3 # Learning rate for causal discovery module
    causal_regularization_coeff: float = 0.1 # Coefficient for causal regularization loss
    causal_graph_complexity_penalty: float = 0.01 # Penalty for complex causal graphs

    # Federated Learning Parameters
    federated_rounds: int = 50 # Number of federated communication rounds
    client_epochs: int = 5 # Local training epochs per client
    client_batch_size: int = 64 # Batch size for client-side training
    server_aggregation_method: str = "FedAvg" # Aggregation method
    client_participation_rate: float = 0.7 # Fraction of clients participating each round
    differential_privacy_scale: float = 0.01 # Noise scale for differential privacy (if used)

    # Training Parameters
    n_episodes: int = 1000 # Total episodes for training
    n_steps_per_episode: int = 200 # Max steps per episode
    replay_buffer_size: int = 1_000_000
    batch_size: int = 32 # Batch size for world model training (imagined experience)
    grad_clip_norm: float = 100.0
    model_learning_rate: float = 6e-4
    value_learning_rate: float = 8e-5
    discount_factor: float = 0.99
    lambda_factor: float = 0.95 # For GAE and Retrace
    horizon: int = 15 # Planning horizon for MPC
    free_nats: float = 3.0 # For KL divergence in RSSM
    kl_loss_scale: float = 1.0
    reward_loss_scale: float = 1.0
    observation_loss_scale: float = 1.0

    # Logging and Checkpointing
    log_interval: int = 10 # Log every N episodes
    checkpoint_interval: int = 50 # Save checkpoint every N federated rounds
    save_dir: str = "results"

    # Experiment specific
    action_noise_std: float = 0.1
    exploration_epsilon: float = 0.1 # For epsilon-greedy in exploration














