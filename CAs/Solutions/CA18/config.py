"""
Configuration module for CA18 - Advanced Reinforcement Learning Paradigms

This module contains all hyperparameter configurations for the various
advanced RL paradigms implemented in CA18.
"""

# World Models Configuration
WORLD_MODELS_CONFIG = {
    # Environment parameters
    'state_dim': 4,
    'action_dim': 2,
    
    # Data collection
    'n_episodes_data_collection': 50,  # Number of episodes to collect random data
    
    # Training batches
    'batch_size': 16,  # Batch size for training
    'seq_length': 15,  # Sequence length for training sequences
    
    # World model architecture
    'state_dim_wm': 20,  # Latent state dimension
    'hidden_dim_wm': 100,  # Hidden dimension for RNN
    'embed_dim_wm': 256,  # Embedding dimension for encoder
    
    # Training parameters
    'n_epochs_wm': 30,  # Number of training epochs
    'learning_rate_wm': 1e-3,  # Learning rate for world model training
    
    # MPC planning
    'horizon': 10,  # Planning horizon
    'n_candidates': 50,  # Number of candidate action sequences
    'n_iterations': 5,  # CEM iterations
    'n_elite': 10,  # Elite samples for CEM
    
    # Evaluation
    'n_episodes_eval': 10,  # Number of evaluation episodes
    'max_steps': 50,  # Maximum steps per episode
}

# Multi-Agent RL Configuration
MULTI_AGENT_CONFIG = {
    # Environment parameters
    'n_agents': 3,  # Number of agents in the environment
    'obs_dim': 8,  # Observation dimension per agent
    'action_dim': 2,  # Action dimension per agent
    
    # Agent architecture
    'hidden_dim_agent': 64,  # Hidden dimension for agent networks
    
    # Training parameters
    'n_episodes_ma': 200,  # Number of training episodes
    'buffer_size': 1000000,  # Replay buffer size
    'batch_size_ma': 64,  # Training batch size
    'gamma_ma': 0.95,  # Discount factor
    'tau': 0.01,  # Soft update parameter for target networks
    'lr_actor': 1e-4,  # Actor learning rate
    'lr_critic': 1e-3,  # Critic learning rate
    
    # Exploration
    'noise_std': 0.1,  # Standard deviation for exploration noise
    
    # Evaluation
    'n_episodes_eval_ma': 10,  # Number of evaluation episodes
    
    # Attention mechanism
    'attention_heads': 4,  # Number of attention heads
    'attention_embed_dim': 32,  # Attention embedding dimension
    
    # Communication
    'communication_dim': 16,  # Communication message dimension
}

# Causal RL Configuration
CAUSAL_RL_CONFIG = {
    # Environment parameters
    'state_dim_causal': 3,  # State dimension (X, Y, Z)
    'action_dim_causal': 1,  # Action dimension
    
    # Causal discovery
    'n_samples_causal': 500,  # Number of samples for causal discovery
    'alpha_causal': 0.05,  # Significance level for causal discovery
    
    # World model architecture
    'obs_dim_causal': 3,  # Observation dimension
    'action_dim_causal_model': 1,  # Action dimension for causal model
    'hidden_dim_causal': 64,  # Hidden dimension for causal world model
    'latent_dim_causal': 16,  # Latent dimension for causal representations
    
    # Training parameters
    'n_epochs_causal': 50,  # Number of training epochs
    'batch_size_causal': 32,  # Training batch size
    'learning_rate_causal': 1e-3,  # Learning rate
    'causal_constraint_weight': 0.1,  # Weight for causal constraint loss
}

# Quantum RL Configuration
QUANTUM_RL_CONFIG = {
    # Environment parameters
    'n_qubits': 2,  # Number of qubits in quantum environment
    'state_dim_quantum': 4,  # State dimension (2^n_qubits)
    'action_dim_quantum': 2,  # Action dimension (rotations per qubit)
    
    # Quantum Q-Learning
    'n_episodes_q': 100,  # Number of training episodes
    'learning_rate_q': 0.1,  # Learning rate
    'discount_factor_q': 0.95,  # Discount factor
    'exploration_rate_q': 1.0,  # Initial exploration rate
    'exploration_decay_q': 0.995,  # Exploration decay rate
    'min_exploration_rate': 0.01,  # Minimum exploration rate
    
    # Quantum Actor-Critic
    'n_episodes_ac': 100,  # Number of training episodes
    'actor_hidden_dim_q': 32,  # Actor hidden dimension
    'critic_hidden_dim_q': 32,  # Critic hidden dimension
    'learning_rate_ac': 1e-3,  # Learning rate for actor-critic
    'gamma_ac': 0.99,  # Discount factor for actor-critic
}

# Federated RL Configuration
FEDERATED_RL_CONFIG = {
    # Client configuration
    'n_clients': 5,  # Number of federated clients
    'clients_per_round': 3,  # Number of clients participating per round
    
    # Training parameters
    'n_rounds': 20,  # Number of communication rounds
    'local_epochs': 5,  # Local training epochs per client
    'local_batch_size': 32,  # Local batch size
    'learning_rate_fed': 1e-3,  # Learning rate for federated learning
    
    # Privacy parameters
    'use_differential_privacy': True,  # Whether to use differential privacy
    'epsilon': 1.0,  # Privacy budget
    'delta': 1e-5,  # Privacy parameter
    'clip_norm': 1.0,  # Gradient clipping norm
    
    # Communication
    'compression_rate': 0.1,  # Model compression rate for communication
    'secure_aggregation': True,  # Use secure aggregation
}

# Integration Configuration
INTEGRATION_CONFIG = {
    # Integrated Environment
    'n_agents_integrated': 2,  # Number of agents in integrated environment
    'obs_dim_integrated': 6,  # Observation dimension
    'action_dim_integrated': 2,  # Action dimension
    'max_steps_integrated': 100,  # Maximum steps per episode
    
    # Paradigm Integration
    'n_episodes_integration': 10,  # Number of episodes for integration demo
    
    # Quantum Agent in Integration
    'n_qubits_integration': 2,  # Qubits for quantum agent
    'learning_rate_integration': 0.1,  # Learning rate for quantum agent
    'gamma_integration': 0.95,  # Discount factor for quantum agent
    'n_layers_integration': 1,  # Number of layers in quantum circuit
    
    # Federated Learning in Integration
    'n_clients': 3,  # Number of federated clients
    'n_rounds': 10,  # Number of communication rounds
    'local_epochs': 5,  # Local training epochs per client
    'episodes_per_client': 20,  # Episodes per client per round
    
    # Hybrid Agent
    'state_dim_hybrid': 16,  # State dimension for hybrid agent
    'hidden_dim_hybrid': 32,  # Hidden dimension for hybrid agent
    'embed_dim_hybrid': 64,  # Embedding dimension for hybrid agent
}

# Advanced Safety Configuration
SAFETY_CONFIG = {
    # Safety constraints
    'safety_threshold': 0.9,  # Safety constraint threshold
    'constraint_penalty': 10.0,  # Penalty for constraint violation
    
    # Shield parameters
    'shield_type': 'probabilistic',  # Type of safety shield
    'shield_confidence': 0.95,  # Confidence level for probabilistic shield
    
    # Verification
    'verification_samples': 1000,  # Number of samples for safety verification
    'verification_horizon': 10,  # Horizon for safety verification
    
    # Robustness
    'adversarial_epsilon': 0.1,  # Epsilon for adversarial robustness
    'noise_level': 0.05,  # Noise level for robustness testing
}

# Experimental Configuration
EXPERIMENT_CONFIG = {
    # Random seeds
    'seed': 42,  # Random seed for reproducibility
    'torch_seed': 42,  # PyTorch random seed
    'numpy_seed': 42,  # NumPy random seed
    
    # Logging
    'log_interval': 10,  # Episodes between logging
    'save_interval': 50,  # Episodes between model saves
    'eval_interval': 20,  # Episodes between evaluations
    
    # Visualization
    'plot_interval': 50,  # Episodes between plots
    'save_plots': True,  # Whether to save plots
    'plot_dir': './plots/',  # Directory for plots
    
    # Model saving
    'save_models': True,  # Whether to save models
    'model_dir': './models/',  # Directory for models
    
    # Results
    'save_results': True,  # Whether to save results
    'results_dir': './results/',  # Directory for results
}

# Device Configuration
DEVICE_CONFIG = {
    'use_cuda': False,  # Whether to use CUDA if available
    'device': 'cpu',  # Device to use ('cpu' or 'cuda')
    'num_threads': 4,  # Number of CPU threads
    'precision': 'float32',  # Numerical precision
}

def get_config(paradigm: str) -> dict:
    """
    Get configuration for a specific paradigm
    
    Args:
        paradigm: Name of the paradigm ('world_models', 'multi_agent', 
                  'causal', 'quantum', 'federated', 'integration', 'safety')
    
    Returns:
        Configuration dictionary for the specified paradigm
    """
    configs = {
        'world_models': WORLD_MODELS_CONFIG,
        'multi_agent': MULTI_AGENT_CONFIG,
        'causal': CAUSAL_RL_CONFIG,
        'quantum': QUANTUM_RL_CONFIG,
        'federated': FEDERATED_RL_CONFIG,
        'integration': INTEGRATION_CONFIG,
        'safety': SAFETY_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
        'device': DEVICE_CONFIG,
    }
    
    if paradigm not in configs:
        raise ValueError(f"Unknown paradigm: {paradigm}. "
                        f"Available: {list(configs.keys())}")
    
    return configs[paradigm]

def update_config(paradigm: str, updates: dict) -> dict:
    """
    Update configuration for a specific paradigm
    
    Args:
        paradigm: Name of the paradigm
        updates: Dictionary of updates to apply
    
    Returns:
        Updated configuration dictionary
    """
    config = get_config(paradigm).copy()
    config.update(updates)
    return config

def print_config(paradigm: str):
    """Print configuration for a specific paradigm"""
    config = get_config(paradigm)
    print(f"\n{'='*60}")
    print(f"Configuration for {paradigm.upper()}")
    print(f"{'='*60}")
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print(f"{'='*60}\n")

# Set random seeds
def set_seeds(seed: int = None):
    """Set random seeds for reproducibility"""
    if seed is None:
        seed = EXPERIMENT_CONFIG['seed']
    
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"✅ Random seeds set to {seed}")
