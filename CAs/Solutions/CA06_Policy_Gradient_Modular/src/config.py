import torch

class Config:
    # General
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # Environment
    ENV_NAME_DISCRETE = "CartPole-v1"
    ENV_NAME_CONTINUOUS = "Pendulum-v1"

    # Network
    HIDDEN_DIM = 128

    # REINFORCE Agent
    REINFORCE_LR = 1e-3
    REINFORCE_GAMMA = 0.99
    REINFORCE_EPISODES = 1000

    # REINFORCE with Baseline Agent
    REINFORCE_BASELINE_LR_POLICY = 1e-3
    REINFORCE_BASELINE_LR_VALUE = 1e-3
    REINFORCE_BASELINE_GAMMA = 0.99
    REINFORCE_BASELINE_EPISODES = 1000

    # Actor-Critic Agent
    ACTOR_CRITIC_LR_ACTOR = 1e-4
    ACTOR_CRITIC_LR_CRITIC = 1e-3
    ACTOR_CRITIC_GAMMA = 0.99
    ACTOR_CRITIC_EPISODES = 1000

    # PPO Agent (Discrete)
    PPO_LR = 3e-4
    PPO_GAMMA = 0.99
    PPO_EPS_CLIP = 0.2
    PPO_K_EPOCHS = 4
    PPO_EPISODES = 1000

    # PPO Agent (Continuous)
    CONTINUOUS_PPO_LR = 3e-4
    CONTINUOUS_PPO_GAMMA = 0.99
    CONTINUOUS_PPO_EPS_CLIP = 0.2
    CONTINUOUS_PPO_K_EPOCHS = 4
    CONTINUOUS_PPO_EPISODES = 1000

    # Training
    MAX_TIMESTEPS = 500
    PRINT_INTERVAL = 10

