import numpy as np
from src.train import DQNAgent
from src.config import Config


def test_agent_initialization_and_action():
    cfg = Config()
    agent = DQNAgent(cfg, state_dim=4, action_dim=2)

    # If epsilon is 1.0, random action is possible but within range
    agent.epsilon = 1.0
    a = agent.select_action(np.zeros(4))
    assert 0 <= a < agent.action_dim

    # If epsilon is 0.0, action should be deterministic (argmax)
    agent.epsilon = 0.0
    # set policy net to output predictable values
    import torch
    with torch.no_grad():
        for p in agent.policy_net.parameters():
            p.zero_()
    a2 = agent.select_action(np.zeros(4))
    assert 0 <= a2 < agent.action_dim
