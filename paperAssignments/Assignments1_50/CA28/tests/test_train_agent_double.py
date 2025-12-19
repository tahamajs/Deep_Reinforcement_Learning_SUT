import numpy as np
from src.config import Config
from src.train import DQNAgent


def test_agent_double_dqn_flag():
    cfg = Config()
    cfg.double_dqn = True
    cfg.batch_size = 4

    agent = DQNAgent(cfg, state_dim=4, action_dim=2)

    for i in range(8):
        s = np.zeros(4) + i
        agent.memory.push(s, i % 2, float(i), s + 0.1, False)

    # Should not raise
    agent.optimize_model()
    assert hasattr(agent, "policy_net")
