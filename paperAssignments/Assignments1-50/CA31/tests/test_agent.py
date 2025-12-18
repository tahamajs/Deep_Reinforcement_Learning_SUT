"""Tests for agent module."""
import torch
from src.agent import A2CAgent


def test_agent_initialization():
    """Test A2C agent initialization."""
    config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    agent = A2CAgent(num_inputs=4, num_actions=2, config=config)
    assert agent.model is not None
    assert agent.optimizer is not None


def test_compute_returns():
    """Test computing discounted returns."""
    config = {"gamma": 0.9}
    agent = A2CAgent(num_inputs=4, num_actions=2, config=config)
    rewards = [1.0, 2.0, 3.0]
    dones = [False, False, True]
    next_value = torch.tensor(0.0)

    returns = agent.compute_returns(rewards, dones, next_value)
    expected = torch.tensor([1 + 0.9*2 + 0.9**2*3, 2 + 0.9*3, 3])
    assert torch.allclose(returns, expected)