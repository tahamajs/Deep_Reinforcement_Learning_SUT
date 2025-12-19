"""Tests for model module."""
import torch
from src.model import ActorCritic


def test_actor_critic_forward():
    """Test forward pass of ActorCritic."""
    model = ActorCritic(num_inputs=4, num_outputs=2)
    state = torch.randn(1, 4)
    action_logits, value = model(state)

    assert action_logits.shape == (1, 2)
    assert value.shape == (1, 1)


def test_get_action():
    """Test action sampling."""
    model = ActorCritic(num_inputs=4, num_outputs=2)
    state = torch.randn(1, 4)
    action, log_prob, value = model.get_action(state)

    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert value.shape == (1, 1)