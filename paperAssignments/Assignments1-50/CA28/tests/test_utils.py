import numpy as np
import pytest
from src.utils import set_seed, ReplayBuffer

def test_set_seed():
    """Test that setting seed doesn't raise an error."""
    set_seed(42)
    # In a real test, you might check reproducibility of random outputs
    assert True

def test_replay_buffer():
    """Test ReplayBuffer functionality."""
    buffer = ReplayBuffer(10)
    state = np.array([1, 2, 3, 4])
    action = 0
    reward = 1.0
    next_state = np.array([1, 2, 3, 4])
    done = False

    # Test push and length
    buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 1

    # Add a few more items and test sampling
    for i in range(5):
        buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 6

    # Test sample
    states, actions, rewards, next_states, dones = buffer.sample(3)
    assert states.shape == (3, 4)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert next_states.shape == (3, 4)
    assert dones.shape == (3,)

def test_replay_buffer_sample_too_large():
    buffer = ReplayBuffer(3)
    state = np.zeros(4)
    for _ in range(2):
        buffer.push(state, 0, 0.0, state, False)
    with pytest.raises(ValueError):
        buffer.sample(4)
