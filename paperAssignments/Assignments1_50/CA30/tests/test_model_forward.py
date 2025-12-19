import numpy as np

from ca30.model import BaseModel


def test_numpy_forward_shape():
    m = BaseModel(input_dim=4, hidden_dim=8, output_dim=3, backend="numpy", seed=0)
    x = np.zeros((2, 4), dtype=np.float32)
    out = m.forward(x)
    assert out.shape == (2, 3)


def test_forward_determinism():
    m1 = BaseModel(input_dim=4, hidden_dim=8, output_dim=3, backend="numpy", seed=0)
    m2 = BaseModel(input_dim=4, hidden_dim=8, output_dim=3, backend="numpy", seed=0)
    x = np.random.RandomState(0).randn(2, 4).astype(np.float32)
    o1 = m1.forward(x)
    o2 = m2.forward(x)
    assert np.allclose(o1, o2)


def test_torch_backend_fallback():
    # If torch unavailable this should not raise on import or construction
    m = BaseModel(input_dim=4, hidden_dim=8, output_dim=3)
    x = np.zeros((2, 4), dtype=np.float32)
    out = m.forward(x)
    assert out.shape[0] == 2
