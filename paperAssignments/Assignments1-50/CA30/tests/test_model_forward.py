import numpy as np

from ca30.model import BaseModel


def test_numpy_forward_shape():
    m = BaseModel(input_dim=4, hidden_dim=8, output_dim=3, backend="numpy")
    x = np.zeros((2, 4), dtype=np.float32)
    out = m.forward(x)
    assert out.shape == (2, 3)


def test_torch_backend_fallback():
    # If torch unavailable this should not raise on import or construction
    m = BaseModel(input_dim=4, hidden_dim=8, output_dim=3)
    x = np.zeros((2, 4), dtype=np.float32)
    out = m.forward(x)
    assert out.shape[0] == 2
