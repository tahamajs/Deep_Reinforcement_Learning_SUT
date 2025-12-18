from typing import Tuple
import numpy as np
from .model import BaseModel


def make_synthetic_batch(batch_size: int, input_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randn(batch_size, input_dim).astype(np.float32)
    y = (np.sum(X, axis=1) > 0).astype(int)
    return X, y


def simple_train_epoch(model: BaseModel, batch_size: int, input_dim: int):
    """A tiny training loop used for examples and tests. This is NOT meant for production experiments.

    It performs a forward pass and a dummy parameter update (only if numpy model) to validate the training flow.
    """
    X, y = make_synthetic_batch(batch_size, input_dim)
    out = model.forward(X)
    # For numpy model we can apply a tiny gradient step on W2 for example.
    if model.backend == "numpy":
        # compute simple gradient for W2 using mean squared error to dummy targets
        targets = np.zeros_like(out)
        targets[np.arange(len(y)), y] = 1.0
        grad = 2 * (out - targets) / out.shape[0]
        # backprop into W2 approximated through h
        h = np.maximum(X.dot(model.W1) + model.b1, 0.0)
        dW2 = h.T.dot(grad)
        model.W2 -= 1e-3 * dW2
    return out
