from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import torch
from torch.optim.optimizer import Optimizer


class Sophia(Optimizer):
    r"""
    A lightweight implementation of the Sophia-style diagonal Hessian
    preconditioner described in ICLR 2024 (diagonal Hessian EMA + clip).

    Per-parameter state:
      - m: EMA of gradients
      - h: EMA of Hessian-diag proxy (e.g., squared gradients)

    Hyperparameters (per param-group):
      - lr: learning rate
      - beta1: EMA decay for first moment (default 0.965)
      - beta2: EMA decay for Hessian proxy (default 0.99)
      - gamma: curvature scaling applied to h (default 0.1)
      - eps: floor for denom to avoid division by zero (default 1e-12)
      - clip: max absolute value for the normalized step (default 1.0)

    This optimizer intentionally implements a simple, interpretable update:
      m_t = beta1 * m_{t-1} + (1-beta1) * g_t
      h_t = beta2 * h_{t-1} + (1-beta2) * \hat{h}_t  (e.g., g_t**2)
      step = clip(m_t / max(gamma * h_t, eps), -clip, clip) * lr
      theta <- theta - step
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-4,
        beta1: float = 0.965,
        beta2: float = 0.99,
        gamma: float = 0.1,
        eps: float = 1e-12,
        clip: float = 1.0,
        hessian_estimator: str = "squared_grad",
        hutchinson_samples: int = 1,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2: {}".format(beta2))

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            gamma=gamma,
            eps=eps,
            clip=clip,
            hessian_estimator=hessian_estimator,
            hutchinson_samples=hutchinson_samples,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None):
        """
        Performs a single optimization step.
        If `closure` is provided, it should recompute the loss and return it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            gamma = group["gamma"]
            eps = group["eps"]
            clip_val = group["clip"]
            estimator = group.get("hessian_estimator", "squared_grad")
            hutchinson_samples = int(group.get("hutchinson_samples", 1))

            params = [p for p in group["params"] if p is not None]
            # If using Hutchinson estimator we need the loss/graph; require closure
            if estimator == "hutchinson":
                if closure is None:
                    raise RuntimeError(
                        "Hutchinson estimator requires a closure that returns the loss (for second-order computation)"
                    )
                # evaluate closure to get loss with create_graph True
                loss = closure()
                # compute gradients with create_graph for Hutchinson
                grads = torch.autograd.grad(
                    loss, params, create_graph=True, allow_unused=True
                )
            else:
                grads = [p.grad.data if p.grad is not None else None for p in params]

            for p, grad in zip(params, grads):
                if p.grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")

                state = self.state[p]
                # State initialization
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                    state["h"] = torch.zeros_like(p.data)

                m = state["m"]
                h = state["h"]
                # convert grad to tensor if None (safety)
                if grad is None:
                    grad = torch.zeros_like(p.data)

                # Update moments (keep in param dtype)
                m.mul_(beta1).add_(grad, alpha=(1.0 - beta1))

                # Hessian proxy update
                if estimator == "hutchinson":
                    # perform Hutchinson diag estimate: Hv * v elementwise with random v
                    hv_acc = torch.zeros_like(p.data)
                    for _ in range(max(1, hutchinson_samples)):
                        v = (
                            torch.randint(
                                0,
                                2,
                                p.data.shape,
                                device=p.data.device,
                                dtype=p.data.dtype,
                            )
                            * 2
                            - 1
                        )
                        # compute Hessian-vector product Hv via autograd: grad_outputs = v
                        Hv = torch.autograd.grad(
                            grad,
                            p,
                            grad_outputs=v,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        if Hv is None or Hv[0] is None:
                            # fallback to squared grad if Hv unavailable
                            hv_est = grad * grad
                        else:
                            Hv = Hv[0]
                            hv_est = Hv * v
                        hv_acc.add_(hv_est)
                    hv_mean = hv_acc.div(float(max(1, hutchinson_samples)))
                    h.mul_(beta2).add_(hv_mean, alpha=(1.0 - beta2))
                else:
                    # Use squared gradients as a cheap Hessian proxy
                    h.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))

                # Denominator: scaled Hessian estimate with numerical floor
                denom = torch.maximum(
                    gamma * h, torch.tensor(eps, device=h.device, dtype=h.dtype)
                )
                # Normalized step
                step = m.div(denom)
                # Clip step magnitude and scale by lr
                if clip_val is not None and clip_val > 0.0:
                    step = torch.clamp(step, -clip_val, clip_val)
                p.data.add_(step, alpha=-lr)

        return loss

    def load_state_dict(self, state_dict):
        # Use default loader but ensure tensors map correctly to device
        super().load_state_dict(state_dict)













