from __future__ import annotations

from typing import Dict, Any

import torch

from .optim.sophia import Sophia


def build_optimizer(
    model: torch.nn.Module, optim_name: str = "sophia", lr: float = 3e-4, **kwargs
) -> torch.optim.Optimizer:
    """
    Simple builder that demonstrates integrating Sophia into an agent.

    Per-parameter options example:
      - set different hessian estimator for biases or batchnorm params
    """
    if optim_name.lower() == "sophia":
        # per-parameter customization: disable Hessian estimator for bias/BN params (cheap)
        param_groups = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                name.endswith(".bias")
                or "bn" in name.lower()
                or "batchnorm" in name.lower()
            ):
                param_groups.append(
                    {"params": [p], "hessian_estimator": "squared_grad", "beta1": 0.9}
                )
            else:
                param_groups.append({"params": [p]})

        # global defaults can be overridden via kwargs
        defaults = dict(lr=lr)
        defaults.update(kwargs)
        # instantiate Sophia with param groups
        opt = Sophia(param_groups, **defaults)
        return opt
    else:
        # fallback to Adam
        return torch.optim.Adam(model.parameters(), lr=lr)


def example_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    use_amp: bool = True,
):
    """
    Example single training step showing AMP-friendly forward/backward,
    while ensuring Hessian accumulation uses FP32 to avoid precision issues.
    If optimizer is Sophia and uses Hutchinson, pass closure to optimizer.step().
    """
    scaler = (
        torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None
    )

    def closure():
        # compute loss with autocast if available
        if use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                out = model(batch["obs"])
                loss = torch.nn.functional.mse_loss(out, batch["target"])
        else:
            out = model(batch["obs"])
            loss = torch.nn.functional.mse_loss(out, batch["target"])
        # ensure grads are computed with create_graph when necessary is handled by optimizer
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward(
                create_graph=getattr(optimizer, "hessian_estimator", None)
                == "hutchinson"
            )
        else:
            loss.backward(
                create_graph=getattr(optimizer, "hessian_estimator", None)
                == "hutchinson"
            )
        return loss

    # If the optimizer requires closure (Hutchinson), provide it to step()
    if hasattr(optimizer, "param_groups") and any(
        g.get("hessian_estimator", "squared_grad") == "hutchinson"
        for g in optimizer.param_groups
    ):
        loss = closure()
        # optimizer.step will expect closure to be callable; wrap to recompute loss if needed
        optimizer.step(lambda: loss)
    else:
        loss = closure()
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    return loss











