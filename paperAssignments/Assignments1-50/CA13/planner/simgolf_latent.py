from __future__ import annotations
from typing import Any, Callable, List, Tuple
import torch


class Branch:
    """Simple container for a simulated branch."""

    def __init__(
        self, ret: float, traj: List[Tuple[torch.Tensor, torch.Tensor, float, float]]
    ):
        self.ret = float(ret)
        self.traj = traj

    def __lt__(self, other: "Branch"):
        return self.ret < other.ret


def _actor_action(actor: Any, z: torch.Tensor) -> torch.Tensor:
    """
    Generic wrapper to get an action from the actor.
    Tries common method names and shapes output to a 1D action tensor.
    """
    # Prefer a sampling method if available
    for name in ("sample", "act", "forward", "__call__"):
        fn = getattr(actor, name, None)
        if callable(fn):
            out = fn(z)
            # If actor returns tuple (action, ...), take first
            if isinstance(out, tuple):
                out = out[0]
            return out
    raise AttributeError(
        "Actor does not expose a callable action method (sample/act/forward)."
    )


@torch.no_grad()
def simulate_branches(
    rssm: Any,
    actor: Any,
    value_fn: Callable[[torch.Tensor], torch.Tensor],
    z_saved: torch.Tensor,
    cfg: Any,
) -> List[Branch]:
    """
    Simulate B branches from latent checkpoint z_saved using provided rssm/actor/value.

    Expected minimal interfaces:
      - actor.sample(z) or actor(z) -> action tensor
      - rssm.step(z, a) or rssm.imagine_step(z, a) -> (z_next, reward, gamma)
      - value_fn(z) -> scalar tensor value

    Returns a list of Branch objects sorted by descending return.
    """
    branches: List[Branch] = []
    device = z_saved.device
    B = int(getattr(cfg, "B", getattr(cfg, "branching_factor", 8)))
    H = int(getattr(cfg, "H", getattr(cfg, "horizon", 12)))
    for b in range(B):
        z = z_saved.clone().to(device)
        ret = 0.0
        disc = 1.0
        traj = []
        for h in range(H):
            a = _actor_action(actor, z)
            # ensure action on same device
            if isinstance(a, torch.Tensor):
                a = a.to(device)
            # rssm step: try several method names
            step_fn = None
            for name in ("step", "imagine_step", "transition", "predict"):
                if hasattr(rssm, name):
                    step_fn = getattr(rssm, name)
                    break
            if step_fn is None:
                # expect rssm to be a callable taking (z,a)
                if callable(rssm):
                    step_fn = rssm
                else:
                    raise AttributeError(
                        "rssm has no step/imagine_step/predict method and is not callable."
                    )

            out = step_fn(z, a)
            # Accept (z_next, r, gamma) or dict-like
            if isinstance(out, tuple) and len(out) >= 3:
                z, r, gamma = out[0], float(out[1]), float(out[2])
            elif isinstance(out, dict):
                z, r, gamma = out["z"], float(out["r"]), float(out.get("gamma", 1.0))
            else:
                # fallback: assume (z_next, r)
                z, r = out
                gamma = 1.0
            # accumulate return with potential penalization handled externally
            ret = ret + disc * float(r)
            disc = disc * float(gamma)
            traj.append(
                (
                    z.detach().clone(),
                    a.detach().clone() if isinstance(a, torch.Tensor) else a,
                    float(r),
                    float(gamma),
                )
            )
            # if gamma is zero, break early
            if float(gamma) == 0.0:
                break
        # bootstrap with value
        try:
            v = value_fn(z)
            if isinstance(v, torch.Tensor):
                v = float(v.detach().cpu())
        except Exception:
            v = 0.0
        ret = ret + disc * v
        branches.append(Branch(ret, traj))
    branches.sort(key=lambda br: br.ret, reverse=True)
    return branches
