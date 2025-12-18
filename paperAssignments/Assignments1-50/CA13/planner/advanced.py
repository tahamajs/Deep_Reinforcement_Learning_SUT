from __future__ import annotations
from typing import Any, Callable, List, Tuple
import torch

"""
Advanced SimGolf latent planner with optional CEM and uncertainty-penalized returns.

This module implements `simulate_branches_advanced` which mirrors the simple
simulator but adds:
 - cfg.cem.{enabled,pop,elite,iters} for CEM-based action sequence search
 - cfg.kappa for uncertainty penalty on predicted rewards (uses out['sigma_r'] if provided)
 - returns Branch objects compatible with simgolf_latent.Branch
"""


class Branch:
    def __init__(
        self, ret: float, traj: List[Tuple[torch.Tensor, torch.Tensor, float, float]]
    ):
        self.ret = float(ret)
        self.traj = traj


def _actor_action(actor: Any, z: torch.Tensor) -> torch.Tensor:
    for name in ("sample", "act", "forward", "__call__"):
        fn = getattr(actor, name, None)
        if callable(fn):
            out = fn(z)
            if isinstance(out, tuple):
                out = out[0]
            return out
    raise AttributeError("Actor missing callable sample/act/forward")


@torch.no_grad()
def simulate_branches_advanced(
    rssm: Any,
    actor: Any,
    value_fn: Callable[[torch.Tensor], torch.Tensor],
    z_saved: torch.Tensor,
    cfg: Any,
) -> List[Branch]:
    device = z_saved.device
    B = int(getattr(cfg, "B", getattr(cfg, "branching_factor", 8)))
    H = int(getattr(cfg, "H", getattr(cfg, "horizon", 12)))
    kappa = float(getattr(cfg, "kappa", 0.0))

    def _run_cem(z0: torch.Tensor):
        pop = int(getattr(cfg.cem, "pop", 64))
        elite = int(getattr(cfg.cem, "elite", max(4, pop // 8)))
        iters = int(getattr(cfg.cem, "iters", 3))
        with torch.no_grad():
            a0 = _actor_action(actor, z0)
        action_dim = a0.view(-1).shape[0] if isinstance(a0, torch.Tensor) else 1
        mean_seq = torch.stack(
            [_actor_action(actor, z0).view(-1) for _ in range(H)], dim=0
        ).to(device)
        std_seq = torch.ones_like(mean_seq)
        for _ in range(iters):
            eps = torch.randn((pop, H, action_dim), device=device)
            samples = mean_seq.unsqueeze(0) + std_seq.unsqueeze(0) * eps
            returns = torch.zeros(pop, device=device)
            for p in range(pop):
                z = z0.clone()
                disc = 1.0
                ret_p = 0.0
                for h in range(H):
                    a = samples[p, h].view(1, -1)
                    step_fn = None
                    for name in ("step", "imagine_step", "transition", "predict"):
                        if hasattr(rssm, name):
                            step_fn = getattr(rssm, name)
                            break
                    if step_fn is None:
                        if callable(rssm):
                            step_fn = rssm
                        else:
                            break
                    out = step_fn(z, a)
                    if isinstance(out, tuple) and len(out) >= 3:
                        z, r, gamma = out[0], float(out[1]), float(out[2])
                    elif isinstance(out, dict):
                        z, r, gamma = (
                            out["z"],
                            float(out["r"]),
                            float(out.get("gamma", 1.0)),
                        )
                    else:
                        z, r = out
                        gamma = 1.0
                    ret_p = ret_p + disc * float(r)
                    disc = disc * float(gamma)
                    if float(gamma) == 0.0:
                        break
                returns[p] = ret_p
            topk = torch.topk(returns, elite, largest=True)
            elites = samples[topk.indices]
            mean_seq = elites.mean(dim=0)
            std_seq = elites.std(dim=0) + 1e-6
        return [mean_seq[h].view(1, -1).detach() for h in range(H)]

    branches = []
    for _ in range(B):
        z = z_saved.clone().to(device)
        traj = []
        ret = 0.0
        disc = 1.0
        cem_actions = None
        if getattr(cfg, "cem", None) and getattr(cfg.cem, "enabled", False):
            try:
                cem_actions = _run_cem(z)
            except Exception:
                cem_actions = None
        for h in range(H):
            a = cem_actions[h] if cem_actions is not None else _actor_action(actor, z)
            if isinstance(a, torch.Tensor):
                a = a.to(device)
            step_fn = None
            for name in ("step", "imagine_step", "transition", "predict"):
                if hasattr(rssm, name):
                    step_fn = getattr(rssm, name)
                    break
            if step_fn is None:
                if callable(rssm):
                    step_fn = rssm
                else:
                    raise AttributeError("rssm missing step/imagine_step")
            out = step_fn(z, a)
            if isinstance(out, tuple) and len(out) >= 3:
                z, r, gamma = out[0], float(out[1]), float(out[2])
                sigma_r = 0.0
            elif isinstance(out, dict):
                z, r, gamma = out["z"], float(out["r"]), float(out.get("gamma", 1.0))
                sigma_r = float(out.get("sigma_r", 0.0))
            else:
                z, r = out
                gamma = 1.0
                sigma_r = 0.0
            r_pen = float(r) - kappa * float(sigma_r)
            ret = ret + disc * r_pen
            disc = disc * float(gamma)
            traj.append(
                (
                    z.detach().clone(),
                    a.detach().clone() if isinstance(a, torch.Tensor) else a,
                    float(r),
                    float(gamma),
                )
            )
            if float(gamma) == 0.0:
                break
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












