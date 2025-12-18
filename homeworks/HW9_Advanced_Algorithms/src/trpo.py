from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .config import CFG


def flat_params(parameters):
    return torch.cat([p.view(-1) for p in parameters])


def set_flat_params(parameters, flat):
    offset = 0
    for p in parameters:
        num = p.numel()
        p.data.copy_(flat[offset:offset+num].view_as(p))
        offset += num


class TRPOAgent:
    def __init__(self, policy: nn.Module, value_fn: nn.Module, config: CFG.__class__ = CFG):
        self.policy = policy
        self.value_fn = value_fn
        self.max_kl = config.trpo_max_kl
        self.damping = config.trpo_damping
        self.cg_iters = config.trpo_cg_iters
        self.backtrack_iters = config.trpo_backtrack_iters
        self.backtrack_coeff = config.trpo_backtrack_coeff
        self.value_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=1e-3)

    def surrogate_loss(self, states, actions, advantages, old_log_probs):
        probs = self.policy(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return (ratio * advantages).mean()

    def kl_divergence(self, states, old_probs):
        new_probs = self.policy(states)
        kl = (old_probs * (torch.log(old_probs + 1e-8) - torch.log(new_probs + 1e-8))).sum(dim=-1).mean()
        return kl

    def fisher_vector_product(self, states, vector, old_probs):
        kl = self.kl_divergence(states, old_probs)
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grads = torch.cat([g.view(-1) for g in grads])
        kl_v = (flat_grads * vector).sum()
        fvp = torch.autograd.grad(kl_v, self.policy.parameters())
        fvp_flat = torch.cat([g.contiguous().view(-1) for g in fvp])
        return fvp_flat + self.damping * vector

    def conjugate_gradient(self, fvp_fn: Callable, b: torch.Tensor, iters: int = None):
        if iters is None:
            iters = self.cg_iters
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(iters):
            Ap = fvp_fn(p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def update(self, states, actions, rewards, dones, advantages, old_log_probs, old_probs):
        # compute gradient of surrogate
        loss = self.surrogate_loss(states, actions, advantages, old_log_probs)
        grads = torch.autograd.grad(loss, self.policy.parameters())
        g = torch.cat([g.view(-1) for g in grads]).detach()

        def fvp(v):
            return self.fisher_vector_product(states, v, old_probs)

        step_dir = self.conjugate_gradient(fvp, g)
        shs = 0.5 * torch.dot(step_dir, fvp(step_dir))
        lm = torch.sqrt((2 * self.max_kl) / (shs + 1e-8))
        full_step = lm * step_dir
        expected_improve = torch.dot(g, full_step)

        old_params = flat_params(self.policy.parameters())

        success = False
        for i in range(self.backtrack_iters):
            step_frac = self.backtrack_coeff ** i
            new_params = old_params + step_frac * full_step
            set_flat_params(self.policy.parameters(), new_params)
            new_loss = self.surrogate_loss(states, actions, advantages, old_log_probs)
            kl = self.kl_divergence(states, old_probs)
            actual_improve = new_loss - loss
            expected_improve_frac = expected_improve * step_frac
            improve_ratio = (actual_improve / (expected_improve_frac + 1e-8)).item()
            if improve_ratio > 0.1 and kl <= self.max_kl:
                success = True
                break

        if not success:
            set_flat_params(self.policy.parameters(), old_params)

        # fit value function
        returns = advantages + self.value_fn(states).squeeze().detach()
        for _ in range(5):
            value_loss = ((self.value_fn(states).squeeze() - returns) ** 2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        return {"surrogate_loss": loss.item(), "kl": self.kl_divergence(states, old_probs).item(), "line_search_success": success}


