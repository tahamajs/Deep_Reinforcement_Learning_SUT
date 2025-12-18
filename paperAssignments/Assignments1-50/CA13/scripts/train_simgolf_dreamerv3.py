#!/usr/bin/env python3
"""
Training script scaffold for DreamerV3 + SimGolf latent planner integration.

This script is intentionally framework-agnostic: it expects the user to supply
their DreamerV3 model objects that implement the minimal interfaces described
in the README. The script demonstrates how to hook the planner into a training
loop and perform planner-triggered imagined updates.
"""
from __future__ import annotations
import argparse
import yaml
import time
from pathlib import Path
from typing import Any
import torch

from planner import CheckpointBuffer, simulate_branches, should_trigger, TriggerConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    p.add_argument("--workdir", type=str, default="./runs/simgolf", help="output dir")
    return p.parse_args()


def load_config(path: str) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class DummyWorldModel:
    """Minimal RSSM-like stub implementing imagine step for testing planner integration."""

    def __init__(self, z_dim=32):
        self.z_dim = z_dim

    def step(self, z, a):
        # deterministic linear transition for stub testing
        z_next = z + 0.01 * (
            a.reshape(z.shape) if isinstance(a, torch.Tensor) else torch.zeros_like(z)
        )
        r = 0.0
        gamma = 1.0
        return z_next, r, gamma


class DummyActor:
    def __init__(self, action_dim=1):
        self.action_dim = action_dim

    def sample(self, z):
        # simple random actions for stub testing
        return torch.randn((z.shape[0], self.action_dim), device=z.device) * 0.1


class DummyValue:
    def __call__(self, z):
        return torch.zeros(1, device=z.device)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    planner_cfg = argparse.Namespace(**cfg.get("planner", {}))
    trig_cfg = TriggerConfig(
        cooldown=int(planner_cfg.cooldown),
        trigger_td=float(planner_cfg.trigger_td),
        trigger_unc=float(planner_cfg.trigger_unc),
        trigger_ent_low=float(planner_cfg.get("trigger_entropy_low", 0.3)),
        trigger_ent_high=float(planner_cfg.get("trigger_entropy_high", 2.0)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer = CheckpointBuffer(
        capacity=int(planner_cfg.buffer_size or 1024), device=device
    )

    # These would normally be loaded/trained DreamerV3 components
    rssm = DummyWorldModel(z_dim=32)
    actor = DummyActor(action_dim=1)
    value_fn = DummyValue()

    last_trigger = None
    step = 0
    total_steps = int(cfg.get("training", {}).get("total_steps", 200000))
    z = torch.zeros((1, 32), device=device)

    print(
        f"Starting training scaffold for {total_steps} steps. Planner enabled={planner_cfg.enabled}"
    )
    start_time = time.time()
    while step < total_steps:
        # Simulate environment interaction (user should replace with real env loop)
        # Here we just step the dummy world and randomly push checkpoints based on dummy criteria
        a = actor.sample(z)
        z_next, r, gamma = rssm.step(z, a)
        # compute dummy TD error / entropy / unc (user should compute from real models)
        td_err = float(torch.rand(1).item())
        unc = float(torch.rand(1).item()) * 0.1
        entropy = float(torch.rand(1).item()) * 1.0

        # maybe save checkpoint if td_err high
        if td_err > 0.8:
            score = td_err
            buffer.push(z.detach().clone(), score=score, step=step)

        # check trigger
        if planner_cfg.enabled and should_trigger(
            td_err, unc, entropy, trig_cfg, last_trigger, step
        ):
            last_trigger = step
            samples = buffer.sample(k=1, prioritized=True) if len(buffer) > 0 else []
            if samples:
                z_saved = samples[0]["z"]
                branches = simulate_branches(
                    rssm, actor, value_fn, z_saved, planner_cfg
                )
                # user should update actor/critic from branches here
                top_return = branches[0].ret if branches else 0.0
                print(
                    f"[step {step}] Planner ran: branches={len(branches)}, top_return={top_return:.3f}"
                )
        # advance
        z = z_next
        step += 1
        if step % 10000 == 0:
            elapsed = time.time() - start_time
            print(
                f"Step {step}/{total_steps} elapsed {elapsed:.1f}s buffer_size={len(buffer)}"
            )

    print("Training scaffold finished.")


if __name__ == "__main__":
    main()












