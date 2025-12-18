from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def collect_episode(env, policy, device="cpu", render: bool = False) -> Dict[str, List]:
    """Run a single episode with the provided env and policy.

    Args:
        env: an OpenAI Gym / Gymnasium environment
        policy: a policy with an `act(obs_tensor)` method returning actions
        device: device string for torch tensors
        render: whether to render the environment (disabled by default)
    Returns:
        A dict with lists: observations, actions, rewards, dones, log_probs
    """
    obs = env.reset()
    # gymnasium returns tuple (obs, info) for newer versions
    if isinstance(obs, tuple):
        obs = obs[0]
    observations, actions, rewards, dones, log_probs = [], [], [], [], []
    done = False
    while not done:
        obs_tensor = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device)
        dist = policy.get_action_dist(obs_tensor)
        action = dist.sample().cpu().numpy()
        lp = dist.log_prob(torch.tensor(action)).detach().cpu().numpy()
        step = env.step(int(action))
        # gym returns different shapes depending on version
        if len(step) == 5:
            next_obs, reward, terminated, truncated, info = step
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, info = step
        observations.append(obs)
        actions.append(int(action))
        rewards.append(float(reward))
        dones.append(bool(done))
        log_probs.append(float(lp))
        obs = next_obs
        if render:
            env.render()
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "log_probs": log_probs,
    }












