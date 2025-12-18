from typing import Any, Dict, List, Tuple
import gym
import torch


def collect_episode(
    env: gym.Env, policy, device: str = "cpu", max_steps: int = 1000
) -> Dict[str, Any]:
    """Run a single episode using the provided policy.

    Args:
        env: an OpenAI Gym environment.
        policy: a policy object with method `get_action(obs_tensor)` that
                returns (action, log_prob).
        device: device string for tensors
        max_steps: maximum number of steps to run the episode

    Returns:
        A dict containing lists: observations, actions, rewards, log_probs, dones
    """
    reset_out = env.reset()
    # gymnasium returns (obs, info); older gym returns obs
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    observations = []
    actions = []
    rewards = []
    log_probs = []
    dones = []

    for t in range(max_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action, logp = policy.get_action(obs_tensor)
        # convert action to pythonic type if tensor
        if isinstance(action, torch.Tensor):
            action_to_env = int(action.item())
            actions.append(action_to_env)
        else:
            action_to_env = int(action)
            actions.append(action_to_env)
        observations.append(obs)
        log_probs.append(float(logp.detach().cpu().item()))
        step_out = env.step(action_to_env)
        # gymnasium: (obs, reward, terminated, truncated, info)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            # gym older API: (obs, reward, done, info)
            next_obs, reward, done, info = step_out
        rewards.append(float(reward))
        dones.append(bool(done))
        obs = next_obs
        if done:
            break

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs,
        "dones": dones,
    }


def collect_n_episodes(
    env: gym.Env, policy, n: int, device: str = "cpu", max_steps: int = 1000
) -> List[Dict[str, Any]]:
    """Collect n episodes and return a list of episode dicts."""
    episodes = []
    for _ in range(n):
        episodes.append(
            collect_episode(env, policy, device=device, max_steps=max_steps)
        )
    return episodes


