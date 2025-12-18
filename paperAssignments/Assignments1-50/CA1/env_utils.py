"""Environment compatibility helpers for Gym and Gymnasium.

Provides unified reset_env and step_env functions that hide API differences between
`gym` and `gymnasium` (different return signatures for reset/step).
"""
from typing import Any, Tuple


def reset_env(env: Any, seed: int | None = None) -> Any:
    """Reset environment and return observation only.

    Works for both Gym (obs) and Gymnasium (obs, info) signatures.
    If seed is provided, attempts to pass it to env.reset.
    """
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        # older gym may not accept seed kw
        out = env.reset()

    if isinstance(out, tuple):
        # gymnasium returns (obs, info)
        return out[0]
    return out


def step_env(env: Any, action: Any) -> Tuple[Any, float, bool, dict]:
    """Step environment and return (obs, reward, done, info).

    Handles Gym step signatures:
      - (obs, reward, done, info)
      - (obs, reward, terminated, truncated, info) -> done = terminated or truncated
    """
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    elif len(out) == 4:
        obs, reward, done, info = out
        done = bool(done)
    else:
        # Unexpected signature: try to unpack safely
        try:
            obs = out[0]
            reward = float(out[1])
            done = bool(out[2])
            info = out[3] if len(out) > 3 else {}
        except Exception:
            raise RuntimeError(f"Unrecognized env.step return signature: {out}")

    return obs, reward, done, info
