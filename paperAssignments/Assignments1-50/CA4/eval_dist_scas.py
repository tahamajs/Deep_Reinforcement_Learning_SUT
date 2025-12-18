import numpy as np
from typing import Callable, Dict


def evaluate(policy: Callable, env, episodes: int = 5) -> Dict[str, float]:
    """
    Minimal evaluation loop. `policy` should expose an `act_eval(state)` method.
    This function is intentionally lightweight and import-safe.
    """
    scores = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            a = policy.act_eval(s)
            s, r, done, info = env.step(a)
            ep_reward += float(r)
        scores.append(ep_reward)
    return {"score": float(np.mean(scores)), "score_std": float(np.std(scores))}










