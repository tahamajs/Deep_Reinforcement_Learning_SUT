import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import the helper package used by the assignment. If it's not on sys.path
# (e.g., running from the src/ folder), add the Q2-VI-PI folder to sys.path as a fallback.
try:
    import deeprl_hw2q2.lake_envs as lake_env
except ModuleNotFoundError:
    pkg_root = (
        pathlib.Path(__file__).resolve().parents[1] / "hw2-VI-PI-DQN" / "Q2-VI-PI"
    )
    if pkg_root.exists():
        sys.path.insert(0, str(pkg_root))
        import deeprl_hw2q2.lake_envs as lake_env
    else:
        # Re-raise with helpful message
        raise


def display_policy_letters(env, policy, save_path=None):
    """Displays a policy as letters, as required by problem 2.2 & 2.6

    Parameters
    ----------
    env: gym.core.Environment
    policy: np.ndarray, with shape (env.nS)
    """
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_env.action_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)

    for row in range(env.nrow):
        print("".join(policy_letters[row, :]))

    # Optionally save a simple image of the policy letters
    if save_path is not None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(env.ncol, env.nrow))
        ax.axis("off")
        table = ax.table(
            cellText=policy_letters.tolist(),
            loc="center",
            cellLoc="center",
        )
        table.scale(1, 1)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close(fig)


def value_func_heatmap(env, value_func, save_path=None):
    """Visualize a policy as a heatmap, as required by problem 2.3 & 2.5

    Note that you might need:
        import matplotlib.pyplot as plt
        import seaborn as sns

    Parameters
    ----------
    env: gym.core.Environment
    value_func: np.ndarray, with shape (env.nS)
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        np.reshape(value_func, [env.nrow, env.ncol]),
        annot=False,
        linewidths=0.5,
        cmap="YlGnBu",
        ax=ax,
        yticklabels=np.arange(1, env.nrow + 1)[::-1],
        xticklabels=np.arange(1, env.nrow + 1),
    )
    # Save image if requested (default pictures/ inside assignment folder)
    if save_path is None:
        pics_dir = pathlib.Path(__file__).resolve().parents[1] / "pictures"
        pics_dir.mkdir(exist_ok=True)
        save_path = str(pics_dir / f"value_func_heatmap_{env.nrow}x{env.ncol}.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return save_path
