import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import deeprl_hw2q2.lake_envs as lake_env
def display_policy_letters(env, policy):
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
def value_func_heatmap(env, value_func):
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
    plt.show()
    return None