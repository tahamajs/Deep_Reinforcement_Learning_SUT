import numpy as np
import time
def sample_trajectory(
    env, policy, max_path_length, render=False, render_mode=("rgb_array")
):

    if render:
        env.render(mode="human")
    ob, _ = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:
            if "rgb_array" in render_mode:
                if hasattr(env, "sim"):
                    if "track" in env.env.model.camera_names:
                        image_obs.append(
                            env.sim.render(camera_name="track", height=500, width=500)[
                                ::-1
                            ]
                        )
                    else:
                        image_obs.append(env.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if "human" in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        obs.append(ob)
        ac = policy.get_action(ob)
        acs.append(ac)
        ob, rew, terminated, truncated, info = env.step(ac)
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)
        done = terminated or truncated
        rollout_done = done or (steps >= max_path_length)
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)
def sample_trajectories(
    env,
    policy,
    min_timesteps_per_batch,
    max_path_length,
    render=False,
    render_mode=("rgb_array"),
):

    timesteps_left = min_timesteps_per_batch
    timesteps_this_batch = 0
    paths = []

    while timesteps_this_batch < min_timesteps_per_batch:
        paths.append(
            sample_trajectory(env, policy, max_path_length, render, render_mode)
        )
        timesteps_this_batch += get_pathlength(paths[-1])

    return paths, timesteps_this_batch
def sample_n_trajectories(
    env, policy, ntraj, max_path_length, render=False, render_mode=("rgb_array")
):
    paths = []
    for n in range(ntraj):
        paths.append(
            sample_trajectory(env, policy, max_path_length, render, render_mode)
        )

    return paths
def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }
def convert_listofrollouts(paths):
    """
    Take a list of rollout dictionaries
    and return separate arrays,
    where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )
def get_pathlength(path):
    return len(path["reward"])