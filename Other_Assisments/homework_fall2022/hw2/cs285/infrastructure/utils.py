import numpy as np
import time
import copy
def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]
    true_states = perform_actions(env, action_sequence)['observation']
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)

        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)
def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):

    if render and 'human' in render_mode:
        env.render(mode='human')

    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0

    while True:
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    inner_env = getattr(env, 'env', env)
                    if hasattr(inner_env, 'model') and 'track' in inner_env.model.camera_names:
                        image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode and hasattr(env, 'model') and hasattr(env.model, 'opt'):
                env.render(mode='human')
                time.sleep(env.model.opt.timestep)

        obs.append(ob)
        action = policy.get_action(ob)
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.squeeze(0)
        acs.append(action)
        ob, rew, done, _ = env.step(action)

        next_obs.append(ob)
        rewards.append(rew)
        steps += 1

        rollout_done = done or (steps >= max_path_length)
        terminals.append(rollout_done)
        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts using policy
        until we have collected min_timesteps_per_batch steps
    """
    timesteps_this_batch = 0
    paths = []

    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        path_length = get_pathlength(path)
        timesteps_this_batch += path_length
        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts using policy
    """
    paths = []
    for _ in range(ntraj):
        paths.append(sample_trajectory(env, policy, max_path_length, render, render_mode))

    return paths
def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}
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
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards
def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp)
    mean_data = np.mean(data, axis=0)
    mean_data[mean_data == 0] = 0.000001
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data