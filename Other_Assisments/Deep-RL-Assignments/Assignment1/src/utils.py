import numpy as np
import gymnasium as gym
def action_to_one_hot(env, action):
    action_vec = np.zeros(env.action_space.n)
    action_vec[action] = 1
    return action_vec
def generate_episode(env, policy):
    """Collects one rollout from the policy in an environment."""
    done = False
    state, info = env.reset()
    states = [state]
    actions = []
    rewards = []
    while not done:
        action = policy.predict(state.reshape(1, -1)).argmax()
        actions.append(action_to_one_hot(env, action))
        next_state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        if not done:
            states.append(next_state)
        state = next_state
    return np.array(states), np.array(actions), np.array(rewards)
def generate_dagger_episode(env, policy, expert_policy):
    """Collects one rollout for DAgger."""
    done = False
    state, info = env.reset()
    states = [state]
    actions = []
    rewards = []
    while not done:
        action = policy.predict(state.reshape(1, -1)).argmax()
        expert_action = expert_policy.predict(state.reshape(1, -1)).argmax()
        actions.append(action_to_one_hot(env, expert_action))
        next_state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        if not done:
            states.append(next_state)
        state = next_state
    return np.array(states), np.array(actions), np.array(rewards)
def generate_GAIL_episode(env, policy, discriminator=None):
    """Collects one rollout for GAIL."""
    done = False
    state, info = env.reset()
    states = [state]
    actions = []
    rewards = []
    while not done:
        action = policy.predict(state.reshape(1, -1)).argmax()
        actions.append(action_to_one_hot(env, action))
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if discriminator is None:
            rewards.append(reward)
        else:
            rewards.append(discriminator(state, action))
        if not done:
            states.append(next_state)
        state = next_state
    env.close()
    return np.array(states), np.array(actions), np.array(rewards)
def get_x_position_histogram(states):
    x_vec = [s[0] for s in states]
    bins = np.linspace(-2.4, 2.4, 11)
    hist, _ = np.histogram(x_vec, bins=bins, density=True)
    return hist
def TV_distance(expert_states, student_states):
    expert_hist = get_x_position_histogram(expert_states)
    student_hist = get_x_position_histogram(student_states)
    return 0.5 * np.sum(np.abs(expert_hist - student_hist))