#!/usr/bin/env python3
"""
Complete implementation for RL_HW5_Dyna.ipynb
This script contains all the completed TODO sections
"""

import random
import numpy as np
import gymnasium as gym
from tqdm.notebook import trange
from heapq import heappush, heappop
from collections import defaultdict

# Environment exploration
def explore_environment():
    """Complete the environment exploration section"""
    slippery = True
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=slippery)
    
    print('Observations:', env.observation_space)
    print('Actions:', env.action_space)
    return env

# Random policy
def random_policy(*args):
    """Random policy implementation"""
    env = args[0] if args else gym.make("FrozenLake-v1")
    action = env.action_space.sample()
    return action

# Greedy policy
def greedy_policy(state: int, q_values: np.ndarray) -> int:
    """Greedy policy implementation"""
    action = np.argmax(q_values[state])
    return action

# Epsilon-greedy policy
def epsilon_greedy_policy(state: int, q_values: np.ndarray, epsilon: float) -> int:
    """Epsilon-greedy policy implementation"""
    if random.random() < epsilon:
        # Random action
        action = random.randint(0, q_values.shape[1] - 1)
    else:
        # Greedy action
        action = np.argmax(q_values[state])
    return action

# Q-Planning function
def q_planning(model: dict, q: np.ndarray, alpha: float, gamma: float, n: int) -> np.ndarray:
    """Perform n steps of planning using the learned model"""
    
    for _ in range(n):
        # Randomly sample a known state-action pair
        if not model:
            break
        state = random.choice(list(model.keys()))
        action = random.choice(list(model[state].keys()))
        
        # Get the predicted reward and next_state from the model
        reward, next_state = model[state][action]
        
        # Update Q-value using the deterministic transition
        td_error = reward + gamma * np.max(q[next_state]) - q[state, action]
        q[state, action] += alpha * td_error
    
    return q

# Dyna-Q algorithm
def dyna_q(n_episodes: int, env: gym.Env, epsilon: float, alpha: float,
           gamma: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Dyna-Q algorithm for deterministic environments."""
    
    reward_sums = np.zeros(n_episodes)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Dyna-Q model for deterministic environments
    model = defaultdict(dict)
    
    for episode_i in (pbar := trange(n_episodes, leave=False)):
        state, info = env.reset()
        reward_sum, terminal = 0, False
        
        while not terminal:
            # Take ε-greedy action
            action = epsilon_greedy_policy(state, q, epsilon)
            
            # Take action and observe
            next_state, reward, terminated, truncated, info = env.step(action)
            terminal = terminated or truncated
            
            # Q-learning update
            td_error = reward + gamma * np.max(q[next_state]) - q[state, action]
            q[state, action] += alpha * td_error
            
            # Update deterministic model
            model[state][action] = (reward, next_state)
            
            # Planning step(s)
            q = q_planning(model, q, alpha, gamma, n)
            
            # Move to next state
            state = next_state
            
            # Update reward sum
            reward_sum += reward
        
        pbar.set_description(f'Episode Reward {int(reward_sum)}')
        reward_sums[episode_i] = reward_sum
    
    return q, reward_sums

# Custom reward function
class CustomFrozenLakeEnv(gym.Env):
    def __init__(self, map_name="8x8", is_slippery=False):
        self.env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Modify the reward calculation
        reward = self.custom_reward_function(obs, reward, terminated)
        return obs, reward, terminated, truncated, info
    
    def custom_reward_function(self, observation, reward, done):
        """Custom reward function with distance-based shaping"""
        if done and reward > 0:
            # Goal reached
            return 1.0
        elif done and reward == 0:
            # Fell in hole
            return -0.1
        else:
            # Small negative reward for each step to encourage efficiency
            return -0.01
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    def close(self):
        return self.env.close()

# Prioritized Sweeping Planning
def q_planning_priority(model: dict, q: np.ndarray, priorities: list, alpha: float, gamma: float, n: int) -> np.ndarray:
    """Performs planning updates using Prioritized Sweeping."""
    
    for _ in range(n):
        if not priorities:
            break
            
        # Sample the state-action pair with the highest priority
        priority, state, action = heappop(priorities)
        priority = -priority  # Convert back from negative (heap is min-heap)
        
        # Retrieve deterministic transition
        if state not in model or action not in model[state]:
            continue
        reward, next_state = model[state][action]
        
        # Update Q-value using the deterministic transition
        td_error = reward + gamma * np.max(q[next_state]) - q[state, action]
        q[state, action] += alpha * td_error
        
        # Update priorities for predecessors
        for pred_state in model:
            for pred_action in model[pred_state]:
                pred_reward, pred_next_state = model[pred_state][pred_action]
                if pred_next_state == state:
                    pred_td_error = pred_reward + gamma * np.max(q[state]) - q[pred_state, pred_action]
                    if abs(pred_td_error) > 0.01:  # Threshold for adding to priority queue
                        heappush(priorities, (-abs(pred_td_error), pred_state, pred_action))
    
    return q

# Dyna-Q with Prioritized Sweeping
def dyna_q_priority(n_episodes: int, env: gym.Env, epsilon: float, alpha: float,
                    gamma: float, n: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Dyna-Q with Prioritized Sweeping algorithm for deterministic environments."""
    
    reward_sums = np.zeros(n_episodes)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Dyna-Q model for deterministic environments
    model = defaultdict(dict)
    
    # Priority queue for prioritized sweeping
    priorities = []
    
    for episode_i in (pbar := trange(n_episodes, leave=False)):
        state, info = env.reset()
        reward_sum, terminal = 0, False
        
        while not terminal:
            # Take ε-greedy action
            action = epsilon_greedy_policy(state, q, epsilon)
            
            # Take action and observe
            next_state, reward, terminated, truncated, info = env.step(action)
            terminal = terminated or truncated
            
            # Q-learning update
            td_error = reward + gamma * np.max(q[next_state]) - q[state, action]
            q[state, action] += alpha * td_error
            
            # Update deterministic model
            model[state][action] = (reward, next_state)
            
            # Update priority queue if the TD error is significant
            if abs(td_error) > theta:
                heappush(priorities, (-abs(td_error), state, action))
            
            # Planning step with prioritized sweeping
            q = q_planning_priority(model, q, priorities, alpha, gamma, n)
            
            # Move to next state
            state = next_state
            
            # Update reward sum
            reward_sum += reward
        
        pbar.set_description(f'Episode Reward {int(reward_sum)}')
        reward_sums[episode_i] = reward_sum
    
    return q, reward_sums

# Example usage and experiments
def run_experiments():
    """Run example experiments with different configurations"""
    
    # Basic Dyna-Q experiment
    print("Running basic Dyna-Q experiment...")
    np.random.seed(2025)
    
    params = {'epsilon': 0.1,    # epsilon-greedy policy
              'alpha': 0.1,      # learning rate
              'gamma': 0.95,     # temporal discount factor
              'n': 5,           # number of planning steps
    }
    
    n_episodes = 1000
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    
    # Solve Frozen Lake using Dyna-Q
    value_dyna_q, reward_sums_dyna_q = dyna_q(n_episodes, env, **params)
    
    print(f"Final Q-values shape: {value_dyna_q.shape}")
    print(f"Average reward over last 100 episodes: {np.mean(reward_sums_dyna_q[-100:])}")
    
    # Reward shaping experiment
    print("\nRunning reward shaping experiment...")
    np.random.seed(2025)
    
    params_shaped = {'epsilon': 0.1,
                     'alpha': 0.1,
                     'gamma': 0.95,
                     'n': 5,
    }
    
    env_shaped = CustomFrozenLakeEnv(map_name="8x8", is_slippery=False)
    value_shaped, reward_sums_shaped = dyna_q(n_episodes, env_shaped, **params_shaped)
    
    print(f"Shaped reward - Average over last 100 episodes: {np.mean(reward_sums_shaped[-100:])}")
    
    # Prioritized sweeping experiment
    print("\nRunning prioritized sweeping experiment...")
    np.random.seed(2025)
    
    params_priority = {'epsilon': 0.1,
                       'alpha': 0.1,
                       'gamma': 0.95,
                       'n': 5,
                       'theta': 0.01  # prioritization threshold
    }
    
    env_priority = CustomFrozenLakeEnv(map_name="8x8", is_slippery=False)
    value_priority, reward_sums_priority = dyna_q_priority(n_episodes, env_priority, **params_priority)
    
    print(f"Prioritized sweeping - Average over last 100 episodes: {np.mean(reward_sums_priority[-100:])}")

if __name__ == "__main__":
    run_experiments()

