#!/usr/bin/env python3
"""
Script to complete the RL_HW5_Dyna.ipynb notebook
"""

import json
import re

def complete_notebook():
    # Read the original notebook
    with open('/Users/tahamajs/Documents/uni/DRL/docs/homeworks/HW5_Model_Based/code/RL_HW5_Dyna.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Complete each cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            # Environment exploration
            if 'print(\'Observations:\', ...)' in source:
                cell['source'] = [
                    "slippery = True\n",
                    "env = gym.make(\"FrozenLake-v1\", render_mode=\"rgb_array\", is_slippery=slippery)\n",
                    "\n",
                    "# TODO: Print the observation space and action space\n",
                    "print('Observations:', env.observation_space)\n",
                    "print('Actions:', env.action_space)"
                ]
            
            # Random policy
            elif 'def random_policy(*args):' in source and 'action = ...' in source:
                cell['source'] = [
                    "def random_policy(*args):\n",
                    "    # TODO: Select a random action\n",
                    "    action = env.action_space.sample()\n",
                    "    return action"
                ]
            
            # Greedy policy
            elif 'def greedy_policy(state: int, q_values: np.ndarray) -> int:' in source and 'action = ...' in source:
                cell['source'] = [
                    "def greedy_policy(state: int, q_values: np.ndarray) -> int:\n",
                    "    # TODO: Select a greedy action\n",
                    "    action = np.argmax(q_values[state])\n",
                    "    return action"
                ]
            
            # Epsilon-greedy policy
            elif 'def epsilon_greedy_policy(state: int, q_values: np.ndarray, epsilon: float) -> int:' in source and 'action = ...' in source:
                cell['source'] = [
                    "def epsilon_greedy_policy(state: int, q_values: np.ndarray, epsilon: float) -> int:\n",
                    "    # TODO: Select an epsilon-greedy action\n",
                    "    if random.random() < epsilon:\n",
                    "        action = random.randint(0, q_values.shape[1] - 1)\n",
                    "    else:\n",
                    "        action = np.argmax(q_values[state])\n",
                    "    return action"
                ]
            
            # Q-planning function
            elif 'def q_planning(model: dict, q: np.ndarray, alpha: float, gamma: float, n: int) -> np.ndarray:' in source:
                cell['source'] = [
                    "def q_planning(model: dict, q: np.ndarray, alpha: float, gamma: float, n: int) -> np.ndarray:\n",
                    "\n",
                    "    for _ in range(n):\n",
                    "        # Randomly sample a known state-action pair\n",
                    "        if not model:\n",
                    "            break\n",
                    "        state = random.choice(list(model.keys()))\n",
                    "        action = random.choice(list(model[state].keys()))\n",
                    "        \n",
                    "        # Get the predicted reward and next_state from the model\n",
                    "        reward, next_state = model[state][action]\n",
                    "        \n",
                    "        # Update Q-value using the deterministic transition\n",
                    "        td_error = reward + gamma * np.max(q[next_state]) - q[state, action]\n",
                    "        q[state, action] += alpha * td_error\n",
                    "\n",
                    "    return q"
                ]
            
            # Dyna-Q function
            elif 'def dyna_q(n_episodes: int, env: gym.Env, epsilon: float, alpha: float,' in source:
                cell['source'] = [
                    "def dyna_q(n_episodes: int, env: gym.Env, epsilon: float, alpha: float,\n",
                    "           gamma: float, n: int) -> tuple[np.ndarray, np.ndarray]:\n",
                    "    \"\"\"Dyna-Q algorithm for deterministic environments.\"\"\"\n",
                    "\n",
                    "    reward_sums = np.zeros(n_episodes)\n",
                    "    q = np.zeros((env.observation_space.n, env.action_space.n))\n",
                    "\n",
                    "    # Dyna-Q model for deterministic environments\n",
                    "    model = defaultdict(dict)\n",
                    "\n",
                    "    for episode_i in (pbar := trange(n_episodes, leave=False)):\n",
                    "        state, info = env.reset()\n",
                    "        reward_sum, terminal = 0, False\n",
                    "\n",
                    "        while not terminal:\n",
                    "            # Take ε-greedy action\n",
                    "            action = epsilon_greedy_policy(state, q, epsilon)\n",
                    "            \n",
                    "            # Take action and observe\n",
                    "            next_state, reward, terminated, truncated, info = env.step(action)\n",
                    "            terminal = terminated or truncated\n",
                    "            \n",
                    "            # Q-learning update\n",
                    "            td_error = reward + gamma * np.max(q[next_state]) - q[state, action]\n",
                    "            q[state, action] += alpha * td_error\n",
                    "            \n",
                    "            # Update deterministic model\n",
                    "            model[state][action] = (reward, next_state)\n",
                    "            \n",
                    "            # Planning step(s)\n",
                    "            q = q_planning(model, q, alpha, gamma, n)\n",
                    "            \n",
                    "            # Move to next state\n",
                    "            state = next_state\n",
                    "            \n",
                    "            # Update reward sum\n",
                    "            reward_sum += reward\n",
                    "\n",
                    "        pbar.set_description(f'Episode Reward {int(reward_sum)}')\n",
                    "        reward_sums[episode_i] = reward_sum\n",
                    "\n",
                    "    return q, reward_sums"
                ]
            
            # Experiment parameters
            elif "'epsilon': ..." in source:
                cell['source'] = [
                    "# set for reproducibility\n",
                    "np.random.seed(2025)\n",
                    "\n",
                    "# parameters needed by our policy and learning rule\n",
                    "params = {'epsilon': 0.1,    # epsilon-greedy policy\n",
                    "          'alpha': 0.1,      # learning rate\n",
                    "          'gamma': 0.95,     # temporal discount factor\n",
                    "          'n': 5,           # number of planning steps\n",
                    "}\n",
                    "\n",
                    "# episodes/trials\n",
                    "n_episodes = 1000\n",
                    "\n",
                    "# environment initialization\n",
                    "env = gym.make(\"FrozenLake-v1\", map_name=\"8x8\", is_slippery=False)\n",
                    "\n",
                    "# Solve Frozen Lake using Dyna-Q\n",
                    "value_dyna_q, reward_sums_dyna_q = dyna_q(n_episodes, env, **params)\n",
                    "\n",
                    "# Plot the results\n",
                    "plot_performance(env, value_dyna_q, reward_sums_dyna_q)"
                ]
            
            # Custom reward function
            elif 'def custom_reward_function(self, observation, reward, done):' in source:
                cell['source'] = [
                    "from gymnasium.envs.toy_text import FrozenLakeEnv\n",
                    "\n",
                    "class CustomFrozenLakeEnv(FrozenLakeEnv):\n",
                    "    def step(self, action):\n",
                    "        obs, reward, terminated, truncated, info = super().step(action)\n",
                    "        # Modify the reward calculation\n",
                    "        reward = self.custom_reward_function(obs, reward, terminated)\n",
                    "        return obs, reward, terminated, truncated, info\n",
                    "\n",
                    "    def custom_reward_function(self, observation, reward, done):\n",
                    "        # TODO: Define your custom reward logic here\n",
                    "        if done and reward > 0:\n",
                    "            custom_reward = 1.0\n",
                    "        elif done and reward == 0:\n",
                    "            custom_reward = -0.1\n",
                    "        else:\n",
                    "            custom_reward = -0.01\n",
                    "        return custom_reward"
                ]
            
            # Reward shaping experiment
            elif "'epsilon': ..." in source and 'CustomFrozenLakeEnv' in source:
                cell['source'] = [
                    "# set for reproducibility\n",
                    "np.random.seed(2025)\n",
                    "\n",
                    "# parameters needed by our policy and learning rule\n",
                    "params = {'epsilon': 0.1,   # epsilon-greedy policy\n",
                    "          'alpha': 0.1,     # learning rate\n",
                    "          'gamma': 0.95,    # temporal discount factor\n",
                    "          'n': 5,          # number of planning steps\n",
                    "}\n",
                    "\n",
                    "# episodes/trials\n",
                    "n_episodes = 1000\n",
                    "\n",
                    "# environment initialization\n",
                    "env = CustomFrozenLakeEnv(map_name=\"8x8\", is_slippery=False)\n",
                    "\n",
                    "# Solve Frozen Lake using Dyna-Q\n",
                    "value_dyna_q, reward_sums_dyna_q = dyna_q(n_episodes, env, **params)\n",
                    "\n",
                    "# Plot the results\n",
                    "plot_performance(env, value_dyna_q, reward_sums_dyna_q)"
                ]
            
            # Prioritized sweeping planning
            elif 'def q_planning_priority(model: dict, q: np.ndarray, priorities: list, alpha: float, gamma: float, n: int) -> np.ndarray:' in source:
                cell['source'] = [
                    "def q_planning_priority(model: dict, q: np.ndarray, priorities: list, alpha: float, gamma: float, n: int) -> np.ndarray:\n",
                    "    \"\"\"Performs planning updates using Prioritized Sweeping.\"\"\"\n",
                    "\n",
                    "    for _ in range(n):\n",
                    "        if not priorities:\n",
                    "            break\n",
                    "            \n",
                    "        # Sample the state-action pair with the highest priority\n",
                    "        priority, state, action = heappop(priorities)\n",
                    "        priority = -priority  # Convert back from negative (heap is min-heap)\n",
                    "        \n",
                    "        # Retrieve deterministic transition\n",
                    "        if state not in model or action not in model[state]:\n",
                    "            continue\n",
                    "        reward, next_state = model[state][action]\n",
                    "        \n",
                    "        # Update Q-value using the deterministic transition\n",
                    "        td_error = reward + gamma * np.max(q[next_state]) - q[state, action]\n",
                    "        q[state, action] += alpha * td_error\n",
                    "        \n",
                    "        # Update priorities for predecessors\n",
                    "        for pred_state in model:\n",
                    "            for pred_action in model[pred_state]:\n",
                    "                pred_reward, pred_next_state = model[pred_state][pred_action]\n",
                    "                if pred_next_state == state:\n",
                    "                    pred_td_error = pred_reward + gamma * np.max(q[state]) - q[pred_state, pred_action]\n",
                    "                    if abs(pred_td_error) > 0.01:  # Threshold for adding to priority queue\n",
                    "                        heappush(priorities, (-abs(pred_td_error), pred_state, pred_action))\n",
                    "\n",
                    "    return q"
                ]
            
            # Dyna-Q with prioritized sweeping
            elif 'def dyna_q_priority(n_episodes: int, env: gym.Env, epsilon: float, alpha: float,' in source:
                cell['source'] = [
                    "def dyna_q_priority(n_episodes: int, env: gym.Env, epsilon: float, alpha: float,\n",
                    "                    gamma: float, n: int, theta: float) -> tuple[np.ndarray, np.ndarray]:\n",
                    "    \"\"\"Dyna-Q with Prioritized Sweeping algorithm for deterministic environments.\"\"\"\n",
                    "\n",
                    "    reward_sums = np.zeros(n_episodes)\n",
                    "    q = np.zeros((env.observation_space.n, env.action_space.n))\n",
                    "\n",
                    "    # Dyna-Q model for deterministic environments\n",
                    "    model = defaultdict(dict)\n",
                    "\n",
                    "    # Priority queue for prioritized sweeping\n",
                    "    priorities = []\n",
                    "\n",
                    "    for episode_i in (pbar := trange(n_episodes, leave=False)):\n",
                    "        state, info = env.reset()\n",
                    "        reward_sum, terminal = 0, False\n",
                    "\n",
                    "        while not terminal:\n",
                    "            # Take ε-greedy action\n",
                    "            action = epsilon_greedy_policy(state, q, epsilon)\n",
                    "            \n",
                    "            # Take action and observe\n",
                    "            next_state, reward, terminated, truncated, info = env.step(action)\n",
                    "            terminal = terminated or truncated\n",
                    "            \n",
                    "            # Q-learning update\n",
                    "            td_error = reward + gamma * np.max(q[next_state]) - q[state, action]\n",
                    "            q[state, action] += alpha * td_error\n",
                    "            \n",
                    "            # Update deterministic model\n",
                    "            model[state][action] = (reward, next_state)\n",
                    "            \n",
                    "            # Update priority queue if the TD error is significant\n",
                    "            if abs(td_error) > theta:\n",
                    "                heappush(priorities, (-abs(td_error), state, action))\n",
                    "            \n",
                    "            # Planning step with prioritized sweeping\n",
                    "            q = q_planning_priority(model, q, priorities, alpha, gamma, n)\n",
                    "            \n",
                    "            # Move to next state\n",
                    "            state = next_state\n",
                    "            \n",
                    "            # Update reward sum\n",
                    "            reward_sum += reward\n",
                    "\n",
                    "        pbar.set_description(f'Episode Reward {int(reward_sum)}')\n",
                    "        reward_sums[episode_i] = reward_sum\n",
                    "\n",
                    "    return q, reward_sums"
                ]
            
            # Final experiment with prioritized sweeping
            elif "'theta': ..." in source:
                cell['source'] = [
                    "# set for reproducibility\n",
                    "np.random.seed(2025)\n",
                    "\n",
                    "# parameters needed by our policy and learning rule\n",
                    "params = {'epsilon': 0.1,   # epsilon-greedy policy\n",
                    "          'alpha': 0.1,     # learning rate\n",
                    "          'gamma': 0.95,    # temporal discount factor\n",
                    "          'n': 5,          # number of planning steps\n",
                    "          'theta': 0.01     # prioritization threshold\n",
                    "}\n",
                    "\n",
                    "# episodes/trials\n",
                    "n_episodes = 1000\n",
                    "\n",
                    "# environment initialization\n",
                    "env = CustomFrozenLakeEnv(map_name=\"8x8\", is_slippery=False)\n",
                    "\n",
                    "# Solve Frozen Lake using Dyna-Q with Prioritized Sweeping\n",
                    "value_dyna_q, reward_sums_dyna_q = dyna_q_priority(n_episodes, env, **params)\n",
                    "\n",
                    "# Plot the results\n",
                    "plot_performance(env, value_dyna_q, reward_sums_dyna_q)"
                ]
    
    # Write the completed notebook
    with open('/Users/tahamajs/Documents/uni/DRL/docs/homeworks/HW5_Model_Based/code/RL_HW5_Dyna.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("Notebook completed successfully!")

if __name__ == "__main__":
    complete_notebook()

