import json

# Read the notebook
with open('/Users/tahamajs/Documents/uni/DRL/docs/homeworks/archive/sp23/hws/HW1/SP23_RL_HW1/RL_HW1_CartPole.ipynb', 'r') as f:
    notebook = json.load(f)

# Update cell 0 (first markdown cell) - Add comprehensive introduction
notebook['cells'][0]['source'] = [
    "# Deep Q-Network (DQN) for CartPole-v1\n",
    "\n",
    "## Overview\n",
    "This notebook implements a Deep Q-Network (DQN) to solve the CartPole-v1 environment from OpenAI Gymnasium. DQN is a value-based reinforcement learning algorithm that uses a neural network to approximate the Q-function.\n",
    "\n",
    "### Key Concepts:\n",
    "- **Q-Learning**: Learn the optimal action-value function Q*(s,a)\n",
    "- **Experience Replay**: Store and sample past experiences to break correlation\n",
    "- **Target Network**: Use a separate network for stable Q-value targets\n",
    "- **ε-greedy Policy**: Balance exploration vs exploitation\n",
    "\n",
    "---\n",
    "\n",
    "# Installations and Imports"
]

# Update cell 3 (Utility functions markdown)
notebook['cells'][3]['source'] = [
    "# Utility Functions for Rendering Environment\n",
    "\n",
    "These helper functions allow us to:\n",
    "1. Record agent performance as video\n",
    "2. Embed videos in the notebook for visualization\n",
    "3. Evaluate trained policies visually"
]

# Update cell 4 (embed_mp4 function) - Add docstring
notebook['cells'][4]['source'] = [
    "def embed_mp4(filename):\n",
    "    \"\"\"Convert MP4 video to base64 and embed in HTML for notebook display\"\"\"\n",
    "    video = open(filename,'rb').read()\n",
    "    b64 = base64.b64encode(video)\n",
    "    tag = '''\n",
    "    <video width=\"640\" height=\"480\" controls>\n",
    "    <source src=\"data:video/mp4;base64,{0}\" type=\"video/mp4\">\n",
    "    Your browser does not support the video tag.\n",
    "    </video>'''.format(b64.decode())\n",
    "    \n",
    "    return IPython.display.HTML(tag)"
]

# Update cell 5 (create_policy_eval_video function) - Add docstring
notebook['cells'][5]['source'] = [
    "def create_policy_eval_video(env, policy, filename, num_episodes=1, fps=30):\n",
    "    \"\"\"Create a video of the agent following a given policy\"\"\"\n",
    "    filename = filename + \".mp4\"\n",
    "    with imageio.get_writer(filename, fps=fps) as video:\n",
    "        for _ in range(num_episodes):\n",
    "            state, info = env.reset()\n",
    "            video.append_data(env.render())\n",
    "            while True:\n",
    "                state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)\n",
    "                action = policy(state)\n",
    "                state, reward, terminated, truncated, _ = env.step(action.item())\n",
    "                video.append_data(env.render())\n",
    "                if terminated:\n",
    "                    break\n",
    "    return embed_mp4(filename)"
]

# Update cell 6 (Replay Memory markdown)
notebook['cells'][6]['source'] = [
    "# Replay Memory and Q-Network\n",
    "\n",
    "## Replay Memory\n",
    "Experience Replay is a crucial technique in DQN that:\n",
    "- Stores past experiences (state, action, reward, next_state)\n",
    "- Breaks temporal correlation between consecutive samples\n",
    "- Enables efficient use of past experiences through random sampling\n",
    "\n",
    "## Q-Network Architecture\n",
    "The Q-Network is a feedforward neural network that:\n",
    "- Takes state as input\n",
    "- Outputs Q-values for each possible action\n",
    "- Uses ReLU activations for non-linearity\n",
    "- Has no activation on the output layer (Q-values can be any real number)"
]

# Update cell 7 (ReplayMemory class) - Add docstring
notebook['cells'][7]['source'] = [
    "class ReplayMemory(object):\n",
    "    \"\"\"Experience Replay Memory with fixed capacity\"\"\"\n",
    "    def __init__(self, capacity):\n",
    "        self.memory = deque([], maxlen=capacity)\n",
    "\n",
    "    def push(self, transition):\n",
    "        \"\"\"Save a transition\"\"\"\n",
    "        self.memory.append(transition)\n",
    "\n",
    "    def sample(self, batch_size):\n",
    "        \"\"\"Randomly sample a batch of transitions\"\"\"\n",
    "        return random.sample(self.memory, batch_size)\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.memory)"
]

# Update cell 8 (DQN class) - COMPLETE IMPLEMENTATION
notebook['cells'][8]['source'] = [
    "# Complete the Q-Network below. \n",
    "# The Q-Network takes a state as input and the output is a vector so that each element is the q-value for an action.\n",
    "\n",
    "class DQN(nn.Module):\n",
    "    \"\"\"Deep Q-Network with 3 fully connected layers\"\"\"\n",
    "    def __init__(self, n_observations, n_actions):\n",
    "        super(DQN, self).__init__()\n",
    "        # ==================================== Your Code (Begin) ====================================\n",
    "        # Define a simple feedforward neural network with 3 layers\n",
    "        # Architecture: Input -> 128 -> 128 -> Output\n",
    "        self.layer1 = nn.Linear(n_observations, 128)\n",
    "        self.layer2 = nn.Linear(128, 128)\n",
    "        self.layer3 = nn.Linear(128, n_actions)\n",
    "        # ==================================== Your Code (End) ====================================\n",
    "\n",
    "    def forward(self, x):\n",
    "        # ==================================== Your Code (Begin) ====================================\n",
    "        # Forward pass through the network with ReLU activations\n",
    "        x = F.relu(self.layer1(x))\n",
    "        x = F.relu(self.layer2(x))\n",
    "        return self.layer3(x)  # No activation on output layer (Q-values can be any real number)\n",
    "        # ==================================== Your Code (End) ===================================="
]

# Update cell 9 (Policies markdown)
notebook['cells'][9]['source'] = [
    "# Action Selection Policies\n",
    "\n",
    "## Greedy Policy\n",
    "Selects the action with the highest Q-value. Used during evaluation.\n",
    "\n",
    "## ε-Greedy Policy\n",
    "Balances exploration and exploitation:\n",
    "- With probability ε: select a random action (exploration)\n",
    "- With probability 1-ε: select the best action (exploitation)\n",
    "- ε decays over time: start with high exploration, gradually shift to exploitation"
]

# Update cell 11 (greedy_policy function) - COMPLETE IMPLEMENTATION
notebook['cells'][11]['source'] = [
    "# This function takes in a state and returns the best action according to your q-network.\n",
    "# Don't forget \"torch.no_grad()\". We don't want gradient flowing through our network. \n",
    "\n",
    "# state shape: (1, state_size) -> output shape: (1, 1)  \n",
    "def greedy_policy(qnet, state):\n",
    "    # ==================================== Your Code (Begin) ====================================\n",
    "    with torch.no_grad():\n",
    "        # Get Q-values for all actions and select the action with maximum Q-value\n",
    "        return qnet(state).max(1)[1].view(1, 1)\n",
    "    # ==================================== Your Code (End) ===================================="
]

# Update cell 12 (e_greedy_policy function) - COMPLETE IMPLEMENTATION
notebook['cells'][12]['source'] = [
    "# state shape: (1, state_size) -> output shape: (1, 1)\n",
    "# Don't forget \"torch.no_grad()\". We don't want gradient flowing through our network.\n",
    "\n",
    "def e_greedy_policy(qnet, state, current_timestep):\n",
    "    \"\"\"Epsilon-greedy action selection with exponential decay\"\"\"\n",
    "    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * current_timestep / EPS_DECAY)\n",
    "    # ==================================== Your Code (Begin) ====================================\n",
    "    # With probability \"eps_threshold\" choose a random action \n",
    "    # and with probability 1-\"eps_threshold\" choose the best action according to your Q-Network.\n",
    "    \n",
    "    sample = random.random()\n",
    "    if sample > eps_threshold:\n",
    "        # Exploitation: choose best action\n",
    "        with torch.no_grad():\n",
    "            return qnet(state).max(1)[1].view(1, 1)\n",
    "    else:\n",
    "        # Exploration: choose random action\n",
    "        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)\n",
    "    # ==================================== Your Code (End) ===================================="
]

# Update cell 13 (Initial setup markdown)
notebook['cells'][13]['source'] = [
    "# Initial Setup\n",
    "\n",
    "## Hyperparameters:\n",
    "- **BATCH_SIZE**: Number of transitions sampled from replay buffer (128)\n",
    "- **GAMMA**: Discount factor for future rewards (0.99)\n",
    "- **EPS_START**: Initial exploration rate (0.9)\n",
    "- **EPS_END**: Final exploration rate (0.05)\n",
    "- **EPS_DECAY**: Rate of exponential decay of epsilon (1000)\n",
    "- **TAU**: Soft update coefficient for target network (0.005)\n",
    "- **LR**: Learning rate for optimizer (1e-4)\n",
    "\n",
    "## Components:\n",
    "- **Environment**: CartPole-v1\n",
    "- **Q-Network**: Main network for learning\n",
    "- **Target Network**: Stable network for computing targets\n",
    "- **Optimizer**: Adam optimizer\n",
    "- **Memory**: Replay buffer with capacity 10,000"
]

# Update cell 14 (Initial setup code)
notebook['cells'][14]['source'] = [
    "BATCH_SIZE = 128\n",
    "GAMMA = 0.99\n",
    "EPS_START = 0.9\n",
    "EPS_END = 0.05\n",
    "EPS_DECAY = 1000\n",
    "TAU = 0.005\n",
    "LR = 1e-4\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "env = gym.make(\"CartPole-v1\", render_mode='rgb_array')\n",
    "n_actions = env.action_space.n\n",
    "state, info = env.reset()\n",
    "n_observations = len(state)\n",
    "q_network = DQN(n_observations, n_actions).to(device)\n",
    "target_network = DQN(n_observations, n_actions).to(device)\n",
    "target_network.load_state_dict(q_network.state_dict())  # Initialize target network with same weights\n",
    "optimizer = optim.Adam(q_network.parameters(), lr=LR)\n",
    "memory = ReplayMemory(10000)\n",
    "\n",
    "print(f\"Device: {device}\")\n",
    "print(f\"State space: {n_observations}\")\n",
    "print(f\"Action space: {n_actions}\")\n",
    "print(\"\\nRandom agent before training:\")\n",
    "create_policy_eval_video(env, lambda s: greedy_policy(q_network, s), \"random_agent\")"
]

# Update cell 15 (Training markdown)
notebook['cells'][15]['source'] = [
    "# Training Loop\n",
    "\n",
    "## DQN Algorithm Steps:\n",
    "\n",
    "### For each episode:\n",
    "1. **Reset environment** and get initial state\n",
    "2. **Interact with environment**:\n",
    "   - Select action using ε-greedy policy\n",
    "   - Execute action and observe reward and next state\n",
    "   - Store transition in replay memory\n",
    "   \n",
    "3. **Learn from experience**:\n",
    "   - Sample random batch from replay memory\n",
    "   - Compute predicted Q-values: Q(s,a) using q_network\n",
    "   - Compute target Q-values: r + γ × max_a' Q(s',a') using target_network\n",
    "   - Compute loss (e.g., Huber loss or MSE)\n",
    "   - Update q_network via gradient descent\n",
    "   \n",
    "4. **Soft update target network**:\n",
    "   - θ' ← τθ + (1-τ)θ'\n",
    "   - Slowly update target network to track q_network\n",
    "   \n",
    "5. **Track performance**:\n",
    "   - Record episode duration and total reward"
]

# Update cell 16 (Training loop) - COMPLETE IMPLEMENTATION
notebook['cells'][16]['source'] = [
    "Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))\n",
    "\n",
    "num_episodes = 200\n",
    "episode_returns = []\n",
    "episode_durations = []\n",
    "\n",
    "for i_episode in range(num_episodes):\n",
    "\n",
    "    # ==================================== Your Code (Begin) ====================================\n",
    "    # 1. Start a new episode\n",
    "    state, info = env.reset()\n",
    "    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)\n",
    "    \n",
    "    total_reward = 0\n",
    "    t = 0\n",
    "    \n",
    "    for t in count():\n",
    "        # 2. Run the environment for 1 step using e-greedy policy\n",
    "        action = e_greedy_policy(q_network, state, i_episode * 500 + t)\n",
    "        observation, reward, terminated, truncated, _ = env.step(action.item())\n",
    "        reward = torch.tensor([reward], device=device)\n",
    "        total_reward += reward.item()\n",
    "        \n",
    "        if terminated:\n",
    "            next_state = None\n",
    "        else:\n",
    "            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)\n",
    "        \n",
    "        # 3. Add the (state, action, next_state, reward) to replay memory\n",
    "        memory.push(Transition(state, action, next_state, reward))\n",
    "        \n",
    "        # Move to next state\n",
    "        state = next_state\n",
    "        \n",
    "        # 4. Optimize your q_network for 1 iteration\n",
    "        if len(memory) >= BATCH_SIZE:\n",
    "            # 4.1 Sample one batch from replay memory\n",
    "            transitions = memory.sample(BATCH_SIZE)\n",
    "            batch = Transition(*zip(*transitions))\n",
    "            \n",
    "            # Create masks for non-final states\n",
    "            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), \n",
    "                                         device=device, dtype=torch.bool)\n",
    "            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])\n",
    "            \n",
    "            state_batch = torch.cat(batch.state)\n",
    "            action_batch = torch.cat(batch.action)\n",
    "            reward_batch = torch.cat(batch.reward)\n",
    "            \n",
    "            # 4.2 Compute predicted state-action values using q_network\n",
    "            # Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken\n",
    "            state_action_values = q_network(state_batch).gather(1, action_batch)\n",
    "            \n",
    "            # 4.3 Compute expected state-action values using target_network\n",
    "            next_state_values = torch.zeros(BATCH_SIZE, device=device)\n",
    "            with torch.no_grad():\n",
    "                next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0]\n",
    "            # Compute the expected Q values: r + gamma * max_a' Q(s', a')\n",
    "            expected_state_action_values = (next_state_values * GAMMA) + reward_batch\n",
    "            \n",
    "            # 4.4 Compute loss function and optimize q_network for 1 step\n",
    "            # Huber loss is less sensitive to outliers than MSE\n",
    "            criterion = nn.SmoothL1Loss()\n",
    "            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))\n",
    "            \n",
    "            # Optimize the model\n",
    "            optimizer.zero_grad()\n",
    "            loss.backward()\n",
    "            # Gradient clipping to prevent exploding gradients\n",
    "            torch.nn.utils.clip_grad_value_(q_network.parameters(), 100)\n",
    "            optimizer.step()\n",
    "        \n",
    "        if terminated or truncated:\n",
    "            episode_durations.append(t + 1)\n",
    "            episode_returns.append(total_reward)\n",
    "            break\n",
    "    \n",
    "    # 5. Soft update the weights of target_network\n",
    "    # θ′ ← τ θ + (1 −τ )θ′\n",
    "    target_net_state_dict = target_network.state_dict()\n",
    "    policy_net_state_dict = q_network.state_dict()\n",
    "    for key in policy_net_state_dict:\n",
    "        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)\n",
    "    target_network.load_state_dict(target_net_state_dict)\n",
    "    \n",
    "    # Print progress every 10 episodes\n",
    "    if (i_episode + 1) % 10 == 0:\n",
    "        avg_duration = sum(episode_durations[-10:]) / 10\n",
    "        avg_return = sum(episode_returns[-10:]) / 10\n",
    "        print(f'Episode {i_episode+1}/{num_episodes} | Avg Duration: {avg_duration:.1f} | Avg Return: {avg_return:.1f}')\n",
    "\n",
    "    # ==================================== Your Code (End) ====================================  \n",
    "\n",
    "print('\\nTraining Complete!')\n",
    "print(f'Final Average Duration (last 10 episodes): {sum(episode_durations[-10:]) / 10:.1f}')\n",
    "print(f'Final Average Return (last 10 episodes): {sum(episode_returns[-10:]) / 10:.1f}')\n",
    "\n",
    "# Plot results\n",
    "plt.figure(figsize=(12, 5))\n",
    "\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(range(1, num_episodes+1), episode_durations)\n",
    "plt.xlabel('Episode')\n",
    "plt.ylabel('Duration')\n",
    "plt.title('Episode Durations Over Training')\n",
    "plt.grid(True)\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(range(1, num_episodes+1), episode_returns)\n",
    "plt.xlabel('Episode')\n",
    "plt.ylabel('Total Return')\n",
    "plt.title('Episode Returns Over Training')\n",
    "plt.grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]

# Add new markdown cell for evaluation
notebook['cells'].insert(17, {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Evaluation\n",
        "\n",
        "Now let's visualize how well our trained agent performs!\n",
        "Compare this video to the random agent from before training."
    ]
})

# Update cell 17 (now 18) (Render trained model)
notebook['cells'][18]['source'] = [
    "# Render trained model\n",
    "print(\"Trained agent performance:\")\n",
    "create_policy_eval_video(env, lambda s: greedy_policy(q_network, s), \"trained_agent\")"
]

# Add new markdown cell for discussion
notebook['cells'].insert(19, {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Analysis and Discussion\n",
        "\n",
        "## What We Learned:\n",
        "1. **Experience Replay** helps break correlation between consecutive samples\n",
        "2. **Target Network** provides stable learning targets\n",
        "3. **ε-greedy exploration** balances exploration and exploitation\n",
        "4. **Soft updates** of target network prevent instability\n",
        "\n",
        "## Expected Results:\n",
        "- CartPole is considered solved when average reward ≥ 195 over 100 episodes\n",
        "- The agent should learn to balance the pole for longer durations\n",
        "- Training curves should show increasing episode durations\n",
        "\n",
        "## Further Improvements:\n",
        "- **Double DQN**: Reduce overestimation bias\n",
        "- **Dueling DQN**: Separate value and advantage streams\n",
        "- **Prioritized Experience Replay**: Sample important transitions more frequently\n",
        "- **Noisy Networks**: Add parametric noise for exploration"
    ]
})

# Update cell 19 (now 20) (empty cell to model save cell)
notebook['cells'][20]['source'] = [
    "# Optional: Save the trained model\n",
    "torch.save({\n",
    "    'q_network_state_dict': q_network.state_dict(),\n",
    "    'target_network_state_dict': target_network.state_dict(),\n",
    "    'optimizer_state_dict': optimizer.state_dict(),\n",
    "    'episode_durations': episode_durations,\n",
    "    'episode_returns': episode_returns,\n",
    "}, 'dqn_cartpole.pth')\n",
    "\n",
    "print(\"Model saved as 'dqn_cartpole.pth'\")"
]

# Write the updated notebook
with open('/Users/tahamajs/Documents/uni/DRL/docs/homeworks/archive/sp23/hws/HW1/SP23_RL_HW1/RL_HW1_CartPole.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook updated successfully!")
