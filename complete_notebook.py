#!/usr/bin/env python3
"""
Script to complete the PPO vs DDPG notebook with all implementations
"""

import json
import sys

# Read the notebook
notebook_path = 'docs/homeworks/archive/sp23/hws/HW2/SP23_RL_HW2/RL_HW2_PPO_vs_DDPG_COMPLETE.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells")

# Find and replace code cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Cell 17: init_layer_uniform and network classes
        if 'def init_layer_uniform' in source and 'Your Code (Begin)' in source:
            print(f"Updating cell {i}: init_layer_uniform and network classes")
            cell['source'] = [
                "def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:\n",
                "    \"\"\"Init uniform parameters on the single layer.\"\"\"\n",
                "    # ==================================== Your Code (Begin) ====================================\n",
                "    layer.weight.data.uniform_(-init_w, init_w)\n",
                "    layer.bias.data.uniform_(-init_w, init_w)\n",
                "    return layer\n",
                "    # ==================================== Your Code (End) ====================================\n",
                "\n",
                "\n",
                "class PPOActor(nn.Module):\n",
                "    def __init__(\n",
                "        self, \n",
                "        in_dim: int, \n",
                "        out_dim: int, \n",
                "        log_std_min: int = -20,\n",
                "        log_std_max: int = 0,\n",
                "    ):\n",
                "        \"\"\"Initialize.\"\"\"\n",
                "        super(PPOActor, self).__init__()\n",
                "\n",
                "        # ==================================== Your Code (Begin) ====================================\n",
                "        self.log_std_min = log_std_min\n",
                "        self.log_std_max = log_std_max\n",
                "        \n",
                "        # Shared hidden layers\n",
                "        self.hidden1 = nn.Linear(in_dim, 128)\n",
                "        self.hidden2 = nn.Linear(128, 128)\n",
                "        \n",
                "        # Separate output heads for mean and log_std\n",
                "        self.mu_layer = nn.Linear(128, out_dim)\n",
                "        self.log_std_layer = nn.Linear(128, out_dim)\n",
                "        \n",
                "        # Initialize output layers with uniform distribution\n",
                "        init_layer_uniform(self.mu_layer)\n",
                "        init_layer_uniform(self.log_std_layer)\n",
                "        # ==================================== Your Code (End) ====================================\n",
                "\n",
                "\n",
                "    def forward(self, state: torch.Tensor) -> torch.Tensor:\n",
                "        \"\"\"Forward method implementation.\"\"\"\n",
                "        \n",
                "        # ==================================== Your Code (Begin) ====================================\n",
                "        x = F.relu(self.hidden1(state))\n",
                "        x = F.relu(self.hidden2(x))\n",
                "        \n",
                "        # Mean of the action distribution (bounded by tanh)\n",
                "        mu = torch.tanh(self.mu_layer(x))\n",
                "        \n",
                "        # Log standard deviation (clamped for stability)\n",
                "        log_std = self.log_std_layer(x)\n",
                "        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)\n",
                "        std = torch.exp(log_std)\n",
                "        \n",
                "        return mu, std\n",
                "        # ==================================== Your Code (End) ====================================\n",
                "\n",
                "\n",
                "\n",
                "class Critic(nn.Module):\n",
                "    def __init__(self, in_dim: int):\n",
                "        \"\"\"Initialize.\"\"\"\n",
                "        super(Critic, self).__init__()\n",
                "        # ==================================== Your Code (Begin) ====================================\n",
                "        self.hidden1 = nn.Linear(in_dim, 128)\n",
                "        self.hidden2 = nn.Linear(128, 128)\n",
                "        self.out = nn.Linear(128, 1)\n",
                "        \n",
                "        # Initialize output layer\n",
                "        init_layer_uniform(self.out)\n",
                "        # ==================================== Your Code (End) ====================================\n",
                "\n",
                "\n",
                "    def forward(self, state: torch.Tensor) -> torch.Tensor:\n",
                "        \"\"\"Forward method implementation.\"\"\"\n",
                "        # ==================================== Your Code (Begin) ====================================\n",
                "        x = F.relu(self.hidden1(state))\n",
                "        x = F.relu(self.hidden2(x))\n",
                "        value = self.out(x)\n",
                "        \n",
                "        return value\n",
                "        # ==================================== Your Code (End) ====================================\n"
            ]
        
        # Cell for DDPG networks
        elif 'class DDPGActor' in source and '# 1. set the hidden layers' in source:
            print(f"Updating cell {i}: DDPG networks")
            cell['source'] = [
                "class DDPGActor(nn.Module):\n",
                "    def __init__(\n",
                "        self, \n",
                "        in_dim: int, \n",
                "        out_dim: int,\n",
                "        init_w: float = 3e-3,\n",
                "    ):\n",
                "        \"\"\"Initialize.\"\"\"\n",
                "        super(DDPGActor, self).__init__()\n",
                "        \n",
                "        # ==================================== Your Code (Begin) ====================================\n",
                "        # 1. set the hidden layers\n",
                "        self.hidden1 = nn.Linear(in_dim, 128)\n",
                "        self.hidden2 = nn.Linear(128, 128)\n",
                "        self.out = nn.Linear(128, out_dim)\n",
                "        \n",
                "        # 2. init hidden layers uniformly\n",
                "        init_layer_uniform(self.out, init_w)\n",
                "        # ==================================== Your Code (End) ====================================\n",
                "\n",
                "    def forward(self, state: torch.Tensor) -> torch.Tensor:\n",
                "        \"\"\"Forward method implementation.\"\"\"\n",
                "        # ==================================== Your Code (Begin) ====================================\n",
                "        # use a tanh function as a ativation function for output layer\n",
                "        x = F.relu(self.hidden1(state))\n",
                "        x = F.relu(self.hidden2(x))\n",
                "        action = torch.tanh(self.out(x))\n",
                "        \n",
                "        return action\n",
                "        # ==================================== Your Code (End) ====================================\n",
                "    \n",
                "    \n",
                "class DDPGCritic(nn.Module):\n",
                "    def __init__(\n",
                "        self, \n",
                "        in_dim: int, \n",
                "        init_w: float = 3e-3,\n",
                "    ):\n",
                "        \"\"\"Initialize.\"\"\"\n",
                "        super(DDPGCritic, self).__init__()\n",
                "        \n",
                "        # ==================================== Your Code (Begin) ====================================\n",
                "        # 1. set the hidden layers\n",
                "        self.hidden1 = nn.Linear(in_dim, 128)\n",
                "        self.hidden2 = nn.Linear(128, 128)\n",
                "        self.out = nn.Linear(128, 1)\n",
                "        \n",
                "        # 2. init hidden layers uniformly\n",
                "        init_layer_uniform(self.out, init_w)\n",
                "        # ==================================== Your Code (End) ====================================\n",
                "\n",
                "    def forward(\n",
                "        self, state: torch.Tensor, action: torch.Tensor\n",
                "    ) -> torch.Tensor:\n",
                "        \"\"\"Forward method implementation.\"\"\"\n",
                "        # ==================================== Your Code (Begin) ====================================\n",
                "        # notice that this value function is Q(s, a)\n",
                "        x = torch.cat([state, action], dim=-1)\n",
                "        x = F.relu(self.hidden1(x))\n",
                "        x = F.relu(self.hidden2(x))\n",
                "        value = self.out(x)\n",
                "        \n",
                "        return value\n",
                "        # ==================================== Your Code (End) ====================================\n"
            ]
        
        # Cell for ppo_iter
        elif 'def ppo_iter' in source and 'Yield mini-batches' in source:
            print(f"Updating cell {i}: ppo_iter function")
            cell['source'] = [
                "def ppo_iter(\n",
                "    epoch: int,\n",
                "    mini_batch_size: int,\n",
                "    states: torch.Tensor,\n",
                "    actions: torch.Tensor,\n",
                "    values: torch.Tensor,\n",
                "    log_probs: torch.Tensor,\n",
                "    returns: torch.Tensor,\n",
                "    advantages: torch.Tensor,\n",
                "):\n",
                "    \"\"\"Yield mini-batches.\"\"\"\n",
                "    \n",
                "    # ==================================== Your Code (Begin) ====================================\n",
                "    batch_size = states.size(0)\n",
                "    \n",
                "    for _ in range(epoch):\n",
                "        # Generate random indices for mini-batch sampling\n",
                "        for _ in range(batch_size // mini_batch_size):\n",
                "            rand_ids = np.random.randint(0, batch_size, mini_batch_size)\n",
                "            \n",
                "            yield (\n",
                "                states[rand_ids, :],\n",
                "                actions[rand_ids, :],\n",
                "                values[rand_ids, :],\n",
                "                log_probs[rand_ids, :],\n",
                "                returns[rand_ids, :],\n",
                "                advantages[rand_ids, :],\n",
                "            )\n",
                "    # ==================================== Your Code (End) ====================================\n"
            ]
        
        # Cell for ActionNormalizer
        elif 'class ActionNormalizer' in source and 'Rescale and relocate' in source:
            print(f"Updating cell {i}: ActionNormalizer")
            cell['source'] = [
                "class ActionNormalizer(gym.ActionWrapper):\n",
                "    \"\"\"Rescale and relocate the actions.\"\"\"\n",
                "\n",
                "    def action(self, action: np.ndarray) -> np.ndarray:\n",
                "        \"\"\"Change the range (-1, 1) to (low, high).\"\"\"\n",
                "        # ==================================== Your Code (Begin) ====================================\n",
                "        low = self.action_space.low\n",
                "        high = self.action_space.high\n",
                "        \n",
                "        # Scale from [-1, 1] to [low, high]\n",
                "        action = low + (action + 1.0) * 0.5 * (high - low)\n",
                "        action = np.clip(action, low, high)\n",
                "        \n",
                "        return action\n",
                "        # ==================================== Your Code (End) ====================================\n",
                "\n",
                "    \n",
                "\n",
                "    def reverse_action(self, action: np.ndarray) -> np.ndarray:\n",
                "        \"\"\"Change the range (low, high) to (-1, 1).\"\"\"\n",
                "        # ==================================== Your Code (Begin) ====================================\n",
                "        low = self.action_space.low\n",
                "        high = self.action_space.high\n",
                "        \n",
                "        # Scale from [low, high] to [-1, 1]\n",
                "        action = 2.0 * (action - low) / (high - low) - 1.0\n",
                "        action = np.clip(action, -1.0, 1.0)\n",
                "        \n",
                "        return action\n",
                "        # ==================================== Your Code (End) ====================================\n"
            ]

# Save the updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"\nNotebook updated successfully!")
print(f"Saved to: {notebook_path}")
