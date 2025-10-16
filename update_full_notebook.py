#!/usr/bin/env python3
"""
Comprehensive script to complete all code in the PPO vs DDPG notebook
"""

import json

notebook_path = 'docs/homeworks/archive/sp23/hws/HW2/SP23_RL_HW2/RL_HW2_PPO_vs_DDPG_COMPLETE.ipynb'

print(f"Loading notebook from: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells")

# Code snippets for each section
code_updates = {
    # Networks cell
    "init_layer_uniform": """def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    \"\"\"Init uniform parameters on the single layer.\"\"\"
    # ==================================== Your Code (Begin) ====================================
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer
    # ==================================== Your Code (End) ====================================


class PPOActor(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        \"\"\"Initialize.\"\"\"
        super(PPOActor, self).__init__()

        # ==================================== Your Code (Begin) ====================================
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared hidden layers
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        
        # Separate output heads for mean and log_std
        self.mu_layer = nn.Linear(128, out_dim)
        self.log_std_layer = nn.Linear(128, out_dim)
        
        # Initialize output layers with uniform distribution
        init_layer_uniform(self.mu_layer)
        init_layer_uniform(self.log_std_layer)
        # ==================================== Your Code (End) ====================================


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        \"\"\"Forward method implementation.\"\"\"
        
        # ==================================== Your Code (Begin) ====================================
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        
        # Mean of the action distribution (bounded by tanh)
        mu = torch.tanh(self.mu_layer(x))
        
        # Log standard deviation (clamped for stability)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mu, std
        # ==================================== Your Code (End) ====================================



class Critic(nn.Module):
    def __init__(self, in_dim: int):
        \"\"\"Initialize.\"\"\"
        super(Critic, self).__init__()
        # ==================================== Your Code (Begin) ====================================
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        
        # Initialize output layer
        init_layer_uniform(self.out)
        # ==================================== Your Code (End) ====================================


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        \"\"\"Forward method implementation.\"\"\"
        # ==================================== Your Code (Begin) ====================================
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        
        return value
        # ==================================== Your Code (End) ====================================
""",

    "DDPGActor": """class DDPGActor(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int,
        init_w: float = 3e-3,
    ):
        \"\"\"Initialize.\"\"\"
        super(DDPGActor, self).__init__()
        
        # ==================================== Your Code (Begin) ====================================
        # 1. set the hidden layers
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_dim)
        
        # 2. init hidden layers uniformly       
        init_layer_uniform(self.out, init_w)
        # ==================================== Your Code (End) ====================================

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        \"\"\"Forward method implementation.\"\"\"
        # ==================================== Your Code (Begin) ====================================
        # use a tanh function as a ativation function for output layer 
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = torch.tanh(self.out(x))
        
        return action
        # ==================================== Your Code (End) ====================================
    
    
class DDPGCritic(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        init_w: float = 3e-3,
    ):
        \"\"\"Initialize.\"\"\"
        super(DDPGCritic, self).__init__()
        
        # ==================================== Your Code (Begin) ====================================
        # 1. set the hidden layers
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        
        # 2. init hidden layers uniformly       
        init_layer_uniform(self.out, init_w)
        # ==================================== Your Code (End) ====================================

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        \"\"\"Forward method implementation.\"\"\"
        # ==================================== Your Code (Begin) ====================================
        # notice that this value function is Q(s, a)
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        
        return value
        # ==================================== Your Code (End) ====================================
""",

    "ppo_iter": """def ppo_iter(
    epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    \"\"\"Yield mini-batches.\"\"\"
    
    # ==================================== Your Code (Begin) ====================================
    batch_size = states.size(0)
    
    for _ in range(epoch):
        # Generate random indices for mini-batch sampling
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            
            yield (
                states[rand_ids, :],
                actions[rand_ids, :],
                values[rand_ids, :],
                log_probs[rand_ids, :],
                returns[rand_ids, :],
                advantages[rand_ids, :],
            )
    # ==================================== Your Code (End) ====================================
""",

    "ActionNormalizer": """class ActionNormalizer(gym.ActionWrapper):
    \"\"\"Rescale and relocate the actions.\"\"\"

    def action(self, action: np.ndarray) -> np.ndarray:
        \"\"\"Change the range (-1, 1) to (low, high).\"\"\"
        # ==================================== Your Code (Begin) ====================================
        low = self.action_space.low
        high = self.action_space.high
        
        # Scale from [-1, 1] to [low, high]
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action
        # ==================================== Your Code (End) ====================================

    

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        \"\"\"Change the range (low, high) to (-1, 1).\"\"\"
        # ==================================== Your Code (Begin) ====================================
        low = self.action_space.low
        high = self.action_space.high
        
        # Scale from [low, high] to [-1, 1]
        action = 2.0 * (action - low) / (high - low) - 1.0
        action = np.clip(action, -1.0, 1.0)
        
        return action
        # ==================================== Your Code (End) ====================================
"""
}

# Update cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Check each code pattern and update
        if 'def init_layer_uniform' in source and 'class PPOActor' in source and 'class Critic' in source:
            print(f"Updating cell {i}: PPO/Critic Networks")
            cell['source'] = code_updates["init_layer_uniform"]
            
        elif 'class DDPGActor' in source and 'class DDPGCritic' in source:
            print(f"Updating cell {i}: DDPG Networks")
            cell['source'] = code_updates["DDPGActor"]
            
        elif 'def ppo_iter' in source and 'Yield mini-batches' in source:
            print(f"Updating cell {i}: ppo_iter function")
            cell['source'] = code_updates["ppo_iter"]
            
        elif 'class ActionNormalizer' in source and 'def reverse_action' in source:
            print(f"Updating cell {i}: ActionNormalizer")
            cell['source'] = code_updates["ActionNormalizer"]

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"\n✅ Notebook updated successfully!")
print(f"Saved to: {notebook_path}")
print("\nNext: The PPO and DDPG agent classes need to be completed manually in the notebook.")
print("Please refer to the complete_agents.py file for the full agent implementations.")


