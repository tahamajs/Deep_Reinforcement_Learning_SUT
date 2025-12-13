import torch
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Any

Transition = namedtuple('Transition', ('obs', 'actions', 'reward', 'next_obs', 'done', 'agent_id'))

class ReplayBuffer:
    """A simple replay buffer for storing and sampling experiences."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        """Samples a batch of transitions from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        """Returns the current size of the internal buffer."""
        return len(self.buffer)

class MultiAgentReplayBuffer:
    """A replay buffer specifically designed for multi-agent settings.
    It stores transitions as dictionaries for each agent and can sample batches for all agents.
    """
    def __init__(self, capacity: int, num_agents: int):
        self.buffers = [deque(maxlen=capacity) for _ in range(num_agents)]
        self.capacity = capacity
        self.num_agents = num_agents

    def push(self, obs_n: List[torch.Tensor], actions_n: List[int], reward: float,
             next_obs_n: List[torch.Tensor], done_n: List[bool], global_state: torch.Tensor = None, 
             next_global_state: torch.Tensor = None, messages_n: List[torch.Tensor] = None):
        """Saves a multi-agent transition."
        Args:
            obs_n (List[torch.Tensor]): List of observations for each agent.
            actions_n (List[int]): List of actions for each agent.
            reward (float): Shared reward for cooperative tasks.
            next_obs_n (List[torch.Tensor]): List of next observations for each agent.
            done_n (List[bool]): List of done flags for each agent.
            global_state (torch.Tensor, optional): Global state if available. Defaults to None.
            next_global_state (torch.Tensor, optional): Next global state. Defaults to None.
            messages_n (List[torch.Tensor], optional): List of messages sent by each agent. Defaults to None.
        """
        # Each agent's buffer stores its local view + global shared info
        for i in range(self.num_agents):
            transition_data = {
                'obs': obs_n[i],
                'actions': actions_n[i],
                'reward': reward,  # Shared reward for cooperative
                'next_obs': next_obs_n[i],
                'done': done_n[i],
                'global_state': global_state, # Can be None if not used
                'next_global_state': next_global_state, # Can be None if not used
                'all_actions': actions_n, # All actions from the step
                'messages': messages_n[i] if messages_n else None # Message sent by agent i
            }
            self.buffers[i].append(transition_data)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Samples a batch of multi-agent transitions from the buffer."
        Returns a list of dictionaries, where each dictionary corresponds to an agent's batch.
        """
        if min(len(buf) for buf in self.buffers) < batch_size:
            raise ValueError("Not enough samples in buffer for batch size")
            
        sampled_batches = []
        for i in range(self.num_agents):
            agent_batch = random.sample(self.buffers[i], batch_size)
            # Convert list of dicts to dict of lists/tensors
            batch_dict = {
                key: torch.stack([d[key] for d in agent_batch if d[key] is not None])
                if isinstance(agent_batch[0][key], torch.Tensor) else 
                [d[key] for d in agent_batch if d[key] is not None]
                for key in agent_batch[0]
            }
            sampled_batches.append(batch_dict)
        return sampled_batches
    
    def sample_global_batch(self, batch_size: int) -> Dict[str, Any]:
        """Samples a single global batch, useful for centralized critic updates."
        Assumes all agent buffers contain the same global transitions, just from their perspective.
        """
        if len(self.buffers[0]) < batch_size:
            raise ValueError("Not enough samples in buffer for batch size")
        
        # Sample from the first agent's buffer (assuming all buffers are synchronized for global info)
        batch_dicts = random.sample(self.buffers[0], batch_size)
        
        # Process to extract global information
        global_obs_batch = torch.stack([d['global_state'] for d in batch_dicts if d['global_state'] is not None])
        global_next_obs_batch = torch.stack([d['next_global_state'] for d in batch_dicts if d['next_global_state'] is not None])
        all_actions_batch = torch.tensor([d['all_actions'] for d in batch_dicts])
        rewards_batch = torch.tensor([d['reward'] for d in batch_dicts], dtype=torch.float32).unsqueeze(-1)
        dones_batch = torch.tensor([d['done'] for d in batch_dicts], dtype=torch.float32).unsqueeze(-1)
        
        # Collect individual observations and messages for convenience
        # Note: This assumes `obs_n` etc. in push are lists of individual tensors
        individual_obs_batch = torch.stack([torch.stack([t['obs'] for t in self.buffers[agent_idx] if t['obs'] is not None]) for agent_idx in range(self.num_agents)])
        individual_next_obs_batch = torch.stack([torch.stack([t['next_obs'] for t in self.buffers[agent_idx] if t['next_obs'] is not None]) for agent_idx in range(self.num_agents)])
        
        # Note: Messages can be tricky as they are agent-specific but also part of global context.
        # For now, we'll assume the 'messages' in agent's dict is the message *sent by that agent*
        # and if the critic needs all messages, they would be concatenated from all_actions_batch or passed explicitly.
        all_messages_batch = None # For now, assume global_state handles this or explicit message passing during update.

        return {
            'global_obs': global_obs_batch,
            'global_next_obs': global_next_obs_batch,
            'all_actions': all_actions_batch,
            'rewards': rewards_batch,
            'dones': dones_batch,
            # 'individual_obs': individual_obs_batch, # Potentially useful for actor updates
            # 'individual_next_obs': individual_next_obs_batch, # Potentially useful for actor updates
        }

    def __len__(self) -> int:
        """Returns the current size of the first agent's buffer (assuming all are synchronized)."""
        return len(self.buffers[0])

