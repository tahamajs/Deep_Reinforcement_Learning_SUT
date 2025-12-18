"""
Multi-Agent and Meta-Learning Systems
CA4: Policy Gradient Methods and Neural Networks in RL - Advanced Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import random
from collections import deque, defaultdict
import copy
import math


class MultiAgentPolicyGradient:
    """Multi-Agent Policy Gradient System"""
    
    def __init__(self, num_agents: int, state_size: int, action_size: int,
                 lr: float = 0.001, gamma: float = 0.99, 
                 communication_enabled: bool = True):
        """Initialize multi-agent system
        
        Args:
            num_agents: Number of agents
            state_size: State space dimension
            action_size: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            communication_enabled: Whether agents can communicate
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.communication_enabled = communication_enabled
        
        # Create agents
        self.agents = []
        for i in range(num_agents):
            agent = MultiAgentPGAgent(
                agent_id=i,
                state_size=state_size,
                action_size=action_size,
                lr=lr,
                gamma=gamma
            )
            self.agents.append(agent)
        
        # Communication network
        if communication_enabled:
            self.communication_network = CommunicationNetwork(num_agents)
        
        # Training statistics
        self.episode_rewards = [[] for _ in range(num_agents)]
        self.cooperation_scores = []
        
    def get_actions(self, states: List[np.ndarray]) -> List[int]:
        """Get actions from all agents
        
        Args:
            states: List of states for each agent
            
        Returns:
            List of actions
        """
        actions = []
        for i, (agent, state) in enumerate(zip(self.agents, states)):
            action = agent.get_action(state)
            actions.append(action)
        
        return actions
    
    def update_agents(self, experiences: List[Dict[str, Any]]):
        """Update all agents
        
        Args:
            experiences: List of experiences for each agent
        """
        for i, (agent, exp) in enumerate(zip(self.agents, experiences)):
            agent.update(exp)
    
    def train_episode(self, env) -> Tuple[List[float], Dict[str, Any]]:
        """Train for one episode
        
        Args:
            env: Multi-agent environment
            
        Returns:
            Tuple of (rewards, info)
        """
        states, info = env.reset()
        
        episode_experiences = [[] for _ in range(self.num_agents)]
        episode_rewards = [0.0] * self.num_agents
        
        done = False
        step = 0
        
        while not done and step < 1000:
            # Get actions from all agents
            actions = self.get_actions(states)
            
            # Step environment
            next_states, rewards, dones, truncated, info = env.step(actions)
            
            # Store experiences
            for i in range(self.num_agents):
                experience = {
                    'state': states[i],
                    'action': actions[i],
                    'reward': rewards[i],
                    'next_state': next_states[i],
                    'done': dones[i] or truncated[i]
                }
                episode_experiences[i].append(experience)
                episode_rewards[i] += rewards[i]
            
            states = next_states
            done = all(dones) or all(truncated)
            step += 1
        
        # Update agents
        self.update_agents(episode_experiences)
        
        # Store rewards
        for i, reward in enumerate(episode_rewards):
            self.episode_rewards[i].append(reward)
        
        # Calculate cooperation score
        cooperation_score = self._calculate_cooperation_score(episode_experiences)
        self.cooperation_scores.append(cooperation_score)
        
        return episode_rewards, {
            'cooperation_score': cooperation_score,
            'episode_length': step,
            'total_reward': sum(episode_rewards)
        }
    
    def _calculate_cooperation_score(self, experiences: List[List[Dict]]) -> float:
        """Calculate cooperation score between agents
        
        Args:
            experiences: List of experiences for each agent
            
        Returns:
            Cooperation score
        """
        if not self.communication_enabled:
            return 0.0
        
        # Calculate action correlation
        action_correlations = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                actions_i = [exp['action'] for exp in experiences[i]]
                actions_j = [exp['action'] for exp in experiences[j]]
                
                if len(actions_i) > 1 and len(actions_j) > 1:
                    correlation = np.corrcoef(actions_i, actions_j)[0, 1]
                    if not np.isnan(correlation):
                        action_correlations.append(abs(correlation))
        
        return np.mean(action_correlations) if action_correlations else 0.0
    
    def train(self, env, num_episodes: int = 1000, print_every: int = 100):
        """Train the multi-agent system
        
        Args:
            env: Multi-agent environment
            num_episodes: Number of episodes
            print_every: Print frequency
        """
        for episode in range(num_episodes):
            rewards, info = self.train_episode(env)
            
            if (episode + 1) % print_every == 0:
                avg_rewards = [np.mean(agent_rewards[-print_every:]) 
                              for agent_rewards in self.episode_rewards]
                avg_cooperation = np.mean(self.cooperation_scores[-print_every:])
                
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Rewards: {[f'{r:.2f}' for r in avg_rewards]} | "
                      f"Cooperation: {avg_cooperation:.3f}")


class MultiAgentPGAgent:
    """Individual agent in multi-agent system"""
    
    def __init__(self, agent_id: int, state_size: int, action_size: int,
                 lr: float = 0.001, gamma: float = 0.99):
        """Initialize agent
        
        Args:
            agent_id: Agent identifier
            state_size: State space dimension
            action_size: Action space dimension
            lr: Learning rate
            gamma: Discount factor
        """
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        
    def get_action(self, state: np.ndarray) -> int:
        """Get action from policy
        
        Args:
            state: Current state
            
        Returns:
            Action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        return action
    
    def update(self, experiences: List[Dict[str, Any]]):
        """Update policy using experiences
        
        Args:
            experiences: List of experiences
        """
        if not experiences:
            return
        
        # Calculate returns
        returns = []
        G = 0
        for exp in reversed(experiences):
            G = exp['reward'] + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensors
        states = torch.FloatTensor([exp['state'] for exp in experiences])
        actions = torch.LongTensor([exp['action'] for exp in experiences])
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        logits = self.policy_net(states)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        policy_loss = -(selected_log_probs * returns).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


class CommunicationNetwork:
    """Communication network for multi-agent systems"""
    
    def __init__(self, num_agents: int, communication_range: float = 1.0):
        """Initialize communication network
        
        Args:
            num_agents: Number of agents
            communication_range: Communication range
        """
        self.num_agents = num_agents
        self.communication_range = communication_range
        
        # Communication matrix
        self.communication_matrix = torch.zeros(num_agents, num_agents)
        
        # Message buffers
        self.message_buffers = [[] for _ in range(num_agents)]
        
    def send_message(self, sender_id: int, receiver_id: int, message: torch.Tensor):
        """Send message between agents
        
        Args:
            sender_id: Sender agent ID
            receiver_id: Receiver agent ID
            message: Message tensor
        """
        if receiver_id < len(self.message_buffers):
            self.message_buffers[receiver_id].append({
                'sender': sender_id,
                'message': message,
                'timestamp': len(self.message_buffers[receiver_id])
            })
    
    def receive_messages(self, agent_id: int) -> List[Dict[str, Any]]:
        """Receive messages for an agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of messages
        """
        messages = self.message_buffers[agent_id].copy()
        self.message_buffers[agent_id].clear()
        return messages
    
    def update_communication_matrix(self, agent_positions: List[np.ndarray]):
        """Update communication matrix based on agent positions
        
        Args:
            agent_positions: List of agent positions
        """
        self.communication_matrix.zero_()
        
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    distance = np.linalg.norm(agent_positions[i] - agent_positions[j])
                    if distance <= self.communication_range:
                        self.communication_matrix[i, j] = 1.0


class MetaLearningAgent:
    """Meta-Learning Agent for Policy Gradient Methods"""
    
    def __init__(self, state_size: int, action_size: int, 
                 meta_lr: float = 0.001, inner_lr: float = 0.01,
                 num_inner_steps: int = 5):
        """Initialize meta-learning agent
        
        Args:
            state_size: State space dimension
            action_size: Action space dimension
            meta_lr: Meta-learning rate
            inner_lr: Inner loop learning rate
            num_inner_steps: Number of inner loop steps
        """
        self.state_size = state_size
        self.action_size = action_size
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
        # Meta-network (initialization network)
        self.meta_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.meta_network.parameters(), lr=meta_lr)
        
        # Task-specific networks
        self.task_networks = {}
        
    def adapt_to_task(self, task_id: str, task_experiences: List[Dict[str, Any]]) -> nn.Module:
        """Adapt to a specific task
        
        Args:
            task_id: Task identifier
            task_experiences: Task-specific experiences
            
        Returns:
            Adapted policy network
        """
        # Initialize task-specific network
        task_network = copy.deepcopy(self.meta_network)
        task_optimizer = optim.Adam(task_network.parameters(), lr=self.inner_lr)
        
        # Inner loop adaptation
        for step in range(self.num_inner_steps):
            if not task_experiences:
                break
            
            # Sample batch from experiences
            batch_size = min(32, len(task_experiences))
            batch = random.sample(task_experiences, batch_size)
            
            # Calculate loss
            loss = self._calculate_task_loss(task_network, batch)
            
            # Update task network
            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()
        
        # Store task network
        self.task_networks[task_id] = task_network
        
        return task_network
    
    def _calculate_task_loss(self, network: nn.Module, experiences: List[Dict[str, Any]]) -> torch.Tensor:
        """Calculate task-specific loss
        
        Args:
            network: Policy network
            experiences: Task experiences
            
        Returns:
            Loss tensor
        """
        states = torch.FloatTensor([exp['state'] for exp in experiences])
        actions = torch.LongTensor([exp['action'] for exp in experiences])
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences])
        
        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        logits = network(states)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        loss = -(selected_log_probs * returns).mean()
        
        return loss
    
    def meta_update(self, task_results: List[Dict[str, Any]]):
        """Perform meta-update
        
        Args:
            task_results: Results from multiple tasks
        """
        meta_loss = 0
        
        for task_result in task_results:
            task_id = task_result['task_id']
            task_experiences = task_result['experiences']
            
            if task_id in self.task_networks:
                # Calculate loss on adapted network
                loss = self._calculate_task_loss(self.task_networks[task_id], task_experiences)
                meta_loss += loss
        
        # Meta-update
        if meta_loss > 0:
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
    
    def get_action(self, state: np.ndarray, task_id: str = None) -> int:
        """Get action from policy
        
        Args:
            state: Current state
            task_id: Task identifier
            
        Returns:
            Action
        """
        if task_id and task_id in self.task_networks:
            network = self.task_networks[task_id]
        else:
            network = self.meta_network
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits = network(state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        return action


class TransferLearningAgent:
    """Transfer Learning Agent for Policy Gradient Methods"""
    
    def __init__(self, source_state_size: int, source_action_size: int,
                 target_state_size: int, target_action_size: int,
                 transfer_type: str = 'feature_transfer'):
        """Initialize transfer learning agent
        
        Args:
            source_state_size: Source task state size
            source_action_size: Source task action size
            target_state_size: Target task state size
            target_action_size: Target task action size
            transfer_type: Type of transfer ('feature_transfer', 'policy_transfer')
        """
        self.source_state_size = source_state_size
        self.source_action_size = source_action_size
        self.target_state_size = target_state_size
        self.target_action_size = target_action_size
        self.transfer_type = transfer_type
        
        # Source task network
        self.source_network = nn.Sequential(
            nn.Linear(source_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, source_action_size)
        )
        
        # Target task network
        if transfer_type == 'feature_transfer':
            # Share feature extractor
            self.feature_extractor = nn.Sequential(
                nn.Linear(max(source_state_size, target_state_size), 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
            
            self.source_head = nn.Linear(128, source_action_size)
            self.target_head = nn.Linear(128, target_action_size)
            
        elif transfer_type == 'policy_transfer':
            # Direct policy transfer
            self.target_network = nn.Sequential(
                nn.Linear(target_state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, target_action_size)
            )
        
        # Optimizers
        self.source_optimizer = optim.Adam(self.source_network.parameters(), lr=0.001)
        self.target_optimizer = optim.Adam(self.target_network.parameters(), lr=0.001)
        
    def train_source_task(self, experiences: List[Dict[str, Any]], num_episodes: int = 1000):
        """Train on source task
        
        Args:
            experiences: Source task experiences
            num_episodes: Number of episodes
        """
        for episode in range(num_episodes):
            if not experiences:
                continue
            
            # Sample batch
            batch_size = min(32, len(experiences))
            batch = random.sample(experiences, batch_size)
            
            # Calculate loss
            loss = self._calculate_source_loss(batch)
            
            # Update
            self.source_optimizer.zero_grad()
            loss.backward()
            self.source_optimizer.step()
    
    def transfer_to_target_task(self, target_experiences: List[Dict[str, Any]]):
        """Transfer knowledge to target task
        
        Args:
            target_experiences: Target task experiences
        """
        if self.transfer_type == 'feature_transfer':
            # Initialize target head with source head weights
            with torch.no_grad():
                self.target_head.weight.data = self.source_head.weight.data[:self.target_action_size, :]
                self.target_head.bias.data = self.source_head.bias.data[:self.target_action_size]
        
        elif self.transfer_type == 'policy_transfer':
            # Copy relevant weights from source to target
            with torch.no_grad():
                for target_param, source_param in zip(self.target_network.parameters(), 
                                                    self.source_network.parameters()):
                    if target_param.shape == source_param.shape:
                        target_param.data.copy_(source_param.data)
    
    def _calculate_source_loss(self, experiences: List[Dict[str, Any]]) -> torch.Tensor:
        """Calculate source task loss
        
        Args:
            experiences: Source task experiences
            
        Returns:
            Loss tensor
        """
        states = torch.FloatTensor([exp['state'] for exp in experiences])
        actions = torch.LongTensor([exp['action'] for exp in experiences])
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences])
        
        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        logits = self.source_network(states)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        loss = -(selected_log_probs * returns).mean()
        
        return loss
    
    def get_target_action(self, state: np.ndarray) -> int:
        """Get action for target task
        
        Args:
            state: Current state
            
        Returns:
            Action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if self.transfer_type == 'feature_transfer':
                features = self.feature_extractor(state_tensor)
                logits = self.target_head(features)
            else:
                logits = self.target_network(state_tensor)
            
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        return action


class DistributedTrainingSystem:
    """Distributed Training System for Policy Gradient Methods"""
    
    def __init__(self, num_workers: int, state_size: int, action_size: int,
                 lr: float = 0.001, gamma: float = 0.99):
        """Initialize distributed training system
        
        Args:
            num_workers: Number of worker processes
            state_size: State space dimension
            action_size: Action space dimension
            lr: Learning rate
            gamma: Discount factor
        """
        self.num_workers = num_workers
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        
        # Central policy network
        self.central_policy = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Worker networks
        self.worker_networks = []
        for i in range(num_workers):
            worker_net = copy.deepcopy(self.central_policy)
            self.worker_networks.append(worker_net)
        
        # Optimizers
        self.central_optimizer = optim.Adam(self.central_policy.parameters(), lr=lr)
        self.worker_optimizers = [optim.Adam(worker.parameters(), lr=lr) 
                                 for worker in self.worker_networks]
        
        # Experience buffers
        self.experience_buffers = [deque(maxlen=10000) for _ in range(num_workers)]
        
        # Training statistics
        self.worker_rewards = [[] for _ in range(num_workers)]
        self.central_rewards = []
        
    def collect_experiences(self, worker_id: int, env, num_episodes: int = 10):
        """Collect experiences from a worker
        
        Args:
            worker_id: Worker identifier
            env: Environment
            num_episodes: Number of episodes
        """
        worker_net = self.worker_networks[worker_id]
        experience_buffer = self.experience_buffers[worker_id]
        
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0
            episode_experiences = []
            
            done = False
            step = 0
            
            while not done and step < 1000:
                # Get action from worker network
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    logits = worker_net(state_tensor)
                    probs = F.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                
                # Step environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Store experience
                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': terminated or truncated
                }
                episode_experiences.append(experience)
                experience_buffer.append(experience)
                
                state = next_state
                episode_reward += reward
                done = terminated or truncated
                step += 1
            
            self.worker_rewards[worker_id].append(episode_reward)
    
    def update_worker(self, worker_id: int, batch_size: int = 32):
        """Update a worker network
        
        Args:
            worker_id: Worker identifier
            batch_size: Batch size
        """
        worker_net = self.worker_networks[worker_id]
        experience_buffer = self.experience_buffers[worker_id]
        optimizer = self.worker_optimizers[worker_id]
        
        if len(experience_buffer) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(experience_buffer, batch_size)
        
        # Calculate loss
        loss = self._calculate_loss(worker_net, batch)
        
        # Update worker
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def update_central_policy(self, batch_size: int = 32):
        """Update central policy network
        
        Args:
            batch_size: Batch size
        """
        # Collect experiences from all workers
        all_experiences = []
        for buffer in self.experience_buffers:
            all_experiences.extend(list(buffer))
        
        if len(all_experiences) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(all_experiences, batch_size)
        
        # Calculate loss
        loss = self._calculate_loss(self.central_policy, batch)
        
        # Update central policy
        self.central_optimizer.zero_grad()
        loss.backward()
        self.central_optimizer.step()
        
        # Synchronize worker networks with central policy
        self._synchronize_workers()
    
    def _synchronize_workers(self):
        """Synchronize worker networks with central policy"""
        for worker_net in self.worker_networks:
            worker_net.load_state_dict(self.central_policy.state_dict())
    
    def _calculate_loss(self, network: nn.Module, experiences: List[Dict[str, Any]]) -> torch.Tensor:
        """Calculate policy gradient loss
        
        Args:
            network: Policy network
            experiences: List of experiences
            
        Returns:
            Loss tensor
        """
        states = torch.FloatTensor([exp['state'] for exp in experiences])
        actions = torch.LongTensor([exp['action'] for exp in experiences])
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences])
        
        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        logits = network(states)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        loss = -(selected_log_probs * returns).mean()
        
        return loss
    
    def train_distributed(self, env, num_episodes: int = 1000, 
                         update_frequency: int = 10, print_every: int = 100):
        """Train using distributed system
        
        Args:
            env: Environment
            num_episodes: Number of episodes
            update_frequency: Update frequency
            print_every: Print frequency
        """
        for episode in range(num_episodes):
            # Collect experiences from all workers
            for worker_id in range(self.num_workers):
                self.collect_experiences(worker_id, env, 1)
            
            # Update workers
            if episode % update_frequency == 0:
                for worker_id in range(self.num_workers):
                    self.update_worker(worker_id)
                
                # Update central policy
                self.update_central_policy()
            
            # Print progress
            if (episode + 1) % print_every == 0:
                avg_rewards = [np.mean(rewards[-print_every:]) 
                              for rewards in self.worker_rewards]
                print(f"Episode {episode + 1:4d} | "
                      f"Worker Avg Rewards: {[f'{r:.2f}' for r in avg_rewards]}")


def create_multi_agent_system(num_agents: int, state_size: int, action_size: int, 
                             **kwargs) -> MultiAgentPolicyGradient:
    """Create multi-agent system
    
    Args:
        num_agents: Number of agents
        state_size: State space dimension
        action_size: Action space dimension
        **kwargs: Additional arguments
        
    Returns:
        Multi-agent system
    """
    return MultiAgentPolicyGradient(num_agents, state_size, action_size, **kwargs)


def create_meta_learning_agent(state_size: int, action_size: int, **kwargs) -> MetaLearningAgent:
    """Create meta-learning agent
    
    Args:
        state_size: State space dimension
        action_size: Action space dimension
        **kwargs: Additional arguments
        
    Returns:
        Meta-learning agent
    """
    return MetaLearningAgent(state_size, action_size, **kwargs)


def create_transfer_learning_agent(source_state_size: int, source_action_size: int,
                                 target_state_size: int, target_action_size: int,
                                 **kwargs) -> TransferLearningAgent:
    """Create transfer learning agent
    
    Args:
        source_state_size: Source task state size
        source_action_size: Source task action size
        target_state_size: Target task state size
        target_action_size: Target task action size
        **kwargs: Additional arguments
        
    Returns:
        Transfer learning agent
    """
    return TransferLearningAgent(source_state_size, source_action_size,
                               target_state_size, target_action_size, **kwargs)


def create_distributed_system(num_workers: int, state_size: int, action_size: int,
                            **kwargs) -> DistributedTrainingSystem:
    """Create distributed training system
    
    Args:
        num_workers: Number of workers
        state_size: State space dimension
        action_size: Action space dimension
        **kwargs: Additional arguments
        
    Returns:
        Distributed training system
    """
    return DistributedTrainingSystem(num_workers, state_size, action_size, **kwargs)

