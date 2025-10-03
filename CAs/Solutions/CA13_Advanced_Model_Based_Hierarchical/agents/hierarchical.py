import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class OptionsCriticAgent:
    """Options-Critic agent for hierarchical RL"""
    
    def __init__(self, state_dim, action_dim, num_options=4, hidden_dim=128, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Option policies π(a|s,ω)
        self.option_policies = nn.ModuleList([
            nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_options)
        ]).to(self.device)
        
        # Termination functions β(s,ω)
        self.termination_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_options)
        ]).to(self.device)
        
        # Intra-option Q-functions Q_U(s,ω)
        self.intra_option_q = nn.ModuleList([
                nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_options)
        ]).to(self.device)
        
        # Option-value function Q_Ω(s,ω)
        self.option_value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        ).to(self.device)
        
        # Optimizers
        self.optimizers = {
            'policies': optim.Adam([p for policy in self.option_policies for p in policy.parameters()], lr=lr),
            'termination': optim.Adam([p for term in self.termination_functions for p in term.parameters()], lr=lr),
            'intra_q': optim.Adam([p for q in self.intra_option_q for p in q.parameters()], lr=lr),
            'option_value': optim.Adam(self.option_value.parameters(), lr=lr)
        }
        
        # State tracking
        self.current_option = 0
        self.option_step_count = 0
        self.option_usage = np.zeros(num_options)
        
    def act(self, state, epsilon=0.1):
        """Select action using current option"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Check if current option should terminate
        if self.should_terminate(state):
            self.current_option = self.select_option(state_tensor, epsilon)
            self.option_step_count = 0
            
        # Select action within current option
        with torch.no_grad():
            logits = self.option_policies[self.current_option](state_tensor)
            if random.random() < epsilon:
                action = random.randint(0, self.action_dim - 1)
            else:
                action = logits.argmax().item()
        
        self.option_step_count += 1
        self.option_usage[self.current_option] += 1
        
        return action, self.current_option
    
    def select_option(self, state_tensor, epsilon=0.1):
        """Select new option based on option-value function"""
        with torch.no_grad():
            option_values = self.option_value(state_tensor)
            if random.random() < epsilon:
                return random.randint(0, self.num_options - 1)
            else:
                return option_values.argmax().item()
    
    def should_terminate(self, state):
        """Check if current option should terminate"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            termination_prob = self.termination_functions[self.current_option](state_tensor).item()
            return random.random() < termination_prob
    
    def update(self, experiences):
        """Update all components of the options-critic architecture"""
        if not experiences:
            return
            
        # Convert experiences to tensors
        states = torch.FloatTensor([e['state'] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e['action'] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in experiences]).to(self.device)
        options = torch.LongTensor([e['option'] for e in experiences]).to(self.device)
        terminated = torch.BoolTensor([e.get('terminated', False) for e in experiences]).to(self.device)
        
        # Update intra-option Q-functions
        self.update_intra_option_q(states, actions, rewards, next_states, options, terminated)
        
        # Update option-value function
        self.update_option_value(states, rewards, next_states, options, terminated)
        
        # Update option policies
        self.update_option_policies(states, actions, next_states, options)
        
        # Update termination functions
        self.update_termination_functions(states, next_states, options)
    
    def update_intra_option_q(self, states, actions, rewards, next_states, options, terminated):
        """Update intra-option Q-functions"""
        batch_size = states.size(0)
        
        for i in range(batch_size):
            option = options[i].item()
            state = states[i:i+1]
            next_state = next_states[i:i+1]
            reward = rewards[i]
            term = terminated[i]
            
            # Current Q-value
            current_q = self.intra_option_q[option](state)
            
            # Target Q-value
            if term:
                # Option terminated, use option-value function
                with torch.no_grad():
                    target_q = reward + self.option_value(next_state).max()
            else:
                # Option continues
                with torch.no_grad():
                    target_q = reward + self.intra_option_q[option](next_state)
            
            # Loss
            loss = F.mse_loss(current_q, target_q)
            
            self.optimizers['intra_q'].zero_grad()
            loss.backward()
            self.optimizers['intra_q'].step()
    
    def update_option_value(self, states, rewards, next_states, options, terminated):
        """Update option-value function"""
        batch_size = states.size(0)
        
        for i in range(batch_size):
            option = options[i].item()
            state = states[i:i+1]
            next_state = next_states[i:i+1]
            reward = rewards[i]
            term = terminated[i]
            
            # Current option value
            current_q = self.option_value(state)[0, option]
            
            # Target value
            if term:
                # Option terminated
                with torch.no_grad():
                    target_q = reward + self.option_value(next_state).max()
            else:
                # Option continues
                with torch.no_grad():
                    target_q = reward + self.option_value(next_state)[0, option]
            
            # Loss
            loss = F.mse_loss(current_q, target_q)
            
            self.optimizers['option_value'].zero_grad()
            loss.backward()
            self.optimizers['option_value'].step()
    
    def update_option_policies(self, states, actions, next_states, options):
        """Update option policies using policy gradient"""
        batch_size = states.size(0)
        
        for i in range(batch_size):
            option = options[i].item()
            state = states[i:i+1]
            action = actions[i]
            
            # Policy logits
            logits = self.option_policies[option](state)
            log_prob = F.log_softmax(logits, dim=1)[0, action]
            
            # Advantage (simplified - using intra-option Q as advantage)
            with torch.no_grad():
                advantage = self.intra_option_q[option](state).item()
            
            # Policy loss (negative log probability weighted by advantage)
            policy_loss = -log_prob * advantage
            
            self.optimizers['policies'].zero_grad()
            policy_loss.backward()
            self.optimizers['policies'].step()
    
    def update_termination_functions(self, states, next_states, options):
        """Update termination functions"""
        batch_size = states.size(0)
        
        for i in range(batch_size):
            option = options[i].item()
            state = states[i:i+1]
            next_state = next_states[i:i+1]
            
            # Termination probability
            termination_prob = self.termination_functions[option](state)
            
            # Advantage for termination (simplified)
            with torch.no_grad():
                current_value = self.option_value(state)[0, option]
                next_option_value = self.option_value(next_state).max()
                termination_advantage = next_option_value - current_value
            
            # Termination loss
            termination_loss = -termination_prob * termination_advantage
            
            self.optimizers['termination'].zero_grad()
            termination_loss.backward()
            self.optimizers['termination'].step()


class FeudalAgent:
    """Feudal Networks agent for hierarchical RL"""
    
    def __init__(self, state_dim, action_dim, goal_dim=16, hidden_dim=128, lr=1e-3, temporal_horizon=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.temporal_horizon = temporal_horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Manager network: sets goals
        self.manager = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim)
        ).to(self.device)
        
        # Worker network: executes actions based on goals
        self.worker = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(self.device)
        
        # Optimizers
        self.manager_optimizer = optim.Adam(self.manager.parameters(), lr=lr)
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=lr)
        
        # State tracking
        self.current_goal = torch.zeros(goal_dim).to(self.device)
        self.goal_step_count = 0

    def act(self, state, epsilon=0.1):
        """Select action using feudal hierarchy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Manager sets new goal if needed
        if self.goal_step_count >= self.temporal_horizon:
            with torch.no_grad():
                self.current_goal = self.manager(state_tensor).squeeze()
                self.current_goal = F.normalize(self.current_goal, p=2, dim=0)  # Normalize goal
            self.goal_step_count = 0
        
        # Worker selects action based on state and current goal
        goal_expanded = self.current_goal.unsqueeze(0).expand(state_tensor.size(0), -1)
        worker_input = torch.cat([state_tensor, goal_expanded], dim=1)
        
        with torch.no_grad():
            action_logits = self.worker(worker_input)
            if random.random() < epsilon:
                action = random.randint(0, self.action_dim - 1)
            else:
                action = action_logits.argmax().item()

            self.goal_step_count += 1
        return action

    def update(self, experiences):
        """Update manager and worker networks"""
        if not experiences:
            return
            
        # Convert experiences to tensors
        states = torch.FloatTensor([e['state'] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e['action'] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in experiences]).to(self.device)
        
        # Update worker (policy gradient)
        worker_loss = 0
        for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
            # Use current goal for worker update
            goal_expanded = self.current_goal.unsqueeze(0)
            worker_input = torch.cat([state.unsqueeze(0), goal_expanded], dim=1)
            
            logits = self.worker(worker_input)
            log_prob = F.log_softmax(logits, dim=1)[0, action]
            worker_loss += -log_prob * reward  # Simplified policy gradient
        
        worker_loss = worker_loss / len(experiences)
        self.worker_optimizer.zero_grad()
        worker_loss.backward()
        self.worker_optimizer.step()
        
        # Update manager (goal setting)
        manager_loss = 0
        for i, (state, next_state, reward) in enumerate(zip(states, next_states, rewards)):
            # Manager should set goals that lead to high rewards
            goal = self.manager(state.unsqueeze(0)).squeeze()
            goal = F.normalize(goal, p=2, dim=0)
            
            # Intrinsic reward based on goal achievement (simplified)
            state_change = next_state - state
            intrinsic_reward = torch.dot(goal, state_change)
            
            # Manager loss (encourage goals that lead to positive intrinsic rewards)
            manager_loss += -intrinsic_reward
        
        manager_loss = manager_loss / len(experiences)
        self.manager_optimizer.zero_grad()
        manager_loss.backward()
        self.manager_optimizer.step()