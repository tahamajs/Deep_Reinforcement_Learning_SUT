"""
Advanced Hierarchical RL Algorithms

This module contains advanced hierarchical RL implementations including:
- HIRO (Hierarchical Reinforcement Learning with Off-policy Corrections)
- HAC (Hierarchical Actor-Critic) with advanced subgoal generation
- Option-Critic with continuous options
- Feudal Networks with advanced goal decomposition
- Multi-goal hierarchical RL with curriculum learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical, MultivariateNormal
import random
import copy
from collections import deque
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HIROAgent(nn.Module):
    """HIRO: Hierarchical Reinforcement Learning with Off-policy Corrections."""
    
    def __init__(self, state_dim, action_dim, subgoal_dim, hidden_dim=256):
        super(HIROAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.subgoal_dim = subgoal_dim
        
        # High-level policy (manager)
        self.manager = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, subgoal_dim * 2)  # mean and log_std
        )
        
        # High-level critic
        self.manager_critic = nn.Sequential(
            nn.Linear(state_dim + subgoal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Low-level policy (worker)
        self.worker = nn.Sequential(
            nn.Linear(state_dim + subgoal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )
        
        # Low-level critic
        self.worker_critic = nn.Sequential(
            nn.Linear(state_dim + subgoal_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Goal transition model for off-policy corrections
        self.goal_transition_model = nn.Sequential(
            nn.Linear(state_dim + subgoal_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, subgoal_dim)
        )
        
        self.managers = [self.manager, self.manager_critic]
        self.workers = [self.worker, self.worker_critic]
        
        self.manager_optimizer = optim.Adam(self.managers, lr=3e-4)
        self.worker_optimizer = optim.Adam(self.workers, lr=3e-4)
        
        self.gamma = 0.99
        self.tau = 0.005
        self.subgoal_freq = 10  # Subgoal update frequency
    
    def get_subgoal(self, state, deterministic=False):
        """Get subgoal from manager."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        output = self.manager(state)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        
        if deterministic:
            subgoal = mean
        else:
            dist = Normal(mean, std)
            subgoal = dist.sample()
        
        return subgoal
    
    def get_action(self, state, subgoal, deterministic=False):
        """Get action from worker given state and subgoal."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(subgoal.shape) == 1:
            subgoal = subgoal.unsqueeze(0)
        
        input_tensor = torch.cat([state, subgoal], dim=-1)
        output = self.worker(input_tensor)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        
        return action.clamp(-1, 1)
    
    def compute_intrinsic_reward(self, state, subgoal, next_state):
        """Compute intrinsic reward based on subgoal achievement."""
        goal_distance = torch.norm(next_state - subgoal, dim=-1)
        intrinsic_reward = -goal_distance
        return intrinsic_reward
    
    def off_policy_correction(self, states, actions, subgoals, next_states):
        """Perform off-policy correction for subgoals."""
        corrected_subgoals = []
        
        for i in range(len(states)):
            # Predict next subgoal using goal transition model
            input_tensor = torch.cat([states[i], subgoals[i], actions[i]], dim=-1)
            predicted_subgoal = self.goal_transition_model(input_tensor)
            
            # Use predicted subgoal if it's closer to achieved state
            original_distance = torch.norm(next_states[i] - subgoals[i])
            corrected_distance = torch.norm(next_states[i] - predicted_subgoal)
            
            if corrected_distance < original_distance:
                corrected_subgoals.append(predicted_subgoal)
            else:
                corrected_subgoals.append(subgoals[i])
        
        return torch.stack(corrected_subgoals)
    
    def update_manager(self, states, subgoals, rewards, next_states, dones):
        """Update manager policy."""
        # Compute manager value
        manager_input = torch.cat([states, subgoals], dim=-1)
        manager_values = self.manager_critic(manager_input)
        
        with torch.no_grad():
            next_subgoals = self.get_subgoal(next_states)
            next_manager_input = torch.cat([next_states, next_subgoals], dim=-1)
            next_manager_values = self.manager_critic(next_manager_input)
            target_values = rewards + self.gamma * next_manager_values * (1 - dones)
        
        manager_loss = F.mse_loss(manager_values, target_values)
        
        self.manager_optimizer.zero_grad()
        manager_loss.backward()
        self.manager_optimizer.step()
    
    def update_worker(self, states, subgoals, actions, rewards, next_states, dones):
        """Update worker policy."""
        # Compute worker value
        worker_input = torch.cat([states, subgoals, actions], dim=-1)
        worker_values = self.worker_critic(worker_input)
        
        with torch.no_grad():
            next_actions = self.get_action(next_states, subgoals)
            next_worker_input = torch.cat([next_states, subgoals, next_actions], dim=-1)
            next_worker_values = self.worker_critic(next_worker_input)
            target_values = rewards + self.gamma * next_worker_values * (1 - dones)
        
        worker_loss = F.mse_loss(worker_values, target_values)
        
        self.worker_optimizer.zero_grad()
        worker_loss.backward()
        self.worker_optimizer.step()


class AdvancedHAC(nn.Module):
    """Advanced Hierarchical Actor-Critic with curriculum learning."""
    
    def __init__(self, state_dim, action_dim, num_levels=3, hidden_dim=256):
        super(AdvancedHAC, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_levels = num_levels
        
        # Multi-level policies
        self.policies = nn.ModuleList()
        self.critics = nn.ModuleList()
        
        for level in range(num_levels):
            if level == num_levels - 1:  # Lowest level
                policy = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim * 2)
                )
            else:  # Higher levels
                policy = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, state_dim * 2)  # Subgoal
                )
            
            critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            self.policies.append(policy)
            self.critics.append(critic)
        
        # Curriculum learning components
        self.curriculum_scheduler = CurriculumScheduler()
        self.subgoal_generator = SubgoalGenerator(state_dim, hidden_dim)
        
        self.optimizers = []
        for level in range(num_levels):
            optimizer = optim.Adam(
                list(self.policies[level].parameters()) + list(self.critics[level].parameters()),
                lr=3e-4
            )
            self.optimizers.append(optimizer)
        
        self.gamma = 0.99
        self.tau = 0.005
    
    def get_action(self, state, level, deterministic=False):
        """Get action/subgoal from specified level."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        output = self.policies[level](state)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        
        return action.clamp(-1, 1)
    
    def hierarchical_forward(self, state):
        """Complete hierarchical forward pass."""
        subgoals = []
        actions = []
        
        current_state = state
        for level in range(self.num_levels - 1):
            subgoal = self.get_action(current_state, level)
            subgoals.append(subgoal)
            current_state = subgoal
        
        # Get final action from lowest level
        final_action = self.get_action(state, self.num_levels - 1)
        actions.append(final_action)
        
        return subgoals, actions
    
    def update_level(self, level, states, actions, rewards, next_states, dones):
        """Update specific level."""
        # Compute value
        values = self.critics[level](states)
        
        with torch.no_grad():
            next_values = self.critics[level](next_states)
            target_values = rewards + self.gamma * next_values * (1 - dones)
        
        # Update critic
        critic_loss = F.mse_loss(values, target_values)
        
        # Update policy
        if level == self.num_levels - 1:  # Lowest level
            action_log_probs = self.get_action_log_prob(states, actions, level)
            policy_loss = -(action_log_probs * (target_values - values)).mean()
        else:  # Higher levels
            subgoal_log_probs = self.get_action_log_prob(states, actions, level)
            policy_loss = -(subgoal_log_probs * (target_values - values)).mean()
        
        total_loss = critic_loss + policy_loss
        
        self.optimizers[level].zero_grad()
        total_loss.backward()
        self.optimizers[level].step()
    
    def get_action_log_prob(self, states, actions, level):
        """Get log probability of actions."""
        output = self.policies[level](states)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        return log_probs


class CurriculumScheduler:
    """Curriculum learning scheduler for hierarchical RL."""
    
    def __init__(self):
        self.difficulty_level = 0.0
        self.max_difficulty = 1.0
        self.curriculum_rate = 0.01
    
    def update_difficulty(self, success_rate):
        """Update curriculum difficulty based on success rate."""
        if success_rate > 0.8:
            self.difficulty_level = min(self.difficulty_level + self.curriculum_rate, self.max_difficulty)
        elif success_rate < 0.3:
            self.difficulty_level = max(self.difficulty_level - self.curriculum_rate, 0.0)
    
    def get_task_difficulty(self):
        """Get current task difficulty."""
        return self.difficulty_level


class SubgoalGenerator(nn.Module):
    """Advanced subgoal generation with diversity."""
    
    def __init__(self, state_dim, hidden_dim=256):
        super(SubgoalGenerator, self).__init__()
        self.state_dim = state_dim
        
        self.generator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2)  # mean and log_std
        )
        
        self.diversity_loss_weight = 0.1
    
    def generate_subgoals(self, state, num_subgoals=5):
        """Generate diverse subgoals."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        output = self.generator(state)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        
        subgoals = []
        for _ in range(num_subgoals):
            dist = Normal(mean, std)
            subgoal = dist.sample()
            subgoals.append(subgoal)
        
        return torch.stack(subgoals)
    
    def compute_diversity_loss(self, subgoals):
        """Compute diversity loss to encourage different subgoals."""
        if len(subgoals) < 2:
            return torch.tensor(0.0)
        
        # Compute pairwise distances
        distances = []
        for i in range(len(subgoals)):
            for j in range(i + 1, len(subgoals)):
                dist = torch.norm(subgoals[i] - subgoals[j])
                distances.append(dist)
        
        # Diversity loss: encourage large distances
        diversity_loss = -torch.mean(torch.stack(distances))
        return diversity_loss


class ContinuousOptionCritic(nn.Module):
    """Option-Critic with continuous options."""
    
    def __init__(self, state_dim, action_dim, num_options=4, hidden_dim=256):
        super(ContinuousOptionCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        
        # Option policies
        self.option_policies = nn.ModuleList()
        for _ in range(num_options):
            policy = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
            )
            self.option_policies.append(policy)
        
        # Option termination functions
        self.termination_functions = nn.ModuleList()
        for _ in range(num_options):
            termination = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.termination_functions.append(termination)
        
        # Option selection policy
        self.option_selection = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        
        # Option value functions
        self.option_values = nn.ModuleList()
        for _ in range(num_options):
            value = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.option_values.append(value)
        
        # Master value function
        self.master_value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizers = []
        for i in range(num_options):
            optimizer = optim.Adam(
                list(self.option_policies[i].parameters()) + 
                list(self.termination_functions[i].parameters()) +
                list(self.option_values[i].parameters()),
                lr=3e-4
            )
            self.optimizers.append(optimizer)
        
        self.master_optimizer = optim.Adam(
            list(self.option_selection.parameters()) + list(self.master_value.parameters()),
            lr=3e-4
        )
        
        self.gamma = 0.99
        self.beta = 0.01  # Termination penalty
    
    def select_option(self, state):
        """Select option based on state."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        logits = self.option_selection(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        option = dist.sample()
        
        return option.item(), probs
    
    def get_action(self, state, option, deterministic=False):
        """Get action from selected option."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        output = self.option_policies[option](state)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        
        return action.clamp(-1, 1)
    
    def should_terminate(self, state, option):
        """Check if option should terminate."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        termination_prob = self.termination_functions[option](state)
        return torch.bernoulli(termination_prob).bool()
    
    def get_option_value(self, state, option):
        """Get value of option in state."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        return self.option_values[option](state)
    
    def get_master_value(self, state):
        """Get master value function."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        return self.master_value(state)
    
    def update_option(self, option, states, actions, rewards, next_states, dones, 
                     option_continues, next_options):
        """Update specific option."""
        # Compute option value
        option_values = self.get_option_value(states, option)
        
        with torch.no_grad():
            next_option_values = self.get_option_value(next_states, option)
            master_values = self.get_master_value(next_states)
            
            # Option value target
            option_targets = rewards + self.gamma * (
                option_continues * next_option_values + 
                (1 - option_continues) * master_values
            ) * (1 - dones)
        
        # Update option value
        value_loss = F.mse_loss(option_values, option_targets)
        
        # Update option policy
        action_log_probs = self.get_action_log_prob(states, actions, option)
        policy_loss = -(action_log_probs * (option_targets - option_values)).mean()
        
        # Update termination function
        termination_probs = self.termination_functions[option](states)
        termination_loss = self.beta * termination_probs.mean()
        
        total_loss = value_loss + policy_loss + termination_loss
        
        self.optimizers[option].zero_grad()
        total_loss.backward()
        self.optimizers[option].step()
    
    def get_action_log_prob(self, states, actions, option):
        """Get log probability of actions for option."""
        output = self.option_policies[option](states)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        return log_probs
