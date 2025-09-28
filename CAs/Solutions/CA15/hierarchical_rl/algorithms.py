"""
Hierarchical Reinforcement Learning Components

This module contains implementations of hierarchical RL algorithms including:
- Options framework
- Hierarchical Actor-Critic
- Goal-conditioned RL with Hindsight Experience Replay
- Feudal Networks architecture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Option:
    """Implementation of the Options framework."""

    def __init__(self, policy, initiation_set=None, termination_condition=None, name="option"):
        self.policy = policy
        self.initiation_set = initiation_set
        self.termination_condition = termination_condition
        self.name = name
        self.active_steps = 0
        self.max_steps = 20  # Default timeout

    def can_initiate(self, state):
        """Check if option can be initiated in given state."""
        if self.initiation_set is None:
            return True
        return self.initiation_set(state)

    def should_terminate(self, state):
        """Check if option should terminate in given state."""
        # Timeout termination
        if self.active_steps >= self.max_steps:
            return True

        # Custom termination condition
        if self.termination_condition is not None:
            return self.termination_condition(state)

        return False

    def get_action(self, state):
        """Get action from option policy."""
        self.active_steps += 1
        return self.policy(state)

    def reset(self):
        """Reset option state."""
        self.active_steps = 0


class HierarchicalActorCritic(nn.Module):
    """Hierarchical Actor-Critic with multiple levels."""

    def __init__(self, state_dim, action_dim, num_levels=3, hidden_dim=256):
        super(HierarchicalActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_levels = num_levels

        # High-level (meta) controllers
        self.meta_controllers = nn.ModuleList()
        self.meta_critics = nn.ModuleList()

        # Low-level controllers
        self.low_controllers = nn.ModuleList()
        self.low_critics = nn.ModuleList()

        for level in range(num_levels - 1):
            # Meta controller generates subgoals
            meta_controller = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)  # Subgoal in state space
            )

            # Meta critic evaluates state-goal pairs
            meta_critic = nn.Sequential(
                nn.Linear(state_dim * 2, hidden_dim),  # state + goal
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

            self.meta_controllers.append(meta_controller)
            self.meta_critics.append(meta_critic)

        # Lowest level controller outputs actions
        low_controller = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # state + subgoal
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        low_critic = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.low_controllers.append(low_controller)
        self.low_critics.append(low_critic)

    def forward_meta(self, state, level):
        """Forward pass for meta controller at given level."""
        if level >= len(self.meta_controllers):
            raise ValueError(f"Level {level} exceeds number of meta controllers")

        subgoal = self.meta_controllers[level](state)
        state_goal = torch.cat([state, subgoal], dim=-1)
        value = self.meta_critics[level](state_goal)

        return subgoal, value

    def forward_low(self, state, subgoal):
        """Forward pass for low-level controller."""
        state_subgoal = torch.cat([state, subgoal], dim=-1)

        action_logits = self.low_controllers[0](state_subgoal)
        value = self.low_critics[0](state_subgoal)

        return action_logits, value

    def hierarchical_forward(self, state):
        """Complete hierarchical forward pass."""
        current_goal = state  # Start with state as initial goal
        subgoals = []
        values = []

        # Generate subgoals from top to bottom
        for level in range(len(self.meta_controllers)):
            subgoal, value = self.forward_meta(state, level)
            subgoals.append(subgoal)
            values.append(value)
            current_goal = subgoal

        # Generate action from lowest level
        action_logits, low_value = self.forward_low(state, current_goal)
        values.append(low_value)

        return {
            'subgoals': subgoals,
            'action_logits': action_logits,
            'values': values
        }


class GoalConditionedAgent:
    """Goal-Conditioned RL with Hindsight Experience Replay."""

    def __init__(self, state_dim, action_dim, goal_dim=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim or state_dim

        # Policy conditioned on state and goal
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + self.goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(device)

        # Value function conditioned on state and goal
        self.value_net = nn.Sequential(
            nn.Linear(state_dim + self.goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)

        # Experience buffer for HER
        self.buffer = deque(maxlen=100000)
        self.her_ratio = 0.8  # Proportion of HER samples

        # Goal generation
        self.goal_strategy = "future"  # "future", "episode", "random"

        # Statistics
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'goal_achievements': [],
            'intrinsic_rewards': []
        }

    def goal_distance(self, achieved_goal, desired_goal):
        """Compute distance between achieved and desired goals."""
        return torch.norm(achieved_goal - desired_goal, dim=-1)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """Compute reward based on goal achievement."""
        distance = self.goal_distance(achieved_goal, desired_goal)
        # Sparse reward: +1 if goal achieved, -1 otherwise
        threshold = 0.1
        reward = (distance < threshold).float() * 2 - 1
        return reward

    def get_action(self, state, goal, deterministic=False):
        """Get action conditioned on state and goal."""
        state_tensor = torch.FloatTensor(state).to(device)
        goal_tensor = torch.FloatTensor(goal).to(device)

        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
            goal_tensor = goal_tensor.unsqueeze(0)

        state_goal = torch.cat([state_tensor, goal_tensor], dim=-1)

        with torch.no_grad():
            action_logits = self.policy_net(state_goal)

            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).squeeze()

        return action.cpu().numpy() if len(action.shape) > 0 else action.item()

    def store_episode(self, episode_states, episode_actions, episode_goals, final_achieved_goal):
        """Store episode with HER augmentation."""
        episode_length = len(episode_states)

        # Store original episode
        for t in range(episode_length - 1):
            achieved_goal = episode_states[t+1]  # Use next state as achieved goal
            reward = self.compute_reward(
                torch.FloatTensor(achieved_goal),
                torch.FloatTensor(episode_goals[t])
            ).item()

            self.buffer.append({
                'state': episode_states[t],
                'action': episode_actions[t],
                'reward': reward,
                'next_state': episode_states[t+1],
                'goal': episode_goals[t],
                'achieved_goal': achieved_goal
            })

        # HER: Generate additional samples with different goals
        for t in range(episode_length - 1):
            if np.random.random() < self.her_ratio:
                # Sample future state as goal
                if self.goal_strategy == "future" and t < episode_length - 2:
                    future_idx = np.random.randint(t + 1, episode_length)
                    her_goal = episode_states[future_idx]
                elif self.goal_strategy == "episode":
                    her_goal = final_achieved_goal
                else:  # random
                    her_goal = np.random.randn(self.goal_dim)

                achieved_goal = episode_states[t+1]
                her_reward = self.compute_reward(
                    torch.FloatTensor(achieved_goal),
                    torch.FloatTensor(her_goal)
                ).item()

                self.buffer.append({
                    'state': episode_states[t],
                    'action': episode_actions[t],
                    'reward': her_reward,
                    'next_state': episode_states[t+1],
                    'goal': her_goal,
                    'achieved_goal': achieved_goal
                })

    def train_step(self, batch_size=64):
        """Training step with goal-conditioned experience."""
        if len(self.buffer) < batch_size:
            return 0, 0

        # Sample batch
        batch = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor([exp['state'] for exp in batch]).to(device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(device)
        goals = torch.FloatTensor([exp['goal'] for exp in batch]).to(device)

        # Prepare inputs
        state_goal = torch.cat([states, goals], dim=-1)
        next_state_goal = torch.cat([next_states, goals], dim=-1)

        # Value function loss
        current_values = self.value_net(state_goal).squeeze()
        with torch.no_grad():
            next_values = self.value_net(next_state_goal).squeeze()
            target_values = rewards + 0.99 * next_values

        value_loss = F.mse_loss(current_values, target_values)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Policy loss (actor-critic)
        action_logits = self.policy_net(state_goal)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            advantages = target_values - current_values

        policy_loss = -(selected_log_probs * advantages).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update statistics
        self.training_stats['policy_losses'].append(policy_loss.item())
        self.training_stats['value_losses'].append(value_loss.item())

        # Track goal achievements
        goal_achieved = (rewards > 0).float().mean().item()
        self.training_stats['goal_achievements'].append(goal_achieved)

        return policy_loss.item(), value_loss.item()


class FeudalNetwork(nn.Module):
    """Feudal Networks for Hierarchical RL."""

    def __init__(self, state_dim, action_dim, goal_dim=64, hidden_dim=256):
        super(FeudalNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        # Shared perception module
        self.perception = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Manager network (sets goals)
        self.manager = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim)
        )

        # Worker network (executes actions)
        self.worker = nn.Sequential(
            nn.Linear(hidden_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value functions
        self.manager_critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.worker_critic = nn.Sequential(
            nn.Linear(hidden_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Intrinsic curiosity module
        self.curiosity_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, previous_goal=None):
        """Forward pass through feudal network."""
        # Shared perception
        perception = self.perception(state)

        # Manager generates goal
        goal = self.manager(perception)
        goal = F.normalize(goal, p=2, dim=-1)  # Normalize goal vector

        # Worker takes action conditioned on perception and goal
        if previous_goal is not None:
            # Use previous goal for temporal consistency
            worker_input = torch.cat([perception, previous_goal], dim=-1)
        else:
            worker_input = torch.cat([perception, goal], dim=-1)

        action_logits = self.worker(worker_input)

        # Value functions
        manager_value = self.manager_critic(perception)
        worker_value = self.worker_critic(worker_input)

        return {
            'goal': goal,
            'action_logits': action_logits,
            'manager_value': manager_value,
            'worker_value': worker_value,
            'perception': perception
        }

    def compute_intrinsic_reward(self, current_perception, next_perception, goal):
        """Compute intrinsic reward based on goal achievement."""
        # Cosine similarity between goal and state transition
        state_diff = next_perception - current_perception
        intrinsic_reward = F.cosine_similarity(goal, state_diff, dim=-1)
        return intrinsic_reward