import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from collections import deque
from typing import Tuple, List, Any, Optional, Union
from src.model import DynamicsModel, ModelEnsemble, QNetwork, Actor, Critic, GoalConditionedActor, GoalConditionedCritic, FeudalManager, FeudalWorker
from src.losses import dynamics_model_loss, q_function_loss, intrinsic_reward_loss, policy_gradient_loss, actor_critic_loss
from src.utils import ReplayBuffer, PrioritizedReplayBuffer, RunningStats, to_tensor, get_device, set_seed, env_reset, env_step
from src.config import Config, DynamicsModelConfig, ManagerConfig, WorkerConfig
import torch.nn.functional as F
import math


class DynaQAgent:
    """Dyna-Q agent combining model-free and model-based learning."""

    def __init__(
        self, config: Config,
        state_dim: int,
        action_dim: int,
    ):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = QNetwork(state_dim, action_dim, config.worker.hidden_dim).to(get_device())
        self.target_q_network = QNetwork(state_dim, action_dim, config.worker.hidden_dim).to(get_device())
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config.worker.learning_rate)

        self.dynamics_model = DynamicsModel(state_dim, action_dim, config.dynamics_model.hidden_dim).to(get_device())
        self.model_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=config.dynamics_model.learning_rate)
        self.model_criterion_state = nn.MSELoss()
        self.model_criterion_reward = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(config.worker.replay_buffer_size)
        self.model_buffer = ReplayBuffer(config.worker.replay_buffer_size) # For model training

        self.gamma = config.worker.discount_factor
        self.epsilon_start = config.worker.epsilon_start
        self.epsilon_end = config.worker.epsilon_end
        self.epsilon_decay = config.worker.epsilon_decay
        self.epsilon = self.epsilon_start
        self.planning_steps = config.dynamics_model.planning_horizon
        self.training_steps = 0

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using an epsilon-greedy policy."""
        if training:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           math.exp(-1. * self.training_steps / self.epsilon_decay)

        if random.random() < self.epsilon and training:
            return random.randrange(self.action_dim)
        else:
            self.q_network.eval()
            with torch.no_grad():
                state_tensor = to_tensor(state)
                q_values = self.q_network(state_tensor)
            self.q_network.train()
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store transition in replay buffer and model buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.model_buffer.push(state, action, reward, next_state, done)

    def update_q_function(self):
        """Update the Q-network using sampled transitions."""
        if len(self.replay_buffer) < self.config.general.batch_size:
            return 0.0

        experiences = self.replay_buffer.sample(self.config.general.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = to_tensor(np.array(states))
        actions = to_tensor(np.array(actions), dtype=torch.long)
        rewards = to_tensor(np.array(rewards).reshape(-1, 1))
        next_states = to_tensor(np.array(next_states))
        dones = to_tensor(np.array(dones).reshape(-1, 1), dtype=torch.float32)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = q_function_loss(current_q_values, target_q_values)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return loss.item()

    def update_model(self):
        """Update the dynamics model using sampled transitions."""
        if len(self.model_buffer) < self.config.general.batch_size:
            return 0.0
        
        experiences = self.model_buffer.sample(self.config.general.batch_size)
        states, actions, rewards, next_states, _ = zip(*experiences)

        states = to_tensor(np.array(states))
        actions = to_tensor(np.array(actions), dtype=torch.long)
        rewards = to_tensor(np.array(rewards).reshape(-1, 1))
        next_states = to_tensor(np.array(next_states))
        
        # Convert actions to one-hot for dynamics model if necessary, or just use as is for continuous actions
        # For discrete actions, we might need to one-hot encode them for the dynamics model input
        # Assuming discrete actions are simply integers for now, and the model can handle it.
        # If action_dim is large, one-hot might be better. For small discrete, embedding might be better.
        # For now, let's just pass the action directly to the dynamics model.
        # If action is discrete, convert to one-hot for dynamics model input
        if self.action_dim > 1 and actions.dtype == torch.long: # Assuming discrete action
            actions_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
            predicted_next_states, predicted_rewards = self.dynamics_model(states, actions_one_hot)
        else:
            predicted_next_states, predicted_rewards = self.dynamics_model(states, actions.float())

        loss_state = self.model_criterion_state(predicted_next_states, next_states)
        loss_reward = self.model_criterion_reward(predicted_rewards, rewards)
        loss = loss_state + loss_reward

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return loss.item()


    def planning_step(self):
        """Perform planning steps using the learned dynamics model."""
        if len(self.replay_buffer) < self.config.general.batch_size:
            return
        
        for _ in range(self.planning_steps):
            # 1. Sample a previously observed state-action pair (s, a) from memory.
            # For simplicity, sample from replay_buffer, but for Dyna-Q it could be from a separate model_experience buffer
            state, action, _, _, _ = self.replay_buffer.sample(1)[0]
            state_tensor = to_tensor(state).unsqueeze(0)
            action_tensor = to_tensor(action, dtype=torch.long).unsqueeze(0)

            # Convert action to one-hot for dynamics model if discrete
            if self.action_dim > 1 and action_tensor.dtype == torch.long:
                action_input = F.one_hot(action_tensor, num_classes=self.action_dim).float()
            else:
                action_input = action_tensor.float()

            # 2. Predict the next state s' and reward r using the dynamics model M: s', r = M(s, a).
            with torch.no_grad():
                predicted_next_state, predicted_reward = self.dynamics_model(state_tensor, action_input)
            
            # Convert back to numpy for consistent experience storage if needed, or directly use tensors
            predicted_next_state_np = predicted_next_state.squeeze(0).cpu().numpy()
            predicted_reward_np = predicted_reward.item()
            
            # 3. Update the Q-function using this simulated experience:
            # (s, a, r, s') is a simulated transition
            # Store this in the replay buffer and then update Q-function
            # Note: Dyna-Q typically updates Q directly, but here we add to buffer and use batch update for DQN.
            self.replay_buffer.push(state, action, predicted_reward_np, predicted_next_state_np, False) # Assume not done for planning steps
            self.update_q_function()

    def update_target_network(self):
        """Update the target Q-network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())


class DQNAgent:
    """Deep Q-Network agent for baseline comparison."""

    def __init__(
        self, config: Config,
        state_dim: int,
        action_dim: int,
    ):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = QNetwork(state_dim, action_dim, config.worker.hidden_dim).to(get_device())
        self.target_q_network = QNetwork(state_dim, action_dim, config.worker.hidden_dim).to(get_device())
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.worker.learning_rate)

        self.replay_buffer = ReplayBuffer(config.worker.replay_buffer_size)

        self.gamma = config.worker.discount_factor
        self.epsilon_start = config.worker.epsilon_start
        self.epsilon_end = config.worker.epsilon_end
        self.epsilon_decay = config.worker.epsilon_decay
        self.epsilon = self.epsilon_start
        self.training_steps = 0

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using an epsilon-greedy policy."""
        if training:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           math.exp(-1. * self.training_steps / self.epsilon_decay)

        if random.random() < self.epsilon and training:
            return random.randrange(self.action_dim)
        else:
            self.q_network.eval()
            with torch.no_grad():
                state_tensor = to_tensor(state)
                q_values = self.q_network(state_tensor)
            self.q_network.train()
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """Update the Q-network using sampled transitions."""
        if len(self.replay_buffer) < self.config.general.batch_size:
            return 0.0

        experiences = self.replay_buffer.sample(self.config.general.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = to_tensor(np.array(states))
        actions = to_tensor(np.array(actions), dtype=torch.long)
        rewards = to_tensor(np.array(rewards).reshape(-1, 1))
        next_states = to_tensor(np.array(next_states))
        dones = to_tensor(np.array(dones).reshape(-1, 1), dtype=torch.float32)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = q_function_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update the target Q-network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())


class HierarchicalActorCritic:
    """Hierarchical Actor-Critic with multiple levels."""

    def __init__(
        self, config: Config,
        state_dim: int,
        action_dim: int,
        subgoal_dim: int,
        num_levels: int = 2, # Manager and Worker
    ):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.subgoal_dim = subgoal_dim
        self.num_levels = num_levels # Assuming 2 levels for now: Manager and Worker

        # Manager (high-level)
        self.manager_actor = FeudalManager(state_dim, subgoal_dim, config.manager.hidden_dim).to(get_device())
        self.manager_critic = Critic(state_dim, config.manager.hidden_dim).to(get_device())
        self.manager_optimizer = optim.Adam(list(self.manager_actor.parameters()) + list(self.manager_critic.parameters()), lr=config.manager.learning_rate)
        self.manager_buffer = ReplayBuffer(config.worker.replay_buffer_size) # Manager's experience buffer

        # Worker (low-level)
        self.worker_actor = FeudalWorker(state_dim, action_dim, subgoal_dim, config.worker.hidden_dim).to(get_device())
        self.worker_critic = GoalConditionedCritic(state_dim, subgoal_dim, config.worker.hidden_dim).to(get_device())
        self.worker_optimizer = optim.Adam(list(self.worker_actor.parameters()) + list(self.worker_critic.parameters()), lr=config.worker.learning_rate)
        self.worker_buffer = ReplayBuffer(config.worker.replay_buffer_size) # Worker's experience buffer

        self.gamma_manager = config.manager.discount_factor
        self.gamma_worker = config.worker.discount_factor
        self.update_frequency_worker_steps = config.manager.update_frequency_worker_steps
        self.worker_steps_counter = 0

        # Dynamics model for model-based enhancements
        self.dynamics_model = DynamicsModel(state_dim + action_dim, state_dim, config.dynamics_model.hidden_dim).to(get_device())
        self.model_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=config.dynamics_model.learning_rate)
        self.model_criterion = nn.MSELoss()

    def select_action(self, state: np.ndarray, level: int = 0, subgoal: Optional[np.ndarray] = None) -> Union[int, np.ndarray]:
        """Select an action based on the policy at the specified level."""
        state_tensor = to_tensor(state).unsqueeze(0)

        if level == 0: # Manager
            self.manager_actor.eval()
            with torch.no_grad():
                subgoal_logits = self.manager_actor(state_tensor) # Assuming subgoal is a vector
                # For discrete subgoals, apply softmax and sample. For continuous, directly use output.
                # For now, let's assume continuous subgoals outputted directly.
                subgoal = subgoal_logits.squeeze(0).cpu().numpy()
            self.manager_actor.train()
            return subgoal
        elif level == 1: # Worker
            assert subgoal is not None, "Worker requires a subgoal."
            self.worker_actor.eval()
            with torch.no_grad():
                subgoal_tensor = to_tensor(subgoal).unsqueeze(0)
                action_logits = self.worker_actor(state_tensor, subgoal_tensor)
                # Assuming discrete actions for now, apply softmax and sample
                action = action_logits.argmax().item()
            self.worker_actor.train()
            return action
        else:
            raise ValueError("Invalid level specified.")

    def update_dynamics_model(self, states, actions, next_states, rewards):
        """Update the dynamics model."""
        states = to_tensor(np.array(states))
        actions = to_tensor(np.array(actions), dtype=torch.long)
        rewards = to_tensor(np.array(rewards).reshape(-1, 1))
        next_states = to_tensor(np.array(next_states))

        # For discrete actions, convert to one-hot for dynamics model input
        if self.action_dim > 1 and actions.dtype == torch.long: 
            actions_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
            predicted_next_states, predicted_rewards = self.dynamics_model(states, actions_one_hot)
        else:
            predicted_next_states, predicted_rewards = self.dynamics_model(states, actions.float())

        loss = dynamics_model_loss(predicted_next_states, next_states, predicted_rewards, rewards)

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return loss.item()

    def update_worker(self):
        """Update the worker policy and critic."""
        if len(self.worker_buffer) < self.config.general.batch_size:
            return 0.0, 0.0

        experiences = self.worker_buffer.sample(self.config.general.batch_size)
        states, actions, rewards, next_states, dones, subgoals = zip(*experiences)

        states = to_tensor(np.array(states))
        actions = to_tensor(np.array(actions), dtype=torch.long)
        rewards = to_tensor(np.array(rewards).reshape(-1, 1))
        next_states = to_tensor(np.array(next_states))
        dones = to_tensor(np.array(dones).reshape(-1, 1), dtype=torch.float32)
        subgoals = to_tensor(np.array(subgoals))

        # Worker Critic update
        current_q_values = self.worker_critic(states, subgoals)
        
        with torch.no_grad():
            next_q_values = self.worker_critic(next_states, subgoals) # Assuming target network for critic or same network
            target_q_values = rewards + (self.gamma_worker * next_q_values * (1 - dones))
        
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        self.worker_optimizer.zero_grad()
        critic_loss.backward()
        self.worker_optimizer.step()

        # Worker Actor update (policy gradient)
        # Re-evaluate current actions given the updated critic
        action_logits = self.worker_actor(states, subgoals) # Logits for discrete actions
        dist = torch.distributions.Categorical(logits=action_logits) # For discrete actions
        log_probs = dist.log_prob(actions)

        # Use current critic to estimate advantage
        # For actor update, we typically want to maximize Q-values
        actor_loss = -(log_probs * self.worker_critic(states, subgoals).detach()).mean()

        self.worker_optimizer.zero_grad()
        actor_loss.backward()
        self.worker_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

    def update_manager(self):
        """Update the manager policy and critic."""
        if len(self.manager_buffer) < self.config.general.batch_size:
            return 0.0, 0.0

        experiences = self.manager_buffer.sample(self.config.general.batch_size)
        states, subgoals, rewards, next_states, dones = zip(*experiences)

        states = to_tensor(np.array(states))
        subgoals = to_tensor(np.array(subgoals))
        rewards = to_tensor(np.array(rewards).reshape(-1, 1))
        next_states = to_tensor(np.array(next_states))
        dones = to_tensor(np.array(dones).reshape(-1, 1), dtype=torch.float32)

        # Manager Critic update
        current_q_values = self.manager_critic(states)
        with torch.no_grad():
            next_q_values = self.manager_critic(next_states) # Assuming target network for critic or same network
            target_q_values = rewards + (self.gamma_manager * next_q_values * (1 - dones))
        
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        self.manager_optimizer.zero_grad()
        critic_loss.backward()
        self.manager_optimizer.step()

        # Manager Actor update (policy gradient)
        subgoal_logits = self.manager_actor(states) # Logits for discrete subgoals if applicable, or direct continuous
        # For continuous subgoals, this would be a distribution, e.g., Gaussian
        # For simplicity, assuming direct output as subgoal, and then using a 'pseudo-log_prob' with the critic.
        # A more robust implementation would involve actual policy distributions.

        # A simple approximation for actor loss in continuous action/subgoal space:
        # Directly maximize the Q-value output by the critic for the chosen subgoal
        actor_loss = -(self.manager_critic(states)).mean() # Maximize value function
        # This is a very simplified actor loss. A proper one would involve log_probs and advantages.

        self.manager_optimizer.zero_grad()
        actor_loss.backward()
        self.manager_optimizer.step()

        return actor_loss.item(), critic_loss.item()


    def store_manager_experience(self, state, subgoal, reward, next_state, done):
        self.manager_buffer.push(state, subgoal, reward, next_state, done)

    def store_worker_experience(self, state, action, reward, next_state, done, subgoal):
        self.worker_buffer.push(state, action, reward, next_state, done, subgoal)


class GoalConditionedAgent:
    """Goal-conditioned reinforcement learning agent with HER."""

    def __init__(
        self, config: Config,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
    ):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        self.actor = GoalConditionedActor(state_dim, action_dim, goal_dim, config.worker.hidden_dim).to(get_device())
        self.actor_target = GoalConditionedActor(state_dim, action_dim, goal_dim, config.worker.hidden_dim).to(get_device())
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.worker.learning_rate)

        self.critic = GoalConditionedCritic(state_dim, goal_dim, config.worker.hidden_dim).to(get_device())
        self.critic_target = GoalConditionedCritic(state_dim, goal_dim, config.worker.hidden_dim).to(get_device())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.worker.learning_rate)

        self.replay_buffer = ReplayBuffer(config.worker.replay_buffer_size) # Store (s, a, r, s', done, g)
        self.gamma = config.worker.discount_factor
        self.her_k = config.worker.her_k

    def get_action(self, state: np.ndarray, goal: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """Select an action based on the goal-conditioned policy."""
        self.actor.eval()
        with torch.no_grad():
            state_tensor = to_tensor(state).unsqueeze(0)
            goal_tensor = to_tensor(goal).unsqueeze(0)
            action = self.actor(state_tensor, goal_tensor).squeeze(0).cpu().numpy()
        self.actor.train()

        # Add exploration noise for continuous actions
        if noise_scale > 0:
            action = action + noise_scale * np.random.randn(self.action_dim)
        return np.clip(action, -1.0, 1.0) # Assuming actions are normalized to [-1, 1]

    def store_experience(self, state, action, reward, next_state, done, goal):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done, goal)

    def _add_her_transitions(self, episode_transitions: List[Tuple]):
        """Add HER transitions to the replay buffer."""
        # episode_transitions is a list of (state, action, reward, next_state, done, original_goal)
        for t, (state, action, reward, next_state, done, original_goal) in enumerate(episode_transitions):
            # Relabel with future goals
            for _ in range(self.her_k):
                future_idx = random.randint(t, len(episode_transitions) - 1)
                achieved_goal = episode_transitions[future_idx][3] # next_state of a future transition as new goal
                
                # Recompute reward for the new goal
                new_reward = self._compute_reward(next_state, achieved_goal)
                new_done = bool(new_reward > -0.05) # Assuming positive reward for goal achievement

                self.replay_buffer.push(state, action, new_reward, next_state, new_done, achieved_goal)

    def _compute_reward(self, achieved_state: np.ndarray, goal: np.ndarray, threshold: float = 0.05) -> float:
        """Compute binary reward based on distance to goal."""
        distance = np.linalg.norm(achieved_state - goal)
        return 0.0 if distance < threshold else -1.0

    def update(self):
        """Update actor and critic networks."""
        if len(self.replay_buffer) < self.config.general.batch_size:
            return 0.0, 0.0

        experiences = self.replay_buffer.sample(self.config.general.batch_size)
        states, actions, rewards, next_states, dones, goals = zip(*experiences)

        states = to_tensor(np.array(states))
        actions = to_tensor(np.array(actions))
        rewards = to_tensor(np.array(rewards).reshape(-1, 1))
        next_states = to_tensor(np.array(next_states))
        dones = to_tensor(np.array(dones).reshape(-1, 1), dtype=torch.float32)
        goals = to_tensor(np.array(goals))

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states, goals)
            target_q = self.critic_target(next_states, goals)
            target_q = rewards + (self.gamma * target_q * (1 - dones))

        current_q = self.critic(states, goals)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, goals).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor, tau=0.005)
        self._soft_update(self.critic_target, self.critic, tau=0.005)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


class FeudalNetwork:
    """Feudal Networks with manager-worker hierarchy."""

    def __init__(
        self, config: Config,
        state_dim: int,
        action_dim: int,
        subgoal_dim: int,
    ):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.subgoal_dim = subgoal_dim

        # Manager
        self.manager = FeudalManager(state_dim, subgoal_dim, config.manager.hidden_dim).to(get_device())
        self.manager_optimizer = optim.Adam(self.manager.parameters(), lr=config.manager.learning_rate)

        # Worker
        self.worker = FeudalWorker(state_dim, action_dim, subgoal_dim, config.worker.hidden_dim).to(get_device())
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=config.worker.learning_rate)

        self.gamma = config.general.discount_factor # A general discount factor. Adjust if manager/worker have different gammas
        self.manager_update_freq = config.manager.update_frequency_worker_steps
        self.worker_steps_counter = 0

    def select_action(self, state: np.ndarray, subgoal: Optional[np.ndarray] = None) -> Union[int, np.ndarray]:
        """Select an action based on manager or worker policy."""
        state_tensor = to_tensor(state).unsqueeze(0)

        if self.worker_steps_counter % self.manager_update_freq == 0: # Manager turn
            self.manager.eval()
            with torch.no_grad():
                subgoal_output = self.manager(state_tensor).squeeze(0).cpu().numpy()
            self.manager.train()
            # Store subgoal for worker to use
            self._current_subgoal = subgoal_output
            return subgoal_output # Manager 'action' is a subgoal
        else: # Worker turn
            assert self._current_subgoal is not None, "Worker needs a subgoal from manager."
            self.worker.eval()
            with torch.no_grad():
                subgoal_tensor = to_tensor(self._current_subgoal).unsqueeze(0)
                action_logits = self.worker(state_tensor, subgoal_tensor)
                action = action_logits.argmax().item() # Assuming discrete actions
            self.worker.train()
            return action

    def update(self, states, actions, rewards, next_states, goals, dones):
        """Update manager and worker networks."""
        # Worker Update (similar to goal-conditioned agent)
        # Manager Update (using extrinsic rewards from environment based on subgoals)
        pass # This needs to be implemented properly, likely with separate update functions.


class ModelPredictiveController:
    """Model Predictive Control using learned dynamics."""

    def __init__(
        self, config: Config,
        dynamics_model: DynamicsModel,
        action_dim: int,
    ):
        self.config = config
        self.dynamics_model = dynamics_model
        self.action_dim = action_dim
        self.horizon = config.dynamics_model.planning_horizon
        self.num_candidates = config.dynamics_model.num_planning_candidates
        self.elite_frac = config.dynamics_model.elite_fraction

    def plan(self, state: np.ndarray, num_iterations: int = 5) -> int:
        """Plan the optimal first action using Cross-Entropy Method."""
        action_mean = torch.zeros(self.horizon, self.action_dim, device=get_device())
        action_std = torch.ones(self.horizon, self.action_dim, device=get_device())

        for _ in range(num_iterations):
            # 1. Sample candidate action sequences
            action_sequences = (action_mean + action_std * torch.randn(
                self.num_candidates, self.horizon, self.action_dim, device=get_device()
            )).clamp(-1.0, 1.0) # Assuming continuous actions between -1 and 1

            # 2. Evaluate each sequence using the dynamics model
            rewards = self._evaluate_sequence(to_tensor(state), action_sequences)

            # 3. Select elite sequences
            elite_indices = torch.topk(rewards, int(self.num_candidates * self.elite_frac))[1]
            elite_actions = action_sequences[elite_indices]

            # 4. Update distribution parameters
            action_mean = elite_actions.mean(dim=0)
            action_std = elite_actions.std(dim=0)

        # Return the first action of the best sequence (mean of elite)
        # For discrete actions, this would need to be converted to a discrete action
        return action_mean[0].argmax().item() if self.action_dim > 1 else action_mean[0].cpu().numpy()

    def _evaluate_sequence(self, initial_state: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        """Evaluate the cumulative reward of action sequences using the dynamics model."""
        total_rewards = torch.zeros(self.num_candidates, device=get_device())
        current_states = initial_state.unsqueeze(0).repeat(self.num_candidates, 1) # (num_candidates, state_dim)

        for t in range(self.horizon):
            actions_t = action_sequences[:, t, :]
            
            # For discrete actions, convert to one-hot for dynamics model
            if self.action_dim > 1 and actions_t.dim() == 2 and actions_t.shape[1] == self.action_dim: # Assuming one-hot
                 pass # Already one-hot
            elif self.action_dim > 1: # Assuming action_t is just the action index for discrete action
                actions_t = F.one_hot(actions_t.long(), num_classes=self.action_dim).float()
            
            predicted_next_states, predicted_rewards = self.dynamics_model(current_states, actions_t)
            total_rewards += predicted_rewards
            current_states = predicted_next_states
        return total_rewards


class MCTSNode:
    """Node in Monte Carlo Tree Search."""

    def __init__(self, state: Any, parent: Optional["MCTSNode"] = None, action: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0

    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if node is root."""
        return self.parent is None

    def select_child(self, exploration_constant: float = 1.4) -> "MCTSNode":
        """Select a child node using UCB1 formula."""
        best_score = -float("inf")
        best_child = None

        for child in self.children:
            if child.visits == 0:
                score = float("inf") # Prefer unvisited nodes
            else:
                uct_score = child.value / child.visits + \
                            exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
                score = uct_score
            
            if score > best_score:
                best_score = score
                best_child = child
        return best_child # type: ignore

    def add_child(self, state: Any, action: int) -> "MCTSNode":
        """Add a new child node."""
        child = MCTSNode(state, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, value: float):
        """Update node statistics after a simulation."""
        self.visits += 1
        self.value += value


class MonteCarloTreeSearch:
    """Monte Carlo Tree Search implementation."""

    def __init__(
        self, config: Config,
        dynamics_model: DynamicsModel,
        env_prototype: Any, # A callable that creates a new environment instance
        action_dim: int,
    ):
        self.config = config
        self.dynamics_model = dynamics_model
        self.env_prototype = env_prototype
        self.num_simulations = config.dynamics_model.num_planning_candidates # Reusing config for consistency
        self.exploration_constant = 1.4
        self.max_depth = config.dynamics_model.planning_horizon
        self.action_dim = action_dim

    def search(self, root_state: Any) -> MCTSNode:
        """Perform MCTS search from a given root state."""
        root = MCTSNode(root_state)

        for _ in range(self.num_simulations):
            node = root
            state = root_state
            path = [node]

            # Selection
            while not node.is_leaf() and not self._is_terminal(state):
                node = node.select_child(self.exploration_constant)
                path.append(node)
                state = self._simulate_action(node.parent.state, node.action)[0] # type: ignore

            # Expansion
            if not self._is_terminal(state):
                for action in range(self.action_dim):
                    next_state, _, _, _ = self._simulate_action(state, action)
                    node.add_child(next_state, action)
            
            # Simulation (Rollout)
            rollout_value = self._simulate_rollout(state)

            # Backpropagation
            for node_to_update in reversed(path):
                node_to_update.update(rollout_value)
                # For more sophisticated MCTS, value propagation might be discounted
        return root

    def get_best_action(self, root_state: Any) -> int:
        """Get the best action after MCTS search."""
        root_node = self.search(root_state)
        best_child = max(root_node.children, key=lambda child: child.visits) # Action with most visits
        return best_child.action # type: ignore

    def _is_terminal(self, state: Any) -> bool:
        """Check if a state is terminal (using a dummy env for now)."""
        # This needs to be more robust. For now, a simple check.
        return False # Assuming non-terminal for planning

    def _get_available_actions(self, state: Any) -> List[int]:
        """Get available actions in a state."""
        return list(range(self.action_dim))

    def _simulate_action(self, state: Any, action: int) -> Tuple[Any, float, bool, Dict]:
        """Simulate one step using the dynamics model."""
        state_tensor = to_tensor(state).unsqueeze(0)
        action_tensor = to_tensor(action, dtype=torch.long).unsqueeze(0)

        if self.action_dim > 1 and action_tensor.dtype == torch.long:
            action_input = F.one_hot(action_tensor, num_classes=self.action_dim).float()
        else:
            action_input = action_tensor.float()

        with torch.no_grad():
            predicted_next_state, predicted_reward = self.dynamics_model(state_tensor, action_input)
        
        next_state_np = predicted_next_state.squeeze(0).cpu().numpy()
        reward_np = predicted_reward.item()

        # Placeholder for done and info
        done = False
        info = {}
        return next_state_np, reward_np, done, info

    def _simulate_rollout(self, state: Any, max_steps: int = 10) -> float:
        """Perform a random rollout from a state using dynamics model."""
        current_state = state
        cumulative_reward = 0.0

        for _ in range(max_steps):
            action = random.randrange(self.action_dim) # Random action
            next_state, reward, done, _ = self._simulate_action(current_state, action)
            cumulative_reward += reward
            current_state = next_state
            if done:
                break
        return cumulative_reward


class ModelBasedValueExpansion:
    """Model-Based Value Expansion for planning."""

    def __init__(
        self, config: Config,
        dynamics_model: DynamicsModel,
        reward_model: nn.Module, # Can be part of dynamics model or separate
        value_network: nn.Module,
        action_dim: int,
    ):
        self.config = config
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.value_network = value_network
        self.horizon = config.dynamics_model.planning_horizon
        self.num_samples = config.dynamics_model.num_planning_candidates # Number of rollouts
        self.action_dim = action_dim

    def expand_value(self, state: np.ndarray) -> torch.Tensor:
        """Estimate value of a state by rolling out in the model and using value network."""
        state_tensor = to_tensor(state).unsqueeze(0) # (1, state_dim)
        
        # Generate multiple rollouts
        total_values = []
        for _ in range(self.num_samples):
            current_state = state_tensor.clone()
            cumulative_discounted_reward = 0.0
            discount = 1.0

            for t in range(self.horizon):
                # Sample random action or use a policy
                action = random.randrange(self.action_dim)
                action_tensor = to_tensor(action, dtype=torch.long).unsqueeze(0)

                if self.action_dim > 1 and action_tensor.dtype == torch.long:
                    action_input = F.one_hot(action_tensor, num_classes=self.action_dim).float()
                else:
                    action_input = action_tensor.float()

                with torch.no_grad():
                    predicted_next_state, predicted_reward = self.dynamics_model(current_state, action_input)
                
                cumulative_discounted_reward += discount * predicted_reward.item()
                current_state = predicted_next_state
                discount *= self.config.worker.discount_factor # Using worker gamma for short horizon
            
            # Add value estimate from the end state of the rollout
            with torch.no_grad():
                final_value_estimate = self.value_network(current_state).item()
            total_values.append(cumulative_discounted_reward + discount * final_value_estimate)
        
        return to_tensor(np.mean(total_values))


class WorldModel(nn.Module):
    """Complete world model with encoder, decoder, and dynamics."""

    def __init__(
        self, config: Config,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = config.dynamics_model.hidden_dim

        # Encoder: state -> latent distribution parameters
        self.encoder = MLP(state_dim, latent_dim * 2, self.hidden_dim) # Output mean and log_var
        # Decoder: latent -> state
        self.decoder = MLP(latent_dim, state_dim, self.hidden_dim)
        # Dynamics: latent, action -> next_latent distribution parameters
        self.dynamics = MLP(latent_dim + action_dim, latent_dim * 2, self.hidden_dim)
        # Reward: latent, action -> reward
        self.reward_predictor = MLP(latent_dim + action_dim, 1, self.hidden_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=config.dynamics_model.learning_rate)

    def encode(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state into latent mean and log variance."""
        params = self.encoder(state)
        mean, log_var = params.chunk(2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from latent distribution."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to state."""
        return self.decoder(latent)

    def predict_next_latent(self, latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next latent state distribution parameters."""
        x = torch.cat([latent, action], dim=-1)
        params = self.dynamics(x)
        mean, log_var = params.chunk(2, dim=-1)
        return mean, log_var

    def predict_reward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state and action."""
        x = torch.cat([latent, action], dim=-1)
        return self.reward_predictor(x).squeeze(-1)

    def train_step(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, rewards: torch.Tensor
    ):
        """Train the world model components."""
        self.optimizer.zero_grad()

        # Encode current state
        mean_s, log_var_s = self.encode(states)
        latent_s = self.reparameterize(mean_s, log_var_s)

        # Predict next latent state
        mean_next_s, log_var_next_s = self.predict_next_latent(latent_s, actions)
        latent_next_s_pred = self.reparameterize(mean_next_s, log_var_next_s)

        # Decode predicted next latent state to reconstruct next_state
        reconstructed_next_states = self.decode(latent_next_s_pred)

        # Predict reward
        predicted_rewards = self.predict_reward(latent_s, actions)

        # Loss calculations
        # 1. Reconstruction loss for next state
        reconstruction_loss = F.mse_loss(reconstructed_next_states, next_states)

        # 2. KL divergence loss (optional, for VAE-like training)
        # This part depends on the exact World Model variant. For a simple version, we might skip KL.
        # kl_loss = -0.5 * torch.sum(1 + log_var_s - mean_s.pow(2) - log_var_s.exp())
        kl_loss_next_state = -0.5 * torch.sum(1 + log_var_next_s - mean_next_s.pow(2) - log_var_next_s.exp())
        # For simplicity, just using reconstruction and reward prediction for now. KL for latent consistency.

        # 3. Reward prediction loss
        reward_loss = F.mse_loss(predicted_rewards, rewards)

        # Total loss
        loss = reconstruction_loss + reward_loss + 0.1 * kl_loss_next_state # KL weight is a hyperparameter

        loss.backward()
        self.optimizer.step()
        return loss.item()


class LatentSpacePlanner:
    """Planning in learned latent space."""

    def __init__(
        self, config: Config,
        world_model: WorldModel,
        action_dim: int,
        latent_dim: int = 32,
    ):
        self.config = config
        self.world_model = world_model
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.planning_horizon = config.dynamics_model.planning_horizon
        self.num_candidates = config.dynamics_model.num_planning_candidates
        self.elite_frac = config.dynamics_model.elite_fraction

    def plan(self, state: np.ndarray, num_iterations: int = 5) -> Any:
        """Plan the optimal first action in latent space using CEM."""
        # 1. Encode current state into latent space
        state_tensor = to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            mean_s, log_var_s = self.world_model.encode(state_tensor)
            initial_latent = self.world_model.reparameterize(mean_s, log_var_s)

        action_mean = torch.zeros(self.planning_horizon, self.action_dim, device=get_device())
        action_std = torch.ones(self.planning_horizon, self.action_dim, device=get_device())

        for _ in range(num_iterations):
            # Sample candidate action sequences
            action_sequences = (action_mean + action_std * torch.randn(
                self.num_candidates, self.planning_horizon, self.action_dim, device=get_device()
            )).clamp(-1.0, 1.0) # Assuming continuous actions between -1 and 1

            # Evaluate each sequence using the world model
            rewards = self._evaluate_sequence(initial_latent, action_sequences)

            # Select elite sequences
            elite_indices = torch.topk(rewards, int(self.num_candidates * self.elite_frac))[1]
            elite_actions = action_sequences[elite_indices]

            # Update distribution parameters
            action_mean = elite_actions.mean(dim=0)
            action_std = elite_actions.std(dim=0)

        # Return the first action of the best sequence (mean of elite)
        return action_mean[0].argmax().item() if self.action_dim > 1 else action_mean[0].cpu().numpy()

    def _evaluate_sequence(self, initial_latent: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        """Evaluate the cumulative reward of action sequences using the world model."""
        total_rewards = torch.zeros(self.num_candidates, device=get_device())
        current_latents = initial_latent.repeat(self.num_candidates, 1) # (num_candidates, latent_dim)

        for t in range(self.planning_horizon):
            actions_t = action_sequences[:, t, :]
            
            # Predict reward
            predicted_rewards = self.world_model.predict_reward(current_latents, actions_t)
            total_rewards += predicted_rewards

            # Predict next latent state
            mean_next_l, log_var_next_l = self.world_model.predict_next_latent(current_latents, actions_t)
            current_latents = self.world_model.reparameterize(mean_next_l, log_var_next_l)
            
        return total_rewards






