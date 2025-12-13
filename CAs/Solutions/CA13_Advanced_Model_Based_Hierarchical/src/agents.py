import torch
import torch.optim as optim
import numpy as np
from src.config import WorldModelConfig, ManagerConfig, WorkerConfig, TrainingConfig
from src.models import WorldModel, ManagerActor, ManagerCritic, WorkerActor, WorkerCritic
from src.losses import world_model_loss, manager_loss, worker_loss
from src.data import ReplayBuffer
from typing import Tuple, Dict

class DreamerFuNAgent:
    """
    Dreamer-FuN Agent: Integrates World Model, Manager, and Worker for hierarchical model-based RL.
    """
    def __init__(
        self,
        world_model_config: WorldModelConfig,
        manager_config: ManagerConfig,
        worker_config: WorkerConfig,
        training_config: TrainingConfig,
        action_dim: int, # Will be set by environment
        device: torch.device
    ):
        self.world_model_config = world_model_config
        self.manager_config = manager_config
        self.worker_config = worker_config
        self.training_config = training_config
        self.action_dim = action_dim
        self.device = device

        # Adjust world model config with actual action_dim
        self.world_model_config.action_dim = self.action_dim

        # Initialize World Model
        self.world_model = WorldModel(self.world_model_config).to(self.device)
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), lr=1e-3) # Example LR

        # Initialize Manager
        self.manager_actor = ManagerActor(self.manager_config, self.world_model_config.latent_dim, self.world_model_config.hidden_dim).to(self.device)
        self.manager_critic = ManagerCritic(self.manager_config, self.world_model_config.latent_dim, self.world_model_config.hidden_dim).to(self.device)
        self.manager_actor_optimizer = optim.Adam(self.manager_actor.parameters(), lr=self.manager_config.learning_rate)
        self.manager_critic_optimizer = optim.Adam(self.manager_critic.parameters(), lr=self.manager_config.learning_rate)

        # Initialize Worker
        self.worker_actor = WorkerActor(self.worker_config, self.world_model_config.latent_dim, self.manager_config.goal_dim, self.action_dim, self.world_model_config.hidden_dim).to(self.device)
        self.worker_critic = WorkerCritic(self.worker_config, self.world_model_config.latent_dim, self.manager_config.goal_dim, self.world_model_config.hidden_dim).to(self.device)
        self.worker_actor_optimizer = optim.Adam(self.worker_actor.parameters(), lr=self.worker_config.learning_rate)
        self.worker_critic_optimizer = optim.Adam(self.worker_critic.parameters(), lr=self.worker_config.learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.training_config.replay_buffer_size,
            observation_shape=self.world_model_config.observation_shape,
            action_dim=self.action_dim,
            device=self.device
        )

        # Internal state for recurrent model
        self._prev_latent = torch.zeros(1, self.world_model_config.latent_dim, device=self.device)
        self._prev_hidden = torch.zeros(1, self.world_model_config.hidden_dim, device=self.device)
        self._current_manager_goal = torch.zeros(1, self.manager_config.goal_dim, device=self.device)
        self._worker_steps_taken = 0

    def _get_action_from_policy(self, actor_net: torch.nn.Module, state: torch.Tensor, goal: torch.Tensor = None) -> np.ndarray:
        """
        Helper to get action (or goal) from an actor network.
        Adds exploration noise during training.
        """
        if goal is not None:
            dist = actor_net(state, goal)
        else:
            dist = actor_net(state)

        action = dist.sample()
        # Add exploration noise if training
        # if self.training and self.worker_config.exploration_amount > 0 and goal is not None:
        #     action = action + torch.randn_like(action) * self.worker_config.exploration_amount

        return action.cpu().numpy()

    def act(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Selects an action based on the current observation using the hierarchical policy.
        """
        self.world_model.eval() # World model should be in eval mode during action selection

        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Update latent state of world model
            self._prev_latent, self._prev_hidden = self.world_model.get_latent(
                obs_tensor, self._prev_latent, torch.zeros(1, self.action_dim, device=self.device), self._prev_hidden # Dummy action for initial step
            )

            # Manager sets a new goal if worker has completed its sub-horizon
            if self._worker_steps_taken % self.worker_config.sub_goal_horizon == 0:
                self.manager_actor.eval()
                self._current_manager_goal = self._get_action_from_policy(self.manager_actor, self._prev_latent.squeeze(0))
                self._worker_steps_taken = 0

            # Worker selects an action based on current latent and manager's goal
            self.worker_actor.eval()
            action = self._get_action_from_policy(self.worker_actor, self._prev_latent.squeeze(0), self._current_manager_goal.squeeze(0))

            self._worker_steps_taken += 1

        self.world_model.train() # Set back to train mode after action selection

        return action

    def store_transition(self, observation: np.ndarray, action: np.ndarray, reward: float, done: bool, next_observation: np.ndarray):
        """
        Stores a single environment transition in the replay buffer.
        """
        self.replay_buffer.add(observation, action, reward, done, next_observation)

    def update_world_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Updates the World Model using a batch of experience sequences.
        """
        self.world_model.train()
        self.world_model_optimizer.zero_grad()

        observations = batch['observations']
        actions = batch['actions'][:, :-1] # Actions for T-1 transitions
        rewards = batch['rewards'][:, :-1]

        # Initialize prev_latent and prev_hidden for sequence processing
        # (B, 1, latent_dim), (B, 1, hidden_dim)
        init_latent = torch.zeros(observations.shape[0], self.world_model_config.latent_dim, device=self.device)
        init_hidden = torch.zeros(observations.shape[0], self.world_model_config.hidden_dim, device=self.device)

        priors, posteriors, reconstructed_observations, predicted_rewards = self.world_model(
            observations, actions, (init_latent, init_hidden)
        )

        # Reshape distributions to match (B*T, D) for loss calculation
        prior_dist_flat = dist.Normal(torch.cat([d.mean for d in priors], dim=1), torch.cat([d.stddev for d in priors], dim=1))
        posterior_dist_flat = dist.Normal(torch.cat([d.mean for d in posteriors], dim=1), torch.cat([d.stddev for d in posteriors], dim=1))
        reconstructed_obs_flat = dist.Normal(torch.cat([d.mean for d in reconstructed_observations], dim=1), torch.cat([d.stddev for d in reconstructed_observations], dim=1))
        predicted_rewards_flat = dist.Normal(torch.cat([d.mean for d in predicted_rewards], dim=1), torch.cat([d.stddev for d in predicted_rewards], dim=1))

        true_obs_flat = observations[:, 1:].reshape(-1, *self.world_model_config.observation_shape)
        true_rewards_flat = rewards.reshape(-1, 1)

        total_loss, losses_dict = world_model_loss(
            self.world_model_config,
            prior_dist_flat, posterior_dist_flat,
            reconstructed_obs_flat, predicted_rewards_flat,
            true_obs_flat, true_rewards_flat
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.training_config.gradient_clip_norm)
        self.world_model_optimizer.step()

        return losses_dict

    def update_manager_and_worker(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Updates the Manager and Worker policies using imagined trajectories from the World Model.
        """
        self.manager_actor.train()
        self.manager_critic.train()
        self.worker_actor.train()
        self.worker_critic.train()

        # Get initial latent states for imagination (from the batch, or current agent state)
        # For simplicity, we'll use the last latent state of each sequence in the batch
        # This needs to be refined for proper sequence handling and imagination starting points
        initial_obs = batch['observations'][:, -1] # (B, C, H, W) or (B, D_o)
        initial_action = batch['actions'][:, -1] # (B, D_a)

        with torch.no_grad():
            # Encode the initial observation to get latent state
            init_latent_batch, init_hidden_batch = self.world_model.get_latent(
                initial_obs, 
                torch.zeros(initial_obs.shape[0], self.world_model_config.latent_dim, device=self.device), 
                torch.zeros(initial_obs.shape[0], self.action_dim, device=self.device), 
                torch.zeros(initial_obs.shape[0], self.world_model_config.hidden_dim, device=self.device)
            )

        # --- Train Worker ---
        # Manager sets a goal (can be a randomly sampled one from replay, or from Manager policy)
        # For now, let's assume the manager_actor_output is the target goal for simplicity in training Worker
        manager_goal_batch = self.manager_actor(init_latent_batch) # (B, goal_dim)

        # Imagine worker trajectory
        imagined_actions_worker = []
        imagined_latents_worker = [init_latent_batch]
        imagined_rewards_ext_worker = []

        current_latent_worker = init_latent_batch
        current_hidden_worker = init_hidden_batch

        for _ in range(self.worker_config.sub_goal_horizon):
            worker_action_dist = self.worker_actor(current_latent_worker.detach(), manager_goal_batch.detach())
            worker_action = worker_action_dist.sample()
            imagined_actions_worker.append(worker_action)

            # Predict next latent state and reward using world model
            prior_dist, current_hidden_worker = self.world_model.rssm(current_latent_worker, worker_action, current_hidden_worker)
            current_latent_worker = prior_dist.sample()
            imagined_latents_worker.append(current_latent_worker)
            imagined_rewards_ext_worker.append(self.world_model.reward_model(current_latent_worker).mean)
        
        imagined_latents_worker = torch.stack(imagined_latents_worker[1:]) # (N, B, latent_dim)
        imagined_actions_worker = torch.stack(imagined_actions_worker) # (N, B, action_dim)
        imagined_rewards_ext_worker = torch.stack(imagined_rewards_ext_worker) # (N, B, 1)

        # Reshape for loss function (B, N, D)
        imagined_latents_worker_reshaped = imagined_latents_worker.permute(1, 0, 2)
        imagined_actions_worker_reshaped = imagined_actions_worker.permute(1, 0, 2)
        imagined_rewards_ext_worker_reshaped = imagined_rewards_ext_worker.permute(1, 0, 2)
        
        worker_current_latent_states = imagined_latents_worker_reshaped[:, :-1]
        worker_achieved_latent_states = imagined_latents_worker_reshaped[:, 1:]

        worker_critic_values = self.worker_critic(worker_current_latent_states, manager_goal_batch.unsqueeze(1).expand_as(worker_current_latent_states))
        target_worker_critic_values = self._calculate_target_values(
            imagined_rewards_ext_worker_reshaped, 
            worker_achieved_latent_states, 
            manager_goal_batch.unsqueeze(1).expand_as(worker_achieved_latent_states), 
            self.worker_critic, 
            self.worker_config.discount_factor
        )

        worker_actor_output_dists = [self.worker_actor(l, manager_goal_batch.unsqueeze(0)) for l in worker_current_latent_states.reshape(-1, self.world_model_config.latent_dim)]
        worker_actor_output_dist_flat = dist.Normal(torch.cat([d.mean for d in worker_actor_output_dists], dim=0), torch.cat([d.stddev for d in worker_actor_output_dists], dim=0))


        worker_loss_val, worker_losses_dict = worker_loss(
            self.worker_config,
            worker_actor_output_dist_flat,
            imagined_actions_worker_reshaped.reshape(-1, self.action_dim),
            worker_current_latent_states,
            manager_goal_batch,
            worker_achieved_latent_states,
            imagined_rewards_ext_worker_reshaped,
            worker_critic_values,
            target_worker_critic_values
        )

        self.worker_actor_optimizer.zero_grad()
        self.worker_critic_optimizer.zero_grad()
        worker_loss_val.backward()
        torch.nn.utils.clip_grad_norm_(self.worker_actor.parameters(), self.training_config.gradient_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.worker_critic.parameters(), self.training_config.gradient_clip_norm)
        self.worker_actor_optimizer.step()
        self.worker_critic_optimizer.step()

        # --- Train Manager ---
        # Manager imagines trajectory over H_M steps, each step corresponds to N worker steps
        manager_current_latent = init_latent_batch
        manager_current_hidden = init_hidden_batch
        manager_goals_imagined = []
        manager_intrinsic_rewards_imagined = []

        for _ in range(self.manager_config.goal_horizon):
            manager_goal = self.manager_actor(manager_current_latent.detach()) # Manager proposes a goal
            manager_goals_imagined.append(manager_goal)

            # Simulate N worker steps in imagination to calculate intrinsic reward for manager
            worker_latent_at_goal_end = manager_current_latent
            for _ in range(self.worker_config.sub_goal_horizon):
                worker_action_dist = self.worker_actor(worker_latent_at_goal_end.detach(), manager_goal.detach())
                worker_action = worker_action_dist.sample()
                prior_dist, manager_current_hidden = self.world_model.rssm(worker_latent_at_goal_end, worker_action, manager_current_hidden)
                worker_latent_at_goal_end = prior_dist.sample()
            
            manager_intrinsic_reward = -F.mse_loss(worker_latent_at_goal_end, manager_goal, reduction='none').mean(dim=-1, keepdim=True)
            manager_intrinsic_rewards_imagined.append(manager_intrinsic_reward)

            # Update manager_current_latent for next manager step (after N worker steps)
            manager_current_latent = worker_latent_at_goal_end

        manager_goals_imagined = torch.stack(manager_goals_imagined) # (H_M, B, goal_dim)
        manager_intrinsic_rewards_imagined = torch.stack(manager_intrinsic_rewards_imagined) # (H_M, B, 1)

        # Reshape for loss function (B, H_M, D)
        manager_goals_imagined_reshaped = manager_goals_imagined.permute(1, 0, 2)
        manager_intrinsic_rewards_imagined_reshaped = manager_intrinsic_rewards_imagined.permute(1, 0, 2)

        manager_current_latent_states = init_latent_batch.unsqueeze(1).expand_as(manager_goals_imagined_reshaped[:,:-1])
        manager_achieved_latent_states = manager_goals_imagined_reshaped[:,1:] # Simplified, should be the state after N worker steps

        manager_critic_values = self.manager_critic(manager_current_latent_states)
        target_manager_critic_values = self._calculate_target_values(
            manager_intrinsic_rewards_imagined_reshaped,
            manager_achieved_latent_states, # Should be the achieved state, not next goal
            None, # Manager critic does not take goal as input
            self.manager_critic, 
            self.manager_config.discount_factor
        )

        manager_actor_output = self.manager_actor(manager_current_latent_states.reshape(-1, self.world_model_config.latent_dim))

        manager_loss_val, manager_losses_dict = manager_loss(
            self.manager_config,
            manager_actor_output,
            manager_current_latent_states.reshape(-1, self.world_model_config.latent_dim),
            manager_achieved_latent_states.reshape(-1, self.world_model_config.latent_dim),
            manager_critic_values,
            target_manager_critic_values
        )

        self.manager_actor_optimizer.zero_grad()
        self.manager_critic_optimizer.zero_grad()
        manager_loss_val.backward()
        torch.nn.utils.clip_grad_norm_(self.manager_actor.parameters(), self.training_config.gradient_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.manager_critic.parameters(), self.training_config.gradient_clip_norm)
        self.manager_actor_optimizer.step()
        self.manager_critic_optimizer.step()

        losses_dict.update(worker_losses_dict)
        losses_dict.update(manager_losses_dict)
        return losses_dict

    def _calculate_target_values(self, rewards: torch.Tensor, next_latents: torch.Tensor, next_goals: torch.Tensor, critic_net: torch.nn.Module, discount_factor: float) -> torch.Tensor:
        """
        Calculates target Q-values or V-values for critic training.
        This is a simplified GAE-like or N-step return calculation.
        """
        # rewards: (B, T, 1)
        # next_latents: (B, T, latent_dim)

        targets = []
        with torch.no_grad():
            # Initialize value for the last step
            if next_goals is not None: # Worker critic
                next_value = critic_net(next_latents[:, -1], next_goals[:, -1]).squeeze(-1)
            else: # Manager critic
                next_value = critic_net(next_latents[:, -1]).squeeze(-1)

            target = next_value
            for t in reversed(range(rewards.shape[1])):
                target = rewards[:, t].squeeze(-1) + discount_factor * target
                targets.append(target)
            targets.reverse()
        return torch.stack(targets, dim=1).unsqueeze(-1)

    def reset_episode_state(self):
        """
        Resets the agent's internal recurrent state at the beginning of a new episode.
        """
        self._prev_latent = torch.zeros(1, self.world_model_config.latent_dim, device=self.device)
        self._prev_hidden = torch.zeros(1, self.world_model_config.hidden_dim, device=self.device)
        self._current_manager_goal = torch.zeros(1, self.manager_config.goal_dim, device=self.device)
        self._worker_steps_taken = 0

