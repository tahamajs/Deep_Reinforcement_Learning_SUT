import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Union

from src.models import QNetwork, DuelingQNetwork, NoisyQNetwork, CategoricalQNetwork
from src.data import ReplayBuffer, PrioritizedReplayBuffer
from src.config import DQNConfig
from src.losses import c51_loss


class DQNAgent:
    """
    Vanilla Deep Q-Network Agent.

    Implements the basic DQN algorithm with experience replay and target networks.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: DQNConfig = DQNConfig(),
    ):
        """
        Initializes the DQNAgent.

        Args:
            state_dim: The dimensionality of the input state space.
            action_dim: The dimensionality of the action space.
            config: Configuration object for DQN hyperparameters.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQ
        self.device = torch.device(config.DEVICE)

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LR)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

        # Exploration
        self.epsilon = config.EPSILON_START
        self.epsilon_start = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY

        # Training tracking
        self.losses = []
        self.epsilon_history = []
        self.update_count = 0

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state: The current state of the environment.
            epsilon: The probability of taking a random action. If None, uses agent's current epsilon.

        Returns:
            The selected action.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def update_epsilon(self):
        """
        Updates the epsilon value for exploration decay.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    def compute_q_targets(self, rewards, next_states, dones):
        """
        Computes the target Q-values for standard DQN.
        """
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        return target_q_values

    def train_step(self) -> Optional[float]:
        """
        Performs one training step on a sampled batch of experiences.

        Returns:
            The loss value for the training step, or None if buffer is not full enough.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q values
        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Compute target Q values
        target_q_values = self.compute_q_targets(rewards, next_states, dones)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def train_episode(self, env: gym.Env, max_steps: int = 1000) -> Tuple[float, Dict]:
        """
        Trains the agent for one episode.

        Args:
            env: The Gymnasium environment to train in.
            max_steps: Maximum number of steps per episode.

        Returns:
            A tuple containing the total reward and a dictionary of episode information.
        """
        state, _ = env.reset(seed=DQNConfig.SEED)
        total_reward = 0
        steps = 0
        episode_losses = []

        while steps < max_steps:
            # Select and perform action
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            self.replay_buffer.push(state, action, reward, next_state, done)

            # Train
            loss = self.train_step()
            if loss is not None:
                episode_losses.append(loss)

            # Update exploration
            self.update_epsilon()

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        return total_reward, {
            "steps": steps,
            "avg_loss": np.mean(episode_losses) if episode_losses else 0,
            "epsilon": self.epsilon,
        }

    def evaluate(
        self, env: gym.Env, num_episodes: int = 10, max_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluates the agent's performance over a number of episodes.

        Args:
            env: The Gymnasium environment to evaluate in.
            num_episodes: The number of episodes to run for evaluation.
            max_steps: Maximum number of steps per evaluation episode.

        Returns:
            A dictionary containing evaluation statistics (mean_reward, std_reward, etc.).
        """
        rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset(seed=DQNConfig.SEED)
            episode_reward = 0
            steps = 0

            while steps < max_steps:
                action = self.select_action(state, epsilon=0.0)  # Greedy policy
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            rewards.append(episode_reward)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
        }


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent to reduce overestimation bias.

    Decouples action selection from action evaluation using two Q-networks.
    """

    def compute_q_targets(self, rewards, next_states, dones):
        """
        Computes the target Q-values for Double DQN.
        """
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)

            # Evaluate actions using target network
            next_q_values = (
                self.target_network(next_states).gather(1, next_actions).squeeze(1)
            )
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        return target_q_values

    def analyze_overestimation_bias(
        self, env: gym.Env, num_samples: int = 50
    ) -> Dict[str, float]:
        """
        Analyzes the overestimation bias of the Double DQN agent.

        Compares the Q-values from the online network with the target network's
        evaluation for a given state-action pair.

        Args:
            env: The Gymnasium environment for sampling states.
            num_samples: Number of state-action pairs to sample for analysis.

        Returns:
            A dictionary containing statistics about the overestimation bias.
        """
        biases = []
        for _ in range(num_samples):
            state, _ = env.reset(seed=DQNConfig.SEED)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_online = self.q_network(state_tensor)
                q_target = self.target_network(state_tensor)

                # True Q-value (estimated by target network's evaluation of online action)
                online_action = q_online.argmax(1, keepdim=True)
                true_q = q_target.gather(1, online_action).squeeze(1).item()

                # Estimated Q-value (from online network)
                estimated_q = q_online.max(1)[0].item()

                biases.append(estimated_q - true_q)

        return {
            "mean_bias": np.mean(biases),
            "std_bias": np.std(biases),
            "min_bias": np.min(biases),
            "max_bias": np.max(biases),
        }


class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN Agent.

    Uses a Dueling Network architecture to separate value and advantage streams.
    """

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig = DQNConfig()):
        super().__init__(state_dim, action_dim, config)
        # Replace Q-network with Dueling Q-network
        self.q_network = DuelingQNetwork(
            state_dim, action_dim, config.HIDDEN_DIM
        ).to(self.device)
        self.target_network = DuelingQNetwork(
            state_dim, action_dim, config.HIDDEN_DIM
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Reinitialize optimizer with new parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LR)

    def get_value_and_advantage(self, state: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Retrieves the value and advantage components from a given state.

        Args:
            state: The current state of the environment.

        Returns:
            A tuple containing the estimated value (float) and advantage (np.ndarray).
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            features = self.q_network.feature_layer(state_tensor)
            value = self.q_network.value_stream(features).squeeze(0).cpu().item()
            advantage = (
                self.q_network.advantage_stream(features).squeeze(0).cpu().numpy()
            )
        return value, advantage

    def analyze_value_advantage_decomposition(
        self, env: gym.Env, num_samples: int = 50
    ) -> Dict[str, float]:
        """
        Analyzes the value and advantage components of the Dueling DQN.

        Args:
            env: The Gymnasium environment for sampling states.
            num_samples: Number of states to sample for analysis.

        Returns:
            A dictionary containing statistics about the value and advantage components.
        """
        values = []
        advantages_mean = []
        advantages_std = []

        for _ in range(num_samples):
            state, _ = env.reset(seed=DQNConfig.SEED)
            value, advantage = self.get_value_and_advantage(state)
            values.append(value)
            advantages_mean.append(np.mean(advantage))
            advantages_std.append(np.std(advantage))

        return {
            "mean_value": np.mean(values),
            "std_value": np.std(values),
            "mean_advantage": np.mean(advantages_mean),
            "std_advantage": np.mean(advantages_std),
        }


class DuelingDoubleDQNAgent(DoubleDQNAgent):
    """
    Dueling Double DQN Agent combining both improvements.

    Uses Dueling Network architecture with Double DQN's target Q-value calculation.
    """

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig = DQNConfig()):
        super().__init__(state_dim, action_dim, config)
        # Replace Q-network with Dueling Q-network
        self.q_network = DuelingQNetwork(
            state_dim, action_dim, config.HIDDEN_DIM
        ).to(self.device)
        self.target_network = DuelingQNetwork(
            state_dim, action_dim, config.HIDDEN_DIM
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Reinitialize optimizer with new parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LR)


class NoisyDQNAgent(DQNAgent):
    """
    DQN with Noisy Networks for exploration.

    Replaces epsilon-greedy exploration with parameter-space noise.
    """

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig = DQNConfig()):
        super().__init__(state_dim, action_dim, config)
        # Replace Q-network with Noisy Q-network
        self.q_network = NoisyQNetwork(
            state_dim, action_dim, config.HIDDEN_DIM, config.NOISE_STD
        ).to(self.device)
        self.target_network = NoisyQNetwork(
            state_dim, action_dim, config.HIDDEN_DIM, config.NOISE_STD
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Reinitialize optimizer with new parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LR)

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Selects an action using noisy network exploration.

        Args:
            state: The current state of the environment.
            epsilon: Ignored for NoisyDQNAgent as exploration is handled by noise.

        Returns:
            The selected action.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self) -> Optional[float]:
        """
        Performs one training step. Resets noise in NoisyLinear layers.

        Returns:
            The loss value for the training step, or None if buffer is not full enough.
        """
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        return super().train_step()


class RainbowDQNAgent(DQNAgent):
    """
    Rainbow DQN Agent, combining multiple advanced techniques:
    - Double DQN
    - Prioritized Experience Replay (PER)
    - N-step Q-learning
    - Dueling Networks
    - Distributional RL (C51)
    - Noisy Networks
    """

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig = DQNConfig()):
        super().__init__(state_dim, action_dim, config)
        self.n_steps = config.N_STEPS
        self.V_min = config.V_MIN
        self.V_max = config.V_MAX
        self.n_atoms = config.N_ATOMS
        self.delta_z = (self.V_max - self.V_min) / (self.n_atoms - 1)
        self.support = torch.linspace(self.V_min, self.V_max, self.n_atoms).to(self.device)

        # Replace replay buffer with PrioritizedReplayBuffer
        self.replay_buffer = PrioritizedReplayBuffer(config.REPLAY_BUFFER_SIZE, config.PER_ALPHA)

        # Use Dueling Categorical Noisy Q-Network
        self.q_network = CategoricalQNetwork(
            state_dim, action_dim, config.HIDDEN_DIM, self.n_atoms
        ).to(self.device)
        self.target_network = CategoricalQNetwork(
            state_dim, action_dim, config.HIDDEN_DIM, self.n_atoms
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Reinitialize optimizer with new parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LR)

        self.beta = config.PER_BETA_START
        self.beta_frames = config.PER_BETA_FRAMES
        self.frame_idx = 0

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Selects an action using the expected Q-values from the categorical distribution.
        Noisy networks handle exploration intrinsically, so epsilon is ignored.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Get expected Q-values from the categorical network
            q_values = self.q_network.get_q_values(state_tensor, self.support)
            return q_values.argmax().item()

    def project_distribution(self, next_dist: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Projects the target distribution onto the fixed support of the Categorical Q-Network.

        Args:
            next_dist: The next state's Q-value distribution (probabilities).
            rewards: Immediate rewards.
            dones: Done flags.

        Returns:
            The projected target distribution (probabilities).
        """
        batch_size = next_dist.size(0)
        n_atoms = self.n_atoms
        delta_z = self.delta_z
        support = self.support
        gamma = self.gamma

        # Clamp rewards to the value range for stability
        rewards = torch.clamp(rewards, self.V_min, self.V_max)

        # Calculate projected atoms (tz_j)
        tz_j = rewards.unsqueeze(1) + gamma * support.unsqueeze(0) * (1 - dones.unsqueeze(1))
        tz_j = torch.clamp(tz_j, self.V_min, self.V_max)

        # Compute the projection of tz_j onto the original support z_i
        b_j = (tz_j - self.V_min) / delta_z
        l = b_j.floor().long()
        u = b_j.ceil().long()

        # Distribute probabilities
        m = torch.zeros(batch_size, n_atoms, device=self.device)
        offset = torch.linspace(0, (batch_size - 1) * n_atoms, batch_size).long().unsqueeze(1).expand(batch_size, n_atoms)

        m.flatten().index_add_(0, (l + offset).flatten(), (next_dist * (u.float() - b_j)).flatten())
        m.flatten().index_add_(0, (u + offset).flatten(), (next_dist * (b_j - l.float())).flatten())

        return m

    def compute_q_targets(self, rewards, next_states, dones):
        """
        Computes the target Q-value distribution for Rainbow DQN (C51 + N-step).
        Uses Double DQN idea for action selection.
        """
        with torch.no_grad():
            # N-step return calculation is more complex and usually handled within the buffer for PER
            # For now, we'll keep it as 1-step for simplicity in this target computation for C51
            # Proper N-step return requires modifying the replay buffer and its sampling.

            # Double DQN style action selection
            # Online network selects the action
            online_q_values = self.q_network.get_q_values(next_states, self.support)
            next_actions = online_q_values.argmax(1)

            # Target network evaluates the distribution for the selected action
            target_logits = self.target_network(next_states)
            target_probs = F.softmax(target_logits, dim=-1)
            target_action_probs = target_probs[range(self.batch_size), next_actions, :]

            # Project the target distribution
            projected_target_dist = self.project_distribution(target_action_probs, rewards, dones)

        return projected_target_dist

    def train_step(self) -> Optional[float]:
        """
        Performs one training step for Rainbow DQN.
        Includes PER beta annealing and TD-error update for priorities.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Anneal beta
        self.beta = min(1.0, self.beta + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)

        # Sample batch from PER buffer
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(
            self.batch_size, self.beta
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Reset noisy layers for training
        self.q_network.reset_noise()
        self.target_network.reset_noise()

        # Get predicted logits from online network
        current_logits = self.q_network(states)
        current_action_logits = current_logits[range(self.batch_size), actions.long(), :]

        # Compute target Q-value distribution (C51 + N-step + Double DQN)
        # Note: N-step needs proper implementation within the buffer to return n-step rewards/next_states/dones
        # For this step, we assume rewards and next_states are from 1-step for simplicity, will extend later.
        target_projected_dist = self.compute_q_targets(rewards, next_states, dones)

        # Compute C51 loss, weighted by importance sampling weights
        loss = c51_loss(current_action_logits, target_projected_dist)
        loss = (loss * weights).mean()

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        loss_value = loss.item()
        self.losses.append(loss_value)

        # Update priorities in PER buffer (use current TD-error for this)
        with torch.no_grad():
            # Compute TD-errors for priority update (using 1-step for simplicity here)
            current_q_values = self.q_network.get_q_values(states, self.support).gather(1, actions.unsqueeze(1)).squeeze(1)
            target_q_values_for_td = self.target_network.get_q_values(next_states, self.support).max(1)[0]
            td_errors = (rewards + self.gamma * target_q_values_for_td * (1 - dones) - current_q_values).abs().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)

        return loss_value

    def train_episode(self, env: gym.Env, max_steps: int = 1000) -> Tuple[float, Dict]:
        """
        Trains the Rainbow agent for one episode.
        """
        state, _ = env.reset(seed=DQNConfig.SEED)
        total_reward = 0
        steps = 0
        episode_losses = []

        # Reset noisy layers at the start of each episode for exploration
        if isinstance(self.q_network, NoisyQNetwork) or isinstance(self.q_network, CategoricalQNetwork):
             # Assuming CategoricalQNetwork can also have noisy layers or handles noise internally
            if hasattr(self.q_network, 'reset_noise'):
                self.q_network.reset_noise()
            if hasattr(self.target_network, 'reset_noise'):
                self.target_network.reset_noise()

        # For N-step, we need a temporary buffer to store transitions before pushing to PER
        n_step_buffer = deque(maxlen=self.n_steps)

        while steps < max_steps:
            self.frame_idx += 1 # Increment global frame counter for beta annealing

            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # For N-step: store transitions in a temporary buffer
            n_step_buffer.append((state, action, reward, next_state, done))

            if len(n_step_buffer) == self.n_steps or done:
                # Compute N-step return
                n_step_reward = 0
                for i in range(len(n_step_buffer)):
                    _, _, r, _, _ = n_step_buffer[i]
                    n_step_reward += r * (self.gamma ** i)
                
                # Get final state and done status for the N-step transition
                _, _, _, last_next_state, last_done = n_step_buffer[-1]
                first_state, first_action, _, _, _ = n_step_buffer[0]

                # Push N-step transition to the replay buffer
                self.replay_buffer.push(first_state, first_action, n_step_reward, last_next_state, last_done)
                
                # Clear buffer if episode is done or N-steps reached, and not yet at end of episode
                if done:
                    n_step_buffer.clear()
                elif len(n_step_buffer) == self.n_steps:
                    # Remove the oldest experience to maintain n_steps window
                    n_step_buffer.popleft()


            # Train
            loss = self.train_step()
            if loss is not None:
                episode_losses.append(loss)

            # Epsilon update is implicitly handled by Noisy Networks, but we keep the method
            # for consistency with parent class and potential future use for other exploration.
            if not (isinstance(self.q_network, NoisyQNetwork) or isinstance(self.q_network, CategoricalQNetwork)):
                self.update_epsilon()

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        return total_reward, {
            "steps": steps,
            "avg_loss": np.mean(episode_losses) if episode_losses else 0,
            "epsilon": self.epsilon, # This will be effectively static for noisy nets unless updated elsewhere
            "beta": self.beta
        }
