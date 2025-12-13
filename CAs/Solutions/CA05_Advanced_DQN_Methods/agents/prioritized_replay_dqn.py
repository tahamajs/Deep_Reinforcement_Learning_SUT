import torch
import torch.nn.functional as F
import numpy as np

from CAs.Solutions.CA05_Advanced_DQN_Methods.agents.dqn_base import DQNAgent, Transition
from CAs.Solutions.CA05_Advanced_DQN_Methods.utils.replay_buffers import PrioritizedReplayBuffer


class PrioritizedDQNAgent(DQNAgent):
    """
    DQN Agent with Prioritized Experience Replay (PER).

    PER improves training efficiency by prioritizing experiences with higher TD-error,
    meaning the agent learns more from surprising or important transitions.
    This implementation extends the base DQNAgent to use a PrioritizedReplayBuffer.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the PrioritizedDQNAgent.

        Overrides the replay buffer with a PrioritizedReplayBuffer.

        Args:
            *args: Variable length argument list for DQNAgent.
            **kwargs: Arbitrary keyword arguments for DQNAgent.
        """
        super().__init__(*args, **kwargs)
        # Override replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            kwargs.get("buffer_size", 50000),
            alpha=kwargs.get("priority_alpha", 0.6)
        )
        self.beta_start = kwargs.get("priority_beta_start", 0.4)
        self.beta_frames = kwargs.get("priority_beta_frames", 100000)
        self.beta = self.beta_start

    def update(self) -> float:
        """
        Updates the Q-network using the Prioritized DQN algorithm.

        This involves sampling a batch of transitions with importance sampling weights
        from the Prioritized Replay Buffer, calculating the target Q-values,
        and performing a weighted gradient descent step on the online Q-network.
        Priorities in the buffer are updated based on the new TD-errors.

        Returns:
            float: The loss value from the update step.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Linearly increase beta over time
        self.beta = min(1.0, self.beta_start + self.steps * (1.0 - self.beta_start) / self.beta_frames)

        # Sample batch with priorities and importance sampling weights
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size, beta=self.beta)
        if not transitions:
            return 0.0

        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device) # Importance sampling weights

        # Current Q values from online network
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values (standard DQN for simplicity in PER update, Double DQN can also be combined)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Calculate TD errors for priority updates
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy() + 1e-6 # Add epsilon for non-zero priority

        # Weighted loss using importance sampling weights
        loss = (weights * F.mse_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()
