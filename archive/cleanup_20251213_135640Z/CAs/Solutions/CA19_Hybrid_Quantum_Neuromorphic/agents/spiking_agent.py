import numpy as np
from typing import List

class SpikingAgent:
    """
    A neuromorphic reinforcement learning agent using spiking neural network principles.
    """
    def __init__(self, state_dim: int, action_dim: int, threshold: float = 1.0, 
                 tau_membrane: float = 0.9, learning_rate: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.threshold = threshold
        self.tau_membrane = tau_membrane
        self.learning_rate = learning_rate
        self.membrane_potential = np.zeros(action_dim)
        self.last_spike_time = np.zeros(action_dim)
        self.weights = np.random.randn(state_dim, action_dim) * 0.1
        self.spike_history = []
        self.reward_trace = 0.0
        self.time_step = 0

    def select_action(self, state: np.ndarray) -> int:
        self.time_step += 1
        input_current = np.dot(state, self.weights)
        self.membrane_potential *= self.tau_membrane
        self.membrane_potential += input_current
        spikes = self.membrane_potential >= self.threshold
        if np.any(spikes):
            spiking_neurons = np.where(spikes)[0]
            action = spiking_neurons[np.argmax(self.membrane_potential[spikes])]
            self.membrane_potential[action] = 0.0
            self.last_spike_time[action] = self.time_step
            self.spike_history.append({
                'time': self.time_step,
                'action': action,
                'state': state.copy()
            })
            return action
        else:
            return np.random.randint(self.action_dim)

    def update(self, states: List[np.ndarray], actions: List[int], rewards: List[float]):
        if not states:
            return
        total_reward = sum(rewards)
        self.reward_trace = 0.9 * self.reward_trace + 0.1 * total_reward
        for state, action, reward in zip(states, actions, rewards):
            if reward > 0:
                self.weights[:, action] += self.learning_rate * state * reward
            else:
                self.weights[:, action] -= self.learning_rate * state * abs(reward) * 0.1
        self.weights = np.clip(self.weights, -2.0, 2.0)
        if self.time_step % 100 == 0:
            weight_norm = np.linalg.norm(self.weights, axis=0)
            self.weights /= (weight_norm + 1e-8)

    def get_spike_rate(self) -> float:
        recent_spikes = [s for s in self.spike_history if self.time_step - s['time'] < 50]
        return len(recent_spikes) / 50.0 if recent_spikes else 0.0
