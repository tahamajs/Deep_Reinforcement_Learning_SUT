import numpy as np
class Replay_Memory:

    def __init__(self, state_dim, action_dim, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.states = np.zeros((self.memory_size, state_dim))
        self.next_states = np.zeros((self.memory_size, state_dim))
        self.actions = np.zeros((self.memory_size, 1))
        self.rewards = np.zeros((self.memory_size, 1))
        self.dones = np.zeros((self.memory_size, 1))
        self.ptr = 0
        self.burned_in = False
        self.not_full_yet = True

    def append(self, states, actions, rewards, next_states, dones):
        self.states[self.ptr] = states
        self.actions[self.ptr, 0] = actions
        self.rewards[self.ptr, 0] = rewards
        self.next_states[self.ptr] = next_states
        self.dones[self.ptr, 0] = dones
        self.ptr += 1

        if self.ptr > self.burn_in:
            self.burned_in = True

        if self.ptr >= self.memory_size:
            self.ptr = 0
            self.not_full_yet = False

    def sample_batch(self, batch_size=32):
        if self.not_full_yet:
            idxs = np.random.choice(self.ptr, batch_size, False)
        else:
            idxs = np.random.choice(self.memory_size, batch_size, False)

        states = self.states[idxs]
        next_states = self.next_states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]
        return states, actions, rewards, next_states, dones