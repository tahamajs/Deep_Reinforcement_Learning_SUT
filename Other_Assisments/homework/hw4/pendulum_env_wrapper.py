"""
Pendulum Environment Wrapper for Model-Based RL

This wraps gym's Pendulum-v0 to provide a cost_fn interface compatible with HW4's model-based RL,
allowing experiments without MuJoCo.

Author: GitHub Copilot
Date: October 4, 2025
"""

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import gym


class PendulumEnvWrapper:
    """Wraps Pendulum-v0 to match HalfCheetah interface for HW4."""
    
    def __init__(self):
        self._env = gym.make('Pendulum-v0')
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        
    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        return self._env.step(action)
    
    def render(self):
        return self._env.render()
    
    @staticmethod
    def cost_fn(states, actions, next_states):
        """
        Cost function for Pendulum.
        
        Pendulum reward is: -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
        We negate this to get cost (lower cost = better).
        
        Args:
            states: shape [batch_size, 3] (cos(theta), sin(theta), theta_dt)
            actions: shape [batch_size, 1]
            next_states: shape [batch_size, 3]
        
        Returns:
            costs: shape [batch_size] or scalar
        """
    is_tf = tf.is_tensor(states)
    is_single_state = (len(states.get_shape()) == 1) if is_tf else (len(np.shape(states)) == 1)
        
        if is_single_state:
            states = states[None, ...]
            actions = actions[None, ...]
            next_states = next_states[None, ...]
        
        # Extract components: [cos(theta), sin(theta), theta_dt]
        cos_theta = states[:, 0]
        sin_theta = states[:, 1]
        theta_dt = states[:, 2]
        
        # Compute angle from cos/sin
        if is_tf:
            theta = tf.atan2(sin_theta, cos_theta)
            angle_cost = tf.square(theta)
            velocity_cost = 0.1 * tf.square(theta_dt)
            action_cost = 0.001 * tf.reduce_sum(tf.square(actions), axis=1)
        else:
            theta = np.arctan2(sin_theta, cos_theta)
            angle_cost = theta ** 2
            velocity_cost = 0.1 * (theta_dt ** 2)
            action_cost = 0.001 * np.sum(actions ** 2, axis=1)
        
        costs = angle_cost + velocity_cost + action_cost
        
        if is_single_state:
            costs = costs[0]
        
        return costs
