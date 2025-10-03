"""
Behavioral Cloning Implementation

This module implements behavioral cloning for imitation learning.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import tensorflow as tf

# TensorFlow 2.x compatibility
if hasattr(tf, '__version__') and int(tf.__version__.split('.')[0]) >= 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import numpy as np
from tf_util import function, initialize
class BehavioralCloning:
    """Behavioral Cloning agent."""

    def __init__(self, env, learning_rate=1e-3, hidden_sizes=[100, 100]):
        """Initialize the BC agent.

        Args:
            env: Gym environment
            learning_rate: Learning rate for optimizer
            hidden_sizes: List of hidden layer sizes
        """
        self.env = env
        self.learning_rate = learning_rate
        self.hidden_sizes = hidden_sizes
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self._build_network()
        self.sess = tf.Session()
        initialize()

    def _build_network(self):
        """Build the neural network for policy."""
        self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_dim])
        self.act_ph = tf.placeholder(tf.float32, [None, self.act_dim])
        layer = self.obs_ph
        for size in self.hidden_sizes:
            layer = tf.compat.v1.layers.dense(layer, size, activation=tf.nn.relu)

        self.predicted_actions = tf.compat.v1.layers.dense(layer, self.act_dim, activation=None)
        self.loss = tf.reduce_mean(tf.square(self.predicted_actions - self.act_ph))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def train(self, observations, actions, epochs=100, batch_size=64):
        """Train the policy using expert data.

        Args:
            observations: Expert observations (N, obs_dim)
            actions: Expert actions (N, act_dim)
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            dict: Training history
        """
        # Ensure actions are 2D
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        
        losses = []

        for epoch in range(epochs):

            indices = np.random.permutation(len(observations))
            obs_shuffled = observations[indices]
            act_shuffled = actions[indices]

            epoch_loss = 0
            num_batches = 0
            for i in range(0, len(observations), batch_size):
                obs_batch = obs_shuffled[i : i + batch_size]
                act_batch = act_shuffled[i : i + batch_size]

                loss_val, _ = self.sess.run(
                    [self.loss, self.train_op],
                    feed_dict={self.obs_ph: obs_batch, self.act_ph: act_batch},
                )
                epoch_loss += loss_val
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return {"losses": losses}

    def get_action(self, observation):
        """Get action for given observation.

        Args:
            observation: Single observation

        Returns:
            action: Predicted action
        """
        obs = observation.reshape(1, -1)
        action = self.sess.run(self.predicted_actions, feed_dict={self.obs_ph: obs})
        return action[0]

    def evaluate(self, num_episodes=10, render=False):
        """Evaluate the trained policy.

        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes

        Returns:
            dict: Evaluation results
        """
        returns = []

        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = self.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1

                if render:
                    self.env.render()

                if steps >= self.env.spec.timestep_limit:
                    break

            returns.append(total_reward)
            print(f"Episode {episode + 1}: Return = {total_reward:.2f}")

        return {
            "returns": returns,
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
        }