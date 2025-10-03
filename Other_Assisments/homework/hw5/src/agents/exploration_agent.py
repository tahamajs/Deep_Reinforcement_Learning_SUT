"""
Exploration Agent

This module contains exploration methods with density models and reward bonuses.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np
try:
    import tensorflow.compat.v1 as tf  # type: ignore
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf  # type: ignore
class DensityModel:
    """Density model for exploration."""

    def __init__(self, state_dim, hidden_dims=[64, 64], learning_rate=1e-3):
        """Initialize density model.

        Args:
            state_dim: State dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
        """
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        self._build_model()

    def _build_model(self):
        """Build the density model network."""
        self.state_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        x = self.state_ph
        for dim in self.hidden_dims:
            x = tf.layers.dense(x, dim, activation=tf.nn.relu)
        self.log_density = tf.layers.dense(x, 1)
        self.target_ph = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.log_density, labels=self.target_ph
            )
        )
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def fit(self, states, sess, epochs=100, batch_size=512):
        """Fit density model to data.

        Args:
            states: State data
            sess: TensorFlow session
            epochs: Number of training epochs
            batch_size: Batch size
        """
        n_samples = len(states)

        for epoch in range(epochs):

            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_states = states[batch_indices]
                targets = np.ones((len(batch_states), 1))

                feed_dict = {self.state_ph: batch_states, self.target_ph: targets}

                sess.run(self.train_op, feed_dict=feed_dict)

    def predict_log_density(self, states, sess):
        """Predict log density for states.

        Args:
            states: States to evaluate
            sess: TensorFlow session

        Returns:
            Log density predictions
        """
        feed_dict = {self.state_ph: states}
        return sess.run(self.log_density, feed_dict=feed_dict)
class ExplorationAgent:
    """Exploration agent with reward bonuses."""

    def __init__(
        self,
        state_dim,
        bonus_coeff=1.0,
        density_hidden_dims=[64, 64],
        density_learning_rate=1e-3,
    ):
        """Initialize exploration agent.

        Args:
            state_dim: State dimension
            bonus_coeff: Bonus coefficient
            density_hidden_dims: Hidden dimensions for density model
            density_learning_rate: Learning rate for density model
        """
        self.state_dim = state_dim
        self.bonus_coeff = bonus_coeff
        self.density_model = DensityModel(
            state_dim, density_hidden_dims, density_learning_rate
        )
        self.sess = None

    def init_tf_sess(self):
        """Initialize TensorFlow session."""
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def fit_density_model(self, states, epochs=100, batch_size=512):
        """Fit density model to states.

        Args:
            states: State data for fitting
            epochs: Training epochs
            batch_size: Batch size
        """
        self.density_model.fit(states, self.sess, epochs, batch_size)

    def compute_reward_bonus(self, states):
        """Compute exploration bonus for states.

        Args:
            states: States to compute bonus for

        Returns:
            Reward bonuses
        """
        log_density = self.density_model.predict_log_density(states, self.sess)

        bonus = -log_density.flatten()
        return bonus

    def modify_reward(self, rewards, states):
        """Modify rewards with exploration bonus.

        Args:
            rewards: Original rewards
            states: Corresponding states

        Returns:
            Modified rewards
        """
        bonus = self.compute_reward_bonus(states)
        new_rewards = rewards + self.bonus_coeff * bonus
        return new_rewards
class DiscreteExplorationAgent(ExplorationAgent):
    """Discrete exploration agent for discrete action spaces."""

    def __init__(
        self,
        state_dim,
        num_actions,
        bonus_coeff=1.0,
        density_hidden_dims=[64, 64],
        density_learning_rate=1e-3,
    ):
        """Initialize discrete exploration agent.

        Args:
            state_dim: State dimension
            num_actions: Number of discrete actions
            bonus_coeff: Bonus coefficient
            density_hidden_dims: Hidden dimensions for density model
            density_learning_rate: Learning rate for density model
        """
        super().__init__(
            state_dim, bonus_coeff, density_hidden_dims, density_learning_rate
        )
        self.num_actions = num_actions
        self.state_action_dim = state_dim + num_actions
        self.state_action_density = DensityModel(
            self.state_action_dim, density_hidden_dims, density_learning_rate
        )

    def fit_density_model(self, states, actions, epochs=100, batch_size=512):
        """Fit density model to state-action pairs.

        Args:
            states: State data
            actions: Action data (one-hot encoded)
            epochs: Training epochs
            batch_size: Batch size
        """

        state_actions = np.concatenate([states, actions], axis=1)
        self.state_action_density.fit(state_actions, self.sess, epochs, batch_size)

    def compute_reward_bonus(self, states, actions):
        """Compute exploration bonus for state-action pairs.

        Args:
            states: States
            actions: Actions (one-hot encoded)

        Returns:
            Reward bonuses
        """
        state_actions = np.concatenate([states, actions], axis=1)
        log_density = self.state_action_density.predict_log_density(
            state_actions, self.sess
        )
        bonus = -log_density.flatten()
        return bonus

    def modify_reward(self, rewards, states, actions):
        """Modify rewards with exploration bonus.

        Args:
            rewards: Original rewards
            states: States
            actions: Actions (one-hot encoded)

        Returns:
            Modified rewards
        """
        bonus = self.compute_reward_bonus(states, actions)
        new_rewards = rewards + self.bonus_coeff * bonus
        return new_rewards