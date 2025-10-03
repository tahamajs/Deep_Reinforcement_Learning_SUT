"""
Soft Actor-Critic (SAC) Agent

This module contains the SAC agent implementation with actor, critic, and value networks.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np
try:
    import tensorflow.compat.v1 as tf  # type: ignore
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf  # type: ignore
from collections import deque
import random


def v1_dense(x, units, activation=None, name="dense"):
    """Lightweight Dense layer compatible with TF v1 graph mode and Keras 3.

    Avoids tf.layers/keras dependencies which are removed in Keras 3.
    """
    input_dim = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable(
            "kernel",
            shape=[input_dim, units],
            initializer=tf.glorot_uniform_initializer(),
        )
        b = tf.get_variable(
            "bias",
            shape=[units],
            initializer=tf.zeros_initializer(),
        )
        z = tf.matmul(x, w) + b
        return activation(z) if activation is not None else z
class ReplayBuffer:
    """Experience replay buffer for SAC."""

    def __init__(self, max_size=1000000):
        """Initialize replay buffer.

        Args:
            max_size: Maximum buffer size
        """
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_states": np.array(next_states),
            "dones": np.array(dones),
        }

    def size(self):
        """Return current buffer size."""
        return len(self.buffer)
class SACAgent:
    """Soft Actor-Critic Agent."""

    def __init__(
        self,
        env,
        hidden_sizes=[256, 256],
        learning_rate=3e-3,
        alpha=1.0,
        batch_size=256,
        discount=0.99,
        tau=0.01,
        reparameterize=False,
        buffer_size=1000000,
    ):
        """Initialize SAC agent.

        Args:
            env: Environment
            hidden_sizes: Hidden layer sizes for networks
            learning_rate: Learning rate for optimization
            alpha: Temperature parameter for entropy
            batch_size: Batch size for training
            discount: Discount factor
            tau: Soft update coefficient
            reparameterize: Whether to use reparameterization trick
            buffer_size: Replay buffer size
        """
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.reparameterize = reparameterize
        self.replay_buffer = ReplayBuffer(buffer_size)
        self._build_networks()
        self.sess = None

    def _build_networks(self):
        """Build actor, critic, and value networks."""

        self.state_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.action_ph = tf.placeholder(tf.float32, [None, self.action_dim])
        self.reward_ph = tf.placeholder(tf.float32, [None])
        self.next_state_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.done_ph = tf.placeholder(tf.float32, [None])
        self.actor_mean, self.actor_log_std = self._build_actor(self.state_ph)
        self.actor_action, self.actor_log_prob = self._sample_action(
            self.actor_mean, self.actor_log_std
        )
        self.q1 = self._build_critic(self.state_ph, self.action_ph, "q1")
        self.q2 = self._build_critic(self.state_ph, self.action_ph, "q2")
        self.target_q1 = self._build_critic(
            self.next_state_ph, self.actor_action, "target_q1", reuse=False
        )
        self.target_q2 = self._build_critic(
            self.next_state_ph, self.actor_action, "target_q2", reuse=False
        )
        self.value = self._build_value(self.state_ph, "value")
        self.target_value = self._build_value(self.next_state_ph, "target_value")
        self._build_target_updates()
        self._build_losses()
        self._build_optimizers()

    def _build_actor(self, state):
        """Build actor network (policy)."""
        with tf.variable_scope("actor"):
            x = state
            for i, size in enumerate(self.hidden_sizes):
                x = v1_dense(x, size, activation=tf.nn.relu, name=f"h{i}")

            mean = v1_dense(x, self.action_dim, activation=tf.nn.tanh, name="mean")
            mean = mean * self.action_high

            log_std = v1_dense(
                x, self.action_dim, activation=tf.nn.tanh, name="log_std"
            )
            log_std = tf.clip_by_value(log_std, -20, 2)

            return mean, log_std

    def _build_critic(self, state, action, scope, reuse=False):
        """Build critic network (Q-function)."""
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.concat([state, action], axis=1)
            for i, size in enumerate(self.hidden_sizes):
                x = v1_dense(x, size, activation=tf.nn.relu, name=f"h{i}")
            return v1_dense(x, 1, name="out")

    def _build_value(self, state, scope, reuse=False):
        """Build value network."""
        with tf.variable_scope(scope, reuse=reuse):
            x = state
            for i, size in enumerate(self.hidden_sizes):
                x = v1_dense(x, size, activation=tf.nn.relu, name=f"h{i}")
            return v1_dense(x, 1, name="out")

    def _sample_action(self, mean, log_std):
        """Sample action from policy distribution."""
        if self.reparameterize:

            std = tf.exp(log_std)
            noise = tf.random_normal(tf.shape(mean))
            action = mean + std * noise
        else:

            dist = tf.distributions.Normal(mean, tf.exp(log_std))
            action = dist.sample()
        action = tf.tanh(action) * self.action_high
        log_prob = self._compute_log_prob(mean, log_std, action)

        return action, log_prob

    def _compute_log_prob(self, mean, log_std, action):
        """Compute log probability of action."""

        unsquashed_action = tf.atanh(
            tf.clip_by_value(action / self.action_high, -0.999, 0.999)
        )
        dist = tf.distributions.Normal(mean, tf.exp(log_std))
        log_prob = dist.log_prob(unsquashed_action)
        log_prob -= tf.reduce_sum(
            tf.log(1 - tf.tanh(unsquashed_action) ** 2 + 1e-6), axis=1
        )

        return log_prob

    def _build_target_updates(self):
        """Build target network update operations."""

        actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "actor")
        q1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q1")
        q2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q2")
        value_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value")
        target_q1_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "target_q1"
        )
        target_q2_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "target_q2"
        )
        target_value_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "target_value"
        )
        self.update_target_q1 = tf.group(
            *[
                tf.assign(target_var, self.tau * var + (1 - self.tau) * target_var)
                for var, target_var in zip(q1_vars, target_q1_vars)
            ]
        )
        self.update_target_q2 = tf.group(
            *[
                tf.assign(target_var, self.tau * var + (1 - self.tau) * target_var)
                for var, target_var in zip(q2_vars, target_q2_vars)
            ]
        )
        self.update_target_value = tf.group(
            *[
                tf.assign(target_var, self.tau * var + (1 - self.tau) * target_var)
                for var, target_var in zip(value_vars, target_value_vars)
            ]
        )

    def _build_losses(self):
        """Build loss functions."""

        target_q = tf.minimum(self.target_q1, self.target_q2)
        q_target = (
            self.reward_ph + self.discount * (1 - self.done_ph) * self.target_value
        )

        self.q1_loss = tf.reduce_mean((self.q1 - q_target) ** 2)
        self.q2_loss = tf.reduce_mean((self.q2 - q_target) ** 2)
        q_pred = tf.minimum(self.q1, self.q2)
        self.value_loss = tf.reduce_mean(
            (self.value - (q_pred - self.alpha * self.actor_log_prob)) ** 2
        )
        self.actor_loss = tf.reduce_mean(self.alpha * self.actor_log_prob - q_pred)

    def _build_optimizers(self):
        """Build optimizers."""
        self.q1_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.q2_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.value_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.q1_train_op = self.q1_optimizer.minimize(self.q1_loss)
        self.q2_train_op = self.q2_optimizer.minimize(self.q2_loss)
        self.value_train_op = self.value_optimizer.minimize(self.value_loss)
        self.actor_train_op = self.actor_optimizer.minimize(self.actor_loss)

    def init_tf_sess(self):
        """Initialize TensorFlow session."""
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(
            [
                tf.assign(target_var, var)
                for target_var, var in zip(
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_q1"),
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q1"),
                )
            ]
        )
        self.sess.run(
            [
                tf.assign(target_var, var)
                for target_var, var in zip(
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_q2"),
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q2"),
                )
            ]
        )
        self.sess.run(
            [
                tf.assign(target_var, var)
                for target_var, var in zip(
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_value"),
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value"),
                )
            ]
        )

    def get_action(self, state, deterministic=False):
        """Get action from policy."""
        if deterministic:

            feed_dict = {self.state_ph: state.reshape(1, -1)}
            action = self.sess.run(self.actor_mean, feed_dict=feed_dict)[0]
        else:

            feed_dict = {self.state_ph: state.reshape(1, -1)}
            action = self.sess.run(self.actor_action, feed_dict=feed_dict)[0]
        return np.clip(action, self.action_low, self.action_high)

    def train_step(self):
        """Perform one training step."""
        if self.replay_buffer.size() < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)

        feed_dict = {
            self.state_ph: batch["states"],
            self.action_ph: batch["actions"],
            self.reward_ph: batch["rewards"],
            self.next_state_ph: batch["next_states"],
            self.done_ph: batch["dones"],
        }
        self.sess.run([self.q1_train_op, self.q2_train_op], feed_dict=feed_dict)
        self.sess.run(self.value_train_op, feed_dict=feed_dict)
        self.sess.run(self.actor_train_op, feed_dict=feed_dict)
        self.sess.run(
            [self.update_target_q1, self.update_target_q2, self.update_target_value]
        )

    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)