"""
Meta-Learning Agent

This module contains meta-learning algorithms for few-shot adaptation.

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


def v1_dense(x, units, activation=None, name="dense"):
    """Lightweight Dense layer compatible with TF v1 graph mode and Keras 3."""
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
    """Replay buffer for meta-learning."""

    def __init__(self, max_size=10000):
        """Initialize replay buffer.

        Args:
            max_size: Maximum buffer size
        """
        self.buffer = deque(maxlen=max_size)

    def add(self, trajectory):
        """Add trajectory to buffer.

        Args:
            trajectory: Dictionary with 'states', 'actions', 'rewards', 'dones'
        """
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        """Sample batch of trajectories."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def size(self):
        """Return buffer size."""
        return len(self.buffer)
class MetaLearningAgent:
    """Meta-learning agent for few-shot adaptation."""

    def __init__(
        self,
        env,
        hidden_sizes=[64, 64],
        learning_rate=1e-3,
        meta_learning_rate=1e-3,
        adaptation_steps=5,
        meta_batch_size=4,
        discount=0.99,
    ):
        """Initialize meta-learning agent.

        Args:
            env: Environment
            hidden_sizes: Hidden layer sizes
            learning_rate: Inner loop learning rate
            meta_learning_rate: Outer loop learning rate
            adaptation_steps: Number of adaptation steps
            meta_batch_size: Meta batch size
            discount: Discount factor
        """
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        # Support both continuous and discrete action spaces
        if hasattr(env.action_space, "shape") and env.action_space.shape is not None and len(env.action_space.shape) > 0:
            self.action_dim = env.action_space.shape[0]
            self.discrete = False
        else:
            self.action_dim = env.action_space.n
            self.discrete = True
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.adaptation_steps = adaptation_steps
        self.meta_batch_size = meta_batch_size
        self.discount = discount
        self.replay_buffer = ReplayBuffer()
        self._build_networks()
        self.sess = None

    def _build_networks(self):
        """Build policy network and meta-learning components."""

        self.state_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.action_ph = tf.placeholder(tf.float32, [None, self.action_dim])
        self.advantage_ph = tf.placeholder(tf.float32, [None])
        self.old_log_prob_ph = tf.placeholder(tf.float32, [None])
        self.action_logits, self.value = self._build_policy(self.state_ph)
        if self.discrete:
            self.action_dist = tf.distributions.Categorical(logits=self.action_logits)
            self.sampled_action = self.action_dist.sample()
            self.log_prob = self.action_dist.log_prob(self.sampled_action)
        else:
            # For continuous actions, model Gaussian policy (simple baseline)
            self.action_mean = self.action_logits
            self.action_log_std = tf.get_variable(
                "action_log_std", shape=[self.action_dim], initializer=tf.zeros_initializer()
            )
            dist = tf.distributions.Normal(self.action_mean, tf.exp(self.action_log_std))
            self.sampled_action = dist.sample()
            self.log_prob = tf.reduce_sum(dist.log_prob(self.sampled_action), axis=1)
        ratio = tf.exp(self.log_prob - self.old_log_prob_ph)
        clipped_ratio = tf.clip_by_value(ratio, 0.8, 1.2)
        self.policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * self.advantage_ph, clipped_ratio * self.advantage_ph)
        )
        self.value_loss = tf.reduce_mean((self.value - self.advantage_ph) ** 2)
        self.total_loss = self.policy_loss + 0.5 * self.value_loss
        self.optimizer = tf.train.AdamOptimizer(self.meta_learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss)

    def _build_policy(self, state):
        """Build policy network."""
        with tf.variable_scope("policy"):
            x = state
            for i, size in enumerate(self.hidden_sizes):
                x = v1_dense(x, size, activation=tf.nn.tanh, name=f"h{i}")
            action_logits = v1_dense(x, self.action_dim, name="act")
            value = v1_dense(x, 1, name="val")

            return action_logits, tf.squeeze(value)

    def init_tf_sess(self):
        """Initialize TensorFlow session."""
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state, deterministic=False):
        """Get action from policy.

        Args:
            state: Current state
            deterministic: Whether to return deterministic action

        Returns:
            Action and log probability
        """
        feed_dict = {self.state_ph: state.reshape(1, -1)}
        action, log_prob = self.sess.run(
            [self.sampled_action, self.log_prob], feed_dict=feed_dict
        )
        return action[0], log_prob[0]

    def compute_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute generalized advantage estimates.

        Args:
            rewards: Rewards
            values: Value estimates
            dones: Done flags
            gamma: Discount factor
            lam: GAE lambda

        Returns:
            Advantages
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                next_advantage = 0
            else:
                next_value = values[t + 1] if t + 1 < len(values) else 0
                next_advantage = advantages[t + 1] if t + 1 < len(advantages) else 0

            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = delta + gamma * lam * next_advantage

        return advantages

    def adapt_to_task(self, task_data, adaptation_steps=None):
        """Adapt policy to a new task.

        Args:
            task_data: Task trajectory data
            adaptation_steps: Number of adaptation steps

        Returns:
            Adapted policy parameters
        """
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps
        original_params = self._get_policy_params()
        for step in range(adaptation_steps):

            states = task_data["states"]
            actions = task_data["actions"]
            rewards = task_data["rewards"]
            dones = task_data["dones"]
            feed_dict = {self.state_ph: states}
            values = self.sess.run(self.value, feed_dict=feed_dict)

            advantages = self.compute_advantages(rewards, values, dones, self.discount)
            feed_dict = {self.state_ph: states, self.action_ph: actions}
            old_log_probs = self.sess.run(self.log_prob, feed_dict=feed_dict)
            feed_dict = {
                self.state_ph: states,
                self.action_ph: actions,
                self.advantage_ph: advantages,
                self.old_log_prob_ph: old_log_probs,
            }

            self.sess.run(self.train_op, feed_dict=feed_dict)
        adapted_params = self._get_policy_params()
        self._set_policy_params(original_params)

        return adapted_params

    def _get_policy_params(self):
        """Get current policy parameters."""
        policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy")
        return self.sess.run(policy_vars)

    def _set_policy_params(self, params):
        """Set policy parameters."""
        policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy")
        assign_ops = [tf.assign(var, param) for var, param in zip(policy_vars, params)]
        self.sess.run(assign_ops)

    def meta_train_step(self):
        """Perform one meta-training step."""
        if self.replay_buffer.size() < self.meta_batch_size:
            return
        task_batch = self.replay_buffer.sample(self.meta_batch_size)

        meta_gradients = []
        original_params = self._get_policy_params()

        for task_data in task_batch:

            adapted_params = self.adapt_to_task(task_data)
            pass
        self._set_policy_params(original_params)

    def add_task_trajectory(self, trajectory):
        """Add task trajectory to replay buffer.

        Args:
            trajectory: Dictionary with trajectory data
        """
        self.replay_buffer.add(trajectory)
class MAMLAgent(MetaLearningAgent):
    """Model-Agnostic Meta-Learning agent."""

    def __init__(
        self,
        env,
        hidden_sizes=[64, 64],
        learning_rate=1e-3,
        meta_learning_rate=1e-3,
        adaptation_steps=5,
        meta_batch_size=4,
        discount=0.99,
    ):
        """Initialize MAML agent."""
        super().__init__(
            env,
            hidden_sizes,
            learning_rate,
            meta_learning_rate,
            adaptation_steps,
            meta_batch_size,
            discount,
        )

    def adapt_to_task(self, task_data, adaptation_steps=None):
        """MAML adaptation with second-order gradients."""
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps
        original_params = self._get_policy_params()
        for step in range(adaptation_steps):

            states = task_data["states"]
            actions = task_data["actions"]
            rewards = task_data["rewards"]
            dones = task_data["dones"]
            feed_dict = {self.state_ph: states}
            values = self.sess.run(self.value, feed_dict=feed_dict)

            advantages = self.compute_advantages(rewards, values, dones, self.discount)
            feed_dict = {self.state_ph: states, self.action_ph: actions}
            old_log_probs = self.sess.run(self.log_prob, feed_dict=feed_dict)
            feed_dict = {
                self.state_ph: states,
                self.action_ph: actions,
                self.advantage_ph: advantages,
                self.old_log_prob_ph: old_log_probs,
            }

            grads = tf.gradients(
                self.total_loss,
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy"),
            )
            computed_grads = self.sess.run(grads, feed_dict=feed_dict)
            policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy")
            current_params = self.sess.run(policy_vars)

            new_params = []
            for param, grad in zip(current_params, computed_grads):
                if grad is not None:
                    new_params.append(param - self.learning_rate * grad)
                else:
                    new_params.append(param)

            self._set_policy_params(new_params)
        adapted_params = self._get_policy_params()
        self._set_policy_params(original_params)

        return adapted_params