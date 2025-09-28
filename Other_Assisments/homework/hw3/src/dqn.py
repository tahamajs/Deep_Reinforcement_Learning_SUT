"""
DQN Neural Network Components

This module contains neural network architectures for Deep Q-Learning.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers
from dqn_utils import huber_loss


def build_q_network(input_shape, num_actions, scope, reuse=False):
    """Build a Q-network for DQN.

    Args:
        input_shape: Shape of input observations
        num_actions: Number of possible actions
        scope: Variable scope for the network
        reuse: Whether to reuse variables

    Returns:
        q_values: Q-values for each action
        network_vars: List of network variables
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Convolutional layers for image inputs (Atari)
        if len(input_shape) == 3:
            # Input: (height, width, channels * frame_history)
            obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
            obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0

            # Convolutional layers
            conv1 = layers.conv2d(obs_t_float, 32, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
            conv2 = layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu, scope="conv2")
            conv3 = layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu, scope="conv3")

            # Flatten and fully connected layers
            flattened = layers.flatten(conv3)
            fc1 = layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu, scope="fc1")
            q_values = layers.fully_connected(fc1, num_actions, activation_fn=None, scope="q_values")

        # Fully connected layers for low-dimensional inputs (LunarLander)
        else:
            obs_t_ph = tf.placeholder(tf.float32, [None] + list(input_shape))
            fc1 = layers.fully_connected(obs_t_ph, 64, activation_fn=tf.nn.relu, scope="fc1")
            fc2 = layers.fully_connected(fc1, 64, activation_fn=tf.nn.relu, scope="fc2")
            q_values = layers.fully_connected(fc2, num_actions, activation_fn=None, scope="q_values")

        network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    return obs_t_ph, q_values, network_vars


class DQNAgent:
    """Deep Q-Learning agent with experience replay and target networks."""

    def __init__(self, env, optimizer_spec, session, exploration_schedule,
                 replay_buffer_size=1000000, batch_size=32, gamma=0.99,
                 learning_starts=50000, learning_freq=4, frame_history_len=4,
                 target_update_freq=10000, grad_norm_clipping=10, double_q=True):
        """Initialize DQN agent.

        Args:
            env: Gym environment
            optimizer_spec: Optimizer specification
            session: TensorFlow session
            exploration_schedule: Epsilon-greedy exploration schedule
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            learning_starts: Timesteps before learning begins
            learning_freq: Learning frequency
            frame_history_len: Number of frames to stack
            target_update_freq: Target network update frequency
            grad_norm_clipping: Gradient norm clipping
            double_q: Whether to use double Q-learning
        """
        self.env = env
        self.optimizer_spec = optimizer_spec
        self.sess = session
        self.exploration = exploration_schedule
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.frame_history_len = frame_history_len
        self.target_update_freq = target_update_freq
        self.grad_norm_clipping = grad_norm_clipping
        self.double_q = double_q

        # Environment properties
        if len(self.env.observation_space.shape) == 1:
            # Low-dimensional observations (LunarLander)
            input_shape = self.env.observation_space.shape
        else:
            # Image observations (Atari)
            img_h, img_w, img_c = self.env.observation_space.shape
            input_shape = (img_h, img_w, frame_history_len * img_c)

        self.num_actions = self.env.action_space.n

        # Build Q-networks
        self.obs_t_ph, self.q_t, self.q_func_vars = build_q_network(
            input_shape, self.num_actions, "q_func")
        self.obs_tp1_ph, self.q_tp1, self.target_q_func_vars = build_q_network(
            input_shape, self.num_actions, "target_q_func")

        # Placeholders for training
        self.act_t_ph = tf.placeholder(tf.int32, [None])
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        # Q-learning target computation
        if self.double_q:
            # Double Q-learning: use online network to select actions
            q_tp1_online = self.q_tp1  # This is from target network scope
            q_tp1_online_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_func")
            # We need to compute q_tp1 using online network for action selection
            _, q_tp1_online_values, _ = build_q_network(
                input_shape, self.num_actions, "q_func", reuse=True)
            best_actions = tf.argmax(q_tp1_online_values, axis=1)
            q_tp1_best = tf.reduce_sum(self.q_tp1 * tf.one_hot(best_actions, self.num_actions), axis=1)
        else:
            # Standard Q-learning
            q_tp1_best = tf.reduce_max(self.q_tp1, axis=1)

        # Target Q-values
        q_t_selected = tf.reduce_sum(self.q_t * tf.one_hot(self.act_t_ph, self.num_actions), axis=1)
        q_tp1_target = self.rew_t_ph + self.gamma * q_tp1_best * (1 - self.done_mask_ph)

        # Bellman error and loss
        self.total_error = tf.reduce_mean(huber_loss(q_t_selected - q_tp1_target))

        # Optimization
        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = self.optimizer_spec.constructor(
            learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        gradients = optimizer.compute_gradients(self.total_error, var_list=self.q_func_vars)
        clipped_gradients = [(tf.clip_by_norm(grad, self.grad_norm_clipping), var)
                           for grad, var in gradients if grad is not None]
        self.train_fn = optimizer.apply_gradients(clipped_gradients)

        # Target network update
        update_target_fn = []
        for var, var_target in zip(sorted(self.q_func_vars, key=lambda v: v.name),
                                   sorted(self.target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

        # Initialize replay buffer
        from dqn_utils import ReplayBuffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len,
                                        lander=(len(env.observation_space.shape) == 1))

        # Training state
        self.model_initialized = False
        self.num_param_updates = 0
        self.t = 0
        self.last_obs = self.env.reset()

    def select_action(self, obs):
        """Select action using epsilon-greedy policy."""
        eps = self.exploration.value(self.t)

        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.model_initialized:
                q_values = self.sess.run(self.q_t, feed_dict={self.obs_t_ph: [obs]})
                return np.argmax(q_values)
            else:
                return random.randint(0, self.num_actions - 1)

    def update_model(self):
        """Perform experience replay and train the network."""
        if (self.t > self.learning_starts and
            self.t % self.learning_freq == 0 and
            self.replay_buffer.can_sample(self.batch_size)):

            # Sample batch from replay buffer
            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask_batch = \
                self.replay_buffer.sample(self.batch_size)

            # Initialize model if needed
            if not self.model_initialized:
                from dqn_utils import initialize_interdependent_variables
                initialize_interdependent_variables(self.sess, tf.global_variables(), {
                    self.obs_t_ph: obs_t_batch,
                    self.obs_tp1_ph: obs_tp1_batch,
                })
                self.model_initialized = True

            # Train the network
            self.sess.run(self.train_fn, feed_dict={
                self.obs_t_ph: obs_t_batch,
                self.act_t_ph: act_t_batch,
                self.rew_t_ph: rew_t_batch,
                self.obs_tp1_ph: obs_tp1_batch,
                self.done_mask_ph: done_mask_batch,
                self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t)
            })

            # Update target network periodically
            if self.num_param_updates % self.target_update_freq == 0:
                self.sess.run(self.update_target_fn)

            self.num_param_updates += 1

    def step_env(self):
        """Take one step in the environment and store transition."""
        # Select action
        encoded_obs = self.replay_buffer.encode_recent_observation()
        action = self.select_action(encoded_obs)

        # Take step in environment
        obs, reward, done, _ = self.env.step(action)

        # Store transition in replay buffer
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # Reset if episode ended
        if done:
            self.last_obs = self.env.reset()
            self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        else:
            self.replay_buffer_idx = self.replay_buffer.store_frame(obs)

        self.t += 1