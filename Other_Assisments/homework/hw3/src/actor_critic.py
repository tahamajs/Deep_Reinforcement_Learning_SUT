"""
Actor-Critic Agent Implementation

This module contains the Actor-Critic agent with policy and value networks.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def build_mlp(input_placeholder, output_size, scope, n_layers, size,
              activation=tf.tanh, output_activation=None):
    """Build a multi-layer perceptron.

    Args:
        input_placeholder: Input tensor
        output_size: Size of output layer
        scope: Variable scope
        n_layers: Number of hidden layers
        size: Size of hidden layers
        activation: Activation for hidden layers
        output_activation: Activation for output layer

    Returns:
        Output tensor
    """
    with tf.variable_scope(scope):
        out = input_placeholder
        for i in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation,
                                name=f"layer_{i}")
        out = tf.layers.dense(out, output_size, activation=output_activation,
                            name="output")
        return out


def pathlength(path):
    """Get the length of a trajectory path."""
    return len(path["reward"])


class ActorCriticAgent:
    """Actor-Critic agent with separate policy and value networks."""

    def __init__(self, ob_dim, ac_dim, discrete, n_layers, size, learning_rate,
                 num_target_updates, num_grad_steps_per_target_update,
                 gamma, normalize_advantages, max_path_length,
                 min_timesteps_per_batch, animate):
        """Initialize Actor-Critic agent.

        Args:
            ob_dim: Observation dimension
            ac_dim: Action dimension
            discrete: Whether action space is discrete
            n_layers: Number of hidden layers
            size: Size of hidden layers
            learning_rate: Learning rate
            num_target_updates: Number of target updates
            num_grad_steps_per_target_update: Gradient steps per target update
            gamma: Discount factor
            normalize_advantages: Whether to normalize advantages
            max_path_length: Maximum path length
            min_timesteps_per_batch: Minimum timesteps per batch
            animate: Whether to animate episodes
        """
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.discrete = discrete
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.num_target_updates = num_target_updates
        self.num_grad_steps_per_target_update = num_grad_steps_per_target_update
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch
        self.animate = animate

    def init_tf_sess(self):
        """Initialize TensorFlow session."""
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                                 intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()
        tf.global_variables_initializer().run()

    def define_placeholders(self):
        """Define placeholders for training."""
        self.sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)

        if self.discrete:
            self.sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            self.sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)

        self.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)

        return self.sy_ob_no, self.sy_ac_na, self.sy_adv_n

    def policy_forward_pass(self, sy_ob_no):
        """Build policy network.

        Args:
            sy_ob_no: Observation placeholder

        Returns:
            Policy parameters (logits for discrete, mean/log_std for continuous)
        """
        if self.discrete:
            sy_logits_na = build_mlp(sy_ob_no, self.ac_dim, "policy",
                                   self.n_layers, self.size)
            return sy_logits_na
        else:
            sy_mean = build_mlp(sy_ob_no, self.ac_dim, "policy",
                              self.n_layers, self.size)
            sy_logstd = tf.get_variable("logstd", shape=[self.ac_dim],
                                      initializer=tf.zeros_initializer())
            return (sy_mean, sy_logstd)

    def sample_action(self, policy_parameters):
        """Sample action from policy distribution.

        Args:
            policy_parameters: Policy parameters

        Returns:
            Sampled action
        """
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_sampled_ac = tf.multinomial(sy_logits_na, 1)
            sy_sampled_ac = tf.squeeze(sy_sampled_ac, axis=1)
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_std = tf.exp(sy_logstd)
            sy_sampled_ac = sy_mean + sy_std * tf.random_normal(tf.shape(sy_mean))

        return sy_sampled_ac

    def get_log_prob(self, policy_parameters, sy_ac_na):
        """Compute log probability of actions.

        Args:
            policy_parameters: Policy parameters
            sy_ac_na: Actions

        Returns:
            Log probabilities
        """
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=sy_ac_na, logits=sy_logits_na)
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_std = tf.exp(sy_logstd)
            sy_var = tf.square(sy_std)

            # Multivariate Gaussian log probability
            sy_diff = sy_ac_na - sy_mean
            sy_quad_form = tf.reduce_sum(tf.square(sy_diff) / sy_var, axis=1)
            sy_log_det = tf.reduce_sum(sy_logstd, axis=1)
            sy_logprob_n = -0.5 * (self.ac_dim * tf.log(2 * np.pi) + sy_quad_form) - sy_log_det

        return sy_logprob_n

    def build_computation_graph(self):
        """Build the computation graph."""
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()

        # Policy network
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        # Actor loss and update
        actor_loss = tf.reduce_sum(-self.sy_logprob_n * self.sy_adv_n)
        self.actor_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(actor_loss)

        # Critic network
        self.critic_prediction = tf.squeeze(build_mlp(
            self.sy_ob_no, 1, "critic", self.n_layers, self.size))
        self.critic_loss = tf.losses.mean_squared_error(self.sy_target_n, self.critic_prediction)
        self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

    def sample_trajectories(self, itr, env):
        """Sample trajectories until we have enough timesteps."""
        timesteps_this_batch = 0
        paths = []

        while True:
            animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)

            if timesteps_this_batch > self.min_timesteps_per_batch:
                break

        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        """Sample a single trajectory."""
        ob = env.reset()
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        steps = 0

        while True:
            if animate_this_episode:
                env.render()
                import time
                time.sleep(0.1)

            obs.append(ob)

            # Sample action
            ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: [ob]})
            ac = ac[0] if not self.discrete else ac
            acs.append(ac)

            ob, rew, done, _ = env.step(ac)
            next_obs.append(ob)
            rewards.append(rew)
            terminals.append(1.0 if done else 0.0)

            steps += 1

            if done or steps > self.max_path_length:
                break

        path = {
            "observation": np.array(obs, dtype=np.float32),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)
        }

        return path

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        """Estimate advantages using critic network."""
        # Compute Q(s,a) = r(s,a) + gamma * V(s') * (1 - terminal)
        v_next = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: next_ob_no})
        q_n = re_n + self.gamma * v_next * (1 - terminal_n)

        # Compute V(s) for baseline
        v_current = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: ob_no})

        # Advantage = Q - V
        adv_n = q_n - v_current

        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)

        return adv_n

    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n):
        """Update critic network."""
        # Target values: r + gamma * V(s') * (1 - terminal)
        v_next = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: next_ob_no})
        target_n = re_n + self.gamma * v_next * (1 - terminal_n)

        # Update critic for multiple steps
        for _ in range(self.num_target_updates):
            for _ in range(self.num_grad_steps_per_target_update):
                self.sess.run(self.critic_update_op, feed_dict={
                    self.sy_ob_no: ob_no,
                    self.sy_target_n: target_n
                })

    def update_actor(self, ob_no, ac_na, adv_n):
        """Update actor network."""
        self.sess.run(self.actor_update_op, feed_dict={
            self.sy_ob_no: ob_no,
            self.sy_ac_na: ac_na,
            self.sy_adv_n: adv_n
        })