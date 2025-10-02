"""
Neural Network Components for Policy Gradient Methods

This module contains neural network building utilities and policy network implementations.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import tensorflow as tf
import numpy as np
def build_mlp(
    input_placeholder,
    output_size,
    scope,
    n_layers,
    size,
    activation=tf.tanh,
    output_activation=None,
):
    """
    Builds a feedforward neural network

    arguments:
        input_placeholder: placeholder variable for the state (batch_size, input_size)
        output_size: size of the output layer
        scope: variable scope of the network
        n_layers: number of hidden layers
        size: dimension of the hidden layer
        activation: activation of the hidden layers
        output_activation: activation of the output layer

    returns:
        output placeholder of the network (the result of a forward pass)
    """
    with tf.variable_scope(scope):
        layer = input_placeholder
        for i in range(n_layers):
            layer = tf.layers.dense(
                layer,
                size,
                activation=activation,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
            )

        output = tf.layers.dense(
            layer,
            output_size,
            activation=output_activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
        )
        return output
class PolicyNetwork:
    """Policy network for policy gradient methods."""

    def __init__(self, ob_dim, ac_dim, discrete, n_layers, size, learning_rate):
        """Initialize the policy network.

        Args:
            ob_dim: dimension of observation space
            ac_dim: dimension of action space
            discrete: whether action space is discrete
            n_layers: number of hidden layers
            size: size of hidden layers
            learning_rate: learning rate for optimizer
        """
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.discrete = discrete
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate

        self._build_network()

    def _build_network(self):
        """Build the policy network."""

        self.sy_ob_no = tf.placeholder(
            shape=[None, self.ob_dim], name="ob", dtype=tf.float32
        )
        self.sy_ac_na = tf.placeholder(
            shape=[None] if self.discrete else [None, self.ac_dim],
            name="ac",
            dtype=tf.int32 if self.discrete else tf.float32,
        )
        self.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        if self.discrete:
            self.policy_parameters = self.policy_forward_pass_discrete(self.sy_ob_no)
        else:
            self.policy_parameters = self.policy_forward_pass_continuous(self.sy_ob_no)
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)
        self.loss = -tf.reduce_mean(self.sy_logprob_n * self.sy_adv_n)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def policy_forward_pass_discrete(self, sy_ob_no):
        """Policy forward pass for discrete action spaces."""
        sy_logits_na = build_mlp(
            sy_ob_no, self.ac_dim, "policy", self.n_layers, self.size
        )
        return sy_logits_na

    def policy_forward_pass_continuous(self, sy_ob_no):
        """Policy forward pass for continuous action spaces."""
        sy_mean = build_mlp(sy_ob_no, self.ac_dim, "policy", self.n_layers, self.size)
        sy_logstd = tf.get_variable(
            "logstd", shape=[self.ac_dim], initializer=tf.zeros_initializer()
        )
        return (sy_mean, sy_logstd)

    def sample_action(self, policy_parameters):
        """Sample action from policy distribution."""
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
        """Get log probability of actions under policy."""
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=sy_ac_na, logits=sy_logits_na
            )
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_std = tf.exp(sy_logstd)
            sy_var = tf.square(sy_std)
            sy_logprob_n = -0.5 * tf.reduce_sum(
                tf.square(sy_ac_na - sy_mean) / sy_var
                + 2 * sy_logstd
                + np.log(2 * np.pi),
                axis=1,
            )
        return sy_logprob_n
class ValueNetwork:
    """Value network (baseline) for policy gradient methods."""

    def __init__(self, ob_dim, n_layers, size, learning_rate):
        """Initialize the value network.

        Args:
            ob_dim: dimension of observation space
            n_layers: number of hidden layers
            size: size of hidden layers
            learning_rate: learning rate for optimizer
        """
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate

        self._build_network()

    def _build_network(self):
        """Build the value network."""
        self.sy_ob_no = tf.placeholder(
            shape=[None, self.ob_dim], name="ob", dtype=tf.float32
        )
        self.sy_target_n = tf.placeholder(shape=[None], name="target", dtype=tf.float32)
        self.baseline_prediction = tf.squeeze(
            build_mlp(self.sy_ob_no, 1, "baseline", self.n_layers, self.size)
        )
        self.baseline_loss = tf.reduce_mean(
            tf.square(self.baseline_prediction - self.sy_target_n)
        )
        self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.baseline_loss
        )