"""
Base Neural Network Components

This module contains reusable neural network building blocks for RL algorithms.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

try:
    import tensorflow.compat.v1 as tf  # type: ignore
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf  # type: ignore
import numpy as np
def build_mlp(
    input_tensor,
    output_dim,
    scope,
    n_layers=2,
    hidden_dim=256,
    activation=tf.nn.relu,
    output_activation=None,
    reuse=False,
):
    """
    Build a multi-layer perceptron.

    Args:
        input_tensor: Input tensor
        output_dim: Output dimension
        scope: Variable scope
        n_layers: Number of hidden layers
        hidden_dim: Hidden layer dimension
        activation: Hidden layer activation
        output_activation: Output layer activation
        reuse: Whether to reuse variables

    Returns:
        Output tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        x = input_tensor
        for i in range(n_layers):
            x = tf.layers.dense(
                x, hidden_dim, activation=activation, name=f"hidden_{i}"
            )
        x = tf.layers.dense(x, output_dim, activation=output_activation, name="output")

        return x
def build_policy_network(
    state_tensor,
    action_dim,
    scope,
    hidden_sizes=[256, 256],
    activation=tf.nn.relu,
    reuse=False,
):
    """
    Build a policy network (stochastic policy).

    Args:
        state_tensor: State input tensor
        action_dim: Action dimension
        scope: Variable scope
        hidden_sizes: Hidden layer sizes
        activation: Activation function
        reuse: Whether to reuse variables

    Returns:
        Tuple of (mean, log_std) tensors
    """
    with tf.variable_scope(scope, reuse=reuse):
        x = state_tensor
        for i, size in enumerate(hidden_sizes):
            x = tf.layers.dense(x, size, activation=activation, name=f"hidden_{i}")
        mean = tf.layers.dense(x, action_dim, name="mean")
        log_std = tf.layers.dense(x, action_dim, name="log_std")
        log_std = tf.clip_by_value(log_std, -20, 2)

        return mean, log_std
def build_value_network(
    state_tensor, scope, hidden_sizes=[256, 256], activation=tf.nn.relu, reuse=False
):
    """
    Build a value network.

    Args:
        state_tensor: State input tensor
        scope: Variable scope
        hidden_sizes: Hidden layer sizes
        activation: Activation function
        reuse: Whether to reuse variables

    Returns:
        Value tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        x = state_tensor
        for i, size in enumerate(hidden_sizes):
            x = tf.layers.dense(x, size, activation=activation, name=f"hidden_{i}")
        value = tf.layers.dense(x, 1, name="value")

        return tf.squeeze(value)
def build_q_network(
    state_tensor,
    action_tensor,
    scope,
    hidden_sizes=[256, 256],
    activation=tf.nn.relu,
    reuse=False,
):
    """
    Build a Q-network (state-action value function).

    Args:
        state_tensor: State input tensor
        action_tensor: Action input tensor
        scope: Variable scope
        hidden_sizes: Hidden layer sizes
        activation: Activation function
        reuse: Whether to reuse variables

    Returns:
        Q-value tensor
    """
    with tf.variable_scope(scope, reuse=reuse):

        x = tf.concat([state_tensor, action_tensor], axis=-1)
        for i, size in enumerate(hidden_sizes):
            x = tf.layers.dense(x, size, activation=activation, name=f"hidden_{i}")
        q_value = tf.layers.dense(x, 1, name="q_value")

        return tf.squeeze(q_value)
def sample_action(mean, log_std, reparameterize=True):
    """
    Sample action from policy distribution.

    Args:
        mean: Mean tensor
        log_std: Log standard deviation tensor
        reparameterize: Whether to use reparameterization trick

    Returns:
        Tuple of (action, log_prob) tensors
    """
    std = tf.exp(log_std)
    dist = tf.distributions.Normal(mean, std)

    if reparameterize:

        eps = tf.random.normal(tf.shape(mean))
        action = mean + std * eps
    else:
        action = dist.sample()
    log_prob = dist.log_prob(action)
    log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
    return action, log_prob
def squash_action(action, log_prob):
    """
    Apply tanh squashing to action and adjust log probability.

    Args:
        action: Raw action tensor
        log_prob: Log probability tensor

    Returns:
        Tuple of (squashed_action, adjusted_log_prob) tensors
    """
    squashed_action = tf.tanh(action)
    squash_correction = tf.reduce_sum(
        tf.log(1 - squashed_action**2 + 1e-6), axis=-1, keepdims=True
    )
    adjusted_log_prob = log_prob - squash_correction

    return squashed_action, adjusted_log_prob