"""
Policy Gradient Agent Implementation

This module contains the main policy gradient agent with trajectory sampling and training.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np
import tensorflow as tf
from .networks import PolicyNetwork, ValueNetwork
def pathlength(path):
    """Get the length of a trajectory path."""
    return len(path["reward"])
class PolicyGradientAgent:
    """Policy Gradient agent with optional neural network baseline."""

    def __init__(
        self,
        ob_dim,
        ac_dim,
        discrete,
        n_layers,
        size,
        learning_rate,
        gamma,
        reward_to_go,
        nn_baseline,
        normalize_advantages,
        min_timesteps_per_batch,
        max_path_length,
        animate,
    ):
        """Initialize the policy gradient agent.

        Args:
            ob_dim: observation dimension
            ac_dim: action dimension
            discrete: whether action space is discrete
            n_layers: number of hidden layers in networks
            size: size of hidden layers
            learning_rate: learning rate for optimization
            gamma: discount factor
            reward_to_go: whether to use reward-to-go
            nn_baseline: whether to use neural network baseline
            normalize_advantages: whether to normalize advantages
            min_timesteps_per_batch: minimum timesteps per batch
            max_path_length: maximum path length
            animate: whether to animate episodes
        """
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.discrete = discrete
        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.nn_baseline = nn_baseline
        self.normalize_advantages = normalize_advantages
        self.min_timesteps_per_batch = min_timesteps_per_batch
        self.max_path_length = max_path_length
        self.animate = animate
        self.policy_net = PolicyNetwork(
            ob_dim, ac_dim, discrete, n_layers, size, learning_rate
        )
        if nn_baseline:
            self.value_net = ValueNetwork(ob_dim, n_layers, size, learning_rate)
        self.sess = None

    def init_tf_sess(self):
        """Initialize TensorFlow session."""
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def sample_trajectories(self, itr, env):
        """Sample trajectories until we have enough timesteps.

        Args:
            itr: current iteration number
            env: gym environment

        Returns:
            paths: list of trajectory paths
            timesteps_this_batch: total timesteps in batch
        """
        timesteps_this_batch = 0
        paths = []

        while True:
            animate_this_episode = len(paths) == 0 and (itr % 10 == 0) and self.animate
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)

            if timesteps_this_batch > self.min_timesteps_per_batch:
                break

        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        """Sample a single trajectory.

        Args:
            env: gym environment
            animate_this_episode: whether to render this episode

        Returns:
            path: trajectory path dictionary
        """
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0

        while True:
            if animate_this_episode:
                env.render()
                import time

                time.sleep(0.1)

            obs.append(ob)
            ac = self.sess.run(
                self.policy_net.sy_sampled_ac,
                feed_dict={self.policy_net.sy_ob_no: [ob]},
            )
            ac = ac[0] if not self.discrete else ac
            acs.append(ac)

            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1

            if done or steps > self.max_path_length:
                break

        path = {
            "observation": np.array(obs, dtype=np.float32),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),
        }
        return path

    def sum_of_rewards(self, re_n):
        """Compute Q-values using Monte Carlo estimation.

        Args:
            re_n: list of reward arrays for each path

        Returns:
            q_n: Q-values for all timesteps
        """
        q_n = []

        for re in re_n:
            if self.reward_to_go:

                q_path = []
                for t in range(len(re)):
                    q_t = 0
                    for t_prime in range(t, len(re)):
                        q_t += (self.gamma ** (t_prime - t)) * re[t_prime]
                    q_path.append(q_t)
                q_n.extend(q_path)
            else:

                total_return = sum(
                    self.gamma**t_prime * re[t_prime] for t_prime in range(len(re))
                )
                q_n.extend([total_return] * len(re))

        return np.array(q_n)

    def compute_advantage(self, ob_no, q_n):
        """Compute advantages by subtracting baseline from Q-values.

        Args:
            ob_no: observations
            q_n: Q-values

        Returns:
            adv_n: advantages
        """
        if self.nn_baseline:

            b_n = self.sess.run(
                self.value_net.baseline_prediction,
                feed_dict={self.value_net.sy_ob_no: ob_no},
            )
            b_n = b_n * np.std(q_n) + np.mean(q_n)
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        return adv_n

    def estimate_return(self, ob_no, re_n):
        """Estimate returns and advantages.

        Args:
            ob_no: observations
            re_n: rewards for each path

        Returns:
            q_n: Q-values
            adv_n: advantages
        """
        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)

        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n):
        """Update policy and baseline parameters.

        Args:
            ob_no: observations
            ac_na: actions
            q_n: Q-values
            adv_n: advantages
        """

        if self.nn_baseline:
            target_n = (q_n - np.mean(q_n)) / (np.std(q_n) + 1e-8)
            self.sess.run(
                self.value_net.baseline_update_op,
                feed_dict={
                    self.value_net.sy_ob_no: ob_no,
                    self.value_net.sy_target_n: target_n,
                },
            )
        self.sess.run(
            self.policy_net.update_op,
            feed_dict={
                self.policy_net.sy_ob_no: ob_no,
                self.policy_net.sy_ac_na: ac_na,
                self.policy_net.sy_adv_n: adv_n,
            },
        )