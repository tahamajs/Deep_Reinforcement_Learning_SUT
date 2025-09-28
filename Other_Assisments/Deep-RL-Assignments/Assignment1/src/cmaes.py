# Author: Taha Majlesi - 810101504, University of Tehran

import numpy as np
import cma
from src.utils import generate_episode


class CMAES:
    def __init__(self, env, num_params, sigma=0.1):
        self.env = env
        self.num_params = num_params
        self.sigma = sigma

    def train(self, num_rollouts, num_epochs=100):
        def evaluate_policy(params):
            policy = (
                lambda state: np.dot(
                    params[
                        : self.env.observation_space.shape[0] * self.env.action_space.n
                    ].reshape(
                        self.env.action_space.n, self.env.observation_space.shape[0]
                    ),
                    state,
                )
                + params[
                    self.env.observation_space.shape[0] * self.env.action_space.n :
                ]
            )
            total_reward = 0
            for _ in range(num_rollouts):
                _, _, rewards = generate_episode(self.env, policy)
                total_reward += np.sum(rewards)
            return -total_reward / num_rollouts  # Minimize negative reward

        es = cma.CMAEvolutionStrategy(np.random.randn(self.num_params), self.sigma)
        es.optimize(evaluate_policy, iterations=num_epochs)
        return es.result.xbest

    def evaluate(self, params, num_rollouts=50):
        policy = (
            lambda state: np.dot(
                params[
                    : self.env.observation_space.shape[0] * self.env.action_space.n
                ].reshape(self.env.action_space.n, self.env.observation_space.shape[0]),
                state,
            )
            + params[self.env.observation_space.shape[0] * self.env.action_space.n :]
        )
        total_rewards = []
        for _ in range(num_rollouts):
            _, _, rewards = generate_episode(self.env, policy)
            total_rewards.append(np.sum(rewards))
        return np.mean(total_rewards), np.std(total_rewards)
