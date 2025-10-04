import numpy as np

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """
        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)
        train_log = self.actor.update(
            observations,
            actions,
            advantages,
            q_values=q_values,
        )
        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """
        if not self.reward_to_go:
            q_values = np.concatenate([
                self._discounted_return(np.array(path_rewards))
                for path_rewards in rewards_list
            ])
        else:
            q_values = np.concatenate([
                self._discounted_cumsum(np.array(path_rewards))
                for path_rewards in rewards_list
            ])

        return q_values

    def estimate_advantage(self, obs: np.ndarray, rews_list: np.ndarray, q_values: np.ndarray, terminals: np.ndarray):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            assert values_unnormalized.ndim == q_values.ndim
            values = values_unnormalized * np.std(q_values) + np.mean(q_values)

            if self.gae_lambda is not None:

                advantages_list = []
                start = 0
                for path_rewards in rews_list:
                    path_rewards = np.array(path_rewards)
                    path_len = len(path_rewards)
                    end = start + path_len
                    path_values = values[start:end]
                    path_values = np.append(path_values, 0)
                    path_terminals = terminals[start:end]
                    path_advantages = np.zeros(path_len)
                    gae = 0
                    for t in reversed(range(path_len)):
                        nonterminal = 1 - path_terminals[t]
                        delta = path_rewards[t] + self.gamma * path_values[t + 1] * nonterminal - path_values[t]
                        gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
                        path_advantages[t] = gae
                    advantages_list.append(path_advantages)
                    start = end
                advantages = np.concatenate(advantages_list)

            else:

                advantages = q_values - values
        else:
            advantages = q_values.copy()
        if self.standardize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages
    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)
    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        rewards = np.array(rewards)
        discounts = np.power(self.gamma, np.arange(len(rewards)))
        total_return = np.sum(discounts * rewards)
        list_of_discounted_returns = np.ones(len(rewards)) * total_return

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        rewards = np.array(rewards)
        discounted_cumsums = np.zeros(len(rewards))
        running_total = 0
        for t in reversed(range(len(rewards))):
            running_total = rewards[t] + self.gamma * running_total
            discounted_cumsums[t] = running_total

        return discounted_cumsums