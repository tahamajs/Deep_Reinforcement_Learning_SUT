import numpy as np
import torch

from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *

class PGAgent:
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.lamb = self.agent_params['lambda']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.device = self.agent_params['device']
        self.actor = MLPPolicyPG(self.agent_params['ac_dim'],
                                 self.agent_params['ob_dim'],
                                 self.agent_params['n_layers'],
                                 self.agent_params['size'],
                                 self.agent_params['device'],
                                 discrete=self.agent_params['discrete'],
                                 learning_rate=self.agent_params['learning_rate'],
                                 nn_baseline=self.agent_params['nn_baseline']
                                 )
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, obs, acs, rews_list, next_obs, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.

            ----------------------------------------------------------------------------------

            Recall that the expression for the policy gradient PG is

                PG = E_{tau} [sum_{t=0}^{T-1} grad log pi(a_t|s_t) * (Q_t - b_t )]

                where
                tau=(s_0, a_0, s_1, a_1, s_2, a_2, ...) is a trajectory,
                Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                b_t is a baseline which may depend on s_t,
                and (Q_t - b_t ) is the advantage.

            Thus, the PG update performed by the actor needs (s_t, a_t, q_t, adv_t),
                and that is exactly what this function provides.

            ----------------------------------------------------------------------------------
        """
        q_values = self.calculate_q_vals(rews_list)
        advantage_values = self.estimate_advantage(obs, q_values, rews_list)
        loss = self.actor.update(obs, acs, qvals = q_values, adv_n = advantage_values)
        return loss

    def calculate_q_vals(self, rews_list):

        """
            Monte Carlo estimation of the Q function.

            arguments:
                rews_list: length: number of sampled rollouts
                    Each element corresponds to a particular rollout,
                    and contains an array of the rewards for every step of that particular rollout

            returns:
                q_values: shape: (sum/total number of steps across the rollouts)
                    Each entry corresponds to the estimated q(s_t,a_t) value
                    of the corresponding obs/ac point at time t.

        """
        if not self.reward_to_go:
            q_values = np.concatenate([self._discounted_return(r) for r in rews_list])
        else:
            q_values = np.concatenate([self._discounted_cumsum(r) for r in rews_list])

        return q_values

    def estimate_advantage(self, obs, q_values, rewards):

        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values
        """
        if self.lamb < 1:
            with torch.no_grad():
                GAE = []
                indice = 0
                V = self.actor.baseline_mlp(torch.Tensor(obs).to(self.device)).view(-1).cpu().detach().numpy()

                for rew_list in rewards:
                    GAE.append(self.generalized_advantage_estimator(V[indice: indice + len(rew_list)], rew_list))
                    indice += len(rew_list)

                adv_n = np.concatenate(GAE)

        elif self.nn_baseline:
            with torch.no_grad():
                b_n_unnormalized = self.actor.baseline_mlp(torch.Tensor(obs).to(self.device)).cpu().detach().numpy()
                b_n = b_n_unnormalized * np.std(q_values) + np.mean(q_values)
                adv_n = q_values - b_n
        else:
            adv_n = q_values.copy()
        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)

        return adv_n
    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)
    def generalized_advantage_estimator(self, V, rewards):
        V = np.append(V, [0])
        delta = rewards + (self.gamma * V[1:]) - V[:-1]

        GAE = []

        for start_time_index in range(len(delta)):
            indices = np.arange(start_time_index, len(delta))
            discounts = np.power(self.gamma * self.lamb, indices - start_time_index)
            discounted_rtg = delta[start_time_index:] * discounts
            GAE.append(np.sum(discounted_rtg))
        return np.array(GAE)
    def _discounted_return(self, rewards):
        """
            Helper function

            Input: a list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^{T-1} gamma^t' r_{t'}
                note that all entries of this output are equivalent
                because each index t is a sum from 0 to T-1 (and doesnt involve t)
        """
        indices = np.arange(len(rewards))
        discounts = np.power(self.gamma, indices)
        discounted_rewards = rewards * discounts
        sum_of_discounted_rewards = np.sum(discounted_rewards)
        list_of_discounted_returns = np.ones(len(rewards)) * sum_of_discounted_rewards

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Input:
                a list of length T
                a list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single rollout of length T
            Output:
                a list of length T
                a list where the entry in each index t is sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
        """

        all_discounted_cumsums = []
        for start_time_index in range(len(rewards)):
            indices = np.arange(start_time_index, len(rewards))
            discounts = np.power(self.gamma, indices - start_time_index)
            discounted_rtg = rewards[indices] * discounts
            sum_discounted_rtg = np.sum(discounted_rtg)
            all_discounted_cumsums.append(sum_discounted_rtg)
        list_of_discounted_cumsums = np.array(all_discounted_cumsums)
        return list_of_discounted_cumsums