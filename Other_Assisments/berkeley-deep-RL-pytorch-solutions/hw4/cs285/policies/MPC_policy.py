import numpy as np

class MPCPolicy:
    def __init__(self,
        env,
        ac_dim,
        dyn_models,
        horizon,
        N,
        **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None

        self.ob_dim = self.env.observation_space.shape[0]
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        random_action_sequences = np.random.uniform(self.low, self.high, (num_sequences, horizon, self.ac_dim))
        return random_action_sequences

    def get_action(self, obs):

        if self.data_statistics is None:

            return self.sample_action_sequences(num_sequences = 1, horizon = 1)[0, 0]
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)
        predicted_rewards_per_ens = []

        for model in self.dyn_models:
            sim_obs = np.tile(obs, (self.N, 1))
            model_rewards = np.zeros(self.N)

            for t in range(self.horizon):
                rew, _ = self.env.get_reward(sim_obs, candidate_action_sequences[:, t, :])
                model_rewards += rew
                sim_obs = model.get_prediction(sim_obs, candidate_action_sequences[:, t, :], self.data_statistics)
            predicted_rewards_per_ens.append(model_rewards)
        predicted_rewards = np.mean(predicted_rewards_per_ens, axis = 0)
        best_index = np.argmax(predicted_rewards)
        best_action_sequence = candidate_action_sequences[best_index]
        action_to_take = best_action_sequence[0]
        return action_to_take