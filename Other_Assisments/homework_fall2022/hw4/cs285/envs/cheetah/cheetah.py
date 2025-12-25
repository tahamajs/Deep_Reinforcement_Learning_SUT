import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class HalfCheetahEnv(MujocoEnv, utils.EzPickle):

    def __init__(self):
        # Newer Gymnasium's MujocoEnv requires an observation_space argument.
        # Provide a temporary small Box and then set the proper observation_space
        # after initialization using a sample observation.
        dummy_obs_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        super(HalfCheetahEnv, self).__init__("half_cheetah.xml", 5, observation_space=dummy_obs_space)
        utils.EzPickle.__init__(self)

        # Now compute and set the correct observation space
        obs = self._get_obs()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64)

        self.action_dim = self.ac_dim = self.action_space.shape[0]
        self.observation_dim = self.obs_dim = self.observation_space.shape[0]

    def get_reward(self, observations, actions):
        """get reward/s of given (observations, actions) datapoint or datapoints

        Args:
            observations: (batchsize, obs_dim) or (obs_dim,)
            actions: (batchsize, ac_dim) or (ac_dim,)

        Return:
            r_total: reward of this (o,a) pair, dimension is (batchsize,1) or (1,)
            done: True if env reaches terminal state, dimension is (batchsize,1) or (1,)
        """
        self.reward_dict = {}
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, axis=0)
            actions = np.expand_dims(actions, axis=0)
            batch_mode = False
        else:
            batch_mode = True
        xvel = observations[:, 9].copy()
        body_angle = observations[:, 2].copy()
        front_leg = observations[:, 6].copy()
        front_shin = observations[:, 7].copy()
        front_foot = observations[:, 8].copy()
        zeros = np.zeros((observations.shape[0],)).copy()
        leg_range = 0.2
        shin_range = 0
        foot_range = 0
        penalty_factor = 10
        self.reward_dict["run"] = xvel

        front_leg_rew = zeros.copy()
        front_leg_rew[front_leg > leg_range] = -penalty_factor
        self.reward_dict["leg"] = front_leg_rew

        front_shin_rew = zeros.copy()
        front_shin_rew[front_shin > shin_range] = -penalty_factor
        self.reward_dict["shin"] = front_shin_rew

        front_foot_rew = zeros.copy()
        front_foot_rew[front_foot > foot_range] = -penalty_factor
        self.reward_dict["foot"] = front_foot_rew
        self.reward_dict["r_total"] = (
            self.reward_dict["run"]
            + self.reward_dict["leg"]
            + self.reward_dict["shin"]
            + self.reward_dict["foot"]
        )
        dones = zeros.copy()
        if not batch_mode:
            return self.reward_dict["r_total"][0], dones[0]
        return self.reward_dict["r_total"], dones

    def get_score(self, obs):
        xposafter = obs[0]
        return xposafter

    def step(self, action):
        # Gymnasium expects step to return: obs, reward, terminated, truncated, info
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        rew, done = self.get_reward(ob, action)
        score = self.get_score(ob)
        info = {
            "obs_dict": self.obs_dict,
            "rewards": self.reward_dict,
            "score": score,
        }
        terminated = bool(done)
        truncated = False
        return ob, rew, terminated, truncated, info

    def _get_obs(self):

        self.obs_dict = {}
        # gymnasium's MuJoCoEnv stores state in self.data
        self.obs_dict["joints_pos"] = self.data.qpos.ravel().copy()
        self.obs_dict["joints_vel"] = self.data.qvel.ravel().copy()
        # compute/approximate center of mass of torso; for smoke tests use zeros
        self.obs_dict["com_torso"] = np.zeros((3,))

        return np.concatenate(
            [
                self.obs_dict["joints_pos"],
                self.obs_dict["joints_vel"],
                self.obs_dict["com_torso"],
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
