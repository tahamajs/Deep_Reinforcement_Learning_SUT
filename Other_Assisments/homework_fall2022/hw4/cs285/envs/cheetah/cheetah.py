import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import MujocoEnv


class HalfCheetahEnv(MujocoEnv, utils.EzPickle):

    def __init__(self):

        MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

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
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        rew, done = self.get_reward(ob, action)
        score = self.get_score(ob)
        env_info = {
            "obs_dict": self.obs_dict,
            "rewards": self.reward_dict,
            "score": score,
        }
        return ob, rew, done, env_info

    def _get_obs(self):

        self.obs_dict = {}
        self.obs_dict["joints_pos"] = self.sim.data.qpos.flat.copy()
        self.obs_dict["joints_vel"] = self.sim.data.qvel.flat.copy()
        self.obs_dict["com_torso"] = self.get_body_com("torso").flat.copy()

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
