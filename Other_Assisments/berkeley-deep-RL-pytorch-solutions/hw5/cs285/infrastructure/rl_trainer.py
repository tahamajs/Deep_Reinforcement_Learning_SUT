import time

from collections import OrderedDict
import pickle
import numpy as np
import torch
import gymnasium as gym
import os
import sys

import cs285.envs
from cs285.infrastructure.utils import *
from cs285.infrastructure.logger import Logger
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40
class RL_Trainer(object):

    def __init__(self, params):
        self.params = params
        self.logger = Logger(self.params["logdir"])
        seed = self.params["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.params["env_name"] == "PointMass-v0":
            from cs285.envs.pointmass import PointMass

            self.env = PointMass()
        else:
            self.env = gym.make(self.params["env_name"])
        self.params["agent_params"]["env_name"] = self.params["env_name"]

        self.max_path_length = (
            self.params["max_path_length"] or self.env.spec.max_episode_steps
        )
        self.params["ep_len"] = self.params["ep_len"] or self.env.spec.max_episode_steps
        self.params["agent_params"]["discrete"] = isinstance(
            self.env.action_space, gym.spaces.Discrete
        )
        self.params["agent_params"]["ob_dim"] = self.env.observation_space.shape[0]
        self.params["agent_params"]["ac_dim"] = (
            self.env.action_space.n
            if self.params["agent_params"]["discrete"]
            else self.env.action_space.shape[0]
        )
        agent_class = self.params["agent_class"]
        self.agent = agent_class(self.env, self.params["agent_params"])

    def run_training_loop(self, n_iter, policy):
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************" % itr)
            if self.params["scalar_log_freq"] == -1:
                self.logmetrics = False
            elif itr % self.params["scalar_log_freq"] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False
            paths, envsteps_this_batch = self.collect_training_trajectories(
                itr, policy, self.params["batch_size"]
            )

            self.total_envsteps += envsteps_this_batch
            self.agent.add_to_replay_buffer(paths)
            loss, ex2_vars = self.train_agent()
            if self.logmetrics:

                print("\nBeginning logging procedure...")
                self.perform_logging(itr, paths, policy, loss, ex2_vars)
    def collect_training_trajectories(self, itr, policy, batch_size):
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(
            self.env,
            policy,
            batch_size,
            self.max_path_length,
            self.params["render"],
            itr,
        )

        return paths, envsteps_this_batch

    def train_agent(self):

        for train_step in range(self.params["num_agent_train_steps_per_iter"]):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = (
                self.agent.sample(self.params["batch_size"])
            )

            loss, ex2_vars = self.agent.train(
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch
            )
        return loss, ex2_vars
    def perform_logging(self, itr, paths, eval_policy, loss, ex2_vars):

        if self.logmetrics:

            train_returns = [path["reward"].sum() for path in paths]
            train_ep_lens = [len(path["reward"]) for path in paths]
            logs = OrderedDict()

            if ex2_vars != None:
                logs["Log_Likelihood_Average"] = np.mean(ex2_vars[0])
                logs["KL_Divergence_Average"] = np.mean(ex2_vars[1])
                logs["ELBO_Average"] = np.mean(ex2_vars[2])

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            if isinstance(loss, dict):
                logs.update(loss)
            else:
                logs["Training loss"] = loss

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            self.logger.flush()