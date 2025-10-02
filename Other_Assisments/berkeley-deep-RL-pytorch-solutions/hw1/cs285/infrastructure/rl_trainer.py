import time
from collections import OrderedDict
import pickle
import numpy as np
import gymnasium as gym
import os
import torch

from cs285.infrastructure.utils import *
from cs285.infrastructure.logger import Logger

import matplotlib.pyplot as plt
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40
class RL_Trainer(object):

    def __init__(self, params):
        self.params = params
        self.logger = Logger(self.params["logdir"])
        seed = self.params["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env = gym.make(self.params["env_name"])
        self.params["ep_len"] = self.params["ep_len"] or self.env.spec.max_episode_steps
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params["agent_params"]["discrete"] = discrete
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params["agent_params"]["ac_dim"] = ac_dim
        self.params["agent_params"]["ob_dim"] = ob_dim
        self.params["agent_params"]["device"] = self.params["device"]
        if "model" in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        else:
            self.fps = self.env.metadata.get("render_fps", 30)
        agent_class = self.params["agent_class"]
        self.agent = agent_class(self.env, self.params["agent_params"])

    def run_training_loop(
        self,
        n_iter,
        collect_policy,
        eval_policy,
        initial_expertdata=None,
        relabel_with_expert=False,
        start_relabel_with_expert=1,
        expert_policy=None,
    ):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************" % itr)
            if (
                itr % self.params["video_log_freq"] == 0
                and self.params["video_log_freq"] != -1
            ):
                self.log_video = True
            else:
                self.log_video = False
            if itr % self.params["scalar_log_freq"] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False
            training_returns = self.collect_training_trajectories(
                itr, initial_expertdata, collect_policy, self.params["batch_size"]
            )
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch
            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)
            self.agent.add_to_replay_buffer(paths)
            self.train_agent()
            if self.log_video or self.log_metrics:
                print("\nBeginning logging procedure...")
                self.perform_logging(itr, paths, eval_policy, train_video_paths)
                print("\nSaving agent's actor...")
                self.agent.actor.save(self.params["logdir"] + "/policy_itr_" + str(itr))
    def collect_training_trajectories(
        self, itr, load_initial_expertdata, collect_policy, batch_size
    ):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        if itr == 0:
            with open(load_initial_expertdata, "rb") as f:
                loaded_paths = pickle.load(f)
            return loaded_paths, 0, None
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(
            self.env, collect_policy, batch_size, self.params["ep_len"]
        )
        train_video_paths = None
        if self.log_video:
            print("\nCollecting train rollouts to be used for saving videos...")
            train_video_paths = sample_n_trajectories(
                self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True
            )

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        print("\nTraining agent using sampled data from replay buffer...")
        for train_step in range(self.params["num_agent_train_steps_per_iter"]):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = (
                self.agent.sample(self.params["train_batch_size"])
            )
            self.agent.train(
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch
            )

    def do_relabel_with_expert(self, expert_policy, paths):
        print(
            "\nRelabelling collected observations with labels from an expert policy..."
        )
        for i in range(len(paths)):
            paths[i]["action"] = (
                expert_policy.get_action(paths[i]["observation"]).detach().numpy()
            )
        return paths
    def perform_logging(self, itr, paths, eval_policy, train_video_paths):
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = sample_trajectories(
            self.env, eval_policy, self.params["eval_batch_size"], self.params["ep_len"]
        )
        if self.log_video and train_video_paths != None:
            print("\nCollecting video rollouts eval")
            eval_video_paths = sample_n_trajectories(
                self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True
            )
            print("\nSaving train rollouts as videos...")
            self.logger.log_paths_as_videos(
                train_video_paths,
                itr,
                fps=self.fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="train_rollouts",
            )
            self.logger.log_paths_as_videos(
                eval_video_paths,
                itr,
                fps=self.fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )
        if self.log_metrics:
            if paths != None:
                train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
            if paths != None:
                train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            if paths != None:
                logs["Train_AverageReturn"] = np.mean(train_returns)
                logs["Train_StdReturn"] = np.std(train_returns)
                logs["Train_MaxReturn"] = np.max(train_returns)
                logs["Train_MinReturn"] = np.min(train_returns)
                logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

                logs["Train_EnvstepsSoFar"] = self.total_envsteps
                logs["TimeSinceStart"] = time.time() - self.start_time

                if itr == 0:
                    self.initial_return = np.mean(train_returns)
                logs["Initial_DataCollection_AverageReturn"] = self.initial_return
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            self.logger.flush()