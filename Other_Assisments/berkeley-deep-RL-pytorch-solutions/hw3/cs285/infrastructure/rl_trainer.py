import time

from collections import OrderedDict
import pickle
import numpy as np
import torch
import gymnasium as gym
import os
import sys
from gymnasium import wrappers

from cs285.infrastructure.utils import *
from cs285.infrastructure.logger import Logger

from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure.dqn_utils import get_wrapper_by_name
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
        if "env_wrappers" in self.params:
            self.env = params["env_wrappers"](self.env)
            self.mean_episode_reward = -float("nan")
            self.best_mean_episode_reward = -float("inf")
        self.params["ep_len"] = self.params["ep_len"] or self.env.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params["ep_len"]
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

        img = len(self.env.observation_space.shape) > 2

        self.params["agent_params"]["discrete"] = discrete
        ob_dim = (
            self.env.observation_space.shape
            if img
            else self.env.observation_space.shape[0]
        )
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params["agent_params"]["ac_dim"] = ac_dim
        self.params["agent_params"]["ob_dim"] = ob_dim
        if "model" in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        elif "env_wrappers" in self.params:
            self.fps = 30
        else:
            self.fps = self.env.env.metadata["video.frames_per_second"]
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
            if (
                itr % self.params["video_log_freq"] == 0
                and self.params["video_log_freq"] != -1
            ):
                self.logvideo = True
            else:
                self.logvideo = False
            if self.params["scalar_log_freq"] == -1:
                self.logmetrics = False
            elif itr % self.params["scalar_log_freq"] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False
            if isinstance(self.agent, DQNAgent):

                self.agent.step_env()
                envsteps_this_batch = 1
                train_video_paths = None
                paths = None
            else:
                paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr,
                        initial_expertdata,
                        collect_policy,
                        self.params["batch_size"],
                    )
                )

            self.total_envsteps += envsteps_this_batch
            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)
            self.agent.add_to_replay_buffer(paths)
            loss = self.train_agent()
            if self.logvideo or self.logmetrics:

                print("\nBeginning logging procedure...")
                if isinstance(self.agent, DQNAgent):
                    self.perform_dqn_logging()
                else:
                    self.perform_logging(
                        itr, paths, eval_policy, train_video_paths, loss
                    )
                if self.params["save_params"]:
                    print("\nSaving agent's actor...")
                    self.agent.actor.save(
                        self.params["logdir"] + "/policy_itr_" + str(itr)
                    )
                    self.agent.critic.save(
                        self.params["logdir"] + "/critic_itr_" + str(itr)
                    )
    def collect_training_trajectories(
        self, itr, load_initial_expertdata, collect_policy, batch_size
    ):

        if itr == 0 and load_initial_expertdata:
            with open(load_initial_expertdata, "rb") as f:
                loaded_paths = pickle.load(f)
            return loaded_paths, 0, None

        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(
            self.env, collect_policy, batch_size, self.params["ep_len"]
        )

        train_video_paths = None
        if self.logvideo:
            print("\nCollecting train rollouts to be used for saving videos...")
            train_video_paths = sample_n_trajectories(
                self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True
            )

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):

        for train_step in range(self.params["num_agent_train_steps_per_iter"]):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = (
                self.agent.sample(self.params["train_batch_size"])
            )
            loss = self.agent.train(
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch
            )

        return loss
    def perform_dqn_logging(self):
        monitor = get_wrapper_by_name(self.env, "Monitor")
        if monitor is not None:
            episode_rewards = monitor.get_episode_rewards()
        else:
            episode_rewards = []
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(
                self.best_mean_episode_reward, self.mean_episode_reward
            )

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = time.time() - self.start_time
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        sys.stdout.flush()

        for key, value in logs.items():
            print("{} : {}".format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print("Done logging...\n\n")

        self.logger.flush()

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, loss):
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = sample_trajectories(
            self.env, eval_policy, self.params["eval_batch_size"], self.params["ep_len"]
        )
        if self.logvideo and train_video_paths != None:
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
        if self.logmetrics:

            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

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