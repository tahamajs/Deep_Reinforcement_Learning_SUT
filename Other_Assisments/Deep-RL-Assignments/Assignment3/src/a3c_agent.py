import os
import json
import argparse
from datetime import datetime
from collections import deque

import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from src.preprocessing import preprocess
from src.model import ActorCritic
class A3C:
    """Implementation of N-step Asynchronous Advantage Actor Critic"""

    def __init__(self, args, env, train=True):
        self.args = args
        self.set_random_seeds()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(env)
        self.environment_name = env
        self.policy = ActorCritic(4, self.env.action_space.n)
        self.policy.apply(self.initialize_weights)
        self.critic = ActorCritic(4, self.env.action_space.n)
        self.critic.apply(self.initialize_weights)
        self.eps = 1e-10
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=args.policy_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.timestamp = datetime.now().strftime("a2c-breakout-%Y-%m-%d_%H-%M-%S")
        self.weights_path = "models/%s/%s" % (self.environment_name, self.timestamp)
        if args.weights_path:
            self.load_model()
        self.policy.to(self.device)
        self.critic.to(self.device)
        if args.render:
            self.policy.eval()
            self.generate_episode(render=True)
            self.plot()
            return
        self.rewards_data = []
        if train:

            self.logdir = "logs/%s/%s" % (self.environment_name, self.timestamp)
            self.summary_writer = SummaryWriter(self.logdir)
            with open(self.logdir + "/training_parameters.json", "w") as f:
                json.dump(vars(self.args), f, indent=4)

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def set_random_seeds(self):
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        torch.backends.cudnn.benchmark = True

    def save_model(self, epoch):
        """Helper function to save model state and weights."""
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "rewards_data": self.rewards_data,
                "epoch": epoch,
            },
            os.path.join(self.weights_path, "model_%d.h5" % epoch),
        )

    def load_model(self):
        """Helper function to load model state and weights."""
        if os.path.isfile(self.args.weights_path):
            print("=> Loading checkpoint", self.args.weights_path)
            self.checkpoint = torch.load(self.args.weights_path)
            self.policy.load_state_dict(self.checkpoint["policy_state_dict"])
            self.policy_optimizer.load_state_dict(self.checkpoint["policy_optimizer"])
            self.critic.load_state_dict(self.checkpoint["critic_state_dict"])
            self.critic_optimizer.load_state_dict(self.checkpoint["critic_optimizer"])
            self.rewards_data = self.checkpoint["rewards_data"]
        else:
            raise Exception("No checkpoint found at %s" % self.args.weights_path)

    def train(self):
        """Trains the model on a single episode using REINFORCE."""
        for epoch in range(self.args.num_episodes):

            returns, log_probs, value_function, train_rewards = self.generate_episode()
            self.summary_writer.add_scalar(
                "train/cumulative_rewards", train_rewards, epoch
            )
            self.summary_writer.add_scalar(
                "train/trajectory_length", returns.size()[0], epoch
            )
            self.policy_optimizer.zero_grad()
            policy_loss = ((returns - value_function.detach()) * -log_probs).mean()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss = F.mse_loss(returns, value_function)
            critic_loss.backward()
            self.critic_optimizer.step()
            if epoch % self.args.test_interval == 0:
                self.policy.eval()
                print("\nTesting")
                rewards = [
                    self.generate_episode(test=True)
                    for epoch in range(self.args.test_episodes)
                ]
                rewards_mean, rewards_std = np.mean(rewards), np.std(rewards)
                print(
                    "Test Rewards (Mean): %.3f | Test Rewards (Std): %.3f\n"
                    % (rewards_mean, rewards_std)
                )
                self.rewards_data.append([epoch, rewards_mean, rewards_std])
                self.summary_writer.add_scalar("test/rewards_mean", rewards_mean, epoch)
                self.summary_writer.add_scalar("test/rewards_std", rewards_std, epoch)
                self.policy.train()
            if epoch % self.args.log_interval == 0:
                print(
                    "Epoch: {0:05d}/{1:05d} | Policy Loss: {2:.3f} | Value Loss: {3:.3f}".format(
                        epoch, self.args.num_episodes, policy_loss, critic_loss
                    )
                )
                self.summary_writer.add_scalar("train/policy_loss", policy_loss, epoch)
                self.summary_writer.add_scalar("train/critic_loss", critic_loss, epoch)
            if epoch % self.args.save_interval == 0:
                self.save_model(epoch)

        self.save_model(epoch)
        self.summary_writer.close()

    def generate_episode(self, gamma=0.99, test=False, render=False, max_iters=10000):
        """
        Generates an episode by executing the current policy in the given env.
        Returns:
        - a list of states, indexed by time epoch
        - a list of actions, indexed by time epoch
        - a list of cumulative discounted returns, indexed by time epoch
        """
        iters = 0
        done = False
        state = self.env.reset()
        if render:
            save_path = "videos/%s/epoch-%s" % (
                self.environment_name,
                self.checkpoint["epoch"],
            )
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            monitor = gym.wrappers.Monitor(self.env, save_path, force=True)

        batches = []
        states = [torch.zeros(84, 84, device=self.device).float()] * 3
        rewards, returns = [], []
        actions, log_probs = [], []

        while not done:

            states.append(
                torch.tensor(preprocess(state), device=self.device).float().squeeze(0)
            )
            batches.append(torch.stack(states[-4:]))
            action_probs = self.policy.forward(batches[-1].unsqueeze(0)).squeeze(0)
            if test and self.args.det_eval:
                action = torch.argmax(action_probs)
            else:
                action = torch.argmax(
                    torch.distributions.Multinomial(logits=action_probs).sample()
                )
            actions.append(action)
            log_probs.append(action_probs[action])
            if render:
                monitor.render()
            state, reward, done, _ = self.env.step(action.cpu().numpy())
            rewards.append(reward)
            iters += 1
            if iters > max_iters:
                break
        cum_rewards = np.sum(rewards)
        if render:
            monitor.close()
            print("\nCumulative Rewards:", cum_rewards)
            return
        if test:
            return cum_rewards
        rewards = np.array(rewards) / self.args.reward_normalizer
        values = []
        minibatches = torch.split(torch.stack(batches), 256)
        for minibatch in minibatches:
            values.append(self.critic.forward(minibatch, action=False).squeeze(1))
        values = torch.cat(values)
        discounted_values = values * gamma**self.args.n
        n_step_rewards = np.zeros((1, self.args.n))
        for i in reversed(range(rewards.shape[0])):
            if i + self.args.n >= rewards.shape[0]:
                V_end = 0
            else:
                V_end = discounted_values[i + self.args.n]
            n_step_rewards[0, :-1] = n_step_rewards[0, 1:] * gamma
            n_step_rewards[0, -1] = rewards[i]

            n_step_return = (
                torch.tensor(n_step_rewards.sum(), device=self.device).unsqueeze(0)
                + V_end
            )
            returns.append(n_step_return)

        return (
            torch.stack(returns[::-1]).detach().squeeze(1),
            torch.stack(log_probs),
            values.squeeze(),
            cum_rewards,
        )

    def plot(self):
        """Save the plot."""
        filename = os.path.join(
            "plots", *self.args.weights_path.split("/")[-2:]
        ).replace(".h5", ".png")
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        data = np.asarray(self.rewards_data)
        plt.errorbar(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            lw=2.5,
            elinewidth=1.5,
            ecolor="grey",
            barsabove=True,
            capthick=2,
            capsize=3,
        )
        plt.title("Cumulative Rewards (Mean/Std) Plot for A3C Algorithm")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Cumulative Rewards")
        plt.grid()
        plt.savefig(filename, dpi=300)
        plt.show()