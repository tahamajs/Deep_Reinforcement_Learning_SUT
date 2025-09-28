"""
DDPG Agent implementation with support for DDPG, TD3, and HER algorithms.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from .actor import ActorNetwork
from .critic import CriticNetwork, CriticNetworkTD3
from .replay_buffer import ReplayBuffer


class EpsilonNormalActionNoise:
    """Action noise for exploration in continuous action spaces."""

    def __init__(self, mu, sigma, epsilon):
        """Initialize the action noise.

        Args:
            mu: (float) mean of the noise
            sigma: (float) standard deviation of the noise
            epsilon: (float) probability of taking random action
        """
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, action):
        """Add noise to the action.

        Args:
            action: (np.ndarray) action to add noise to

        Returns:
            noisy_action: (np.ndarray) action with added noise
        """
        if np.random.uniform() > self.epsilon:
            return np.clip(
                action + np.random.normal(self.mu, self.sigma, action.shape), -1.0, 1.0
            )
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPGAgent:
    """DDPG Agent with support for DDPG, TD3, and HER algorithms."""

    def __init__(self, args, env, outfile_name):
        """Initialize the DDPG agent.

        Args:
            args: (argparse.Namespace) command line arguments
            env: (gym.Env) environment instance
            outfile_name: (str) name of the output file
        """
        self.action_dim = len(env.action_space.low)
        self.state_dim = len(env.observation_space.low)
        np.random.seed(1337)

        self.env = env
        self.args = args
        self.outfile = outfile_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.weights_path = f"models/{self.timestamp}"

        # Data for plotting
        self.rewards_data = []  # n * [epoch, mean(returns), std(returns)]
        self.count = 0

        # Initialize components
        self.action_selector = EpsilonNormalActionNoise(0, 0.05, self.args.epsilon)
        self.memory = ReplayBuffer(
            args.buffer_size, args.burn_in, self.state_dim, self.action_dim, self.device
        )

        self.actor = ActorNetwork(
            self.state_dim,
            self.action_dim,
            self.args.batch_size,
            self.args.tau,
            self.args.actor_lr,
            self.device,
            args.custom_init,
        )

        if self.args.algorithm in ["ddpg", "her"]:
            self.critic = CriticNetwork(
                self.state_dim,
                self.action_dim,
                self.args.batch_size,
                self.args.tau,
                self.args.critic_lr,
                self.args.gamma,
                self.device,
                args.custom_init,
            )
        elif self.args.algorithm == "td3":
            self.critic = CriticNetworkTD3(
                self.state_dim,
                self.action_dim,
                self.args.batch_size,
                self.args.tau,
                self.args.critic_lr,
                self.args.gamma,
                self.device,
                args.custom_init,
            )

        # Load model if weights path is provided
        if args.weights_path:
            self.load_model()

        # Initialize training components if training
        if args.train:
            self.logdir = f"logs/{self.timestamp}"
            self.imgdir = f"imgs/{self.timestamp}"
            os.makedirs(self.imgdir, exist_ok=True)
            self.summary_writer = SummaryWriter(self.logdir)

            # Save hyperparameters
            with open(f"{self.logdir}/training_parameters.json", "w") as f:
                json.dump(vars(self.args), f, indent=4)

    def save_model(self, epoch):
        """Save model state and weights.

        Args:
            epoch: (int) current training epoch
        """
        os.makedirs(self.weights_path, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.actor.policy.state_dict(),
                "policy_target_state_dict": self.actor.policy_target.state_dict(),
                "policy_optimizer": self.actor.policy_optimizer.state_dict(),
                "critic_state_dict": self.critic.critic.state_dict(),
                "critic_target_state_dict": self.critic.critic_target.state_dict(),
                "critic_optimizer": self.critic.critic_optimizer.state_dict(),
                "rewards_data": self.rewards_data,
                "epoch": epoch,
            },
            os.path.join(self.weights_path, f"model_{epoch}.h5"),
        )

    def load_model(self):
        """Load model state and weights."""
        if os.path.isfile(self.args.weights_path):
            print(f"=> Loading checkpoint {self.args.weights_path}")
            checkpoint = torch.load(self.args.weights_path)
            self.actor.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.actor.policy_target.load_state_dict(
                checkpoint["policy_target_state_dict"]
            )
            self.actor.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
            self.critic.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic.critic_target.load_state_dict(
                checkpoint["critic_target_state_dict"]
            )
            self.critic.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            self.rewards_data = checkpoint["rewards_data"]
        else:
            raise Exception(f"No checkpoint found at {self.args.weights_path}")

    def evaluate(self, num_episodes):
        """Evaluate the policy without exploration noise.

        Args:
            num_episodes: (int) number of evaluation episodes

        Returns:
            success_rate: (float) fraction of successful episodes
            average_return: (float) average cumulative return
            std_return: (float) standard deviation of returns
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))

        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False

            while not done:
                s_vec.append(s_t)
                with torch.no_grad():
                    a_t = self.actor.policy(
                        torch.tensor(s_t, device=self.device).float()
                    )
                new_s, r_t, done, info = self.env.step(a_t.cpu().numpy())

                if done and "goal" in info.get("done", ""):
                    success = True

                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1

            success_vec.append(success)
            test_rewards.append(total_reward)

            # Plot trajectories for first 9 episodes if rendering is enabled
            if i < 9 and self.args.render:
                plt.subplot(3, 3, i + 1)
                s_vec = np.array(s_vec)
                pusher_vec = s_vec[:, :2]
                puck_vec = s_vec[:, 2:4]
                goal_vec = s_vec[:, 4:]
                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], "-o", label="pusher")
                plt.plot(puck_vec[:, 0], puck_vec[:, 1], "-o", label="puck")
                plt.plot(
                    goal_vec[:, 0], goal_vec[:, 1], "*", label="goal", markersize=10
                )
                plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], "k-", linewidth=3)
                plt.fill_between(
                    [-1, 6], [-1, -1], [6, 6], alpha=0.1, color="g" if success else "r"
                )
                plt.xlim([-1, 6])
                plt.ylim([-1, 6])
                if i == 0:
                    plt.legend(
                        loc="lower left", fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0)
                    )
                if i == 8:
                    plt.savefig(os.path.join(self.imgdir, str(self.count)))
                    self.count += 1

        return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards)

    def train(self, num_episodes):
        """Train the DDPG agent.

        Args:
            num_episodes: (int) number of training episodes
        """
        for i in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False
            step = 0
            critic_loss = 0
            trajectory_data = []
            state = torch.tensor(state, device=self.device).float()

            while not done:
                # Select action with exploration noise
                with torch.no_grad():
                    action = self.actor.policy(state)
                    env_action = self.action_selector(action.cpu().numpy())
                    action = torch.tensor(env_action, device=self.device).float()

                # Take step in environment
                next_state, reward, done, info = self.env.step(env_action)
                next_state = torch.tensor(next_state, device=self.device).float()

                # Store transition in replay buffer
                self.memory.add(
                    state,
                    action,
                    torch.tensor(reward, device=self.device),
                    next_state,
                    torch.tensor(done, device=self.device),
                )

                # Store trajectory data for HER
                if self.args.algorithm == "her":
                    trajectory_data.append(
                        [
                            state.detach().cpu().numpy(),
                            action.detach().cpu().numpy(),
                            reward,
                            next_state.detach().cpu().numpy(),
                            done,
                        ]
                    )

                total_reward += reward
                step += 1

                if not done:
                    state = next_state.clone().detach()

            # Add HER experience if using HER
            if self.args.algorithm == "her":
                self.add_hindsight_replay_experience(trajectory_data)

            # Train networks if buffer is ready
            if self.memory.burned_in:
                if self.args.algorithm in ["ddpg", "her"]:
                    critic_loss, policy_loss, new_metric = self.train_ddpg()
                    self.summary_writer.add_scalar("train/policy_loss", policy_loss, i)
                    self.summary_writer.add_scalar(
                        "train/new_metric", new_metric.mean(), i
                    )

                elif self.args.algorithm == "td3":
                    critic_loss, policy_loss = self.train_td3(i)
                    if i % self.args.policy_update_frequency == 0:
                        self.summary_writer.add_scalar(
                            "train/policy_loss", policy_loss, i
                        )

            # Logging
            if self.memory.burned_in and i % self.args.log_interval == 0:
                print(f"Episode {i}: Total reward = {total_reward}")
                print(f"\tTD loss = {critic_loss / step:.2f}")
                self.summary_writer.add_scalar("train/trajectory_length", step, i)
                self.summary_writer.add_scalar("train/critic_loss", critic_loss, i)

            # Evaluation
            if i % self.args.test_interval == 0:
                successes, mean_rewards, std_rewards = self.evaluate(10)
                self.rewards_data.append([i, mean_rewards, std_rewards])

                self.summary_writer.add_scalar("test/success", successes, i)
                self.summary_writer.add_scalar("test/rewards_mean", mean_rewards, i)
                self.summary_writer.add_scalar("test/rewards_std", std_rewards, i)

                print(
                    f"Evaluation: success = {successes:.2f}; return = {mean_rewards:.2f}"
                )
                with open(self.outfile, "a") as f:
                    f.write(f"{successes:.2f}, {mean_rewards:.2f},\n")

            # Save model
            if i % self.args.save_interval == 0:
                self.save_model(i)

        self.save_model(i)
        self.summary_writer.close()

    def add_hindsight_replay_experience(self, trajectory):
        """Add hindsight experience replay (HER) transitions.

        Args:
            trajectory: (list) list of [state, action, reward, next_state, done] tuples
        """
        # Get new goal location (last location of box)
        goal = trajectory[-1][3][2:4]

        # Relabel trajectory with new goal
        for state, action, reward, next_state, done in trajectory:
            state[-2:] = goal.copy()
            next_state[-2:] = goal.copy()
            reward = self.env._HER_calc_reward(state)
            if reward == 0:
                done = True

            self.memory.add(
                torch.tensor(state, device=self.device),
                torch.tensor(action, device=self.device),
                torch.tensor(reward, device=self.device),
                torch.tensor(next_state, device=self.device),
                torch.tensor(done, device=self.device),
            )

            if reward == 0:
                break

    def train_ddpg(self):
        """Train using DDPG algorithm."""
        for _ in range(self.args.num_update_iters):
            states, actions, rewards, next_states, dones = self.memory.get_batch(
                self.args.batch_size
            )
            next_actions = self.actor.policy_target(next_states).detach()
            critic_loss = self.critic.train(
                states, actions, rewards, next_states, dones, next_actions
            )
            new_Q_value = self.critic.critic(states, self.actor.policy(states))
            policy_loss = self.actor.train(new_Q_value)

            self.critic.update_target()
            self.actor.update_target()

        return critic_loss, policy_loss, new_Q_value

    def train_td3(self, i):
        """Train using TD3 algorithm.

        Args:
            i: (int) current episode number

        Returns:
            critic_loss: (float) critic loss
            policy_loss: (float) policy loss
        """
        for j in range(self.args.num_update_iters):
            states, actions, rewards, next_states, dones = self.memory.get_batch(
                self.args.batch_size
            )
            next_actions = self.noise_regularization(
                self.actor.policy_target(next_states).detach().cpu().numpy()
            )
            next_actions = torch.tensor(next_actions, device=self.device).float()

            critic_loss = self.critic.train(
                states, actions, rewards, next_states, dones, next_actions
            )

            policy_loss = 0
            if (
                i * self.args.num_update_iters + j
            ) % self.args.policy_update_frequency == 0:
                policy_loss = self.actor.train(
                    self.critic.critic.get_Q(states, self.actor.policy(states))
                )

                self.critic.update_target()
                self.actor.update_target()

        return critic_loss, policy_loss

    def noise_regularization(self, next_actions):
        """Add noise regularization for TD3.

        Args:
            next_actions: (np.ndarray) next actions

        Returns:
            regularized_actions: (np.ndarray) actions with added noise
        """
        noise = np.clip(
            np.random.normal(
                0,
                self.args.target_action_sigma,
                (self.args.batch_size, self.action_dim),
            ),
            -self.args.clip,
            self.args.clip,
        )
        return np.clip(next_actions + noise, -1.0, 1.0)

    def plot(self):
        """Plot training results."""
        filename = os.path.join(
            "plots", *self.args.weights_path.split("/")[-2:]
        ).replace(".h5", ".png")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Plot error bars with mean and std of rewards
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
        plt.title("Cumulative Rewards (Mean/Std) Plot for DDPG Algorithm")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Cumulative Rewards")
        plt.grid()
        plt.savefig(filename, dpi=300)
        plt.show()
