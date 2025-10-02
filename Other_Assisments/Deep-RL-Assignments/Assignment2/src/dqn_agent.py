import copy
import json
import os
import keras
import numpy as np
import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
from tensorboardX import SummaryWriter
from src.q_network import QNetwork
from src.replay_memory import Replay_Memory
class DQN_Agent:
    def __init__(self, args):
        self.args = args
        self.environment_name = self.args.env
        self.render = self.args.render
        self.epsilon = args.epsilon
        self.network_update_freq = args.network_update_freq
        self.log_freq = args.log_freq
        self.test_freq = args.test_freq
        self.save_freq = args.save_freq
        self.learning_rate = self.args.learning_rate
        if self.environment_name == "CartPole-v0":
            self.env = gym.make(self.environment_name)
            self.discount_factor = 0.99
            self.num_episodes = 5000
        elif self.environment_name == "MountainCar-v0":
            self.env = gym.make(self.environment_name)
            self.discount_factor = 1.00
            self.num_episodes = 10000
        else:
            raise Exception("Unknown Environment")
        self.q_network = QNetwork(
            args,
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.learning_rate,
        )
        self.target_q_network = QNetwork(
            args,
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.learning_rate,
        )
        self.memory = Replay_Memory(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            memory_size=self.args.memory_size,
        )
        self.rewards = []
        self.td_error = []
        self.batch = list(range(32))
        self.logdir = "logs/%s/%s" % (
            self.environment_name,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        self.summary_writer = SummaryWriter(self.logdir)
        with open(self.logdir + "/hyperparameters.json", "w") as outfile:
            json.dump(vars(self.args), outfile, indent=4)

    def epsilon_greedy_policy(self, q_values, epsilon):

        p = np.random.uniform(0, 1)
        if p < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(q_values)

    def greedy_policy(self, q_values):
        return np.argmax(q_values)

    def train(self):
        self.burn_in_memory()
        for step in range(self.num_episodes):

            self.generate_episode(
                policy=self.epsilon_greedy_policy,
                mode="train",
                epsilon=self.epsilon,
                frameskip=self.args.frameskip,
            )
            if step % self.test_freq == 0:
                test_reward, test_error = self.test(episodes=20)
                self.rewards.append([test_reward, step])
                self.td_error.append([test_error, step])
                self.summary_writer.add_scalar("test/reward", test_reward, step)
                self.summary_writer.add_scalar("test/td_error", test_error, step)
            if step % self.network_update_freq == 0:
                self.hard_update()
            if step % self.log_freq == 0:
                print("Step: {0:05d}/{1:05d}".format(step, self.num_episodes))
            if step % self.save_freq == 0:
                self.q_network.save_model_weights(step)

            step += 1
            self.epsilon_decay()
            if step % int(self.num_episodes / 3) == 0 and self.args.render:

                self.q_network.save_model_weights(step)

        self.summary_writer.export_scalars_to_json(
            os.path.join(self.logdir, "all_scalars.json")
        )
        self.summary_writer.close()

    def train_dqn(self):

        state, action, rewards, next_state, done = self.memory.sample_batch(
            batch_size=32
        )
        _y = rewards + self.discount_factor * np.multiply(
            (1 - done),
            np.amax(
                self.target_q_network.model.predict_on_batch(next_state),
                axis=1,
                keepdims=True,
            ),
        )
        y = self.q_network.model.predict_on_batch(state)
        y[self.batch, action.squeeze().astype(int)] = _y.squeeze()
        history = self.q_network.model.fit(
            state, y, epochs=1, batch_size=32, verbose=False
        )
        loss = history.history["loss"][-1]
        acc = history.history["acc"][-1]
        return loss, acc

    def train_double_dqn(self):

        state, action, rewards, next_state, done = self.memory.sample_batch(
            batch_size=32
        )
        next_action = np.argmax(
            self.q_network.model.predict_on_batch(next_state), axis=1
        )
        _y = rewards + self.discount_factor * np.multiply(
            (1 - done),
            self.target_q_network.model.predict_on_batch(next_state)[
                self.batch, next_action
            ].reshape(-1, 1),
        )
        y = self.q_network.model.predict_on_batch(state)
        y[self.batch, action.squeeze().astype(int)] = _y.squeeze()
        history = self.q_network.model.fit(
            state, y, epochs=1, batch_size=32, verbose=False
        )
        loss = history.history["loss"][-1]
        acc = history.history["acc"][-1]
        return loss, acc

    def hard_update(self):
        self.target_q_network.model.set_weights(self.q_network.model.get_weights())

    def test(self, model_file=None, episodes=100):
        cum_reward = []
        td_error = []
        for count in range(episodes):
            reward, error = self.generate_episode(
                policy=self.epsilon_greedy_policy,
                mode="test",
                epsilon=0.05,
                frameskip=self.args.frameskip,
            )
            cum_reward.append(reward)
            td_error.append(error)
        cum_reward = np.array(cum_reward)
        td_error = np.array(td_error)
        print(
            "\nTest Rewards: {0} | TD Error: {1:.4f}\n".format(
                np.mean(cum_reward), np.mean(td_error)
            )
        )
        return np.mean(cum_reward), np.mean(td_error)

    def burn_in_memory(self):

        while not self.memory.burned_in:
            self.generate_episode(
                policy=self.epsilon_greedy_policy,
                mode="burn_in",
                epsilon=self.epsilon,
                frameskip=self.args.frameskip,
            )
        print("Burn Complete!")

    def generate_episode(self, policy, epsilon, mode="train", frameskip=1):
        """
        Collects one rollout from the policy in an environment.
        """
        done = False
        state = self.env.reset()
        rewards = 0
        q_values = self.q_network.model.predict(state.reshape(1, -1))
        td_error = []
        while not done:
            action = policy(q_values, epsilon)
            i = 0
            while (i < frameskip) and not done:
                next_state, reward, done, info = self.env.step(action)
                rewards += reward
                i += 1
            next_q_values = self.q_network.model.predict(next_state.reshape(1, -1))
            if mode in ["train", "burn_in"]:
                self.memory.append(state, action, reward, next_state, done)
            else:
                td_error.append(
                    abs(
                        reward
                        + self.discount_factor * (1 - done) * np.max(next_q_values)
                        - q_values
                    )
                )
            if not done:
                state = copy.deepcopy(next_state)
                q_values = copy.deepcopy(next_q_values)
            if mode == "train":
                if self.args.double_dqn:
                    self.train_double_dqn()
                else:
                    self.train_dqn()

        return rewards, np.mean(td_error)

    def plots(self):
        """
        Plots:
        1) Avg Cummulative Test Reward over 20 Plots
        2) TD Error
        """
        reward, time = zip(*self.rewards)
        plt.figure(figsize=(8, 3))
        plt.subplot(121)
        plt.title("Cummulative Reward")
        plt.plot(time, reward)
        plt.xlabel("iterations")
        plt.ylabel("rewards")
        plt.legend()
        plt.ylim([0, None])

        loss, time = zip(*self.td_error)
        plt.subplot(122)
        plt.title("Loss")
        plt.plot(time, loss)
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.show()

    def epsilon_decay(self, initial_eps=1.0, final_eps=0.05):
        if self.epsilon > final_eps:
            factor = (initial_eps - final_eps) / 10000
            self.epsilon -= factor