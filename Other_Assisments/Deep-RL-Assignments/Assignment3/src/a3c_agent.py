import os
import json
import argparse
from datetime import datetime
from collections import deque
import torch.multiprocessing as mp
import multiprocessing as _mp

# Use fork start method on macOS/Linux to avoid pickling non-picklable objects
# (e.g., SummaryWriter file handles) when spawning worker processes.
try:
    mp.set_start_method("fork")
except RuntimeError:
    # start method may have already been set by another part of the program;
    # in that case, we leave it as-is.
    pass

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

    def __init__(self, args, env_name, train=True):  # Ensure this says env_name
        self.args = args
        self.env_name = env_name
        self.set_random_seeds()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Setup Environment
        import shimmy  # Ensure shimmy is imported for Atari

        # self.env = gym.make(env)
        # self.environment_name = env
        self.args = args
        self.env_name = env_name  # Store the string, don't keep the env object
        self.set_random_seeds()
        # 2. Setup Global Model (The missing part)
        # We use one unified ActorCritic network for A3C
        temp_env = gym.make(env_name)
        num_actions = temp_env.action_space.n
        temp_env.close()

        self.global_model = ActorCritic(4, num_actions)
        # CRITICAL: Share memory allows workers to update the same weights
        self.global_model.share_memory()

        # 3. Setup Optimizer (Shared optimizer is often needed for A3C)
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=args.policy_lr)

        self.timestamp = datetime.now().strftime("a3c-breakout-%Y-%m-%d_%H-%M-%S")
        self.global_episode = mp.Value("i", 0)

        self.weights_path = "models/%s/%s" % (self.env_name, self.timestamp)

        # Load weights if provided
        if args.weights_path:
            self.load_model()
        self.global_episode_reward = mp.Value("d", 0.0)  # 'd' for double (float)
        self.global_avg_reward = mp.Value("d", 0.0)  # 'd' for double (float)
        self.best_score = mp.Value(
            "d", -float("inf")
        )  # Track the best score seen so far

        # No need to move global_model to GPU for A3C training usually
        # (workers run on CPU, model stays in shared CPU memory)
        # If you are strictly on CPU, this is fine.

        if args.render:
            # For rendering we might just use the global model
            self.global_model.eval()
            self.generate_episode(render=True)
            self.plot()
            return

        self.rewards_data = []
        if train:
            self.logdir = "logs/%s/%s" % (self.env_name, self.timestamp)
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

    def test_model_worker(self, global_model, episode, lock):
        """Test the model during training"""
        with lock:
            test_rewards = []
            for _ in range(self.args.test_episodes):
                reward = self.generate_episode_test(global_model)
                test_rewards.append(reward)

            rewards_mean, rewards_std = np.mean(test_rewards), np.std(test_rewards)
            print(
                f"\nTest Results | Episode {episode} | Mean: {rewards_mean:.2f} | Std: {rewards_std:.2f}"
            )

            self.rewards_data.append([episode, rewards_mean, rewards_std])
            self.summary_writer.add_scalar("test/rewards_mean", rewards_mean, episode)
            self.summary_writer.add_scalar("test/rewards_std", rewards_std, episode)

    def generate_episode_test(self, model):
        """Generate a test episode"""
        env = gym.make(self.env_name)
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(
                preprocess(state), dtype=torch.float32
            ).unsqueeze(0)
            policy, _ = model(state_tensor)
            action = torch.argmax(policy).item()

            state, reward, done, _, _ = env.step(action)
            total_reward += reward

        env.close()
        return total_reward

    def test_model(self):
        """Test the loaded model with rendering"""
        env = gym.make(self.env_name, render_mode="human")
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 10000:
            state_tensor = torch.tensor(
                preprocess(state), dtype=torch.float32
            ).unsqueeze(0)
            policy, _ = self.global_model(state_tensor)
            action = torch.argmax(policy).item()

            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1

            env.render()

        print(f"Test completed. Total reward: {total_reward}")
        env.close()
        self.plot()

    def worker(
        self,
        worker_id,
        global_model,
        optimizer,
        global_episode,
        global_episode_reward,
        global_avg_reward,
        best_score,
        lock,
    ):
        """A3C worker process"""
        torch.manual_seed(self.args.random_seed + worker_id)
        np.random.seed(self.args.random_seed + worker_id)

        # Use a short-lived environment to infer action space for the local model.
        temp_env = gym.make(self.env_name)
        local_model = ActorCritic(4, temp_env.action_space.n)
        local_model.load_state_dict(global_model.state_dict())
        temp_env.close()

        # Each worker should create its own environment for interaction.
        env = gym.make(self.env_name)
        episode_count = 0

        while global_episode.value < self.args.num_episodes:
            local_model.load_state_dict(global_model.state_dict())

            episode_reward = 0
            done = False
            state, _ = env.reset()
            states, actions, rewards, log_probs, values = [], [], [], [], []

            # Generate episode
            for t in range(self.args.n):
                state_tensor = torch.tensor(
                    preprocess(state), dtype=torch.float32
                ).unsqueeze(0)
                states.append(state_tensor)

                policy, value = local_model(state_tensor)
                action = torch.multinomial(torch.exp(policy), 1).item()

                next_state, reward, done, _, _ = env.step(action)

                actions.append(action)
                rewards.append(reward)
                log_probs.append(policy[0, action])
                values.append(value.item())

                episode_reward += reward
                state = next_state

                if done:
                    break

            # Calculate returns and advantages
            if done:
                R = 0
            else:
                state_tensor = torch.tensor(
                    preprocess(state), dtype=torch.float32
                ).unsqueeze(0)
                _, value = local_model(state_tensor)
                R = value.item()

            returns = []
            for r in rewards[::-1]:
                R = r + 0.99 * R
                returns.append(R)
            returns.reverse()

            # Convert to tensors
            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32)
            log_probs = torch.stack(log_probs)

            # Calculate advantages
            advantages = returns - values

            # Calculate losses
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -0.01 * torch.sum(
                torch.exp(log_probs) * log_probs
            )  # Small entropy bonus

            total_loss = policy_loss + 0.5 * value_loss + entropy_loss

            # Update global model
            optimizer.zero_grad()
            total_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)

            # Update global parameters
            for local_param, global_param in zip(
                local_model.parameters(), global_model.parameters()
            ):
                global_param._grad = local_param.grad

            optimizer.step()

            # Update global counters
            with lock:
                global_episode.value += 1
                global_episode_reward.value = episode_reward

                # Update running average
                global_avg_reward.value = (
                    0.9 * global_avg_reward.value + 0.1 * episode_reward
                )

                if episode_reward > best_score.value:
                    best_score.value = episode_reward

                episode_count += 1

                # Logging
                if episode_count % self.args.log_interval == 0:
                    print(
                        f"Worker {worker_id} | Episode {global_episode.value} | Reward: {episode_reward:.2f} | Avg: {global_avg_reward.value:.2f}"
                    )

                # Testing
                if global_episode.value % self.args.test_interval == 0:
                    self.test_model_worker(
                        global_model, int(global_episode.value), lock
                    )

                # Save model
                if global_episode.value % self.args.save_interval == 0:
                    self.save_model_worker(
                        global_model, int(global_episode.value), lock
                    )

        env.close()

    def save_model_worker(self, global_model, epoch, lock):
        """Helper function to save model state and weights."""
        with lock:
            if not os.path.exists(self.weights_path):
                os.makedirs(self.weights_path)
            torch.save(
                {
                    "model_state_dict": global_model.state_dict(),
                    "optimizer_state_dict": self.global_optimizer.state_dict(),
                    "rewards_data": self.rewards_data,
                    "epoch": epoch,
                },
                os.path.join(self.weights_path, f"model_{epoch}.h5"),
            )

    def load_model(self):
        """Helper function to load model state and weights."""
        if os.path.isfile(self.args.weights_path):
            print("=> Loading checkpoint", self.args.weights_path)
            checkpoint = torch.load(self.args.weights_path)
            self.global_model.load_state_dict(checkpoint["model_state_dict"])
            self.global_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.rewards_data = checkpoint.get("rewards_data", [])
        else:
            raise Exception("No checkpoint found at %s" % self.args.weights_path)

    def train(self):
        """Train A3C with multiple asynchronous workers"""
        # Start workers
        num_workers = min(mp.cpu_count(), 8)  # Use up to 8 workers
        lock = mp.Lock()

        workers = []
        # Use a fork-based context for spawning worker processes to avoid
        # pickling the A3C instance (SummaryWriter, file handles, etc.).
        ctx = _mp.get_context("fork")
        for worker_id in range(num_workers):
            p = ctx.Process(
                target=self.worker,
                args=(
                    worker_id,
                    self.global_model,
                    self.optimizer,
                    self.global_episode,
                    self.global_episode_reward,
                    self.global_avg_reward,
                    self.best_score,
                    lock,
                ),
            )
            p.start()
            workers.append(p)

        # Monitor training
        try:
            while self.global_episode.value < self.args.num_episodes:
                time.sleep(1)  # Check every second

                with lock:
                    episode = self.global_episode.value
                    avg_reward = self.global_avg_reward.value
                    best = self.best_score.value

                print(
                    f"\rGlobal Episode: {episode}/{self.args.num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | Best: {best:.2f}",
                    end="",
                    flush=True,
                )

                if episode >= self.args.num_episodes:
                    break

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        # Wait for all workers to finish
        for p in workers:
            p.join()

        print(f"\nTraining completed. Best score: {self.best_score.value:.2f}")
        self.summary_writer.close()

    def plot(self):
        """Save the plot."""
        if not self.rewards_data:
            print("No rewards data to plot")
            return

        filename = os.path.join(
            "plots", self.environment_name, f"{self.timestamp}_rewards.png"
        )
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        data = np.array(self.rewards_data)
        plt.figure(figsize=(10, 6))
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
        plt.title("Test Rewards (Mean/Std) Plot for A3C Algorithm")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Test Rewards")
        plt.grid(True)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
