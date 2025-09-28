"""
Foundation Models Training Utilities

This module contains training utilities and data handling for foundation models:
- Trajectory preprocessing
- Dataset creation
- Training loops
- Evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrajectoryDataset(Dataset):
    """Dataset for trajectory data."""

    def __init__(self, trajectories, max_length=1024, discount_factor=0.99):
        self.trajectories = trajectories
        self.max_length = max_length
        self.discount_factor = discount_factor

        self.processed_trajectories = []
        for traj in trajectories:
            processed = self._process_trajectory(traj)
            if processed is not None:
                self.processed_trajectories.append(processed)

    def _process_trajectory(self, trajectory):
        """Process a single trajectory."""
        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]
        dones = trajectory["dones"]

        if len(states) == 0:
            return None

        returns_to_go = self._compute_returns_to_go(rewards, dones)

        seq_len = min(len(states), self.max_length)

        padded_states = np.zeros((self.max_length, states[0].shape[0]))
        padded_actions = np.zeros((self.max_length, actions[0].shape[0]))
        padded_returns = np.zeros((self.max_length, 1))
        padded_timesteps = np.arange(self.max_length)

        padded_states[:seq_len] = np.array(states[:seq_len])
        padded_actions[:seq_len] = np.array(actions[:seq_len])
        padded_returns[:seq_len] = returns_to_go[:seq_len]

        attention_mask = np.zeros(self.max_length * 3)  # 3 tokens per timestep
        attention_mask[: seq_len * 3] = 1

        return {
            "states": padded_states,
            "actions": padded_actions,
            "returns_to_go": padded_returns,
            "timesteps": padded_timesteps,
            "attention_mask": attention_mask,
            "seq_len": seq_len,
        }

    def _compute_returns_to_go(self, rewards, dones):
        """Compute returns-to-go for a trajectory."""
        returns = []
        running_return = 0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_return = 0
            running_return = reward + self.discount_factor * running_return
            returns.append(running_return)

        returns.reverse()
        return np.array(returns).reshape(-1, 1)

    def __len__(self):
        return len(self.processed_trajectories)

    def __getitem__(self, idx):
        return self.processed_trajectories[idx]


class MultiTaskTrajectoryDataset(Dataset):
    """Dataset for multi-task trajectory data."""

    def __init__(self, task_trajectories, max_length=1024, discount_factor=0.99):
        self.task_trajectories = task_trajectories  # Dict[task_id: trajectories]
        self.max_length = max_length
        self.discount_factor = discount_factor

        self.processed_trajectories = []
        for task_id, trajectories in task_trajectories.items():
            for traj in trajectories:
                processed = self._process_trajectory(traj, task_id)
                if processed is not None:
                    self.processed_trajectories.append(processed)

    def _process_trajectory(self, trajectory, task_id):
        """Process trajectory with task label."""
        processed = TrajectoryDataset(
            [trajectory], self.max_length, self.discount_factor
        )[0]
        if processed is not None:
            processed["task_id"] = task_id
        return processed

    def __len__(self):
        return len(self.processed_trajectories)

    def __getitem__(self, idx):
        return self.processed_trajectories[idx]


class FoundationModelEvaluator:
    """Evaluation utilities for foundation models."""

    def __init__(self, model, env_fn, num_eval_episodes=10):
        self.model = model
        self.env_fn = env_fn
        self.num_eval_episodes = num_eval_episodes

    def evaluate(self, context_trajectories=None, desired_return=None):
        """Evaluate model performance."""
        returns = []
        episode_lengths = []

        for _ in range(self.num_eval_episodes):
            env = self.env_fn()
            state, _ = env.reset()
            done = False
            episode_return = 0
            episode_length = 0

            if context_trajectories is not None:
                context_learner = InContextLearningRL(self.model)
                for traj in context_trajectories:
                    for s, a, r, ns, d in zip(
                        traj["states"],
                        traj["actions"],
                        traj["rewards"],
                        traj["next_states"],
                        traj["dones"],
                    ):
                        context_learner.add_context(s, a, r, ns, d)

            while not done and episode_length < 1000:
                if context_trajectories is not None:
                    action = context_learner.get_action(state, desired_return or 0.0)
                else:
                    action = self.model.get_action_from_state(state)

                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_return += reward
                episode_length += 1
                state = next_state

            returns.append(episode_return)
            episode_lengths.append(episode_length)

        return {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "returns": returns,
            "lengths": episode_lengths,
        }


class FoundationModelTrainer:
    """Enhanced training framework with evaluation and logging."""

    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-2, eval_freq=100):
        self.model = model
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )

        self.eval_freq = eval_freq
        self.training_stats = defaultdict(list)
        self.best_model_state = None
        self.best_eval_score = -float("inf")

    def train(self, train_loader, val_loader=None, num_epochs=100, eval_env_fn=None):
        """Full training loop with evaluation."""
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(train_loader)
            self.training_stats["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.training_stats["val_loss"].append(val_loss)

            if eval_env_fn is not None and epoch % self.eval_freq == 0:
                evaluator = FoundationModelEvaluator(self.model, eval_env_fn)
                eval_results = evaluator.evaluate()
                self.training_stats["eval_returns"].append(eval_results["mean_return"])

                if eval_results["mean_return"] > self.best_eval_score:
                    self.best_eval_score = eval_results["mean_return"]
                    self.best_model_state = self.model.state_dict().copy()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
                if val_loader is not None:
                    print(f"  Val Loss = {val_loss:.4f}")
                if eval_env_fn is not None and epoch % self.eval_freq == 0:
                    print(f"  Eval Return = {eval_results['mean_return']:.2f}")

    def _train_epoch(self, train_loader):
        """Single training epoch."""
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            loss = self._train_step(batch)
            total_loss += loss

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader):
        """Single validation epoch."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_loss(batch)
                total_loss += loss

        return total_loss / len(val_loader)

    def _train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        loss = self._compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def _compute_loss(self, batch):
        """Compute loss for a batch."""
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        returns_to_go = batch["returns_to_go"].to(device)
        timesteps = batch["timesteps"].to(device)

        outputs = self.model(states, actions, returns_to_go, timesteps)

        action_loss = nn.MSELoss()(outputs["action_preds"], actions)
        value_loss = nn.MSELoss()(outputs["value_preds"], returns_to_go)
        return_loss = nn.MSELoss()(outputs["return_preds"], returns_to_go)

        total_loss = action_loss + 0.5 * value_loss + 0.1 * return_loss

        return total_loss

    def save_model(self, path):
        """Save model and training stats."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "training_stats": dict(self.training_stats),
                "best_model_state": self.best_model_state,
                "best_eval_score": self.best_eval_score,
            },
            path,
        )

    def load_model(self, path):
        """Load model and training stats."""
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_stats = defaultdict(list, checkpoint["training_stats"])
        self.best_model_state = checkpoint["best_model_state"]
        self.best_eval_score = checkpoint["best_eval_score"]


def create_trajectory_dataset_from_env(env_fn, num_trajectories=1000, max_steps=1000):
    """Create trajectory dataset by collecting data from environment."""
    trajectories = []

    for _ in range(num_trajectories):
        env = env_fn()
        state, _ = env.reset()
        done = False

        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "next_states": [],
        }

        steps = 0
        while not done and steps < max_steps:
            action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)
            trajectory["next_states"].append(next_state)

            state = next_state
            steps += 1

        trajectories.append(trajectory)

    return TrajectoryDataset(trajectories)


def plot_training_curves(stats, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(stats["train_loss"], label="Train Loss")
    if "val_loss" in stats:
        axes[0, 0].plot(stats["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    if "eval_returns" in stats:
        axes[0, 1].plot(stats["eval_returns"])
        axes[0, 1].set_title("Evaluation Returns")
        axes[0, 1].set_xlabel("Evaluation Step")
        axes[0, 1].set_ylabel("Mean Return")

    if "action_losses" in stats and len(stats["action_losses"]) > 0:
        axes[1, 0].plot(stats["action_losses"], label="Action Loss")
        if "value_losses" in stats:
            axes[1, 0].plot(stats["value_losses"], label="Value Loss")
        if "return_losses" in stats:
            axes[1, 0].plot(stats["return_losses"], label="Return Loss")
        axes[1, 0].set_title("Loss Components")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
