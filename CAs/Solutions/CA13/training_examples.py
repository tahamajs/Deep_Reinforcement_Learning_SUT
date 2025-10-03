"""
Computer Assignment 13: Sample-Efficient Deep RL
Training Examples and Utilities

This file contains training loops and utility functions for:
- Model-free agents (DQN, PPO)
- Model-based agents with learned dynamics
- Sample-efficient methods (Rainbow, SAC)
- Hierarchical RL approaches

Author: DRL Course Team
Institution: Sharif University of Technology
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import time

# Import from local modules
from agents.model_free import ModelFreeAgent, DQNAgent
from agents.model_based import ModelBasedAgent
from buffers.replay_buffer import ReplayBuffer
from utils import get_device


@dataclass
class EpisodeMetrics:
    """Container for per-episode statistics."""
    episode: int
    return_: float
    length: int
    elapsed_sec: float
    mean_loss: Optional[float] = None
    success: Optional[bool] = None
    exploration_rate: Optional[float] = None
    notes: Dict[str, Any] = field(default_factory=dict)


def env_reset(env):
    """Wrapper for gym/gymnasium reset API."""
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
    else:
        observation, info = result, {}
    return observation, info


def env_step(env, action):
    """Wrapper around env.step supporting both Gym and Gymnasium."""
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = terminated or truncated
    elif isinstance(result, tuple) and len(result) == 4:
        observation, reward, done, info = result
    else:
        raise ValueError("Unexpected env.step return format")
    return observation, reward, done, info


def train_dqn_agent(
    env,
    agent: DQNAgent,
    num_episodes: int = 500,
    max_steps: int = 1000,
    eval_interval: int = 50,
):
    """Train DQN agent with detailed logging."""
    episode_rewards = []
    episode_lengths = []
    episode_logs = []
    losses = []

    for episode in range(1, num_episodes + 1):
        state, reset_info = env_reset(env)
        done = False
        ep_reward = 0.0
        ep_length = 0
        ep_losses = []
        start_time = time.time()

        while not done and ep_length < max_steps:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env_step(env, action)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            if loss is not None:
                ep_losses.append(loss)
                losses.append(loss)

            state = next_state
            ep_reward += reward
            ep_length += 1

        elapsed = time.time() - start_time
        
        metrics = EpisodeMetrics(
            episode=episode,
            return_=ep_reward,
            length=ep_length,
            elapsed_sec=elapsed,
            mean_loss=float(np.mean(ep_losses)) if ep_losses else None,
            exploration_rate=agent.epsilon if hasattr(agent, 'epsilon') else None,
            notes={"reset_info": reset_info},
        )
        episode_logs.append(asdict(metrics))
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        if episode % eval_interval == 0:
            mean_return = np.mean(episode_rewards[-eval_interval:])
            print(f"Episode {episode:04d} | Avg Return = {mean_return:.2f} | Length = {ep_length}")

    results = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "losses": losses,
        "episode_logs": episode_logs,
    }

    if episode_logs:
        results["episode_dataframe"] = pd.DataFrame(episode_logs)

    return results


def train_model_based_agent(
    env,
    agent: ModelBasedAgent,
    num_episodes: int = 500,
    max_steps: int = 1000,
    eval_interval: int = 50,
    planning_steps: int = 10,
):
    """Train model-based agent with planning."""
    episode_rewards = []
    episode_lengths = []
    episode_logs = []
    model_losses = []
    q_losses = []

    for episode in range(1, num_episodes + 1):
        state, reset_info = env_reset(env)
        done = False
        ep_reward = 0.0
        ep_length = 0
        ep_model_losses = []
        ep_q_losses = []
        start_time = time.time()

        while not done and ep_length < max_steps:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env_step(env, action)

            agent.store_transition(state, action, reward, next_state, done)
            
            # Update model and Q-function
            model_loss = agent.train_model()
            q_loss = agent.train_q_function()
            
            if model_loss is not None:
                ep_model_losses.append(model_loss)
                model_losses.append(model_loss)
            if q_loss is not None:
                ep_q_losses.append(q_loss)
                q_losses.append(q_loss)
            
            # Planning step
            if hasattr(agent, 'planning_step'):
                agent.planning_step(num_steps=planning_steps)

            state = next_state
            ep_reward += reward
            ep_length += 1

        elapsed = time.time() - start_time
        
        metrics = EpisodeMetrics(
            episode=episode,
            return_=ep_reward,
            length=ep_length,
            elapsed_sec=elapsed,
            mean_loss=float(np.mean(ep_q_losses)) if ep_q_losses else None,
            notes={
                "reset_info": reset_info,
                "model_loss": float(np.mean(ep_model_losses)) if ep_model_losses else None,
            },
        )
        episode_logs.append(asdict(metrics))
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        if episode % eval_interval == 0:
            mean_return = np.mean(episode_rewards[-eval_interval:])
            print(f"Episode {episode:04d} | Avg Return = {mean_return:.2f} | Model Loss = {metrics.notes.get('model_loss', 0):.4f}")

    results = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "q_losses": q_losses,
        "model_losses": model_losses,
        "episode_logs": episode_logs,
    }

    if episode_logs:
        results["episode_dataframe"] = pd.DataFrame(episode_logs)

    return results


def evaluate_agent(env, agent, num_episodes: int = 10, max_steps: int = 1000):
    """Evaluate agent performance."""
    returns = []
    lengths = []
    
    for _ in range(num_episodes):
        state, _ = env_reset(env)
        done = False
        ep_return = 0.0
        ep_length = 0
        
        while not done and ep_length < max_steps:
            action = agent.select_action(state, training=False)
            state, reward, done, _ = env_step(env, action)
            ep_return += reward
            ep_length += 1
        
        returns.append(ep_return)
        lengths.append(ep_length)
    
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
    }
