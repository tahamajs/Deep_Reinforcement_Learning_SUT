import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MultivariateNormal, kl_divergence
import torch.multiprocessing as mp
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, deque, namedtuple
import random
import pickle
import json
import copy
import time
import threading
from typing import Tuple, List, Dict, Optional, Union, NamedTuple, Any
import warnings
from dataclasses import dataclass, field
import math
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools

warnings.filterwarnings("ignore")

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from scipy.optimize import minimize, linprog
from scipy.special import softmax
import cvxpy as cp

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

print(f"🤖 Multi-Agent Reinforcement Learning Environment Setup")
print(f"Device: {device}")
print(f"Available GPUs: {n_gpus}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11

agent_colors = sns.color_palette("Set2", 8)
performance_colors = sns.color_palette("viridis", 6)
sns.set_palette(agent_colors)


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent systems."""

    n_agents: int = 2
    state_dim: int = 10
    action_dim: int = 4
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 100000
    update_freq: int = 10
    communication: bool = False
    message_dim: int = 32
    coordination_mechanism: str = "centralized"  # centralized, decentralized, mixed


@dataclass
class PolicyConfig:
    """Configuration for advanced policy methods."""

    algorithm: str = "PPO"  # PPO, TRPO, SAC, DDPG, TD3
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    minibatch_size: int = 64
    use_gae: bool = True
    gae_lambda: float = 0.95


ma_config = MultiAgentConfig()
policy_config = PolicyConfig()

print("✅ Multi-Agent RL environment setup complete!")
print(
    f"🎯 Configuration: {ma_config.n_agents} agents, {ma_config.coordination_mechanism} coordination"
)
print("🚀 Ready for advanced multi-agent reinforcement learning!")
