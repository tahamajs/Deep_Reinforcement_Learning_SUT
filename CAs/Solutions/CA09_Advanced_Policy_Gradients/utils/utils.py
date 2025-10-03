import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import deque, namedtuple
import random
import time
import math
from typing import List, Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def test_environment_setup():
    """Test basic environment functionality"""
    try:
        env = gym.make("CartPole-v1")
        state, _ = env.reset()
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        env.close()
        print(f"Environment setup successful!")
        print(f"  State shape: {state.shape}")
        print(f"  Action space: {env.action_space}")
        print(f"  Sample reward: {reward}")
    except Exception as e:
        print(f"Environment setup failed: {e}")
