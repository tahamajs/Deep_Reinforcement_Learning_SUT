# -*- coding: utf-8 -*-
# Copyright (C) 2024 Prof. Ulf Nilsson
# This file is part of the DRL Course.

"""
CA5: Advanced Deep Q-Network (DQN) Agents Package

This package provides implementations of various advanced Deep Q-Network (DQN)
algorithms, built upon a common base agent structure.

Modules:
    dqn_base: Contains the base DQNAgent class and the Transition namedtuple.
    double_dqn: Implements the Double DQN algorithm.
    dueling_dqn: Implements the Dueling DQN algorithm.
    prioritized_replay_dqn: Implements DQN with Prioritized Experience Replay.
    rainbow_dqn: Implements the Rainbow DQN algorithm (combining multiple improvements).
"""

from .dqn_base import DQNAgent, Transition
from .double_dqn import DoubleDQNAgent
from .dueling_dqn import DuelingDQNAgent
from .prioritized_replay_dqn import PrioritizedDQNAgent
from .rainbow_dqn import RainbowDQNAgent

__all__ = [
    "DQNAgent",
    "Transition",  # Expose Transition namedtuple
    "DoubleDQNAgent",
    "DuelingDQNAgent",
    "PrioritizedDQNAgent",
    "RainbowDQNAgent",
]
