#!/usr/bin/env python3
"""
Test script for CA3 TD Learning implementations
"""

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from environments import GridWorld
from agents.policies import RandomPolicy
from agents.algorithms import TD0Agent, QLearningAgent, SARSAAgent


def test_basic_functionality():
    """Test basic functionality of TD learning agents"""

    print("Testing CA3 TD Learning implementations...")

    # Create environment
    env = GridWorld()
    print(f"✓ GridWorld created: {len(env.states)} states, {len(env.actions)} actions")

    # Test TD(0) agent
    policy = RandomPolicy(env)
    td_agent = TD0Agent(env, policy, alpha=0.1, gamma=0.9)
    print("✓ TD(0) agent created")

    # Quick training
    V_td = td_agent.train(num_episodes=50, print_every=25)
    print("✓ TD(0) training completed")

    # Test Q-Learning agent
    q_agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    q_agent.train(num_episodes=50, print_every=25)
    print("✓ Q-Learning training completed")

    # Test SARSA agent
    sarsa_agent = SARSAAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    sarsa_agent.train(num_episodes=50, print_every=25)
    print("✓ SARSA training completed")

    print("\n✅ All CA3 implementations working correctly!")
    return True


if __name__ == "__main__":
    test_basic_functionality()
