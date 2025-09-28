import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class TD0Agent:
    """
    TD(0) agent for policy evaluation
    Learns state values V(s) for a given policy
    """

    def __init__(self, env, policy, alpha=0.1, gamma=0.9):
        self.env = env
        self.policy = policy
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

        self.V = defaultdict(float)

        self.episode_rewards = []
        self.value_history = []

    def get_action(self, state):
        """Get action from policy"""
        if hasattr(self.policy, "get_action"):
            return self.policy.get_action(state)
        else:
            valid_actions = self.env.get_valid_actions(state)
            return np.random.choice(valid_actions) if valid_actions else None

    def td_update(self, state, reward, next_state, done):
        """
        Perform TD(0) update
        V(s) ← V(s) + α[R + γV(s') - V(s)]
        """
        if done:
            td_target = reward  # No next state value for terminal states
        else:
            td_target = reward + self.gamma * self.V[next_state]

        td_error = td_target - self.V[state]
        self.V[state] += self.alpha * td_error

        return td_error

    def run_episode(self, max_steps=100):
        """Run one episode and learn"""
        state = self.env.reset()
        episode_reward = 0
        steps = 0

        while steps < max_steps:
            action = self.get_action(state)
            if action is None:
                break

            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            td_error = self.td_update(state, reward, next_state, done)

            state = next_state
            steps += 1

            if done:
                break

        return episode_reward, steps

    def train(self, num_episodes=1000, print_every=100):
        """Train the agent over multiple episodes"""
        print(f"Training TD(0) agent for {num_episodes} episodes...")
        print(f"Learning rate α = {self.alpha}, Discount factor γ = {self.gamma}")

        for episode in range(num_episodes):
            episode_reward, steps = self.run_episode()
            self.episode_rewards.append(episode_reward)

            if episode % 10 == 0:
                self.value_history.append(dict(self.V))

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                print(f"Episode {episode + 1}: Average reward = {avg_reward:.2f}")

        print("Training completed!")
        return self.V

    def get_value_function(self):
        """Get current value function as dictionary"""
        return dict(self.V)


class QLearningAgent:
    """
    Q-Learning agent for finding optimal policy
    Learns Q*(s,a) through off-policy temporal difference learning
    """

    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.Q = defaultdict(lambda: defaultdict(float))

        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        self.q_value_history = []

    def get_action(self, state, explore=True):
        """
        Get action using ε-greedy policy
        """
        if not explore:
            return self.get_greedy_action(state)

        if np.random.random() < self.epsilon:
            valid_actions = self.env.get_valid_actions(state)
            return np.random.choice(valid_actions) if valid_actions else None
        else:
            return self.get_greedy_action(state)

    def get_greedy_action(self, state):
        """Get greedy action (highest Q-value)"""
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return None

        q_values = {action: self.Q[state][action] for action in valid_actions}
        max_q = max(q_values.values())

        best_actions = [action for action, q in q_values.items() if q == max_q]
        return np.random.choice(best_actions)

    def update_q(self, state, action, reward, next_state, done):
        """
        Q-Learning update:
        Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.Q[state][action]

        if done:
            td_target = reward
        else:
            valid_next_actions = self.env.get_valid_actions(next_state)
            if valid_next_actions:
                max_next_q = max([self.Q[next_state][a] for a in valid_next_actions])
            else:
                max_next_q = 0.0
            td_target = reward + self.gamma * max_next_q

        td_error = td_target - current_q
        self.Q[state][action] += self.alpha * td_error

        return td_error

    def run_episode(self, max_steps=200):
        """Run one episode and learn"""
        state = self.env.reset()
        episode_reward = 0
        steps = 0

        while steps < max_steps:
            action = self.get_action(state, explore=True)
            if action is None:
                break

            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            td_error = self.update_q(state, action, reward, next_state, done)

            state = next_state
            steps += 1

            if done:
                break

        return episode_reward, steps

    def train(self, num_episodes=1000, print_every=100):
        """Train the Q-learning agent"""
        print(f"Training Q-Learning agent for {num_episodes} episodes...")
        print(f"Parameters: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")

        for episode in range(num_episodes):
            episode_reward, steps = self.run_episode()

            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)
            self.epsilon_history.append(self.epsilon)

            if episode % 50 == 0:
                q_snapshot = {}
                for state in self.env.states:
                    q_snapshot[state] = dict(self.Q[state])
                self.q_value_history.append(q_snapshot)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                avg_steps = np.mean(self.episode_steps[-print_every:])
                print(
                    f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                    f"Avg Steps = {avg_steps:.1f}, ε = {self.epsilon:.3f}"
                )

        print("Q-Learning training completed!")

    def get_value_function(self):
        """Extract value function V*(s) = max_a Q*(s,a)"""
        V = {}
        for state in self.env.states:
            valid_actions = self.env.get_valid_actions(state)
            if valid_actions:
                V[state] = max([self.Q[state][action] for action in valid_actions])
            else:
                V[state] = 0.0
        return V

    def get_policy(self):
        """Extract optimal policy π*(s) = argmax_a Q*(s,a)"""
        policy = {}
        for state in self.env.states:
            if not self.env.is_terminal(state):
                policy[state] = self.get_greedy_action(state)
        return policy

    def evaluate_policy(self, num_episodes=100):
        """Evaluate learned policy (no exploration)"""
        rewards = []
        steps_list = []

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0

            while steps < 200:
                action = self.get_action(state, explore=False)  # No exploration
                if action is None:
                    break

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            rewards.append(episode_reward)
            steps_list.append(steps)

        return {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_steps": np.mean(steps_list),
            "success_rate": sum(1 for r in rewards if r > 5) / len(rewards),
        }


class SARSAAgent:
    """
    SARSA agent for on-policy control
    Learns Q^π(s,a) for the policy being followed
    """

    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.Q = defaultdict(lambda: defaultdict(float))

        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []

    def get_action(self, state, explore=True):
        """Get action using ε-greedy policy"""
        if not explore:
            return self.get_greedy_action(state)

        if np.random.random() < self.epsilon:
            valid_actions = self.env.get_valid_actions(state)
            return np.random.choice(valid_actions) if valid_actions else None
        else:
            return self.get_greedy_action(state)

    def get_greedy_action(self, state):
        """Get greedy action"""
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return None

        q_values = {action: self.Q[state][action] for action in valid_actions}
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return np.random.choice(best_actions)

    def update_q_sarsa(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
        """
        current_q = self.Q[state][action]

        if done:
            td_target = reward
        else:
            next_q = self.Q[next_state][next_action] if next_action else 0.0
            td_target = reward + self.gamma * next_q

        td_error = td_target - current_q
        self.Q[state][action] += self.alpha * td_error

        return td_error

    def run_episode(self, max_steps=200):
        """Run one episode using SARSA"""
        state = self.env.reset()
        action = self.get_action(state, explore=True)

        episode_reward = 0
        steps = 0

        while steps < max_steps and action is not None:
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            if done:
                next_action = None
            else:
                next_action = self.get_action(next_state, explore=True)

            td_error = self.update_q_sarsa(
                state, action, reward, next_state, next_action, done
            )

            state = next_state
            action = next_action
            steps += 1

            if done:
                break

        return episode_reward, steps

    def train(self, num_episodes=1000, print_every=100):
        """Train SARSA agent"""
        print(f"Training SARSA agent for {num_episodes} episodes...")
        print(f"Parameters: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")

        for episode in range(num_episodes):
            episode_reward, steps = self.run_episode()

            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)
            self.epsilon_history.append(self.epsilon)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                avg_steps = np.mean(self.episode_steps[-print_every:])
                print(
                    f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                    f"Avg Steps = {avg_steps:.1f}, ε = {self.epsilon:.3f}"
                )

        print("SARSA training completed!")

    def get_value_function(self):
        """Extract value function"""
        V = {}
        for state in self.env.states:
            valid_actions = self.env.get_valid_actions(state)
            if valid_actions:
                V[state] = max([self.Q[state][action] for action in valid_actions])
            else:
                V[state] = 0.0
        return V

    def get_policy(self):
        """Extract learned policy"""
        policy = {}
        for state in self.env.states:
            if not self.env.is_terminal(state):
                policy[state] = self.get_greedy_action(state)
        return policy

    def evaluate_policy(self, num_episodes=100):
        """Evaluate learned policy"""
        rewards = []
        steps_list = []

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0

            while steps < 200:
                action = self.get_action(state, explore=False)
                if action is None:
                    break

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            rewards.append(episode_reward)
            steps_list.append(steps)

        return {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_steps": np.mean(steps_list),
            "success_rate": sum(1 for r in rewards if r > 5) / len(rewards),
        }
