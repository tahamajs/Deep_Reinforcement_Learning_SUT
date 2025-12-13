"""
agents.py - Implementation of Temporal Difference Learning Agents.

This module contains the core implementations of TD(0), Q-Learning, and SARSA agents.
It also includes basic policy classes such as RandomPolicy for exploration and
greedy policy extraction.
"""

from typing import Dict, Tuple, List, Any
import numpy as np
from collections import defaultdict
import random

# Assuming GridWorld and config are available in the src package
from .environments import GridWorld
from .config import AgentConfig, ExplorationConfig, SEED

class BasePolicy:
    """Base class for all policies."""
    def __init__(self, env: GridWorld):
        """
        Initializes the BasePolicy with the given environment.

        Args:
            env (GridWorld): The environment instance.
        """
        self.env = env

    def get_action(self, state: Tuple[int, int], q_values: Dict = None) -> str:
        """
        Abstract method to get an action for a given state.
        Must be implemented by subclasses.

        Args:
            state (Tuple[int, int]): The current state.
            q_values (Dict, optional): Q-values if needed for action selection. Defaults to None.

        Returns:
            str: The chosen action.
        """
        raise NotImplementedError

class RandomPolicy(BasePolicy):
    """
    A policy that selects actions uniformly at random from the valid actions.
    """
    def get_action(self, state: Tuple[int, int], q_values: Dict = None) -> str:
        """
        Selects a random action from the environment's valid actions.

        Args:
            state (Tuple[int, int]): The current state.
            q_values (Dict, optional): Not used for RandomPolicy.

        Returns:
            str: A randomly chosen action.
        """
        return np.random.choice(self.env.get_valid_actions(state))

class GreedyPolicy(BasePolicy):
    """
    A policy that selects the action with the maximum Q-value.
    Used for evaluation or when no exploration is desired.
    """
    def get_action(self, state: Tuple[int, int], q_values: Dict[Tuple[int, int], Dict[str, float]]) -> str:
        """
        Selects the greedy action based on the provided Q-values.

        Args:
            state (Tuple[int, int]): The current state.
            q_values (Dict): The Q-value table.

        Returns:
            str: The greedy action.
        """
        if state not in q_values or not q_values[state]:
            return np.random.choice(self.env.get_valid_actions(state)) # Fallback to random if no Q-values

        state_q_values = q_values[state]
        max_q = -float('inf')
        best_action = random.choice(self.env.get_valid_actions(state)) # Initialize with a random valid action

        for action, q_value in state_q_values.items():
            if action in self.env.get_valid_actions(state): # Ensure action is valid for the state
                if q_value > max_q:
                    max_q = q_value
                    best_action = action
                elif q_value == max_q:
                    # Break ties randomly
                    if random.random() < 0.5:
                        best_action = action
        return best_action


class BaseAgent:
    """Base class for all TD learning agents."""

    def __init__(
        self,
        env: GridWorld,
        alpha: float = AgentConfig.ALPHA,
        gamma: float = AgentConfig.GAMMA,
    ):
        """
        Initializes the BaseAgent.

        Args:
            env (GridWorld): The environment instance.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.episode_rewards = []
        self.episode_steps = []
        np.random.seed(SEED)
        random.seed(SEED)

    def get_action(self, state: Tuple[int, int], explore: bool = True) -> str:
        """
        Abstract method to get an action for a given state.
        """
        raise NotImplementedError

    def train(self, num_episodes: int = AgentConfig.NUM_EPISODES, print_every: int = AgentConfig.PRINT_EVERY) -> Any:
        """
        Abstract method for training the agent.
        """
        raise NotImplementedError

    def evaluate_policy(self, num_episodes: int = AgentConfig.NUM_EPISODES) -> Dict[str, Any]:
        """
        Evaluates the agent's current policy over a number of episodes.

        Args:
            num_episodes (int): Number of episodes to run for evaluation.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics like average reward,
                            standard deviation of reward, average steps, and success rate.
        """
        rewards_per_episode = []
        steps_per_episode = []
        successes = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action = self.get_action(state, explore=False)  # Use greedy policy for evaluation
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                state = next_state
                if steps > self.env.size * self.env.size * 2: # Prevent infinite loops in evaluation
                    break

            rewards_per_episode.append(total_reward)
            steps_per_episode.append(steps)
            if self.env.current_state == self.env.goal_state and total_reward > 0: # Ensure positive reward for success
                successes += 1

        avg_reward = np.mean(rewards_per_episode)
        std_reward = np.std(rewards_per_episode)
        avg_steps = np.mean(steps_per_episode)
        success_rate = successes / num_episodes

        return {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "rewards_per_episode": rewards_per_episode,
            "steps_per_episode": steps_per_episode,
        }


class TD0Agent(BaseAgent):
    """
    Implements the TD(0) algorithm for policy evaluation.
    Learns the state-value function V for a given policy.
    """

    def __init__(
        self,
        env: GridWorld,
        policy: BasePolicy,
        alpha: float = AgentConfig.ALPHA,
        gamma: float = AgentConfig.GAMMA,
    ):
        """
        Initializes the TD(0) agent.

        Args:
            env (GridWorld): The environment instance.
            policy (BasePolicy): The policy to evaluate.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
        """
        super().__init__(env, alpha, gamma)
        self.policy = policy
        self.V: Dict[Tuple[int, int], float] = defaultdict(float) # State-value function

    def td_update(
        self, state: Tuple[int, int], reward: float, next_state: Tuple[int, int], done: bool
    ) -> float:
        """
        Performs a single TD(0) update step for the state-value function.

        Args:
            state (Tuple[int, int]): The current state.
            reward (float): The reward received after taking an action from the current state.
            next_state (Tuple[int, int]): The next state.
            done (bool): True if the episode has ended, False otherwise.

        Returns:
            float: The TD error for the update.
        """
        old_value = self.V[state]
        td_target = reward + self.gamma * self.V[next_state] * (1 - int(done))
        td_error = td_target - old_value
        self.V[state] = old_value + self.alpha * td_error
        return td_error

    def train(self, num_episodes: int = AgentConfig.NUM_EPISODES, print_every: int = AgentConfig.PRINT_EVERY) -> Dict[Tuple[int, int], float]:
        """
        Trains the TD(0) agent for a specified number of episodes.

        Args:
            num_episodes (int): The number of episodes to train for.
            print_every (int): Frequency of printing training progress.

        Returns:
            Dict[Tuple[int, int], float]: The learned state-value function V.
        """
        print(f"Training TD(0) agent for {num_episodes} episodes...")
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                action = self.policy.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.td_update(state, reward, next_state, done)
                episode_reward += reward
                steps += 1
                state = next_state

            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Current V[{self.env.start_state}]: {self.V[self.env.start_state]:.2f}")
        print("TD(0) training complete.")
        return self.V

    def get_value_function(self) -> Dict[Tuple[int, int], float]:
        """
        Returns the learned state-value function V.
        """
        return self.V

    def get_policy(self) -> Dict[Tuple[int, int], str]:
        """
        For TD(0) which evaluates a given policy, this simply returns the evaluated policy.
        """
        # Since TD(0) evaluates a fixed policy, we can conceptually return that policy.
        # However, for control algorithms we would derive it from Q-values.
        # For consistency with other agents, we return a representation of the policy.
        policy = {}
        for state in self.env.states:
            if not self.env.is_terminal(state):
                policy[state] = self.policy.get_action(state) # The policy it was evaluating
        return policy

    def get_action(self, state: Tuple[int, int], explore: bool = True) -> str:
        """
        Returns an action based on the agent's policy.
        For TD(0), it uses the predefined policy.
        """
        return self.policy.get_action(state)


class QLearningAgent(BaseAgent):
    """
    Implements the Q-Learning algorithm for off-policy control.
    Learns the optimal action-value function Q*.
    """

    def __init__(
        self,
        env: GridWorld,
        alpha: float = AgentConfig.ALPHA,
        gamma: float = AgentConfig.GAMMA,
        epsilon: float = ExplorationConfig.EPSILON_START,
        epsilon_decay: float = ExplorationConfig.EPSILON_DECAY,
        epsilon_min: float = ExplorationConfig.EPSILON_MIN,
    ):
        """
        Initializes the Q-Learning agent.

        Args:
            env (GridWorld): The environment instance.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate for ε-greedy.
            epsilon_decay (float): Decay rate for epsilon.
            epsilon_min (float): Minimum value for epsilon.
        """
        super().__init__(env, alpha, gamma)
        self.Q: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.greedy_policy_obj = GreedyPolicy(env) # Helper for greedy action selection

    def get_action(self, state: Tuple[int, int], explore: bool = True) -> str:
        """
        Selects an action using an ε-greedy strategy or purely greedily.

        Args:
            state (Tuple[int, int]): The current state.
            explore (bool): If True, use ε-greedy; otherwise, use greedy policy.

        Returns:
            str: The chosen action.
        """
        if explore and random.random() < self.epsilon:
            return np.random.choice(self.env.get_valid_actions(state))
        else:
            return self.greedy_policy_obj.get_action(state, self.Q)

    def update_q(
        self,
        state: Tuple[int, int],
        action: str,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ) -> float:
        """
        Performs a single Q-Learning update step.

        Args:
            state (Tuple[int, int]): The current state.
            action (str): The action taken.
            reward (float): The reward received.
            next_state (Tuple[int, int]): The next state.
            done (bool): True if the episode has ended, False otherwise.

        Returns:
            float: The TD error for the update.
        """
        old_q_value = self.Q[state][action]
        
        # Q-Learning target uses the maximum Q-value for the next state (off-policy)
        max_next_q = 0.0
        if not done and next_state in self.Q:
            max_next_q = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0

        td_target = reward + self.gamma * max_next_q
        td_error = td_target - old_q_value
        self.Q[state][action] = old_q_value + self.alpha * td_error
        return td_error

    def train(self, num_episodes: int = AgentConfig.NUM_EPISODES, print_every: int = AgentConfig.PRINT_EVERY) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Trains the Q-Learning agent for a specified number of episodes.

        Args:
            num_episodes (int): The number of episodes to train for.
            print_every (int): Frequency of printing training progress.

        Returns:
            Dict[Tuple[int, int], Dict[str, float]]: The learned action-value function Q.
        """
        print(f"Training Q-Learning agent for {num_episodes} episodes...")
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                action = self.get_action(state, explore=True)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q(state, action, reward, next_state, done)
                episode_reward += reward
                steps += 1
                state = next_state
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {self.epsilon:.2f}")
        print("Q-Learning training complete.")
        return self.Q

    def get_value_function(self) -> Dict[Tuple[int, int], float]:
        """
        Extracts the state-value function V from the learned Q-values.
        V(s) = max_a Q(s,a)
        """
        V = defaultdict(float)
        for state, actions in self.Q.items():
            if actions:
                V[state] = max(actions.values())
            else:
                V[state] = 0.0
        return V

    def get_policy(self) -> Dict[Tuple[int, int], str]:
        """
        Extracts the optimal policy from the learned Q-values.
        """
        policy = {}
        for state in self.env.states:
            if not self.env.is_terminal(state) and state in self.Q:
                policy[state] = self.greedy_policy_obj.get_action(state, self.Q)
            elif not self.env.is_terminal(state): # Fallback for states not yet visited
                 policy[state] = np.random.choice(self.env.get_valid_actions(state))
        return policy


class SARSAAgent(BaseAgent):
    """
    Implements the SARSA algorithm for on-policy control.
    Learns the action-value function Q for the current policy.
    """

    def __init__(
        self,
        env: GridWorld,
        alpha: float = AgentConfig.ALPHA,
        gamma: float = AgentConfig.GAMMA,
        epsilon: float = ExplorationConfig.EPSILON_START,
        epsilon_decay: float = ExplorationConfig.EPSILON_DECAY,
        epsilon_min: float = ExplorationConfig.EPSILON_MIN,
    ):
        """
        Initializes the SARSA agent.

        Args:
            env (GridWorld): The environment instance.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate for ε-greedy.
            epsilon_decay (float): Decay rate for epsilon.
            epsilon_min (float): Minimum value for epsilon.
        """
        super().__init__(env, alpha, gamma)
        self.Q: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.greedy_policy_obj = GreedyPolicy(env) # Helper for greedy action selection

    def get_action(self, state: Tuple[int, int], explore: bool = True) -> str:
        """
        Selects an action using an ε-greedy strategy or purely greedily.

        Args:
            state (Tuple[int, int]): The current state.
            explore (bool): If True, use ε-greedy; otherwise, use greedy policy.

        Returns:
            str: The chosen action.
        """
        if explore and random.random() < self.epsilon:
            return np.random.choice(self.env.get_valid_actions(state))
        else:
            return self.greedy_policy_obj.get_action(state, self.Q)

    def update_q_sarsa(
        self,
        state: Tuple[int, int],
        action: str,
        reward: float,
        next_state: Tuple[int, int],
        next_action: str,
        done: bool,
    ) -> float:
        """
        Performs a single SARSA update step.

        Args:
            state (Tuple[int, int]): The current state.
            action (str): The action taken.
            reward (float): The reward received.
            next_state (Tuple[int, int]): The next state.
            next_action (str): The action chosen from the next state by the *current* policy.
            done (bool): True if the episode has ended, False otherwise.

        Returns:
            float: The TD error for the update.
        """
        old_q_value = self.Q[state][action]
        
        # SARSA target uses the Q-value for the next state-action pair chosen by the current policy (on-policy)
        next_q_value = self.Q[next_state][next_action] if not done else 0.0

        td_target = reward + self.gamma * next_q_value
        td_error = td_target - old_q_value
        self.Q[state][action] = old_q_value + self.alpha * td_error
        return td_error

    def train(self, num_episodes: int = AgentConfig.NUM_EPISODES, print_every: int = AgentConfig.PRINT_EVERY) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Trains the SARSA agent for a specified number of episodes.

        Args:
            num_episodes (int): The number of episodes to train for.
            print_every (int): Frequency of printing training progress.

        Returns:
            Dict[Tuple[int, int], Dict[str, float]]: The learned action-value function Q.
        """
        print(f"Training SARSA agent for {num_episodes} episodes...")
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.get_action(state, explore=True) # First action
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.get_action(next_state, explore=True) # Next action
                self.update_q_sarsa(state, action, reward, next_state, next_action, done)
                episode_reward += reward
                steps += 1
                state = next_state
                action = next_action # SARSA uses the *next* action to update

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {self.epsilon:.2f}")
        print("SARSA training complete.")
        return self.Q

    def get_value_function(self) -> Dict[Tuple[int, int], float]:
        """
        Extracts the state-value function V from the learned Q-values.
        V(s) = Q(s, a) for the current policy's chosen action.
        """
        V = defaultdict(float)
        for state in self.env.states:
            if not self.env.is_terminal(state) and state in self.Q:
                action = self.greedy_policy_obj.get_action(state, self.Q) # Use greedy for evaluation
                V[state] = self.Q[state][action] if action in self.Q[state] else 0.0
            else:
                V[state] = 0.0
        return V

    def get_policy(self) -> Dict[Tuple[int, int], str]:
        """
        Extracts the optimal policy from the learned Q-values.
        """
        policy = {}
        for state in self.env.states:
            if not self.env.is_terminal(state) and state in self.Q:
                policy[state] = self.greedy_policy_obj.get_action(state, self.Q)
            elif not self.env.is_terminal(state): # Fallback for states not yet visited
                 policy[state] = np.random.choice(self.env.get_valid_actions(state))
        return policy


if __name__ == "__main__":
    # Example Usage:
    env = GridWorld()
    print("\n--- TD(0) Agent Demo ---")
    random_policy = RandomPolicy(env)
    td_agent = TD0Agent(env, random_policy)
    V_td = td_agent.train(num_episodes=100)
    print(f"Learned V for start state: {V_td[env.start_state]:.2f}")
    env.visualize_values(V_td, title="TD(0) Learned Value Function (Random Policy)", filepath="visualizations/td0_v_values.png")

    print("\n--- Q-Learning Agent Demo ---")
    q_agent = QLearningAgent(env)
    q_agent.train(num_episodes=200)
    q_evaluation = q_agent.evaluate_policy(num_episodes=20)
    print(f"Q-Learning Average Reward: {q_evaluation['avg_reward']:.2f}")
    print(f"Q-Learning Success Rate: {q_evaluation['success_rate']*100:.1f}%")
    env.visualize_values(q_agent.get_value_function(), title="Q-Learning Optimal Value Function", policy=q_agent.get_policy(), filepath="visualizations/q_learning_v_values.png")

    print("\n--- SARSA Agent Demo ---")
    sarsa_agent = SARSAAgent(env)
    sarsa_agent.train(num_episodes=200)
    sarsa_evaluation = sarsa_agent.evaluate_policy(num_episodes=20)
    print(f"SARSA Average Reward: {sarsa_evaluation['avg_reward']:.2f}")
    print(f"SARSA Success Rate: {sarsa_evaluation['success_rate']*100:.1f}%")
    env.visualize_values(sarsa_agent.get_value_function(), title="SARSA Learned Value Function", policy=sarsa_agent.get_policy(), filepath="visualizations/sarsa_v_values.png")
