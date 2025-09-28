"""
Causal Reinforcement Learning Agent
Implements agents that leverage causal structure for improved learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
from causal_discovery import CausalGraph
from causal_rl_utils import device


class CausalReasoningNetwork(nn.Module):
    """
    Neural network that performs causal reasoning
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        causal_graph: CausalGraph,
        hidden_dim: int = 128,
    ):
        super(CausalReasoningNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.causal_graph = causal_graph
        self.hidden_dim = hidden_dim

        # Feature extraction for each variable
        self.variable_encoders = nn.ModuleDict()
        for var in causal_graph.variables:
            self.variable_encoders[var] = nn.Sequential(
                nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )

        # Causal reasoning layers
        self.causal_reasoning = nn.Sequential(
            nn.Linear(hidden_dim * len(causal_graph.variables), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal reasoning

        Args:
            state: State tensor

        Returns:
            Tuple of (policy_logits, value_estimate)
        """
        # Encode each variable
        variable_features = []
        for i, var in enumerate(self.causal_graph.variables):
            var_value = (
                state[:, i : i + 1] if len(state.shape) > 1 else state[i : i + 1]
            )
            var_feature = self.variable_encoders[var](var_value)
            variable_features.append(var_feature)

        # Concatenate features
        combined_features = torch.cat(variable_features, dim=-1)

        # Causal reasoning
        reasoned_features = self.causal_reasoning(combined_features)

        # Policy and value outputs
        policy_logits = self.policy_head(reasoned_features)
        value = self.value_head(reasoned_features)

        return policy_logits, value

    def intervene(
        self, state: torch.Tensor, intervention: Dict[str, float]
    ) -> torch.Tensor:
        """
        Perform causal intervention

        Args:
            state: Current state
            intervention: Dictionary of variable -> intervention_value

        Returns:
            Intervened state
        """
        intervened_state = state.clone()

        for var, value in intervention.items():
            if var in self.causal_graph.var_to_idx:
                idx = self.causal_graph.var_to_idx[var]
                intervened_state[0, idx] = value

        return intervened_state


class CausalRLAgent:
    """
    Reinforcement Learning agent that uses causal reasoning
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        causal_graph: CausalGraph,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.causal_graph = causal_graph

        # Create causal reasoning network
        self.network = CausalReasoningNetwork(
            state_dim, action_dim, causal_graph, hidden_dim
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Logging
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.causal_interventions = []

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action using causal reasoning

        Args:
            state: Current state
            deterministic: Whether to select deterministically

        Returns:
            Tuple of (action, log_prob)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = self.network(state_tensor)
            probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=1).item()
            log_prob = dist.log_prob(torch.tensor(action))
        else:
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action))

        return action, log_prob

    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
    ) -> Dict[str, float]:
        """
        Update the agent using causal reasoning

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags

        Returns:
            Dictionary of loss values
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        dones_tensor = torch.FloatTensor(dones).to(device)

        # Get current policy and values
        logits, values = self.network(states_tensor)

        # Compute advantages using causal reasoning
        advantages = self._compute_causal_advantages(
            states_tensor, rewards_tensor, next_states_tensor, dones_tensor, values
        )

        # Policy loss
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        returns = advantages + values.squeeze().detach()
        value_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Logging
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def _compute_causal_advantages(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantages using causal structure

        Args:
            states: State tensors
            rewards: Reward tensors
            next_states: Next state tensors
            dones: Done flags
            values: Value estimates

        Returns:
            Advantage estimates
        """
        # Bootstrap next values
        with torch.no_grad():
            _, next_values = self.network(next_states)
            next_values = next_values.squeeze()

        # Compute TD targets
        targets = rewards + self.gamma * next_values * (1 - dones)

        # Advantages
        advantages = targets - values.squeeze()

        return advantages

    def perform_intervention(
        self, state: np.ndarray, intervention: Dict[str, float]
    ) -> np.ndarray:
        """
        Perform causal intervention on state

        Args:
            state: Current state
            intervention: Intervention dictionary

        Returns:
            Intervened state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        intervened_state = self.network.intervene(state_tensor, intervention)
        self.causal_interventions.append(intervention)

        return intervened_state.squeeze(0).cpu().numpy()

    def counterfactual_reasoning(
        self,
        trajectory: List[Tuple[np.ndarray, int, float, np.ndarray, bool]],
        intervention: Dict[str, float],
    ) -> List[float]:
        """
        Perform counterfactual reasoning on a trajectory

        Args:
            trajectory: List of (state, action, reward, next_state, done) tuples
            intervention: Intervention to apply

        Returns:
            Counterfactual rewards
        """
        counterfactual_rewards = []

        for state, action, reward, next_state, done in trajectory:
            # Intervene on state
            intervened_state = self.perform_intervention(state, intervention)

            # Get action under intervention
            intervened_action, _ = self.select_action(
                intervened_state, deterministic=True
            )

            # Simulate reward (simplified - would need environment model)
            counterfactual_reward = reward * 0.8  # Placeholder
            counterfactual_rewards.append(counterfactual_reward)

        return counterfactual_rewards

    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        """
        Train for one episode

        Args:
            env: Environment
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (episode_reward, steps)
        """
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        states, actions, rewards, next_states, dones = [], [], [], [], []

        while steps < max_steps:
            action, _ = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            episode_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        # Update agent
        if len(states) > 0:
            self.update(states, actions, rewards, next_states, dones)

        self.episode_rewards.append(episode_reward)
        return episode_reward, steps


class CounterfactualRLAgent(CausalRLAgent):
    """
    Agent that uses counterfactual reasoning for improved learning
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counterfactual_experiences = []

    def update_with_counterfactuals(
        self, trajectory: List[Tuple], interventions: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Update using counterfactual reasoning

        Args:
            trajectory: Original trajectory
            interventions: List of interventions to consider

        Returns:
            Loss dictionary
        """
        total_loss = 0
        counterfactual_losses = []

        for intervention in interventions:
            # Generate counterfactual trajectory
            cf_rewards = self.counterfactual_reasoning(trajectory, intervention)

            # Compute counterfactual loss
            cf_loss = torch.tensor(cf_rewards).mean()
            counterfactual_losses.append(cf_loss.item())
            total_loss += cf_loss

        # Combine with standard update
        standard_losses = super().update(
            [t[0] for t in trajectory],
            [t[1] for t in trajectory],
            [t[2] for t in trajectory],
            [t[3] for t in trajectory],
            [t[4] for t in trajectory],
        )

        return {
            **standard_losses,
            "counterfactual_loss": total_loss.item(),
            "avg_counterfactual_loss": np.mean(counterfactual_losses),
        }
