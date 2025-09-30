import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import networkx as nx
from itertools import combinations, permutations
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import random


class CausalGraph:
    """Represents a causal graph structure"""

    def __init__(self, variables: List[str]):
        self.variables = variables
        self.n_vars = len(variables)
        self.var_to_idx = {var: i for i, var in enumerate(variables)}

        self.adj_matrix = np.zeros((self.n_vars, self.n_vars), dtype=bool)

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(variables)

    def add_edge(self, from_var: str, to_var: str):
        """Add directed edge from_var -> to_var"""
        i = self.var_to_idx[from_var]
        j = self.var_to_idx[to_var]
        self.adj_matrix[i, j] = True
        self.graph.add_edge(from_var, to_var)

    def remove_edge(self, from_var: str, to_var: str):
        """Remove directed edge from_var -> to_var"""
        i = self.var_to_idx[from_var]
        j = self.var_to_idx[to_var]
        self.adj_matrix[i, j] = False
        if self.graph.has_edge(from_var, to_var):
            self.graph.remove_edge(from_var, to_var)

    def get_parents(self, var: str) -> List[str]:
        """Get parent variables of var"""
        j = self.var_to_idx[var]
        parent_indices = np.where(self.adj_matrix[:, j])[0]
        return [self.variables[i] for i in parent_indices]

    def get_children(self, var: str) -> List[str]:
        """Get children variables of var"""
        i = self.var_to_idx[var]
        child_indices = np.where(self.adj_matrix[i, :])[0]
        return [self.variables[j] for j in child_indices]

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if ancestor is an ancestor of descendant"""
        return nx.has_path(self.graph, ancestor, descendant)

    def get_markov_blanket(self, var: str) -> Set[str]:
        """Get Markov blanket of variable (parents, children, and co-parents)"""
        parents = set(self.get_parents(var))
        children = set(self.get_children(var))
        co_parents = set()

        for child in children:
            co_parents.update(self.get_parents(child))

        markov_blanket = parents | children | co_parents
        markov_blanket.discard(var)  # Remove the variable itself

        return markov_blanket

    def visualize(self, pos: Optional[Dict] = None, figsize: Tuple = (10, 8)):
        """Visualize the causal graph"""
        plt.figure(figsize=figsize)

        if pos is None:
            pos = nx.spring_layout(self.graph, k=2, iterations=50)

        nx.draw_networkx_nodes(
            self.graph, pos, node_color="lightblue", node_size=1500, alpha=0.8
        )
        nx.draw_networkx_edges(
            self.graph, pos, edge_color="gray", arrows=True, arrowsize=20, width=2
        )
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight="bold")

        plt.title("Causal Graph Structure", fontsize=16, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


class PCCausalDiscovery:
    """PC algorithm for causal discovery"""

    def __init__(self, alpha: float = 0.05, max_cond_set_size: int = 3):
        self.alpha = alpha  # Significance level for independence tests
        self.max_cond_set_size = max_cond_set_size

    def conditional_independence_test(
        self, X: np.ndarray, Y: np.ndarray, Z: Optional[np.ndarray] = None
    ) -> Tuple[bool, float]:
        """Test conditional independence X ⊥ Y | Z using partial correlation"""
        if Z is None or Z.shape[1] == 0:
            corr, p_value = stats.pearsonr(X, Y)
            return p_value > self.alpha, p_value

        n = len(X)
        k = Z.shape[1]

        design_X = np.column_stack([np.ones(n), Z])
        design_Y = np.column_stack([np.ones(n), Z])

        try:
            beta_X = np.linalg.lstsq(design_X, X, rcond=None)[0]
            beta_Y = np.linalg.lstsq(design_Y, Y, rcond=None)[0]

            residual_X = X - design_X @ beta_X
            residual_Y = Y - design_Y @ beta_Y

            if np.var(residual_X) > 1e-10 and np.var(residual_Y) > 1e-10:
                corr, p_value = stats.pearsonr(residual_X, residual_Y)
                return p_value > self.alpha, p_value
            else:
                return True, 1.0  # Perfect dependence through Z
        except:
            return True, 1.0  # Assume independence if test fails

    def discover_structure(self, data: np.ndarray, var_names: List[str]) -> CausalGraph:
        """Discover causal structure using PC algorithm"""
        n_vars = data.shape[1]

        graph = CausalGraph(var_names)
        adjacencies = set()

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                adjacencies.add((i, j))

        for cond_size in range(self.max_cond_set_size + 1):
            to_remove = set()

            for i, j in adjacencies:
                neighbors_i = {
                    k
                    for k, l in adjacencies
                    if (k == i and l != j) or (l == i and k != j)
                }
                neighbors_j = {
                    k
                    for k, l in adjacencies
                    if (k == j and l != i) or (l == j and k != i)
                }

                potential_cond = neighbors_i | neighbors_j
                potential_cond.discard(i)
                potential_cond.discard(j)

                if len(potential_cond) >= cond_size:
                    for cond_set in combinations(potential_cond, cond_size):
                        if len(cond_set) == cond_size:
                            X = data[:, i]
                            Y = data[:, j]
                            Z = data[:, list(cond_set)] if cond_set else None

                            is_independent, p_value = (
                                self.conditional_independence_test(X, Y, Z)
                            )

                            if is_independent:
                                to_remove.add((i, j))
                                break

            adjacencies -= to_remove

        for i, j in adjacencies:
            corr_i = np.mean(
                [
                    abs(np.corrcoef(data[:, i], data[:, k])[0, 1])
                    for k in range(n_vars)
                    if k != i and k != j
                ]
            )
            corr_j = np.mean(
                [
                    abs(np.corrcoef(data[:, j], data[:, k])[0, 1])
                    for k in range(n_vars)
                    if k != i and k != j
                ]
            )

            if corr_i > corr_j:
                graph.add_edge(var_names[i], var_names[j])
            else:
                graph.add_edge(var_names[j], var_names[i])

        return graph


class CausalMechanism(nn.Module):
    """Learn individual causal mechanisms P(X_j | pa(X_j))"""

    def __init__(self, n_parents: int, hidden_dim: int = 64):
        super().__init__()

        if n_parents == 0:
            self.mechanism = nn.Sequential(
                nn.Linear(1, hidden_dim),  # Input is just noise
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # Mean and log-std
            )
        else:
            self.mechanism = nn.Sequential(
                nn.Linear(n_parents, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # Mean and log-std
            )

        self.n_parents = n_parents

    def forward(
        self,
        parents: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from causal mechanism"""

        if self.n_parents == 0:
            if noise is None:
                noise = torch.randn(1, 1)
            params = self.mechanism(noise)
        else:
            if parents is None:
                raise ValueError("Parents required for non-root mechanism")
            params = self.mechanism(parents)

        mean, log_std = params.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-10, 10))

        if noise is None:
            noise = torch.randn_like(mean)

        return mean + std * noise


class CausalWorldModel(nn.Module):
    """World model that respects causal structure"""

    def __init__(
        self,
        causal_graph: CausalGraph,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.causal_graph = causal_graph
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_vars = [f"state_{i}" for i in range(state_dim)]
        self.action_vars = [f"action_{i}" for i in range(action_dim)]

        self.mechanisms = nn.ModuleDict()

        for var in self.state_vars:
            n_parents = state_dim + action_dim
            self.mechanisms[var] = CausalMechanism(n_parents, hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # Current and next state
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state using causal mechanisms"""
        batch_size = state.shape[0]
        next_state_components = []

        for i, var in enumerate(self.state_vars):
            parents = self.causal_graph.get_parents(var)

            if not parents:
                parent_values = torch.cat([state, action], dim=-1)
            else:
                parent_indices = [
                    self.causal_graph.var_to_idx[p]
                    for p in parents
                    if p in self.state_vars
                ]
                action_indices = [
                    self.causal_graph.var_to_idx[p] - self.state_dim
                    for p in parents
                    if p in self.action_vars
                ]

                parent_values_list = []
                if parent_indices:
                    parent_values_list.append(state[:, parent_indices])
                if action_indices:
                    parent_values_list.append(action[:, action_indices])

                if parent_values_list:
                    parent_values = torch.cat(parent_values_list, dim=-1)
                else:
                    parent_values = torch.cat([state, action], dim=-1)

            component = self.mechanisms[var](parent_values)
            next_state_components.append(component)

        next_state = torch.cat(next_state_components, dim=-1)
        return next_state

    def intervene(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        intervention_var: str,
        intervention_value: torch.Tensor,
    ) -> torch.Tensor:
        """Perform intervention do(X = x) and predict outcome"""

        batch_size = state.shape[0]
        next_state_components = []

        for i, var in enumerate(self.state_vars):
            if var == intervention_var:
                next_state_components.append(intervention_value.unsqueeze(-1))
            else:
                parents = self.causal_graph.get_parents(var)

                if intervention_var in parents:
                    parents = [p for p in parents if p != intervention_var]

                if not parents:
                    parent_values = torch.cat([state, action], dim=-1)
                else:
                    parent_indices = [
                        self.causal_graph.var_to_idx[p]
                        for p in parents
                        if p in self.state_vars
                    ]
                    action_indices = [
                        self.causal_graph.var_to_idx[p] - self.state_dim
                        for p in parents
                        if p in self.action_vars
                    ]

                    parent_values_list = []
                    if parent_indices:
                        parent_values_list.append(state[:, parent_indices])
                    if action_indices:
                        parent_values_list.append(action[:, action_indices])

                    if parent_values_list:
                        parent_values = torch.cat(parent_values_list, dim=-1)
                    else:
                        parent_values = torch.cat([state, action], dim=-1)

                component = self.mechanisms[var](parent_values)
                next_state_components.append(component)

        next_state = torch.cat(next_state_components, dim=-1)
        return next_state


class CounterfactualPolicyEvaluator:
    """Evaluate policies using counterfactual reasoning"""

    def __init__(self, causal_world_model: CausalWorldModel):
        self.world_model = causal_world_model

    def counterfactual_value(
        self,
        trajectory: Dict,
        counterfactual_policy: nn.Module,
        original_policy: nn.Module,
        gamma: float = 0.99,
    ) -> float:
        """
        Compute counterfactual value: "What if we had followed counterfactual_policy?"

        Uses three-step process:
        1. Abduction: Infer unobserved confounders from trajectory
        2. Action: Modify actions according to counterfactual policy
        3. Prediction: Predict outcomes under modified actions
        """

        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]

        T = len(states)
        counterfactual_return = 0.0

        for t in range(T):
            state = torch.FloatTensor(states[t]).unsqueeze(0)

            with torch.no_grad():
                cf_action = counterfactual_policy(state)
                cf_action = cf_action.squeeze().numpy()

            if t < T - 1:
                cf_action_tensor = torch.FloatTensor(cf_action).unsqueeze(0)
                cf_next_state = self.world_model(state, cf_action_tensor)

                cf_reward = self._compute_counterfactual_reward(
                    state.numpy(), cf_action, rewards[t]
                )

                counterfactual_return += (gamma**t) * cf_reward

        return counterfactual_return

    def _compute_counterfactual_reward(
        self, state: np.ndarray, cf_action: np.ndarray, observed_reward: float
    ) -> float:
        """Compute counterfactual reward (simplified heuristic)"""
        action_quality = np.linalg.norm(cf_action)  # Simplified metric
        return observed_reward * (1 + 0.1 * action_quality)


class CausalRLAgent:
    """RL Agent that uses causal reasoning for robust learning"""

    def __init__(
        self,
        n_arms: int,
        n_contexts: int,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        buffer_size: int = 10000,
    ):
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.state_dim = n_contexts  # Context as state
        self.action_dim = n_arms

        # Create simple causal graph
        variables = [f"context_{i}" for i in range(n_contexts)] + [
            f"arm_{j}" for j in range(n_arms)
        ]
        self.causal_graph = CausalGraph(variables)

        # Add some causal relationships
        for i in range(n_contexts):
            for j in range(n_arms):
                if np.random.random() > 0.5:  # Random causal links
                    self.causal_graph.add_edge(f"context_{i}", f"arm_{j}")

        self.world_model = CausalWorldModel(
            self.causal_graph, self.state_dim, self.action_dim
        )

        self.policy = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=learning_rate
        )
        self.model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=learning_rate
        )

        self.replay_buffer = []

    def select_action(self, state: np.ndarray) -> int:
        """Select action using causal policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            logits = self.policy(state_tensor)
            action = torch.argmax(logits, dim=-1).item()

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

    def train_step(self) -> Dict[str, float]:
        """Train the agent for one step"""
        if len(self.replay_buffer) < 32:
            return {}

        # Sample batch
        batch = random.sample(self.replay_buffer, 32)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Train world model
        transitions = [
            {"state": s, "action": a, "next_state": ns}
            for s, a, ns in zip(states, actions, next_states)
        ]
        model_loss = self.train_world_model(transitions)

        # Policy and value update
        trajectories = [{"states": states, "actions": actions, "rewards": rewards}]

        policy_loss, value_loss = self.causal_policy_gradient(trajectories)

        return {
            "causal_strength": 0.5,  # Placeholder
            "counterfactual_regret": value_loss,
        }
