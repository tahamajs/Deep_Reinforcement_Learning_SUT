import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import networkx as nx
from scipy.stats import pearsonr


class CausalGraph:
    """Represents causal relationships between variables"""

    def __init__(self, variables: List[str]):
        self.variables = variables
        self.n_vars = len(variables)
        self.var_to_idx = {var: i for i, var in enumerate(variables)}

        # Adjacency matrix: adj_matrix[i][j] = 1 if i -> j
        self.adj_matrix = np.zeros((self.n_vars, self.n_vars), dtype=int)

    def add_edge(self, from_var: str, to_var: str):
        """Add causal edge from_var -> to_var"""
        from_idx = self.var_to_idx[from_var]
        to_idx = self.var_to_idx[to_var]
        self.adj_matrix[from_idx][to_idx] = 1

    def get_parents(self, var: str) -> List[str]:
        """Get parent variables of var"""
        var_idx = self.var_to_idx[var]
        parent_indices = np.where(self.adj_matrix[:, var_idx] == 1)[0]
        return [self.variables[i] for i in parent_indices]

    def get_children(self, var: str) -> List[str]:
        """Get children variables of var"""
        var_idx = self.var_to_idx[var]
        child_indices = np.where(self.adj_matrix[var_idx, :] == 1)[0]
        return [self.variables[i] for i in child_indices]

    def is_d_separated(self, x: str, y: str, z: List[str]) -> bool:
        """Check if x and y are d-separated given z (simplified)"""
        # Simplified d-separation check
        # In practice, this would use a proper d-separation algorithm
        x_idx = self.var_to_idx[x]
        y_idx = self.var_to_idx[y]
        z_indices = [self.var_to_idx[var] for var in z]

        # Check if there's a direct path from x to y not blocked by z
        # This is a simplified version
        return not self._has_unblocked_path(x_idx, y_idx, z_indices)

    def _has_unblocked_path(self, start: int, end: int, blocking: List[int]) -> bool:
        """Simplified path checking (DFS-based)"""
        if start == end:
            return True

        visited = set()
        stack = [start]

        while stack:
            current = stack.pop()
            if current in visited or current in blocking:
                continue

            visited.add(current)

            # Check all connected nodes
            for next_node in range(self.n_vars):
                if (self.adj_matrix[current][next_node] == 1 or
                    self.adj_matrix[next_node][current] == 1):
                    if next_node == end:
                        return True
                    stack.append(next_node)

        return False

    def visualize(self):
        """Simple text visualization of the graph"""
        print("Causal Graph Structure:")
        for i, var in enumerate(self.variables):
            children = self.get_children(var)
            if children:
                print(f"{var} -> {', '.join(children)}")


class CausalDiscovery:
    """Causal structure discovery from data"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Significance level for independence tests

    def pc_algorithm(self, data: np.ndarray, var_names: List[str]) -> CausalGraph:
        """PC Algorithm for causal discovery"""
        n_vars = len(var_names)

        # Initialize complete undirected graph
        skeleton = np.ones((n_vars, n_vars)) - np.eye(n_vars)

        # Phase 1: Skeleton discovery
        for order in range(n_vars - 2):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if skeleton[i][j] == 0:
                        continue

                    # Find all possible conditioning sets of size 'order'
                    neighbors = [k for k in range(n_vars)
                                if k != i and k != j and skeleton[i][k] == 1]

                    if len(neighbors) >= order:
                        from itertools import combinations
                        for cond_set in combinations(neighbors, order):
                            # Test conditional independence
                            if self._test_independence(data, i, j, list(cond_set)):
                                skeleton[i][j] = skeleton[j][i] = 0
                                break

        # Phase 2: Orient edges (simplified)
        graph = CausalGraph(var_names)
        oriented = self._orient_edges(skeleton, data)

        for i in range(n_vars):
            for j in range(n_vars):
                if oriented[i][j] == 1:
                    graph.add_edge(var_names[i], var_names[j])

        return graph

    def _test_independence(self, data: np.ndarray, i: int, j: int,
                          cond_set: List[int]) -> bool:
        """Test conditional independence using correlation (simplified)"""
        # This is a simplified test - in practice, use proper statistical tests

        if len(cond_set) == 0:
            # Unconditional independence
            corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
            return abs(corr) < 0.1  # Simplified threshold

        # Conditional independence using partial correlation
        from scipy.stats import pearsonr

        # Regress out conditioning variables
        X = data[:, [i] + cond_set]
        Y = data[:, j]

        # Simplified partial correlation
        if len(cond_set) == 1:
            # Partial correlation formula for one conditioning variable
            r_ij = np.corrcoef(data[:, i], data[:, j])[0, 1]
            r_ik = np.corrcoef(data[:, i], data[:, cond_set[0]])[0, 1]
            r_jk = np.corrcoef(data[:, j], data[:, cond_set[0]])[0, 1]

            partial_corr = (r_ij - r_ik * r_jk) / np.sqrt((1 - r_ik**2) * (1 - r_jk**2))
            return abs(partial_corr) < 0.1

        return False  # Simplified - assume dependent if complex conditioning

    def _orient_edges(self, skeleton: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Orient edges to create DAG (simplified)"""
        n_vars = skeleton.shape[0]
        oriented = np.zeros_like(skeleton)

        # Simplified orientation: use temporal ordering if available
        # Or use variance-based heuristics
        for i in range(n_vars):
            for j in range(n_vars):
                if skeleton[i][j] == 1:
                    # Simple heuristic: variable with higher variance causes lower variance
                    var_i = np.var(data[:, i])
                    var_j = np.var(data[:, j])

                    if var_i > var_j:
                        oriented[i][j] = 1
                    else:
                        oriented[j][i] = 1

        return oriented


class InterventionalDataset:
    """Dataset with interventional data"""

    def __init__(self):
        self.observational_data = []
        self.interventional_data = {}  # {intervention: data}

    def add_observational(self, data: Dict[str, np.ndarray]):
        """Add observational data"""
        self.observational_data.append(data)

    def add_interventional(self, intervention: str, data: Dict[str, np.ndarray]):
        """Add interventional data"""
        if intervention not in self.interventional_data:
            self.interventional_data[intervention] = []
        self.interventional_data[intervention].append(data)


class CausalWorldModel(nn.Module):
    """World model with causal structure"""

    def __init__(self, causal_graph: CausalGraph, state_dims: Dict[str, int],
                 action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.causal_graph = causal_graph
        self.state_dims = state_dims
        self.action_dim = action_dim
        self.variables = causal_graph.variables

        # Create prediction networks for each variable
        self.predictors = nn.ModuleDict()

        for var in self.variables:
            parents = causal_graph.get_parents(var)
            parent_dim = sum(state_dims[p] for p in parents)

            # Include action in input if relevant
            input_dim = parent_dim + action_dim
            output_dim = state_dims[var]

            self.predictors[var] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, states: Dict[str, torch.Tensor],
                actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict next states following causal structure"""

        predictions = {}

        for var in self.variables:
            parents = self.causal_graph.get_parents(var)

            # Collect parent values
            parent_values = []
            for parent in parents:
                parent_values.append(states[parent])

            # Combine inputs
            if parent_values:
                parent_input = torch.cat(parent_values, dim=-1)
                model_input = torch.cat([parent_input, actions], dim=-1)
            else:
                model_input = actions

            # Predict
            predictions[var] = self.predictors[var](model_input)

        return predictions

    def intervene(self, states: Dict[str, torch.Tensor], actions: torch.Tensor,
                  interventions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict under interventions"""

        # Copy states and apply interventions
        modified_states = {k: v.clone() for k, v in states.items()}
        for var, value in interventions.items():
            modified_states[var] = value

        # Predict following causal structure
        predictions = {}

        for var in self.variables:
            if var in interventions:
                # Interventions fix variables
                predictions[var] = interventions[var]
            else:
                # Predict from causal parents
                parents = self.causal_graph.get_parents(var)
                parent_values = [modified_states[p] for p in parents]

                if parent_values:
                    parent_input = torch.cat(parent_values, dim=-1)
                    model_input = torch.cat([parent_input, actions], dim=-1)
                else:
                    model_input = actions

                predictions[var] = self.predictors[var](model_input)

        return predictions


class CausalPolicyGradient:
    """Policy gradient with causal regularization"""

    def __init__(self, policy: nn.Module, causal_graph: CausalGraph,
                 lr: float = 3e-4, causal_weight: float = 0.1):

        self.policy = policy
        self.causal_graph = causal_graph
        self.causal_weight = causal_weight
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def update(self, states: Dict[str, torch.Tensor], actions: torch.Tensor,
               rewards: torch.Tensor, causal_world_model: CausalWorldModel):
        """Update policy with causal regularization"""

        # Standard policy gradient loss
        log_probs = self.policy.get_log_prob(states, actions)
        policy_loss = -(log_probs * rewards).mean()

        # Causal regularization
        causal_loss = self._compute_causal_regularization(
            states, actions, causal_world_model
        )

        # Total loss
        total_loss = policy_loss + self.causal_weight * causal_loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'causal_loss': causal_loss.item(),
            'total_loss': total_loss.item()
        }

    def _compute_causal_regularization(self, states: Dict[str, torch.Tensor],
                                     actions: torch.Tensor,
                                     causal_world_model: CausalWorldModel) -> torch.Tensor:
        """Compute causal regularization term"""

        # Interventional consistency loss
        consistency_loss = 0
        n_interventions = 0

        for var in self.causal_graph.variables:
            # Create intervention
            intervention_value = torch.randn_like(states[var])
            interventions = {var: intervention_value}

            # Predict under intervention
            pred_intervened = causal_world_model.intervene(states, actions, interventions)

            # Predict with modified state
            modified_states = {k: v.clone() for k, v in states.items()}
            modified_states[var] = intervention_value
            pred_modified = causal_world_model(modified_states, actions)

            # Consistency loss for non-descendants
            for other_var in self.causal_graph.variables:
                if other_var != var and not self._is_descendant(var, other_var):
                    consistency_loss += F.mse_loss(
                        pred_intervened[other_var],
                        pred_modified[other_var]
                    )
                    n_interventions += 1

        return consistency_loss / max(n_interventions, 1)

    def _is_descendant(self, ancestor: str, var: str) -> bool:
        """Check if var is a descendant of ancestor"""
        # Simple DFS to check descendancy
        visited = set()
        stack = self.causal_graph.get_children(ancestor)

        while stack:
            current = stack.pop()
            if current == var:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.causal_graph.get_children(current))

        return False


def create_synthetic_causal_data(n_samples: int = 1000):
    """Create synthetic data with known causal structure"""

    # True causal structure: X1 -> X2 -> X3, X1 -> X3, Action -> X2
    data = {}

    # Exogenous noise
    e1 = np.random.normal(0, 0.5, n_samples)
    e2 = np.random.normal(0, 0.3, n_samples)
    e3 = np.random.normal(0, 0.4, n_samples)

    # Actions
    actions = np.random.uniform(-1, 1, (n_samples, 2))

    # Causal mechanisms
    X1 = e1
    X2 = 0.7 * X1 + 0.5 * actions[:, 0] + e2
    X3 = 0.8 * X2 + 0.3 * X1 + e3

    data['X1'] = X1.reshape(-1, 1)
    data['X2'] = X2.reshape(-1, 1)
    data['X3'] = X3.reshape(-1, 1)

    return data, actions