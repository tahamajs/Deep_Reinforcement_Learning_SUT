"""
Causal Discovery Module for Reinforcement Learning
Implements causal structure learning and reasoning
"""

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Tuple, Optional
from itertools import combinations


class CausalGraph:
    """
    Represents a causal graph with variables and their relationships
    """

    def __init__(self, variables: List[str]):
        """
        Initialize causal graph

        Args:
            variables: List of variable names
        """
        self.variables = variables
        self.var_to_idx = {var: i for i, var in enumerate(variables)}
        self.idx_to_var = {i: var for i, var in enumerate(variables)}
        self.adj_matrix = np.zeros((len(variables), len(variables)), dtype=bool)

    def add_edge(self, cause: str, effect: str):
        """Add a causal edge from cause to effect"""
        if cause not in self.var_to_idx or effect not in self.var_to_idx:
            raise ValueError(f"Variables {cause} or {effect} not in graph")
        i, j = self.var_to_idx[cause], self.var_to_idx[effect]
        self.adj_matrix[i, j] = True

    def remove_edge(self, cause: str, effect: str):
        """Remove a causal edge from cause to effect"""
        if cause not in self.var_to_idx or effect not in self.var_to_idx:
            raise ValueError(f"Variables {cause} or {effect} not in graph")
        i, j = self.var_to_idx[cause], self.var_to_idx[effect]
        self.adj_matrix[i, j] = False

    def get_parents(self, var: str) -> List[str]:
        """Get parent variables of a given variable"""
        if var not in self.var_to_idx:
            raise KeyError(f"Variable {var} not found in graph")
        j = self.var_to_idx[var]
        parent_indices = np.where(self.adj_matrix[:, j])[0]
        return [self.variables[i] for i in parent_indices]

    def get_children(self, var: str) -> List[str]:
        """Get child variables of a given variable"""
        if var not in self.var_to_idx:
            raise KeyError(f"Variable {var} not found in graph")
        i = self.var_to_idx[var]
        child_indices = np.where(self.adj_matrix[i, :])[0]
        return [self.variables[j] for j in child_indices]

    def get_ancestors(self, var: str) -> List[str]:
        """Get all ancestors of a variable"""
        ancestors = set()
        to_visit = [var]

        while to_visit:
            current = to_visit.pop()
            for parent in self.get_parents(current):
                if parent not in ancestors:
                    ancestors.add(parent)
                    to_visit.append(parent)

        return list(ancestors)

    def get_descendants(self, var: str) -> List[str]:
        """Get all descendants of a variable"""
        descendants = set()
        to_visit = [var]

        while to_visit:
            current = to_visit.pop()
            for child in self.get_children(current):
                if child not in descendants:
                    descendants.add(child)
                    to_visit.append(child)

        return list(descendants)

    def is_dag(self) -> bool:
        """Check if the graph is a DAG"""
        try:
            nx.DiGraph(self.adj_matrix).topological_sort()
            return True
        except nx.NetworkXError:
            return False

    def get_topological_order(self) -> List[str]:
        """Get topological ordering of variables"""
        if not self.is_dag():
            raise ValueError("Graph contains cycles")
        order_indices = list(nx.DiGraph(self.adj_matrix).topological_sort())
        return [self.variables[i] for i in order_indices]

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph for visualization"""
        G = nx.DiGraph()
        G.add_nodes_from(self.variables)

        for i in range(len(self.variables)):
            for j in range(len(self.variables)):
                if self.adj_matrix[i, j]:
                    G.add_edge(self.variables[i], self.variables[j])

        return G

    def __str__(self):
        """String representation of the graph"""
        edges = []
        for i in range(len(self.variables)):
            for j in range(len(self.variables)):
                if self.adj_matrix[i, j]:
                    edges.append(f"{self.variables[i]} -> {self.variables[j]}")
        return f"CausalGraph(variables={self.variables}, edges={edges})"


class CausalDiscovery:
    """
    Causal discovery algorithms for learning causal structure from data
    """

    @staticmethod
    def pc_algorithm(
        data: np.ndarray, variable_names: List[str], alpha: float = 0.05
    ) -> CausalGraph:
        """
        PC algorithm for causal discovery

        Args:
            data: Data matrix (n_samples, n_variables)
            variable_names: Names of variables
            alpha: Significance level for independence tests

        Returns:
            Learned causal graph
        """
        n_vars = len(variable_names)
        graph = CausalGraph(variable_names)

        # Start with fully connected undirected graph
        for i, j in combinations(range(n_vars), 2):
            graph.add_edge(variable_names[i], variable_names[j])
            graph.add_edge(variable_names[j], variable_names[i])

        # PC algorithm implementation (simplified)
        # In practice, would use proper conditional independence tests
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Simple correlation-based test (placeholder)
                corr = np.abs(np.corrcoef(data[:, i], data[:, j])[0, 1])
                if corr < 0.1:  # Arbitrary threshold
                    graph.remove_edge(variable_names[i], variable_names[j])
                    graph.remove_edge(variable_names[j], variable_names[i])

        return graph

    @staticmethod
    def ges_algorithm(data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        """
        Greedy Equivalence Search (GES) algorithm

        Args:
            data: Data matrix (n_samples, n_variables)
            variable_names: Names of variables

        Returns:
            Learned causal graph
        """
        # Simplified GES implementation
        graph = CausalGraph(variable_names)

        # Forward phase: add edges
        for i, j in combinations(range(len(variable_names)), 2):
            corr = np.abs(np.corrcoef(data[:, i], data[:, j])[0, 1])
            if corr > 0.3:  # Arbitrary threshold
                # Determine direction based on correlation strength
                if corr > 0.5:
                    graph.add_edge(variable_names[i], variable_names[j])
                else:
                    graph.add_edge(variable_names[j], variable_names[i])

        return graph

    @staticmethod
    def lingam_algorithm(data: np.ndarray, variable_names: List[str]) -> CausalGraph:
        """
        Linear Non-Gaussian Acyclic Model (LiNGAM) for causal discovery

        Args:
            data: Data matrix (n_samples, n_variables)
            variable_names: Names of variables

        Returns:
            Learned causal graph
        """
        from sklearn.linear_model import LinearRegression

        n_vars = len(variable_names)
        graph = CausalGraph(variable_names)

        # Fit linear models and determine causal order
        residuals = np.copy(data)

        for i in range(n_vars):
            # Find the variable with most independent residuals
            residual_vars = np.var(residuals, axis=0)
            target_idx = np.argmin(residual_vars)

            # Regress on remaining variables
            remaining_indices = [j for j in range(n_vars) if j != target_idx]
            if remaining_indices:
                X = data[:, remaining_indices]
                y = data[:, target_idx]

                reg = LinearRegression()
                reg.fit(X, y)

                # Add edges from predictors to target
                for j, coef in zip(remaining_indices, reg.coef_):
                    if abs(coef) > 0.1:  # Threshold
                        graph.add_edge(variable_names[j], variable_names[target_idx])

            # Remove this variable from consideration
            residuals = np.delete(residuals, target_idx, axis=1)
            data = np.delete(data, target_idx, axis=1)
            variable_names = [
                v for k, v in enumerate(variable_names) if k != target_idx
            ]

        return graph
