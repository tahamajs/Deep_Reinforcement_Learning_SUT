import random
import numpy as np
import torch
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Logger:
    """Handles logging of training metrics."""
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = os.path.join(log_dir, experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = open(os.path.join(self.log_dir, "log.txt"), "w")
        self.metrics: Dict[str, List[Any]] = {}

    def log(self, tag: str, value: Any, step: int):
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append((step, value))
        self.log_file.write(f"[{step}] {tag}: {value}\n")
        self.log_file.flush()

    def get_metrics(self) -> Dict[str, List[Any]]:
        return self.metrics

    def close(self):
        self.log_file.close()

class Checkpointing:
    """Saves and loads model checkpoints."""
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, round_idx: int, prefix: str = ""):
        path = os.path.join(self.save_dir, f"{prefix}_checkpoint_{round_idx}.pt")
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "round_idx": round_idx}, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {path}")
        return checkpoint["round_idx"]

class FederatedAggregator:
    """Implements Federated Averaging (FedAvg) aggregation logic."""
    def __init__(self, config):
        self.config = config

    def aggregate_model_weights(self, client_weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not client_weights: # Ensure client_weights is not empty
            return {}
        
        global_weights = client_weights[0].copy()
        for key in global_weights.keys():
            for i in range(1, len(client_weights)):
                global_weights[key] += client_weights[i][key]
            global_weights[key] = torch.div(global_weights[key], len(client_weights))
        return global_weights

    def apply_differential_privacy(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.config.differential_privacy_scale > 0:
            for key in weights.keys():
                noise = torch.randn(weights[key].shape) * self.config.differential_privacy_scale
                weights[key] += noise.to(weights[key].device) # Ensure noise is on the same device
        return weights

class CausalGraphLearner:
    """A module for learning the causal graph."""
    def __init__(self, num_nodes: int, device: torch.device):
        self.num_nodes = num_nodes
        self.device = device
        # Placeholder for causal graph (e.g., adjacency matrix)
        self.causal_graph = torch.eye(num_nodes, device=device) # Initialize as identity (no causal effects)
        print(f"CausalGraphLearner initialized with {num_nodes} nodes.")

    def learn_causal_graph(self, latent_data: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Placeholder for a causal discovery algorithm. Returns an updated adjacency matrix.
        Args:
            latent_data (torch.Tensor): Batched latent states from the world model.
            adjacency_matrix (torch.Tensor): Current adjacency matrix representing causal graph.
        Returns:
            torch.Tensor: Updated adjacency matrix.
        """
        # In a real scenario, this would involve a complex causal discovery algorithm
        # e.g., PC algorithm, GFN-based approaches, or differentiable causal discovery.
        # For this implementation, we'll keep it simple: assume some fixed or pre-learned structure
        # or slowly update based on some criterion. Here, we'll just return the input for now.
        
        # A more advanced implementation would use libraries like causaldag or integrate
        # differentiable causal discovery methods.
        
        # Example of a dummy update (e.g., based on some correlation)
        # For demonstration, let's assume a simple feed-forward causal structure
        # where z_i can cause z_{i+1}
        new_adjacency = adjacency_matrix.clone()
        if latent_data.shape[1] == self.num_nodes: # Ensure latent_data matches num_nodes
            correlation_matrix = torch.corrcoef(latent_data.T)
            # Simple heuristic: if correlation > threshold, assume a causal link
            # This is NOT a real causal discovery method, just illustrative.
            threshold = 0.5
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if correlation_matrix[i, j].abs() > threshold:
                        # Assuming i causes j
                        new_adjacency[i, j] = 1.0
                        new_adjacency[j, i] = 0.0 # Acyclic
        self.causal_graph = new_adjacency
        return self.causal_graph
