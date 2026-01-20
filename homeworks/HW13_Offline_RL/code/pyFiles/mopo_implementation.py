"""
MOPO implementation (ensemble dynamics + penalized rollouts).
Lightweight, intended for integration in notebooks.
"""
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

class DynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        h = self.net(x)
        delta = self.delta_head(h)
        reward = self.reward_head(h)
        next_state = state + delta
        return next_state, reward.squeeze(-1)

class MOPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        lambda_u: float = 1.0,
        lr: float = 1e-3,
    ):
        self.ensemble: List[DynamicsModel] = [
            DynamicsModel(state_dim, action_dim) for _ in range(ensemble_size)
        ]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.ensemble]
        self.lambda_u = lambda_u
    
    def train_dynamics(
        self,
        dataset: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        epochs: int = 10,
    ):
        """Train the dynamics model using the dataset with progress tracking"""
        print(f"\n{'='*50}")
        print(f"Training MOPO Dynamics Model (Ensemble Size: {len(self.ensemble)})")
        print(f"{'='*50}")
        
        for epoch in range(epochs):
            total_loss = 0.0
            epoch_loss = 0.0
            for model_idx, (model, opt) in enumerate(zip(self.ensemble, self.optimizers)):
                model_loss = 0.0
                for s, a, r, s_next, _ in dataset:
                    # Convert to tensors with correct shapes
                    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)  # [1, state_dim]
                    a_tensor = torch.tensor(a, dtype=torch.float32).view(1, 1)    # [1, 1]
                    s_next_tensor = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0)
                    r_tensor = torch.tensor(r, dtype=torch.float32).view(1, 1)
                    
                    # Forward pass
                    pred_next, pred_r = model(s_tensor, a_tensor)
                    
                    # Compute losses
                    next_loss = ((pred_next.squeeze(0) - s_next_tensor) ** 2).mean()
                    r_loss = ((pred_r.squeeze(0) - r_tensor) ** 2).mean()
                    loss = next_loss + r_loss
                    
                    # Backpropagation
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
                    model_loss += loss.item()
                    total_loss += loss.item()
                
                # Print per-model progress
                avg_model_loss = model_loss / len(dataset)
                print(f"  Model {model_idx+1}/{len(self.ensemble)} | Epoch {epoch+1}/{epochs} | Loss: {avg_model_loss:.4f}")
            
            # Print epoch summary
            avg_epoch_loss = total_loss / (len(dataset) * len(self.ensemble))
            print(f"Epoch {epoch+1}/{epochs} | Overall Avg Loss: {avg_epoch_loss:.4f}")
            print(f"{'-'*50}")
                
    def model_rollout(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # <-- ADD COLON HERE IF MISSING
        """Predict next state and reward using ensemble dynamics model."""
        # Remove extra batch dimension (state is [1, 1, state_dim] from select_action)
        state = state.squeeze(0)  # Now [1, state_dim]
        action = action.squeeze(0)  # Now [1, action_dim]
        
        # Predict with ensemble
        preds_next = []
        preds_r = []
        for model in self.ensemble:
            ns, r = model(state, action)  # <-- FIXED: Removed .unsqueeze(0)
            preds_next.append(ns.squeeze(0))
            preds_r.append(r.squeeze(0))
        
        # Compute ensemble statistics
        preds_next = torch.stack(preds_next, dim=0)
        preds_r = torch.stack(preds_r, dim=0)
        next_state_mean = preds_next.mean(dim=0)
        reward_mean = preds_r.mean(dim=0)
        uncertainty = preds_next.std(dim=0).mean()
        penalty = self.lambda_u * uncertainty
        adjusted_reward = reward_mean - penalty
        
        return next_state_mean, adjusted_reward

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select an action using the MOPO algorithm.
        For CartPole (discrete actions), we try both possible actions and select the one with highest expected reward.
        """
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Expand state to batch dimension
        state = state.unsqueeze(0)
        
        # Try both possible actions (0 and 1 for CartPole)
        actions = torch.tensor([0, 1], dtype=torch.float32).view(-1, 1)
        
        # Evaluate both actions
        rewards = []
        for action in actions:
            # Use model_rollout to get the expected reward
            _, reward = self.model_rollout(state, action)
            rewards.append(reward.item())
        
        # Select the action with the highest expected reward
        best_action_idx = torch.argmax(torch.tensor(rewards))
        return actions[best_action_idx].item()