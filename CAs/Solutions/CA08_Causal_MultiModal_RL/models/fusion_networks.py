"""
Multi-Modal Fusion Networks
Neural network architectures for combining different modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional


class EarlyFusionNetwork(nn.Module):
    """Early fusion: Concatenate all modalities before processing"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 128):
        super(EarlyFusionNetwork, self).__init__()
        
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        self.total_input_dim = sum(modal_dims.values())
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(self.total_input_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with early fusion"""
        # Concatenate all modalities
        concatenated = torch.cat([modal_inputs[modality] for modality in self.modal_dims.keys()], dim=-1)
        
        # Project to fusion dimension
        fused = self.input_projection(concatenated)
        
        # Process features
        output = self.feature_processor(fused)
        
        return output


class LateFusionNetwork(nn.Module):
    """Late fusion: Process modalities separately then combine"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 128):
        super(LateFusionNetwork, self).__init__()
        
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        
        # Modality-specific encoders
        self.modal_encoders = nn.ModuleDict()
        for modality, dim in modal_dims.items():
            self.modal_encoders[modality] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim)
            )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * len(modal_dims), fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with late fusion"""
        # Process each modality separately
        modal_features = []
        for modality in self.modal_dims.keys():
            if modality in modal_inputs:
                feature = self.modal_encoders[modality](modal_inputs[modality])
                modal_features.append(feature)
        
        # Concatenate processed features
        concatenated = torch.cat(modal_features, dim=-1)
        
        # Fuse features
        output = self.fusion_layer(concatenated)
        
        return output


class CrossModalAttentionNetwork(nn.Module):
    """Cross-modal attention fusion"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 128, num_heads: int = 4):
        super(CrossModalAttentionNetwork, self).__init__()
        
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Modality-specific encoders
        self.modal_encoders = nn.ModuleDict()
        for modality, dim in modal_dims.items():
            self.modal_encoders[modality] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with cross-modal attention"""
        # Encode all modalities
        modal_features = []
        for modality in self.modal_dims.keys():
            if modality in modal_inputs:
                feature = self.modal_encoders[modality](modal_inputs[modality])
                modal_features.append(feature)
        
        # Stack features for attention
        stacked_features = torch.stack(modal_features, dim=1)  # [batch, num_modalities, fusion_dim]
        
        # Apply self-attention
        attended_features, _ = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Global average pooling
        pooled_features = attended_features.mean(dim=1)  # [batch, fusion_dim]
        
        # Output projection
        output = self.output_projection(pooled_features)
        
        return output


class HierarchicalFusionNetwork(nn.Module):
    """Hierarchical fusion: Progressive combination of modalities"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 128):
        super(HierarchicalFusionNetwork, self).__init__()
        
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        
        # Modality-specific encoders
        self.modal_encoders = nn.ModuleDict()
        for modality, dim in modal_dims.items():
            self.modal_encoders[modality] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim)
            )
        
        # Hierarchical fusion layers
        self.fusion_layers = nn.ModuleList()
        num_modalities = len(modal_dims)
        
        # Progressive fusion
        current_dim = fusion_dim
        for i in range(num_modalities - 1):
            self.fusion_layers.append(nn.Sequential(
                nn.Linear(current_dim + fusion_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim)
            ))
            current_dim = fusion_dim
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with hierarchical fusion"""
        # Encode all modalities
        modal_features = []
        for modality in self.modal_dims.keys():
            if modality in modal_inputs:
                feature = self.modal_encoders[modality](modal_inputs[modality])
                modal_features.append(feature)
        
        # Hierarchical fusion
        if len(modal_features) == 0:
            return torch.zeros(modal_inputs[list(self.modal_dims.keys())[0]].shape[0], self.fusion_dim)
        
        # Start with first modality
        fused = modal_features[0]
        
        # Progressively fuse with other modalities
        for i, feature in enumerate(modal_features[1:], 1):
            # Concatenate current fused representation with next modality
            combined = torch.cat([fused, feature], dim=-1)
            
            # Apply fusion layer
            fused = self.fusion_layers[i-1](combined)
        
        # Final output
        output = self.output_layer(fused)
        
        return output


class AdaptiveFusionNetwork(nn.Module):
    """Adaptive fusion: Learn optimal combination weights"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 128):
        super(AdaptiveFusionNetwork, self).__init__()
        
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        
        # Modality-specific encoders
        self.modal_encoders = nn.ModuleDict()
        for modality, dim in modal_dims.items():
            self.modal_encoders[modality] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim)
            )
        
        # Adaptive weighting network
        self.weight_network = nn.Sequential(
            nn.Linear(fusion_dim * len(modal_dims), fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, len(modal_dims)),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with adaptive fusion"""
        # Encode all modalities
        modal_features = []
        for modality in self.modal_dims.keys():
            if modality in modal_inputs:
                feature = self.modal_encoders[modality](modal_inputs[modality])
                modal_features.append(feature)
        
        if len(modal_features) == 0:
            return torch.zeros(modal_inputs[list(self.modal_dims.keys())[0]].shape[0], self.fusion_dim)
        
        # Concatenate all features for weight computation
        concatenated = torch.cat(modal_features, dim=-1)
        
        # Compute adaptive weights
        weights = self.weight_network(concatenated)  # [batch, num_modalities]
        
        # Weighted combination
        weighted_features = []
        for i, feature in enumerate(modal_features):
            weighted_feature = feature * weights[:, i:i+1]  # [batch, fusion_dim]
            weighted_features.append(weighted_feature)
        
        # Sum weighted features
        fused = sum(weighted_features)
        
        # Output projection
        output = self.output_projection(fused)
        
        return output
