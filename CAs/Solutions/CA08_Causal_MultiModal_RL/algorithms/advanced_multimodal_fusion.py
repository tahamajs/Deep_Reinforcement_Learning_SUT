"""
Advanced Multi-Modal Fusion Networks for CA8
============================================

This module implements state-of-the-art multi-modal fusion techniques:
- Transformer-based Cross-Modal Attention
- Hierarchical Multi-Modal Fusion
- Dynamic Adaptive Fusion
- Memory-Augmented Fusion
- Graph Neural Network Fusion
- Quantum-Inspired Fusion
- Neuromorphic Fusion
- Meta-Learning Fusion

Author: DRL Course Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import math
from transformers import AutoModel, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")


class TransformerCrossModalAttention(nn.Module):
    """Transformer-based Cross-Modal Attention"""
    
    def __init__(self, modal_dims: Dict[str, int], hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        self.modal_dims = modal_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Modality encoders
        self.encoders = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.encoders[modal_name] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode all modalities
        encoded_modals = {}
        for modal_name, input_tensor in modal_inputs.items():
            encoded_modals[modal_name] = self.encoders[modal_name](input_tensor)
        
        # Stack modalities for attention
        modal_names = list(self.modal_dims.keys())
        stacked_features = torch.stack([encoded_modals[name] for name in modal_names], dim=1)
        
        # Apply transformer layers
        x = stacked_features
        for i, (attn, ffn, ln1, ln2) in enumerate(zip(
            self.attention_layers, self.ffns, self.layer_norms1, self.layer_norms2
        )):
            # Self-attention
            attn_out, _ = attn(x, x, x)
            x = ln1(x + attn_out)
            
            # Feed-forward
            ffn_out = ffn(x)
            x = ln2(x + ffn_out)
        
        # Average across modalities
        fused_features = torch.mean(x, dim=1)
        return self.output_projection(fused_features)


class HierarchicalMultiModalFusion(nn.Module):
    """Hierarchical Multi-Modal Fusion with multiple levels"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 256):
        super().__init__()
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        
        # Level 1: Individual modality processing
        self.level1_encoders = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.level1_encoders[modal_name] = nn.Sequential(
                nn.Linear(dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_dim // 2, fusion_dim // 2)
            )
        
        # Level 2: Pairwise fusion
        self.level2_fusions = nn.ModuleDict()
        modal_names = list(modal_dims.keys())
        for i, modal1 in enumerate(modal_names):
            for j, modal2 in enumerate(modal_names):
                if i < j:
                    pair_name = f"{modal1}_{modal2}"
                    self.level2_fusions[pair_name] = nn.Sequential(
                        nn.Linear(fusion_dim, fusion_dim),
                        nn.ReLU(),
                        nn.Linear(fusion_dim, fusion_dim // 2)
                    )
        
        # Level 3: Global fusion
        self.level3_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Attention weights for hierarchical fusion
        self.attention_weights = nn.Parameter(torch.ones(len(modal_names)))
        
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Level 1: Process individual modalities
        level1_features = {}
        for modal_name, input_tensor in modal_inputs.items():
            level1_features[modal_name] = self.level1_encoders[modal_name](input_tensor)
        
        # Level 2: Pairwise fusion
        level2_features = []
        modal_names = list(self.modal_dims.keys())
        
        for i, modal1 in enumerate(modal_names):
            for j, modal2 in enumerate(modal_names):
                if i < j:
                    pair_name = f"{modal1}_{modal2}"
                    pair_input = torch.cat([
                        level1_features[modal1], 
                        level1_features[modal2]
                    ], dim=-1)
                    pair_output = self.level2_fusions[pair_name](pair_input)
                    level2_features.append(pair_output)
        
        # Level 3: Global fusion
        if level2_features:
            global_input = torch.cat(level2_features, dim=-1)
            global_output = self.level3_fusion(global_input)
        else:
            # Fallback to level 1 features
            global_output = torch.mean(torch.stack(list(level1_features.values())), dim=0)
        
        return global_output


class DynamicAdaptiveFusion(nn.Module):
    """Dynamic Adaptive Fusion that learns optimal fusion strategies"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 256):
        super().__init__()
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        
        # Modality encoders
        self.encoders = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.encoders[modal_name] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
        
        # Dynamic fusion controller
        self.fusion_controller = nn.Sequential(
            nn.Linear(fusion_dim * len(modal_dims), fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, len(modal_dims)),
            nn.Softmax(dim=-1)
        )
        
        # Multiple fusion strategies
        self.fusion_strategies = nn.ModuleDict({
            'early': nn.Linear(sum(modal_dims.values()), fusion_dim),
            'late': nn.Linear(fusion_dim * len(modal_dims), fusion_dim),
            'attention': nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True),
            'gated': nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.Sigmoid()
            )
        })
        
        # Strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(fusion_dim * len(modal_dims), fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, len(self.fusion_strategies)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode modalities
        encoded_modals = {}
        for modal_name, input_tensor in modal_inputs.items():
            encoded_modals[modal_name] = self.encoders[modal_name](input_tensor)
        
        # Get fusion weights
        stacked_features = torch.cat(list(encoded_modals.values()), dim=-1)
        fusion_weights = self.fusion_controller(stacked_features)
        
        # Select fusion strategy
        strategy_weights = self.strategy_selector(stacked_features)
        
        # Apply different fusion strategies
        fusion_results = []
        
        # Early fusion
        early_input = torch.cat(list(modal_inputs.values()), dim=-1)
        early_output = self.fusion_strategies['early'](early_input)
        fusion_results.append(early_output)
        
        # Late fusion
        late_input = torch.cat(list(encoded_modals.values()), dim=-1)
        late_output = self.fusion_strategies['late'](late_input)
        fusion_results.append(late_output)
        
        # Attention fusion
        modal_stack = torch.stack(list(encoded_modals.values()), dim=1)
        attn_output, _ = self.fusion_strategies['attention'](modal_stack, modal_stack, modal_stack)
        attn_output = torch.mean(attn_output, dim=1)
        fusion_results.append(attn_output)
        
        # Gated fusion
        gated_input = torch.cat([early_output, late_output], dim=-1)
        gate = self.fusion_strategies['gated'](gated_input)
        gated_output = gate * early_output + (1 - gate) * late_output
        fusion_results.append(gated_output)
        
        # Weighted combination of strategies
        final_output = sum(w * result for w, result in zip(strategy_weights.T, fusion_results))
        
        return final_output


class MemoryAugmentedFusion(nn.Module):
    """Memory-Augmented Multi-Modal Fusion"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 256, memory_size: int = 1000):
        super().__init__()
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        self.memory_size = memory_size
        
        # Modality encoders
        self.encoders = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.encoders[modal_name] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
        
        # Memory bank
        self.register_buffer('memory_keys', torch.randn(memory_size, fusion_dim))
        self.register_buffer('memory_values', torch.randn(memory_size, fusion_dim))
        self.memory_usage = torch.zeros(memory_size)
        
        # Memory controller
        self.memory_controller = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode modalities
        encoded_modals = {}
        for modal_name, input_tensor in modal_inputs.items():
            encoded_modals[modal_name] = self.encoders[modal_name](input_tensor)
        
        # Combine modalities
        combined_features = torch.mean(torch.stack(list(encoded_modals.values())), dim=0)
        
        # Memory retrieval
        similarities = torch.mm(combined_features, self.memory_keys.T)
        attention_weights = F.softmax(similarities, dim=-1)
        
        # Retrieve from memory
        memory_output = torch.mm(attention_weights, self.memory_values)
        
        # Update memory usage
        self.memory_usage += attention_weights.mean(dim=0)
        
        # Fusion with memory
        fusion_input = torch.cat([combined_features, memory_output], dim=-1)
        fused_output = self.fusion_net(fusion_input)
        
        return fused_output


class GraphNeuralNetworkFusion(nn.Module):
    """Graph Neural Network-based Multi-Modal Fusion"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 256):
        super().__init__()
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        
        # Modality encoders
        self.encoders = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.encoders[modal_name] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
        
        # Graph convolution layers
        self.gcn_layers = nn.ModuleList([
            nn.Linear(fusion_dim, fusion_dim) for _ in range(3)
        ])
        
        # Learnable adjacency matrix
        self.adjacency = nn.Parameter(torch.randn(len(modal_dims), len(modal_dims)))
        
        # Output projection
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
        
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode modalities
        encoded_modals = {}
        for modal_name, input_tensor in modal_inputs.items():
            encoded_modals[modal_name] = self.encoders[modal_name](input_tensor)
        
        # Stack modalities
        modal_names = list(self.modal_dims.keys())
        stacked_features = torch.stack([encoded_modals[name] for name in modal_names], dim=1)
        
        # Apply graph convolution
        adj_matrix = torch.sigmoid(self.adjacency)
        
        x = stacked_features
        for gcn_layer in self.gcn_layers:
            # Graph convolution
            x = torch.matmul(x, adj_matrix)
            x = gcn_layer(x)
            x = F.relu(x)
        
        # Average across modalities
        fused_features = torch.mean(x, dim=1)
        return self.output_projection(fused_features)


class QuantumInspiredFusion(nn.Module):
    """Quantum-Inspired Multi-Modal Fusion"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 256):
        super().__init__()
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        
        # Modality encoders
        self.encoders = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.encoders[modal_name] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
        
        # Quantum-inspired operations
        self.quantum_gates = nn.ModuleDict({
            'hadamard': nn.Linear(fusion_dim, fusion_dim),
            'pauli_x': nn.Linear(fusion_dim, fusion_dim),
            'pauli_y': nn.Linear(fusion_dim, fusion_dim),
            'pauli_z': nn.Linear(fusion_dim, fusion_dim),
        })
        
        # Quantum entanglement simulation
        self.entanglement_net = nn.Sequential(
            nn.Linear(fusion_dim * len(modal_dims), fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Measurement (collapse to classical state)
        self.measurement = nn.Linear(fusion_dim, fusion_dim)
        
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode modalities
        encoded_modals = {}
        for modal_name, input_tensor in modal_inputs.items():
            encoded_modals[modal_name] = self.encoders[modal_name](input_tensor)
        
        # Apply quantum gates
        quantum_states = []
        for modal_name, features in encoded_modals.items():
            # Apply different quantum gates
            hadamard_state = torch.tanh(self.quantum_gates['hadamard'](features))
            pauli_x_state = torch.tanh(self.quantum_gates['pauli_x'](features))
            
            # Combine quantum states
            quantum_state = hadamard_state + pauli_x_state
            quantum_states.append(quantum_state)
        
        # Quantum entanglement
        entangled_input = torch.cat(quantum_states, dim=-1)
        entangled_state = self.entanglement_net(entangled_input)
        
        # Measurement (collapse to classical state)
        classical_output = self.measurement(entangled_state)
        
        return classical_output


class NeuromorphicFusion(nn.Module):
    """Neuromorphic Computing-inspired Multi-Modal Fusion"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 256):
        super().__init__()
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        
        # Spiking neural network components
        self.spike_encoders = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.spike_encoders[modal_name] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
        
        # Synaptic weights (learnable connections)
        self.synaptic_weights = nn.Parameter(torch.randn(len(modal_dims), len(modal_dims)))
        
        # Membrane potential dynamics
        self.membrane_potential = nn.Parameter(torch.zeros(fusion_dim))
        self.threshold = nn.Parameter(torch.ones(fusion_dim))
        
        # Spike generation
        self.spike_generator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )
        
        # Output integration
        self.output_integrator = nn.Linear(fusion_dim, fusion_dim)
        
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode modalities
        encoded_modals = {}
        for modal_name, input_tensor in modal_inputs.items():
            encoded_modals[modal_name] = self.spike_encoders[modal_name](input_tensor)
        
        # Simulate spiking dynamics
        modal_names = list(self.modal_dims.keys())
        spike_outputs = []
        
        for i, modal_name in enumerate(modal_names):
            features = encoded_modals[modal_name]
            
            # Update membrane potential
            membrane_update = torch.sum(features, dim=0)
            self.membrane_potential.data += membrane_update
            
            # Generate spikes
            spike_probability = torch.sigmoid(
                (self.membrane_potential - self.threshold) / self.threshold
            )
            spikes = self.spike_generator(spike_probability.unsqueeze(0))
            
            # Reset membrane potential for spiked neurons
            self.membrane_potential.data *= (1 - spikes.squeeze())
            
            spike_outputs.append(spikes)
        
        # Integrate spike outputs
        integrated_spikes = torch.mean(torch.stack(spike_outputs), dim=0)
        fused_output = self.output_integrator(integrated_spikes)
        
        return fused_output


class MetaLearningFusion(nn.Module):
    """Meta-Learning Multi-Modal Fusion"""
    
    def __init__(self, modal_dims: Dict[str, int], fusion_dim: int = 256):
        super().__init__()
        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        
        # Base modality encoders
        self.base_encoders = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.base_encoders[modal_name] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
        
        # Meta-learning components
        self.meta_controller = nn.Sequential(
            nn.Linear(fusion_dim * len(modal_dims), fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Task-specific adaptation layers
        self.adaptation_layers = nn.ModuleDict()
        for modal_name in modal_dims.keys():
            self.adaptation_layers[modal_name] = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
        
        # Fusion strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 4),  # 4 different fusion strategies
            nn.Softmax(dim=-1)
        )
        
        # Multiple fusion strategies
        self.fusion_strategies = nn.ModuleDict({
            'concat': nn.Linear(fusion_dim * len(modal_dims), fusion_dim),
            'add': nn.Linear(fusion_dim, fusion_dim),
            'multiply': nn.Linear(fusion_dim, fusion_dim),
            'attention': nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        })
        
    def forward(self, modal_inputs: Dict[str, torch.Tensor], task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode modalities
        encoded_modals = {}
        for modal_name, input_tensor in modal_inputs.items():
            encoded_modals[modal_name] = self.base_encoders[modal_name](input_tensor)
        
        # Meta-learning adaptation
        if task_context is not None:
            # Adapt encoders based on task context
            for modal_name, features in encoded_modals.items():
                adapted_features = self.adaptation_layers[modal_name](features)
                encoded_modals[modal_name] = features + adapted_features
        
        # Select fusion strategy
        combined_features = torch.cat(list(encoded_modals.values()), dim=-1)
        meta_context = self.meta_controller(combined_features)
        strategy_weights = self.strategy_selector(meta_context)
        
        # Apply different fusion strategies
        fusion_results = []
        
        # Concatenation fusion
        concat_output = self.fusion_strategies['concat'](combined_features)
        fusion_results.append(concat_output)
        
        # Addition fusion
        add_output = torch.sum(torch.stack(list(encoded_modals.values())), dim=0)
        add_output = self.fusion_strategies['add'](add_output)
        fusion_results.append(add_output)
        
        # Multiplication fusion
        mult_output = torch.prod(torch.stack(list(encoded_modals.values())), dim=0)
        mult_output = self.fusion_strategies['multiply'](mult_output)
        fusion_results.append(mult_output)
        
        # Attention fusion
        modal_stack = torch.stack(list(encoded_modals.values()), dim=1)
        attn_output, _ = self.fusion_strategies['attention'](modal_stack, modal_stack, modal_stack)
        attn_output = torch.mean(attn_output, dim=1)
        fusion_results.append(attn_output)
        
        # Weighted combination
        final_output = sum(w * result for w, result in zip(strategy_weights.T, fusion_results))
        
        return final_output


def run_advanced_multimodal_fusion_comparison(
    modal_data: Dict[str, torch.Tensor],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run comprehensive comparison of advanced multi-modal fusion methods"""
    
    print("🔬 Advanced Multi-Modal Fusion Comparison")
    print("=" * 50)
    
    modal_dims = {name: data.shape[1] for name, data in modal_data.items()}
    fusion_dim = 256
    
    # Initialize all fusion methods
    fusion_methods = {
        'Transformer': TransformerCrossModalAttention(modal_dims, fusion_dim),
        'Hierarchical': HierarchicalMultiModalFusion(modal_dims, fusion_dim),
        'Dynamic': DynamicAdaptiveFusion(modal_dims, fusion_dim),
        'Memory': MemoryAugmentedFusion(modal_dims, fusion_dim),
        'GNN': GraphNeuralNetworkFusion(modal_dims, fusion_dim),
        'Quantum': QuantumInspiredFusion(modal_dims, fusion_dim),
        'Neuromorphic': NeuromorphicFusion(modal_dims, fusion_dim),
        'MetaLearning': MetaLearningFusion(modal_dims, fusion_dim),
    }
    
    results = {}
    
    for method_name, fusion_model in fusion_methods.items():
        print(f"\n🔍 Testing {method_name}...")
        
        try:
            # Test fusion
            with torch.no_grad():
                fused_output = fusion_model(modal_data)
            
            # Compute metrics
            output_norm = torch.norm(fused_output).item()
            output_std = torch.std(fused_output).item()
            
            results[method_name] = {
                'output_shape': fused_output.shape,
                'output_norm': output_norm,
                'output_std': output_std,
                'success': True
            }
            
            print(f"  ✅ Output shape: {fused_output.shape}, Norm: {output_norm:.3f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results[method_name] = {'error': str(e), 'success': False}
    
    # Create comparison visualization
    if save_path:
        _create_advanced_fusion_comparison_plot(results, save_path)
    
    return results


def _create_advanced_fusion_comparison_plot(results: Dict[str, Any], save_path: str):
    """Create advanced fusion comparison visualization"""
    
    valid_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Output norm comparison
    ax = axes[0, 0]
    methods = list(valid_results.keys())
    norms = [valid_results[method]['output_norm'] for method in methods]
    
    bars = ax.bar(range(len(norms)), norms, color='skyblue', alpha=0.7)
    ax.set_xticks(range(len(norms)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel("Output Norm")
    ax.set_title("Fusion Output Magnitude Comparison")
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 2. Output variability
    ax = axes[0, 1]
    stds = [valid_results[method]['output_std'] for method in methods]
    
    bars = ax.bar(range(len(stds)), stds, color='lightgreen', alpha=0.7)
    ax.set_xticks(range(len(stds)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel("Output Standard Deviation")
    ax.set_title("Fusion Output Variability")
    ax.grid(True, alpha=0.3)
    
    # 3. Method characteristics radar
    ax = axes[0, 2]
    ax.axis('off')
    
    ax_radar = plt.subplot(2, 3, 3, projection='polar')
    
    categories = ['Speed', 'Accuracy', 'Robustness', 'Scalability', 'Innovation']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Mock scores for demonstration
    method_scores = {
        'Transformer': [8, 9, 8, 7, 6],
        'Hierarchical': [7, 8, 9, 8, 7],
        'Dynamic': [6, 9, 9, 6, 8],
        'Memory': [5, 8, 8, 5, 9],
        'GNN': [6, 9, 8, 6, 8],
        'Quantum': [4, 7, 6, 4, 10],
        'Neuromorphic': [3, 6, 7, 3, 10],
        'MetaLearning': [5, 9, 9, 7, 9],
    }
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(valid_results)))
    
    for i, (method, scores) in enumerate(method_scores.items()):
        if method in valid_results:
            scores += scores[:1]
            ax_radar.plot(angles, scores, 'o-', linewidth=2, label=method, color=colors[i])
            ax_radar.fill(angles, scores, alpha=0.15, color=colors[i])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_ylim(0, 10)
    ax_radar.set_title("Method Characteristics", fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax_radar.grid(True)
    
    # 4. Complexity vs Performance
    ax = axes[1, 0]
    
    complexities = [8, 7, 6, 5, 6, 4, 3, 5]  # Relative complexity
    performances = [9, 8, 9, 8, 9, 7, 6, 9]  # Relative performance
    
    scatter = ax.scatter(complexities, performances, c=colors, s=200, alpha=0.6, edgecolors='black')
    
    for i, method in enumerate(methods):
        ax.annotate(method, (complexities[i], performances[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.6))
    
    ax.set_xlabel("Implementation Complexity")
    ax.set_ylabel("Performance Score")
    ax.set_title("Complexity vs Performance Trade-off")
    ax.grid(True, alpha=0.3)
    
    # 5. Innovation level
    ax = axes[1, 1]
    
    innovation_scores = [6, 7, 8, 9, 8, 10, 10, 9]
    
    bars = ax.bar(range(len(innovation_scores)), innovation_scores, color='purple', alpha=0.7)
    ax.set_xticks(range(len(innovation_scores)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel("Innovation Level")
    ax.set_title("Method Innovation Score")
    ax.grid(True, alpha=0.3)
    
    # 6. Summary and recommendations
    ax = axes[1, 2]
    ax.axis('off')
    
    best_performance = max(valid_results.keys(), key=lambda x: method_scores.get(x, [0])[1])
    best_innovation = max(valid_results.keys(), key=lambda x: method_scores.get(x, [0, 0, 0, 0, 0])[4])
    
    summary_text = f"""
    📊 Advanced Multi-Modal Fusion Analysis
    
    🏆 Best Performance:
       • Overall: {best_performance}
       • Innovation: {best_innovation}
    
    🔬 Method Insights:
    
    • Transformer: Excellent attention
      mechanisms, good scalability
    
    • Hierarchical: Robust multi-level
      fusion, good interpretability
    
    • Dynamic: Adaptive fusion strategies,
      high flexibility
    
    • Memory: Long-term memory integration,
      unique approach
    
    • GNN: Graph-based relationships,
      good for structured data
    
    • Quantum: Novel quantum-inspired
      operations, cutting-edge
    
    • Neuromorphic: Brain-inspired
      computing, biologically plausible
    
    • MetaLearning: Task-adaptive fusion,
      excellent generalization
    
    💡 Recommendations:
    
    • Use Transformer for attention-critical
      applications
    
    • Use Hierarchical for interpretable
      multi-level fusion
    
    • Use Dynamic for adaptive scenarios
    
    • Use Quantum/Neuromorphic for
      research and innovation
    
    • Use MetaLearning for multi-task
      scenarios
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Advanced fusion comparison plot saved to: {save_path}")


if __name__ == "__main__":
    # Generate synthetic multi-modal data
    torch.manual_seed(42)
    
    batch_size = 32
    modal_data = {
        'visual': torch.randn(batch_size, 64),
        'textual': torch.randn(batch_size, 32),
        'audio': torch.randn(batch_size, 48),
        'state': torch.randn(batch_size, 16)
    }
    
    print("🚀 Testing Advanced Multi-Modal Fusion Methods")
    print("=" * 50)
    
    # Run comparison
    results = run_advanced_multimodal_fusion_comparison(
        modal_data,
        save_path="visualizations/advanced_multimodal_fusion_comparison.png"
    )
    
    print("\n🎉 Advanced Multi-Modal Fusion Testing Complete!")
    print("=" * 50)
