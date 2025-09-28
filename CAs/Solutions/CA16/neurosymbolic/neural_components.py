"""
Neural Components for Neurosymbolic RL

This module contains neural network components that interface with symbolic reasoning:
- Neural perception modules
- Symbolic reasoning networks
- Hybrid neural-symbolic architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .knowledge_base import SymbolicKnowledgeBase, SymbolicAtom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralPerceptionModule(nn.Module):
    """Neural network for processing perceptual inputs."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.attention = nn.MultiheadAttention(
            output_dim, num_heads=4, batch_first=True
        )

        self.symbol_grounding = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process perceptual input."""
        features = self.encoder(x)

        attended_features, _ = self.attention(
            features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)

        symbolic_features = self.symbol_grounding(attended_features)

        return symbolic_features


class SymbolicReasoningModule(nn.Module):
    """Neural module for symbolic reasoning."""

    def __init__(
        self, symbol_dim: int = 64, hidden_dim: int = 128, num_symbols: int = 100
    ):
        super().__init__()

        self.symbol_dim = symbol_dim
        self.num_symbols = num_symbols

        self.symbol_embeddings = nn.Embedding(num_symbols, symbol_dim)

        self.predicate_network = nn.Sequential(
            nn.Linear(symbol_dim * 2, hidden_dim),  # subject + object
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # predicate truth value
            nn.Sigmoid(),
        )

        self.rule_network = nn.Sequential(
            nn.Linear(symbol_dim * 3, hidden_dim),  # premise1 + premise2 + conclusion
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # rule confidence
            nn.Sigmoid(),
        )

        self.inference_network = nn.GRU(symbol_dim, hidden_dim, batch_first=True)

        self.logic_and = nn.Sequential(nn.Linear(symbol_dim * 2, symbol_dim), nn.Tanh())
        self.logic_or = nn.Sequential(nn.Linear(symbol_dim * 2, symbol_dim), nn.Tanh())
        self.logic_not = nn.Sequential(nn.Linear(symbol_dim, symbol_dim), nn.Tanh())

    def forward(
        self, symbolic_input: torch.Tensor, reasoning_steps: int = 3
    ) -> torch.Tensor:
        """Perform symbolic reasoning."""
        batch_size = symbolic_input.shape[0]

        reasoning_state = symbolic_input

        for step in range(reasoning_steps):
            output, hidden = self.inference_network(reasoning_state.unsqueeze(1))
            reasoning_state = output.squeeze(1)

            reasoning_state = self._apply_logic_operations(reasoning_state)

        return reasoning_state

    def _apply_logic_operations(self, x: torch.Tensor) -> torch.Tensor:
        """Apply logical operations to reasoning state."""
        split_size = x.shape[-1] // 3

        part1 = x[:, :split_size]
        part2 = x[:, split_size : 2 * split_size]
        part3 = x[:, 2 * split_size :]

        and_result = self.logic_and(torch.cat([part1, part2], dim=-1))

        or_result = self.logic_or(torch.cat([and_result, part3], dim=-1))

        not_result = self.logic_not(or_result)

        return not_result

    def evaluate_predicate(
        self,
        subject_embedding: torch.Tensor,
        object_embedding: torch.Tensor,
        predicate_idx: int,
    ) -> torch.Tensor:
        """Evaluate a predicate between subject and object."""
        predicate_emb = self.symbol_embeddings(
            torch.tensor([predicate_idx], device=device)
        )

        combined = torch.cat([subject_embedding, object_embedding], dim=-1)
        truth_value = self.predicate_network(combined)

        return truth_value.squeeze()

    def evaluate_rule(
        self,
        premise1_emb: torch.Tensor,
        premise2_emb: torch.Tensor,
        conclusion_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate confidence in a logical rule."""
        combined = torch.cat([premise1_emb, premise2_emb, conclusion_emb], dim=-1)
        confidence = self.rule_network(combined)

        return confidence.squeeze()


class NeuralSymbolicInterface:
    """Interface between neural and symbolic components."""

    def __init__(
        self,
        perception_module: NeuralPerceptionModule,
        reasoning_module: SymbolicReasoningModule,
        knowledge_base: SymbolicKnowledgeBase,
    ):
        self.perception = perception_module
        self.reasoning = reasoning_module
        self.kb = knowledge_base

        self.symbol_to_idx = {}
        self.idx_to_symbol = {}
        self.next_symbol_idx = 0

    def perceive_and_reason(
        self, observation: torch.Tensor, query_atoms: List[SymbolicAtom]
    ) -> Dict[str, torch.Tensor]:
        """Process observation and perform reasoning about queries."""
        symbolic_features = self.perception(observation)

        query_embeddings = []
        for atom in query_atoms:
            atom_embedding = self._atom_to_embedding(atom)
            query_embeddings.append(atom_embedding)

        if query_embeddings:
            query_tensor = torch.stack(query_embeddings)
        else:
            query_tensor = torch.empty(0, self.reasoning.symbol_dim, device=device)

        reasoned_features = self.reasoning(
            symbolic_features.unsqueeze(0), query_tensor.unsqueeze(0)
        )

        results = {}
        for i, atom in enumerate(query_atoms):
            if len(reasoned_features) > 0:
                query_emb = self._atom_to_embedding(atom)
                truth_value = self.reasoning.evaluate_predicate(
                    symbolic_features, query_emb, 0  # Using dummy predicate idx
                )
                results[str(atom)] = truth_value
            else:
                results[str(atom)] = torch.tensor(
                    0.5, device=device
                )  # Default uncertainty

        return {
            "perceptual_features": symbolic_features,
            "reasoned_features": (
                reasoned_features.squeeze(0)
                if reasoned_features.numel() > 0
                else symbolic_features
            ),
            "query_results": results,
        }

    def _atom_to_embedding(self, atom: SymbolicAtom) -> torch.Tensor:
        """Convert symbolic atom to embedding."""
        predicate_key = f"pred_{atom.predicate.name}"

        if predicate_key not in self.symbol_to_idx:
            self.symbol_to_idx[predicate_key] = self.next_symbol_idx
            self.idx_to_symbol[self.next_symbol_idx] = predicate_key
            self.next_symbol_idx += 1

        predicate_idx = self.symbol_to_idx[predicate_key]

        arg_embeddings = []
        for arg in atom.args:
            arg_key = f"arg_{arg}"
            if arg_key not in self.symbol_to_idx:
                self.symbol_to_idx[arg_key] = self.next_symbol_idx
                self.idx_to_symbol[self.next_symbol_idx] = arg_key
                self.next_symbol_idx += 1

            arg_idx = self.symbol_to_idx[arg_key]
            arg_emb = self.reasoning.symbol_embeddings(
                torch.tensor([arg_idx], device=device)
            )
            arg_embeddings.append(arg_emb)

        if arg_embeddings:
            combined = torch.cat(
                [
                    self.reasoning.symbol_embeddings(
                        torch.tensor([predicate_idx], device=device)
                    )
                ]
                + arg_embeddings,
                dim=-1,
            )
            embedding = combined.mean(dim=0, keepdim=True)
        else:
            embedding = self.reasoning.symbol_embeddings(
                torch.tensor([predicate_idx], device=device)
            )

        return embedding.squeeze()


class DifferentiableReasoningLayer(nn.Module):
    """Differentiable layer for logical reasoning."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.logic_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output logic value
        )

        self.and_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.or_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.not_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, premises: List[torch.Tensor]) -> torch.Tensor:
        """Perform differentiable logical reasoning."""
        if not premises:
            return torch.tensor(0.0, device=device)

        combined = torch.stack(premises).mean(dim=0)

        and_result = F.linear(combined, self.and_weights)
        or_result = F.linear(combined, self.or_weights)
        not_result = F.linear(combined, self.not_weights)

        logic_output = (and_result + or_result - not_result) / 3

        final_output = self.logic_mlp(logic_output)

        return final_output.squeeze()


class NeuroSymbolicEncoder(nn.Module):
    """Encoder that combines neural and symbolic representations."""

    def __init__(self, input_dim: int, symbol_dim: int = 64, hidden_dim: int = 128):
        super().__init__()

        self.neural_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, symbol_dim),
        )

        self.symbolic_encoder = nn.Sequential(
            nn.Linear(symbol_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, symbol_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(symbol_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, symbol_dim),
        )

        self.attention = nn.MultiheadAttention(
            symbol_dim, num_heads=4, batch_first=True
        )

    def forward(
        self, neural_input: torch.Tensor, symbolic_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode neural and symbolic inputs together."""
        neural_features = self.neural_encoder(neural_input)

        if symbolic_input is not None:
            symbolic_features = self.symbolic_encoder(symbolic_input)

            combined = torch.cat([neural_features, symbolic_features], dim=-1)
            fused = self.fusion(combined)

            attended, _ = self.attention(
                fused.unsqueeze(1), fused.unsqueeze(1), fused.unsqueeze(1)
            )
            output = attended.squeeze(1)
        else:
            output = neural_features

        return output
