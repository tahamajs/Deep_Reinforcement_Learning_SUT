"""
Interpretability and Explainability for Neurosymbolic RL

This module provides tools for understanding and explaining neurosymbolic policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from .knowledge_base import SymbolicKnowledgeBase, LogicalRule
from .policies import NeurosymbolicPolicy


class AttentionExplainer:
    """Explains decisions using attention mechanisms."""
    
    def __init__(self, policy: NeurosymbolicPolicy):
        self.policy = policy
    
    def explain_decision(self, state: torch.Tensor) -> Dict[str, Any]:
        """Explain a single decision using attention weights."""
        with torch.no_grad():
            _, _, interpretability_info = self.policy(state.unsqueeze(0))
            
            # Extract attention information
            attention_weights = interpretability_info['attention_weights']
            attention_probs = interpretability_info['attention_probs']
            
            # Analyze attention patterns
            explanation = {
                'neural_attention': attention_weights[0].squeeze().cpu().numpy(),
                'symbolic_attention': attention_probs.cpu().numpy(),
                'decision_factors': self._identify_decision_factors(attention_weights),
                'confidence': self._compute_confidence(attention_weights)
            }
            
            return explanation
    
    def _identify_decision_factors(self, attention_weights: torch.Tensor) -> List[str]:
        """Identify key factors influencing the decision."""
        factors = []
        
        # Analyze neural attention
        neural_attention = attention_weights[0].squeeze()
        if neural_attention is not None:
            top_indices = torch.topk(neural_attention, k=3).indices
            factors.extend([f"neural_feature_{i}" for i in top_indices.cpu().numpy()])
        
        # Analyze symbolic attention
        symbolic_attention = attention_weights[1] if len(attention_weights) > 1 else None
        if symbolic_attention is not None:
            factors.append("symbolic_reasoning")
        
        return factors
    
    def _compute_confidence(self, attention_weights: torch.Tensor) -> float:
        """Compute confidence in the decision."""
        # Simple confidence based on attention concentration
        neural_attention = attention_weights[0].squeeze()
        if neural_attention is not None:
            entropy = -torch.sum(neural_attention * torch.log(neural_attention + 1e-8))
            confidence = 1.0 - entropy / torch.log(torch.tensor(neural_attention.shape[0]))
            return confidence.item()
        return 0.5


class RuleExtractor:
    """Extracts logical rules from neural networks."""
    
    def __init__(self, policy: NeurosymbolicPolicy, knowledge_base: SymbolicKnowledgeBase):
        self.policy = policy
        self.knowledge_base = knowledge_base
    
    def extract_rules(self, states: torch.Tensor, actions: torch.Tensor, 
                     threshold: float = 0.8) -> List[LogicalRule]:
        """Extract logical rules from policy behavior."""
        extracted_rules = []
        
        with torch.no_grad():
            # Get policy outputs
            action_logits, _, interpretability_info = self.policy(states)
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Analyze decision patterns
            decision_patterns = self._analyze_decision_patterns(states, actions, action_probs)
            
            # Extract rules from patterns
            for pattern in decision_patterns:
                if pattern['confidence'] > threshold:
                    rule = self._pattern_to_rule(pattern)
                    if rule:
                        extracted_rules.append(rule)
        
        return extracted_rules
    
    def _analyze_decision_patterns(self, states: torch.Tensor, actions: torch.Tensor,
                                 action_probs: torch.Tensor) -> List[Dict[str, Any]]:
        """Analyze patterns in decision making."""
        patterns = []
        
        for i in range(states.shape[0]):
            state = states[i]
            action = actions[i]
            probs = action_probs[i]
            
            # Extract state features
            state_features = self._extract_state_features(state)
            
            # Analyze action selection
            action_confidence = probs[action].item()
            
            pattern = {
                'state_features': state_features,
                'action': action.item(),
                'confidence': action_confidence,
                'probabilities': probs.cpu().numpy()
            }
            
            patterns.append(pattern)
        
        return patterns
    
    def _extract_state_features(self, state: torch.Tensor) -> Dict[str, float]:
        """Extract meaningful features from state."""
        features = {}
        
        # Convert state to numpy for analysis
        state_np = state.cpu().numpy()
        
        # Extract basic features
        features['state_norm'] = float(torch.norm(state).item())
        features['state_mean'] = float(torch.mean(state).item())
        features['state_std'] = float(torch.std(state).item())
        
        # Extract position-based features (assuming 2D state)
        if len(state_np) >= 2:
            features['x_position'] = float(state_np[0])
            features['y_position'] = float(state_np[1])
        
        return features
    
    def _pattern_to_rule(self, pattern: Dict[str, Any]) -> Optional[LogicalRule]:
        """Convert a decision pattern to a logical rule."""
        # Simplified rule extraction
        # In practice, this would be more sophisticated
        
        state_features = pattern['state_features']
        action = pattern['action']
        confidence = pattern['confidence']
        
        if confidence < 0.8:
            return None
        
        # Create a simple rule based on state features
        # This is a placeholder implementation
        from .knowledge_base import LogicalPredicate, LogicalRule
        
        # Create predicates
        action_predicate = LogicalPredicate(f"action_{action}", 0)
        state_predicate = LogicalPredicate("state_condition", 1)
        
        # Create rule
        rule = LogicalRule(action_predicate, [state_predicate], weight=confidence)
        
        return rule


class CausalAnalyzer:
    """Analyzes causal relationships in neurosymbolic policies."""
    
    def __init__(self, policy: NeurosymbolicPolicy):
        self.policy = policy
    
    def analyze_causality(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
        """Analyze causal relationships between states and actions."""
        causal_analysis = {
            'causal_strength': {},
            'causal_graph': {},
            'interventions': {}
        }
        
        with torch.no_grad():
            # Get policy outputs
            action_logits, _, interpretability_info = self.policy(states)
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Analyze causal relationships
            for i in range(states.shape[1]):  # For each state dimension
                causal_strength = self._compute_causal_strength(states, actions, i)
                causal_analysis['causal_strength'][f'state_dim_{i}'] = causal_strength
            
            # Analyze interventions
            causal_analysis['interventions'] = self._analyze_interventions(states, actions)
        
        return causal_analysis
    
    def _compute_causal_strength(self, states: torch.Tensor, actions: torch.Tensor, 
                               state_dim: int) -> float:
        """Compute causal strength for a state dimension."""
        # Simple causal strength computation
        # In practice, this would use more sophisticated methods
        
        # Compute correlation between state dimension and actions
        state_values = states[:, state_dim]
        action_values = actions.float()
        
        # Compute correlation
        correlation = torch.corrcoef(torch.stack([state_values, action_values]))[0, 1]
        
        return correlation.item() if not torch.isnan(correlation) else 0.0
    
    def _analyze_interventions(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
        """Analyze the effects of interventions."""
        interventions = {}
        
        with torch.no_grad():
            # Original predictions
            original_logits, _, _ = self.policy(states)
            original_probs = F.softmax(original_logits, dim=-1)
            
            # Test interventions on each state dimension
            for i in range(states.shape[1]):
                # Create intervened states
                intervened_states = states.clone()
                intervened_states[:, i] = 0.0  # Set to zero
                
                # Get predictions for intervened states
                intervened_logits, _, _ = self.policy(intervened_states)
                intervened_probs = F.softmax(intervened_logits, dim=-1)
                
                # Compute intervention effect
                effect = torch.mean(torch.abs(original_probs - intervened_probs))
                interventions[f'intervention_dim_{i}'] = effect.item()
        
        return interventions


class CounterfactualReasoner:
    """Generates counterfactual explanations for decisions."""
    
    def __init__(self, policy: NeurosymbolicPolicy):
        self.policy = policy
    
    def generate_counterfactuals(self, state: torch.Tensor, action: int,
                               num_counterfactuals: int = 5) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations."""
        counterfactuals = []
        
        with torch.no_grad():
            # Original prediction
            original_logits, _, _ = self.policy(state.unsqueeze(0))
            original_probs = F.softmax(original_logits, dim=-1)
            original_action_prob = original_probs[0, action].item()
            
            # Generate counterfactual states
            for i in range(num_counterfactuals):
                counterfactual_state = self._generate_counterfactual_state(state, i)
                
                # Get prediction for counterfactual state
                counterfactual_logits, _, _ = self.policy(counterfactual_state.unsqueeze(0))
                counterfactual_probs = F.softmax(counterfactual_logits, dim=-1)
                counterfactual_action_prob = counterfactual_probs[0, action].item()
                
                # Compute counterfactual effect
                effect = counterfactual_action_prob - original_action_prob
                
                counterfactual = {
                    'original_state': state.cpu().numpy(),
                    'counterfactual_state': counterfactual_state.cpu().numpy(),
                    'original_prob': original_action_prob,
                    'counterfactual_prob': counterfactual_action_prob,
                    'effect': effect,
                    'explanation': self._generate_explanation(state, counterfactual_state, effect)
                }
                
                counterfactuals.append(counterfactual)
        
        return counterfactuals
    
    def _generate_counterfactual_state(self, state: torch.Tensor, index: int) -> torch.Tensor:
        """Generate a counterfactual state."""
        counterfactual_state = state.clone()
        
        # Modify different dimensions based on index
        dim_to_modify = index % state.shape[0]
        modification = 0.1 * (index + 1)  # Small modification
        
        counterfactual_state[dim_to_modify] += modification
        
        return counterfactual_state
    
    def _generate_explanation(self, original_state: torch.Tensor, 
                            counterfactual_state: torch.Tensor, 
                            effect: float) -> str:
        """Generate a textual explanation for the counterfactual."""
        # Find the dimension that changed the most
        diff = torch.abs(counterfactual_state - original_state)
        max_diff_dim = torch.argmax(diff).item()
        max_diff_value = diff[max_diff_dim].item()
        
        if effect > 0:
            explanation = f"Increasing state dimension {max_diff_dim} by {max_diff_value:.3f} would increase the probability of this action by {effect:.3f}."
        else:
            explanation = f"Increasing state dimension {max_diff_dim} by {max_diff_value:.3f} would decrease the probability of this action by {abs(effect):.3f}."
        
        return explanation
