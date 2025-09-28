"""
Symbolic Knowledge Base

This module contains symbolic reasoning components for neurosymbolic RL:
- Logical predicates and rules
- Knowledge base management
- Symbolic inference
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import itertools

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LogicalPredicate:
    """Represents a logical predicate in first-order logic."""

    def __init__(self, name: str, arity: int, domain: Optional[List[Any]] = None):
        self.name = name
        self.arity = arity
        self.domain = domain or []

        # Ground all possible instantiations if domain is provided
        if domain:
            self.groundings = list(itertools.product(domain, repeat=arity))
        else:
            self.groundings = []

    def __call__(self, *args):
        """Evaluate predicate on given arguments."""
        if len(args) != self.arity:
            raise ValueError(
                f"Predicate {self.name} expects {self.arity} arguments, got {len(args)}"
            )

        # For now, return symbolic representation
        return SymbolicAtom(self, args)

    def __str__(self):
        return f"{self.name}/{self.arity}"

    def __repr__(self):
        return f"LogicalPredicate({self.name}, {self.arity})"


class SymbolicAtom:
    """Represents a ground atom (predicate applied to constants)."""

    def __init__(self, predicate: LogicalPredicate, args: Tuple):
        self.predicate = predicate
        self.args = tuple(args)

        if len(args) != predicate.arity:
            raise ValueError(f"Atom arity mismatch: {len(args)} vs {predicate.arity}")

    def __eq__(self, other):
        return (
            isinstance(other, SymbolicAtom)
            and self.predicate.name == other.predicate.name
            and self.args == other.args
        )

    def __hash__(self):
        return hash((self.predicate.name, self.args))

    def __str__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.predicate.name}({args_str})"

    def __repr__(self):
        return f"SymbolicAtom({self.predicate.name}, {self.args})"


class LogicalRule:
    """Represents a logical rule (Horn clause)."""

    def __init__(self, head: SymbolicAtom, body: List[SymbolicAtom]):
        self.head = head
        self.body = body

    def __str__(self):
        body_str = " ∧ ".join(str(atom) for atom in self.body)
        return f"{self.head} ← {body_str}"

    def __repr__(self):
        return f"LogicalRule({self.head}, {self.body})"


class SymbolicKnowledgeBase:
    """Symbolic knowledge base with inference capabilities."""

    def __init__(self):
        self.predicates: Dict[str, LogicalPredicate] = {}
        self.facts: Set[SymbolicAtom] = set()
        self.rules: List[LogicalRule] = []

        # Derived facts from inference
        self.derived_facts: Set[SymbolicAtom] = set()

    def add_predicate(self, predicate: LogicalPredicate):
        """Add a predicate to the knowledge base."""
        self.predicates[predicate.name] = predicate

    def add_fact(self, fact: SymbolicAtom):
        """Add a ground fact to the knowledge base."""
        self.facts.add(fact)

    def add_rule(self, rule: LogicalRule):
        """Add a logical rule to the knowledge base."""
        self.rules.append(rule)

    def query(self, atom: SymbolicAtom) -> bool:
        """Query whether an atom is true in the knowledge base."""
        # Check direct facts
        if atom in self.facts:
            return True

        # Check derived facts
        if atom in self.derived_facts:
            return True

        # Try inference
        return self._infer(atom)

    def _infer(self, query_atom: SymbolicAtom) -> bool:
        """Perform backward chaining inference."""
        # Simple backward chaining for Horn clauses
        for rule in self.rules:
            if rule.head.predicate.name == query_atom.predicate.name:
                # Try to unify
                if self._unify_atoms(rule.head, query_atom):
                    # Check if body is satisfied
                    if all(self.query(atom) for atom in rule.body):
                        self.derived_facts.add(query_atom)
                        return True

        return False

    def _unify_atoms(self, atom1: SymbolicAtom, atom2: SymbolicAtom) -> bool:
        """Check if two atoms can be unified."""
        if atom1.predicate.name != atom2.predicate.name:
            return False

        # For now, simple equality check (no variable unification)
        return atom1.args == atom2.args

    def get_all_facts(self) -> Set[SymbolicAtom]:
        """Get all facts (direct + derived)."""
        all_facts = self.facts.union(self.derived_facts)
        return all_facts

    def clear_derived_facts(self):
        """Clear derived facts (useful for incremental inference)."""
        self.derived_facts.clear()

    def __str__(self):
        lines = ["Knowledge Base:"]
        lines.append("Predicates:")
        for pred in self.predicates.values():
            lines.append(f"  {pred}")

        lines.append("Facts:")
        for fact in self.facts:
            lines.append(f"  {fact}")

        lines.append("Rules:")
        for rule in self.rules:
            lines.append(f"  {rule}")

        return "\n".join(lines)


class PrologStyleKB:
    """Prolog-style knowledge base with more advanced inference."""

    def __init__(self):
        self.kb = SymbolicKnowledgeBase()
        self.variable_counter = 0

    def assert_fact(self, predicate_name: str, *args):
        """Assert a fact in Prolog style: assert_fact('parent', 'alice', 'bob')."""
        if predicate_name not in self.kb.predicates:
            # Infer arity from arguments
            arity = len(args)
            predicate = LogicalPredicate(predicate_name, arity)
            self.kb.add_predicate(predicate)

        predicate = self.kb.predicates[predicate_name]
        atom = SymbolicAtom(predicate, args)
        self.kb.add_fact(atom)

    def assert_rule(
        self, head_pred: str, head_args: List, body: List[Tuple[str, List]]
    ):
        """Assert a rule in Prolog style.

        Example: assert_rule('ancestor', ['X', 'Y'], [('parent', ['X', 'Y']), ('parent', ['X', 'Z'])])
        """
        # Create head atom
        if head_pred not in self.kb.predicates:
            arity = len(head_args)
            predicate = LogicalPredicate(head_pred, arity)
            self.kb.add_predicate(predicate)

        head_predicate = self.kb.predicates[head_pred]
        head_atom = SymbolicAtom(head_predicate, head_args)

        # Create body atoms
        body_atoms = []
        for body_pred, body_args in body:
            if body_pred not in self.kb.predicates:
                arity = len(body_args)
                predicate = LogicalPredicate(body_pred, arity)
                self.kb.add_predicate(predicate)

            body_predicate = self.kb.predicates[body_pred]
            body_atom = SymbolicAtom(body_predicate, body_args)
            body_atoms.append(body_atom)

        rule = LogicalRule(head_atom, body_atoms)
        self.kb.add_rule(rule)

    def query(self, predicate_name: str, *args) -> List[Dict]:
        """Query the knowledge base and return variable bindings."""
        # For now, simple implementation without full unification
        if predicate_name not in self.kb.predicates:
            return []

        predicate = self.kb.predicates[predicate_name]
        query_atom = SymbolicAtom(predicate, args)

        if self.kb.query(query_atom):
            return [{}]  # Empty binding dict for ground queries
        else:
            return []


class RLKnowledgeBase(SymbolicKnowledgeBase):
    """Knowledge base specialized for RL domains."""

    def __init__(self):
        super().__init__()

        # RL-specific predicates
        self._init_rl_predicates()

    def _init_rl_predicates(self):
        """Initialize common RL predicates."""
        # State predicates
        self.add_predicate(LogicalPredicate("at", 2))  # at(agent, location)
        self.add_predicate(LogicalPredicate("has", 2))  # has(agent, object)
        self.add_predicate(LogicalPredicate("adjacent", 2))  # adjacent(loc1, loc2)

        # Action predicates
        self.add_predicate(LogicalPredicate("safe", 1))  # safe(action)
        self.add_predicate(LogicalPredicate("legal", 2))  # legal(action, state)

        # Goal predicates
        self.add_predicate(LogicalPredicate("goal", 1))  # goal(state)
        self.add_predicate(LogicalPredicate("rewarding", 2))  # rewarding(action, state)

    def add_state_facts(self, state_dict: Dict):
        """Add facts from a state dictionary."""
        for key, value in state_dict.items():
            if isinstance(value, list):
                for item in value:
                    self.add_fact(self.predicates["has"](key, item))
            else:
                self.add_fact(self.predicates["at"](key, value))

    def is_safe_action(self, action, state_dict: Dict) -> bool:
        """Check if an action is safe in the given state."""
        self.clear_derived_facts()
        self.add_state_facts(state_dict)

        safe_atom = self.predicates["safe"](action)
        return self.query(safe_atom)

    def get_legal_actions(self, state_dict: Dict) -> List[str]:
        """Get all legal actions in the given state."""
        self.clear_derived_facts()
        self.add_state_facts(state_dict)

        legal_actions = []
        # This would need to be extended based on domain-specific actions
        possible_actions = ["move", "pickup", "drop", "use"]

        for action in possible_actions:
            legal_atom = self.predicates["legal"](action, "current_state")
            if self.query(legal_atom):
                legal_actions.append(action)

        return legal_actions
