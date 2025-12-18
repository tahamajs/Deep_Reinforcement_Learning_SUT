"""Minimal LightZero / mu-zero adapter scaffold.

This file provides a lightweight adapter that demonstrates how a policy+value model
could be exposed to LightZero-style MCTS integration. It is intentionally minimal and
serves as a starting point for a full integration.
"""

from typing import Any, Dict


class MuZeroAdapter:
    """Adapter exposing prediction and dynamics API expected by many mu-zero implementations.

    Methods:
      - initial_inference(observation) -> {"policy": priors, "value": value, "hidden_state": ...}
      - recurrent_inference(hidden_state, action) -> {"policy": priors, "value": value, "next_state": ...}

    This adapter wraps a model providing `policy(state)` and `value(state)` used by PUCT.
    For richer integrations the adapter can be extended to return learned hidden states,
    rewards, and recurrent dynamics outputs compatible with LightZero / mu-zero interfaces.
    """

    def __init__(self, model):
        self.model = model

    def initial_inference(self, observation: Any) -> Dict[str, Any]:
        # observation is a raw state; pass to model
        priors = self.model.policy(observation)
        value = self.model.value(observation)
        # For LightZero-like APIs we return priors under key 'policy' and include a hidden_state
        # which for simple models can be the raw observation or an embedding.
        return {"policy": priors, "value": value, "hidden_state": observation}

    def recurrent_inference(self, hidden_state: Any, action: int) -> Dict[str, Any]:
        # For simple toy models without learned dynamics, step the environment externally
        # and call the model on the resulting state. Here we assume hidden_state encodes
        # the integer state and action is an index mapped to a delta (0->-1,1->0,2->+1).
        next_state = hidden_state + (action - 1)
        priors = self.model.policy(next_state)
        value = self.model.value(next_state)
        return {
            "policy": priors,
            "value": value,
            "next_state": next_state,
            "hidden_state": next_state,
        }














