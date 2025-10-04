"""
Model Definitions and Utilities

This module contains model classes and utilities for storing and managing
learned policies, value functions, and Q-tables.
"""

import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import os
from datetime import datetime


class PolicyModel:
    """
    A model class for storing and managing learned policies.
    """

    def __init__(self, policy_type: str, env_params: Dict, policy_data: Dict):
        self.policy_type = policy_type
        self.env_params = env_params
        self.policy_data = policy_data
        self.created_at = datetime.now()
        self.metadata = {}

    def get_action(self, state: Tuple[int, int]) -> Optional[str]:
        """Get action for a given state"""
        return self.policy_data.get(state)

    def get_action_probabilities(self, state: Tuple[int, int]) -> Dict[str, float]:
        """Get action probabilities for a given state"""
        return self.policy_data.get(state, {})

    def save(self, filepath: str):
        """Save policy model to file"""
        model_data = {
            "policy_type": self.policy_type,
            "env_params": self.env_params,
            "policy_data": self.policy_data,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load policy model from file"""
        with open(filepath, "r") as f:
            model_data = json.load(f)

        model = cls(
            model_data["policy_type"],
            model_data["env_params"],
            model_data["policy_data"],
        )
        model.created_at = datetime.fromisoformat(model_data["created_at"])
        model.metadata = model_data.get("metadata", {})

        return model


class ValueFunctionModel:
    """
    A model class for storing and managing value functions.
    """

    def __init__(self, algorithm: str, env_params: Dict, values: Dict, gamma: float):
        self.algorithm = algorithm
        self.env_params = env_params
        self.values = values
        self.gamma = gamma
        self.created_at = datetime.now()
        self.metadata = {}

    def get_value(self, state: Tuple[int, int]) -> float:
        """Get value for a given state"""
        return self.values.get(state, 0.0)

    def get_max_value(self) -> float:
        """Get maximum value across all states"""
        return max(self.values.values()) if self.values else 0.0

    def get_min_value(self) -> float:
        """Get minimum value across all states"""
        return min(self.values.values()) if self.values else 0.0

    def save(self, filepath: str):
        """Save value function model to file"""
        model_data = {
            "algorithm": self.algorithm,
            "env_params": self.env_params,
            "values": {str(k): v for k, v in self.values.items()},
            "gamma": self.gamma,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load value function model from file"""
        with open(filepath, "r") as f:
            model_data = json.load(f)

        # Convert string keys back to tuples
        values = {eval(k): v for k, v in model_data["values"].items()}

        model = cls(
            model_data["algorithm"],
            model_data["env_params"],
            values,
            model_data["gamma"],
        )
        model.created_at = datetime.fromisoformat(model_data["created_at"])
        model.metadata = model_data.get("metadata", {})

        return model


class QTableModel:
    """
    A model class for storing and managing Q-tables.
    """

    def __init__(
        self, algorithm: str, env_params: Dict, q_table: Dict, learning_params: Dict
    ):
        self.algorithm = algorithm
        self.env_params = env_params
        self.q_table = q_table
        self.learning_params = learning_params
        self.created_at = datetime.now()
        self.metadata = {}

    def get_q_value(self, state: Tuple[int, int], action: str) -> float:
        """Get Q-value for a given state-action pair"""
        return self.q_table.get((state, action), 0.0)

    def get_best_action(
        self, state: Tuple[int, int], valid_actions: List[str]
    ) -> Optional[str]:
        """Get best action for a given state"""
        if not valid_actions:
            return None

        best_action = None
        best_value = float("-inf")

        for action in valid_actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action

        return best_action

    def get_state_values(
        self, state: Tuple[int, int], valid_actions: List[str]
    ) -> Dict[str, float]:
        """Get Q-values for all valid actions in a state"""
        return {action: self.get_q_value(state, action) for action in valid_actions}

    def save(self, filepath: str):
        """Save Q-table model to file"""
        model_data = {
            "algorithm": self.algorithm,
            "env_params": self.env_params,
            "q_table": {str(k): v for k, v in self.q_table.items()},
            "learning_params": self.learning_params,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load Q-table model from file"""
        with open(filepath, "r") as f:
            model_data = json.load(f)

        # Convert string keys back to tuples
        q_table = {eval(k): v for k, v in model_data["q_table"].items()}

        model = cls(
            model_data["algorithm"],
            model_data["env_params"],
            q_table,
            model_data["learning_params"],
        )
        model.created_at = datetime.fromisoformat(model_data["created_at"])
        model.metadata = model_data.get("metadata", {})

        return model


class ModelManager:
    """
    A manager class for handling multiple models and their persistence.
    """

    def __init__(self, base_dir: str = "models"):
        self.base_dir = base_dir
        self.models = {}

        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

    def save_model(self, model: Any, name: str, model_type: str = "generic"):
        """Save a model with a given name and type"""
        model_dir = os.path.join(self.base_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)

        filepath = os.path.join(model_dir, f"{name}.json")
        model.save(filepath)

        self.models[name] = {"type": model_type, "filepath": filepath, "model": model}

    def load_model(self, name: str, model_type: str = "generic"):
        """Load a model by name and type"""
        if name in self.models:
            return self.models[name]["model"]

        model_dir = os.path.join(self.base_dir, model_type)
        filepath = os.path.join(model_dir, f"{name}.json")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model {name} not found")

        if model_type == "policy":
            model = PolicyModel.load(filepath)
        elif model_type == "value_function":
            model = ValueFunctionModel.load(filepath)
        elif model_type == "q_table":
            model = QTableModel.load(filepath)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.models[name] = {"type": model_type, "filepath": filepath, "model": model}

        return model

    def list_models(self, model_type: Optional[str] = None):
        """List all available models"""
        if model_type:
            model_dir = os.path.join(self.base_dir, model_type)
            if os.path.exists(model_dir):
                return [f[:-5] for f in os.listdir(model_dir) if f.endswith(".json")]
            return []
        else:
            all_models = {}
            for model_type in ["policy", "value_function", "q_table"]:
                all_models[model_type] = self.list_models(model_type)
            return all_models

    def delete_model(self, name: str, model_type: str = "generic"):
        """Delete a model"""
        if name in self.models:
            del self.models[name]

        model_dir = os.path.join(self.base_dir, model_type)
        filepath = os.path.join(model_dir, f"{name}.json")

        if os.path.exists(filepath):
            os.remove(filepath)


def create_model_from_policy(
    policy, env, algorithm_name: str = "unknown"
) -> PolicyModel:
    """Create a PolicyModel from a policy object"""
    policy_data = {}

    for state in env.states:
        if state not in env.obstacles and state != env.goal_state:
            action = policy.get_action(state)
            if action:
                policy_data[state] = action

    env_params = {
        "size": env.size,
        "goal_state": env.goal_state,
        "start_state": env.start_state,
        "obstacles": env.obstacles,
        "goal_reward": env.goal_reward,
        "step_reward": env.step_reward,
        "obstacle_reward": env.obstacle_reward,
    }

    return PolicyModel(algorithm_name, env_params, policy_data)


def create_model_from_values(
    values: Dict, env, algorithm_name: str = "unknown", gamma: float = 0.9
) -> ValueFunctionModel:
    """Create a ValueFunctionModel from a value function"""
    env_params = {
        "size": env.size,
        "goal_state": env.goal_state,
        "start_state": env.start_state,
        "obstacles": env.obstacles,
        "goal_reward": env.goal_reward,
        "step_reward": env.step_reward,
        "obstacle_reward": env.obstacle_reward,
    }

    return ValueFunctionModel(algorithm_name, env_params, values, gamma)


def create_model_from_q_table(
    q_table: Dict, env, algorithm_name: str = "unknown", learning_params: Dict = None
) -> QTableModel:
    """Create a QTableModel from a Q-table"""
    if learning_params is None:
        learning_params = {}

    env_params = {
        "size": env.size,
        "goal_state": env.goal_state,
        "start_state": env.start_state,
        "obstacles": env.obstacles,
        "goal_reward": env.goal_reward,
        "step_reward": env.step_reward,
        "obstacle_reward": env.obstacle_reward,
    }

    return QTableModel(algorithm_name, env_params, q_table, learning_params)


if __name__ == "__main__":
    # Example usage
    from environments.environments import GridWorld
    from agents.policies import RandomPolicy

    env = GridWorld()
    policy = RandomPolicy(env)

    # Create and save a policy model
    policy_model = create_model_from_policy(policy, env, "random_policy")

    manager = ModelManager()
    manager.save_model(policy_model, "test_random", "policy")

    # Load the model back
    loaded_model = manager.load_model("test_random", "policy")

    print("Model saved and loaded successfully!")
    print(f"Available models: {manager.list_models()}")
