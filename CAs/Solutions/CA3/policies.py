import numpy as np
from typing import Dict, List, Tuple, Optional


class RandomPolicy:
    """Random policy for testing TD(0)"""

    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        """Return random valid action"""
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return None
        return np.random.choice(valid_actions)
