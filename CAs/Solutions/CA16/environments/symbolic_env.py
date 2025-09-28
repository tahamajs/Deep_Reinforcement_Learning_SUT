"""
Symbolic Grid World Environment

This module provides a symbolic grid world environment for neurosymbolic RL.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict


class SymbolicGridWorld(gym.Env):
    """Symbolic grid world environment for testing neurosymbolic RL."""

    def __init__(self, size: int = 8, num_symbols: int = 5, max_steps: int = 50):
        super().__init__()

        self.size = size
        self.num_symbols = num_symbols
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.MultiDiscrete([size, size]),
                "symbol_positions": spaces.MultiDiscrete([size, size, num_symbols + 1]),
                "facts": spaces.Sequence(spaces.Text(max_length=50)),  # Logical facts
            }
        )

        self.agent_pos = None
        self.symbol_positions = {}  # symbol_id -> (x, y)
        self.step_count = 0

        self.symbols = [f"symbol_{i}" for i in range(num_symbols)]
        self.goals = []  # Goal conditions
        self.rules = []  # Environment rules

        self._init_goals_and_rules()

    def _init_goals_and_rules(self):
        """Initialize symbolic goals and rules."""
        self.goals = [
            "collected(symbol_0)",
            "collected(symbol_1)",
            "at(agent, (7,7))",  # Reach corner
        ]

        self.rules = [
            "adjacent(X,Y) -> can_move_to(X,Y)",
            "has_key(agent) -> can_open_door(door)",
            "collected(symbol_X) -> goal_achieved(symbol_X)",
        ]

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        self.agent_pos = (np.random.randint(self.size), np.random.randint(self.size))

        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        np.random.shuffle(positions)

        self.symbol_positions = {}
        for i, symbol in enumerate(self.symbols):
            self.symbol_positions[symbol] = positions[i]

        if self.agent_pos in self.symbol_positions.values():
            for pos in positions:
                if pos not in self.symbol_positions.values():
                    self.agent_pos = pos
                    break

        self.step_count = 0
        self.collected_symbols = set()

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action in symbolic environment."""
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = self.step_count >= self.max_steps

        if action < 4:  # Movement
            reward += self._move_agent(action)
        else:  # Interact
            reward += self._interact()

        if self._check_symbolic_goals():
            reward += 10.0
            terminated = True

        reward -= 0.1  # Step penalty

        observation = self._get_observation()
        info = {
            "collected_symbols": list(self.collected_symbols),
            "remaining_symbols": len(self.symbols) - len(self.collected_symbols),
            "facts": self._get_current_facts(),
            "goals_satisfied": self._get_goal_satisfaction(),
        }

        return observation, reward, terminated, truncated, info

    def _move_agent(self, direction: int) -> float:
        """Move agent symbolically."""
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            return -0.5  # Boundary penalty

        if not self._can_move_to(new_pos):
            return -0.5  # Invalid move penalty

        self.agent_pos = new_pos

        for symbol, pos in self.symbol_positions.items():
            if pos == self.agent_pos and symbol not in self.collected_symbols:
                self.collected_symbols.add(symbol)
                return 1.0  # Collection reward

        return 0.0

    def _interact(self) -> float:
        """Perform symbolic interaction."""
        current_pos_symbols = [
            symbol
            for symbol, pos in self.symbol_positions.items()
            if pos == self.agent_pos
        ]

        if current_pos_symbols:
            symbol = current_pos_symbols[0]
            if symbol not in self.collected_symbols:
                self.collected_symbols.add(symbol)
                return 2.0  # Interaction reward

        return 0.0

    def _can_move_to(self, position: Tuple[int, int]) -> bool:
        """Check if agent can move to position based on symbolic rules."""
        if not (0 <= position[0] < self.size and 0 <= position[1] < self.size):
            return False

        return True

    def _check_symbolic_goals(self) -> bool:
        """Check if symbolic goals are satisfied."""
        satisfied_goals = 0

        for goal in self.goals:
            if goal.startswith("collected("):
                symbol = goal.split("(")[1].split(")")[0]
                if symbol in self.collected_symbols:
                    satisfied_goals += 1
            elif goal.startswith("at("):
                parts = goal.split("(")[1].split(")")[0].split(", ")
                target_pos = (int(parts[1].strip("()")), int(parts[2].strip("()")))
                if self.agent_pos == target_pos:
                    satisfied_goals += 1

        return satisfied_goals >= len(self.goals)

    def _get_observation(self) -> Dict:
        """Get symbolic observation."""
        return {
            "agent_pos": np.array(self.agent_pos),
            "symbol_positions": self._get_symbol_position_array(),
            "facts": self._get_current_facts(),
        }

    def _get_symbol_position_array(self) -> np.ndarray:
        """Get symbol positions as array."""
        pos_array = np.zeros((self.size, self.size), dtype=int)

        for symbol, pos in self.symbol_positions.items():
            symbol_id = int(symbol.split("_")[1]) + 1  # 1-based indexing
            pos_array[pos[0], pos[1]] = symbol_id

        return pos_array

    def _get_current_facts(self) -> List[str]:
        """Get current logical facts about the environment."""
        facts = []

        facts.append(f"at(agent, {self.agent_pos})")

        for symbol, pos in self.symbol_positions.items():
            facts.append(f"at({symbol}, {pos})")

        for symbol in self.collected_symbols:
            facts.append(f"collected({symbol})")

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                adj_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
                if 0 <= adj_pos[0] < self.size and 0 <= adj_pos[1] < self.size:
                    facts.append(f"adjacent({self.agent_pos}, {adj_pos})")

        return facts

    def _get_goal_satisfaction(self) -> Dict[str, bool]:
        """Get satisfaction status of each goal."""
        satisfaction = {}

        for goal in self.goals:
            if goal.startswith("collected("):
                symbol = goal.split("(")[1].split(")")[0]
                satisfaction[goal] = symbol in self.collected_symbols
            elif goal.startswith("at("):
                parts = goal.split("(")[1].split(")")[0].split(", ")
                target_pos = (int(parts[1].strip("()")), int(parts[2].strip("()")))
                satisfaction[goal] = self.agent_pos == target_pos
            else:
                satisfaction[goal] = False

        return satisfaction

    def render(self, mode="human"):
        """Render the symbolic environment."""
        if mode == "rgb_array":
            return self._render_rgb()

        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        for symbol, pos in self.symbol_positions.items():
            symbol_id = int(symbol.split("_")[1])
            if symbol in self.collected_symbols:
                grid[pos[0]][pos[1]] = f"c{symbol_id}"  # Collected
            else:
                grid[pos[0]][pos[1]] = f"s{symbol_id}"  # Available

        grid[self.agent_pos[0]][self.agent_pos[1]] = "A"

        print("\n".join(" ".join(row) for row in grid))
        print(f"Collected: {sorted(self.collected_symbols)}")
        print(f"Step: {self.step_count}")

        print("Current facts:")
        for fact in self._get_current_facts()[:5]:  # Show first 5 facts
            print(f"  {fact}")

    def _render_rgb(self) -> np.ndarray:
        """Render as RGB array."""
        image = np.zeros((self.size * 10, self.size * 10, 3), dtype=np.uint8)

        for i in range(self.size):
            for j in range(self.size):
                color = [100, 100, 100]  # Gray background

                if (i, j) == self.agent_pos:
                    color = [255, 255, 255]  # White

                for symbol, pos in self.symbol_positions.items():
                    if pos == (i, j):
                        symbol_id = int(symbol.split("_")[1])
                        if symbol in self.collected_symbols:
                            color = [0, 255, 0]  # Green for collected
                        else:
                            color = [255, 0, 0]  # Red for available

                image[i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = color

        return image

    def close(self):
        """Close the environment."""
        pass


class SymbolicGridWorldWithReasoning(SymbolicGridWorld):
    """Symbolic grid world with integrated reasoning capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_steps = []  # Store reasoning traces

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with reasoning trace."""
        pre_reasoning = self._reason_about_action(action)

        observation, reward, terminated, truncated, info = super().step(action)

        post_reasoning = self._reason_about_state()

        reasoning_trace = {
            "pre_action": pre_reasoning,
            "post_action": post_reasoning,
            "action_taken": action,
            "reward": reward,
        }
        self.reasoning_steps.append(reasoning_trace)

        info["reasoning_trace"] = reasoning_trace

        return observation, reward, terminated, truncated, info

    def _reason_about_action(self, action: int) -> Dict[str, Any]:
        """Reason about the potential consequences of an action."""
        reasoning = {
            "action": action,
            "expected_outcome": "move" if action < 4 else "interact",
            "risk_assessment": "low",
            "expected_reward": 0.0,
        }

        if action < 4:
            direction = ["north", "south", "west", "east"][action]
            reasoning["direction"] = direction
            reasoning["expected_position"] = self._predict_position(action)

        return reasoning

    def _reason_about_state(self) -> Dict[str, Any]:
        """Reason about the current state."""
        return {
            "collected_symbols": len(self.collected_symbols),
            "remaining_goals": len(self.goals)
            - sum(self._get_goal_satisfaction().values()),
            "current_facts_count": len(self._get_current_facts()),
            "progress_assessment": self._assess_progress(),
        }

    def _predict_position(self, action: int) -> Tuple[int, int]:
        """Predict position after action."""
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        return (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

    def _assess_progress(self) -> str:
        """Assess current progress toward goals."""
        satisfied = sum(self._get_goal_satisfaction().values())
        total = len(self.goals)

        if satisfied == total:
            return "complete"
        elif satisfied > total * 0.5:
            return "good_progress"
        elif satisfied > 0:
            return "some_progress"
        else:
            return "no_progress"
