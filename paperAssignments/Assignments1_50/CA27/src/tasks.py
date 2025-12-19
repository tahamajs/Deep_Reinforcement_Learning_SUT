"""Task and environment utilities for meta-learning."""
import numpy as np
from typing import List, Protocol, Any
from dataclasses import dataclass


class Task(Protocol):
    """Protocol for RL tasks in meta-learning."""
    env_name: str
    env: Any  # gym.Env
    obs_dim: int
    action_dim: int
    is_discrete: bool

    def reset(self) -> np.ndarray:
        ...

    def step(self, action: Any) -> tuple:
        ...

    def sample_task(self) -> 'Task':
        ...


@dataclass
class CartPoleTask:
    """CartPole task with variable parameters."""
    env_name: str = "CartPole-v1"
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5

    def __post_init__(self):
        """Initialize the environment after dataclass creation."""
        try:
            import gym
            self.env = gym.make(self.env_name)
            # Set custom parameters if supported
            if hasattr(self.env.env, 'gravity'):
                self.env.env.gravity = self.gravity
            if hasattr(self.env.env, 'masscart'):
                self.env.env.masscart = self.masscart
            if hasattr(self.env.env, 'masspole'):
                self.env.env.masspole = self.masspole
            if hasattr(self.env.env, 'length'):
                self.env.env.length = self.length

            self.obs_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.n if isinstance(self.env.action_space, gym.spaces.Discrete) else self.env.action_space.shape[0]
            self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        except ImportError:
            # For testing without gym
            self.env = None
            self.obs_dim = 4  # CartPole default
            self.action_dim = 2
            self.is_discrete = True

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        if self.env:
            return self.env.reset()
        return np.zeros(self.obs_dim)

    def step(self, action: Any):
        """Step the environment."""
        if self.env:
            return self.env.step(action)
        # Dummy step for testing
        return np.zeros(self.obs_dim), 0.0, True, {}

    def sample_task(self) -> 'CartPoleTask':
        """Sample a new task instance (not implemented for simplicity)."""
        return self


class MetaLearningTaskDistribution:
    """Distribution over tasks for meta-learning."""
    def __init__(self, task_class, num_tasks: int = 100, **kwargs):
        self.task_class = task_class
        self.num_tasks = num_tasks
        self.kwargs = kwargs
        self.tasks = [task_class(**kwargs) for _ in range(num_tasks)]

    def sample(self, batch_size: int = 1) -> List[Task]:
        """Sample batch of tasks."""
        return np.random.choice(self.tasks, size=batch_size, replace=False).tolist()

    def sample_random(self, batch_size: int = 1) -> List[Task]:
        """Sample batch of tasks with random parameters."""
        tasks = []
        for _ in range(batch_size):
            # Sample random parameters
            gravity = np.random.uniform(8.0, 11.0)
            masscart = np.random.uniform(0.8, 1.2)
            masspole = np.random.uniform(0.08, 0.12)
            length = np.random.uniform(0.4, 0.6)
            task = self.task_class(
                gravity=gravity,
                masscart=masscart,
                masspole=masspole,
                length=length
            )
            tasks.append(task)
        return tasks