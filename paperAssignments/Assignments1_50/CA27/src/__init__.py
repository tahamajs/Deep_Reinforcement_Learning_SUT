"""Meta-learning algorithms package."""
from .config import MAMLConfig, RL2Config, TaskConfig, ExperimentConfig
from .tasks import CartPoleTask, MetaLearningTaskDistribution
from .utils import Trajectory, collect_trajectory, compute_returns, compute_gae_returns
from .maml import MAML
from .rl2 import RL2Trainer

__all__ = [
    'MAMLConfig', 'RL2Config', 'TaskConfig', 'ExperimentConfig',
    'CartPoleTask', 'MetaLearningTaskDistribution',
    'Trajectory', 'collect_trajectory', 'compute_returns', 'compute_gae_returns',
    'MAML', 'RL2Trainer'
]