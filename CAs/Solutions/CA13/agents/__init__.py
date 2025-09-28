"""
Reinforcement Learning Agents
"""

from .model_free import ModelFreeAgent, DQNAgent
from .model_based import ModelBasedAgent
from .hybrid import HybridDynaAgent
from .imagination_based import ImaginationBasedAgent
from .sample_efficient import SampleEfficientAgent
from .transfer_learning import TransferLearningAgent
from .options_critic import OptionsCriticNetwork, OptionsCriticAgent
from .feudal import FeudalNetwork, FeudalAgent
from .integrated import IntegratedAdvancedAgent

__all__ = [
    'ModelFreeAgent', 'DQNAgent', 'ModelBasedAgent', 'HybridDynaAgent',
    'ImaginationBasedAgent', 'SampleEfficientAgent', 'TransferLearningAgent',
    'OptionsCriticNetwork', 'OptionsCriticAgent', 'FeudalNetwork', 'FeudalAgent',
    'IntegratedAdvancedAgent'
]