"""
Imitation Learning Package

This package contains modular components for behavioral cloning and expert data collection.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

from .expert_data_collector import ExpertDataCollector, load_expert_policy
from .behavioral_cloning import BehavioralCloning

__all__ = ["ExpertDataCollector", "load_expert_policy", "BehavioralCloning"]