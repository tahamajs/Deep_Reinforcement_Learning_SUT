"""
Utilities Package for CA11 - World Models and Latent Dynamics

This package provides utilities for data collection, visualization,
and evaluation of world models and model-based RL agents.
"""

from .data_collection import (
    collect_world_model_data,
    collect_sequence_data,
    collect_rollout_data,
    create_data_loader,
    split_data,
    augment_data
)

from .visualization import (
    plot_world_model_training,
    plot_rssm_training,
    plot_world_model_predictions,
    plot_trajectory_rollout,
    plot_latent_space_analysis,
    plot_dreamer_training,
    plot_comparison_metrics
)

__all__ = [
    # Data collection
    "collect_world_model_data",
    "collect_sequence_data", 
    "collect_rollout_data",
    "create_data_loader",
    "split_data",
    "augment_data",
    
    # Visualization
    "plot_world_model_training",
    "plot_rssm_training",
    "plot_world_model_predictions",
    "plot_trajectory_rollout",
    "plot_latent_space_analysis",
    "plot_dreamer_training",
    "plot_comparison_metrics"
]