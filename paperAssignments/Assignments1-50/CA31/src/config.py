"""Configuration loading utilities."""
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config