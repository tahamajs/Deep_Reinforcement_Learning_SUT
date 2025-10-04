"""Visualizations module for CA14 advanced RL implementations."""

import os

# Ensure visualizations directory exists
VIS_DIR = os.path.dirname(__file__)
if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

__all__ = []
