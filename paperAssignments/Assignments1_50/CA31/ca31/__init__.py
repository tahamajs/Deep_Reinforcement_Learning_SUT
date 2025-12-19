"""Top-level compat package for CA31.

This module re-exports the small `src.ca31` package so tests and users can
`import ca31` directly (a common convention used across the course repos).
"""

from src.ca31 import *  # re-export everything from the implementation package

__all__ = ["utils", "bandit", "train"]
