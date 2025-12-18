"""CA24 package

Lightweight import-safe package top-level markers.
"""

__all__ = ["config", "model", "data", "losses", "utils", "experiment"]

# Keep import-safe: avoid heavy initialization at import time.
__version__ = "0.1.0"
