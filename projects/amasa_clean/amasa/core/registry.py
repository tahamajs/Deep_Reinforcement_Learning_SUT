"""Simple registries for environments, algorithms, and safety components."""
from __future__ import annotations

from typing import Callable, Dict, Any


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, key: str):
        def decorator(factory: Callable[..., Any]):
            if key in self._items:
                raise ValueError(f"{self.name} already contains key '{key}'")
            self._items[key] = factory
            return factory

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            options = ", ".join(sorted(self._items.keys())) or "<empty>"
            raise KeyError(f"Unknown {self.name} key '{key}'. Available: {options}")
        return self._items[key]

    def create(self, key: str, *args, **kwargs):
        return self.get(key)(*args, **kwargs)

    def keys(self):
        return sorted(self._items.keys())


algo_registry = Registry("algorithm")
env_registry = Registry("environment")
safety_registry = Registry("safety")
