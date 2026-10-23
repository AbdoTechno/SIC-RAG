"""Dynamic plugin registry for discovering and instantiating pipeline components."""

from typing import Dict, Type, Callable

class PluginRegistry:
    """Registry pattern implementation for modular component loading."""
    
    def __init__(self):
        self._registry: Dict[str, Dict[str, Callable]] = {}

    def register(self, category: str, name: str) -> Callable:
        """Decorator to register a component class."""
        def decorator(cls: Type) -> Type:
            if category not in self._registry:
                self._registry[category] = {}
            self._registry[category][name] = cls
            return cls
        return decorator

    def get(self, category: str, name: str) -> Callable:
        """Retrieve a registered component by category and name."""
        if category not in self._registry or name not in self._registry[category]:
            raise ValueError(f"Component '{name}' not found in category '{category}'.")
        return self._registry[category][name]

registry = PluginRegistry()
