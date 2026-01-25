"""Registry mechanism for modular architecture."""

from typing import Dict, Type, Any, Optional


class Registry:
    """A registry to map string names to classes."""

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def __repr__(self):
        return f"<Registry {self._name}: {list(self._module_dict.keys())}>"

    def register(self, name: Optional[str] = None):
        """Register a module.
        
        Args:
            name: Name of the module. If None, use class name.
        """
        def _register(cls):
            key = name or cls.__name__
            if key in self._module_dict:
                raise KeyError(f"{key} is already registered in {self._name}")
            self._module_dict[key] = cls
            return cls
        return _register

    def get(self, key: str) -> Type:
        """Get a registered module by name."""
        if key not in self._module_dict:
            raise KeyError(f"{key} is not registered in {self._name}. Available: {list(self._module_dict.keys())}")
        return self._module_dict[key]
    
    def build(self, config: Dict[str, Any], *args, **kwargs) -> Any:
        """Build an instance from config.
        
        Args:
            config: Config dict with 'type' key and optional 'config' key.
            *args, **kwargs: Additional arguments passed to constructor.
        """
        if "type" not in config:
            raise ValueError(f"Config for {self._name} must contain 'type' field")
        
        module_type = config["type"]
        module_cls = self.get(module_type)
        
        # Merge explicitly passed kwargs with config-based kwargs
        # The exact construction logic depends on how the classes are designed.
        # Here we assume classes accept a single 'config' dict or kwargs.
        # But based on our design, modules accept specific config dicts.
        
        module_config = config.get("config", {})
        
        # Support passing the raw config dict or kwargs expansion
        return module_cls(module_config, *args, **kwargs)


# Global Registries
LAYOUT_REGISTRY = Registry("LAYOUT")
VLM_REGISTRY = Registry("VLM")
