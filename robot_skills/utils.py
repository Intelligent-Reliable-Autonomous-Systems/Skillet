"""utils.py.

Utilities for the policies module

Written by Will Solow and Jeff Jewett, 2026
"""

import importlib
import inspect
from typing import Any


def get_subclasses(module_name: str, base_class_name: str) -> dict[str, Any]:
    """Return a list of classes in the given module that inherit from `base_class_name`.

    Args:
        module_name (str): Full module name
        base_class_name (str): Name of the base class to filter by

    Returns:
        List[type]: List of classes in the module that inherit from `base_class_name`.

    """
    module = importlib.import_module(module_name)

    base_class = None
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name == base_class_name:
            base_class = obj
            break
    if base_class is None:
        raise ValueError(f"Base class '{base_class_name}' not found in module '{module_name}'")

    return {
        name: cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, base_class) and cls is not base_class
    }
