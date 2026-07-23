"""A tiny name -> method-class registry.

Usage::

    from .registry import register
    from .base import BaseMethod

    @register("logreg_l2")
    class LogRegL2(BaseMethod):
        ...

Then ``get("logreg_l2")`` returns the class, and ``available()`` lists names.
Importing :mod:`src.methods` populates the registry.
"""
from __future__ import annotations

from typing import Dict, Type

from .base import BaseMethod

_REGISTRY: Dict[str, Type[BaseMethod]] = {}


def register(name: str):
    def _wrap(cls: Type[BaseMethod]):
        if name in _REGISTRY:
            raise ValueError(f"method already registered: {name}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get(name: str) -> Type[BaseMethod]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown method '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available():
    return sorted(_REGISTRY)
