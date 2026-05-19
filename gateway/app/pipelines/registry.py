"""管线注册表 — 管理所有可用管线"""

from typing import Callable, Dict

_pipeline_registry: Dict[str, Callable] = {}


def register_pipeline(name: str):
    """管线注册装饰器"""
    def decorator(func):
        _pipeline_registry[name] = func
        return func
    return decorator


def get_pipeline(name: str) -> Callable | None:
    return _pipeline_registry.get(name)


def list_pipelines() -> list[str]:
    return list(_pipeline_registry.keys())
