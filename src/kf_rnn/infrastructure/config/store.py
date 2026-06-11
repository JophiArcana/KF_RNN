"""Hydra ConfigStore registration and ``_target_`` instantiation helpers."""
from __future__ import annotations

import importlib
from typing import Any

from kf_rnn.infrastructure.config import schema


_REGISTERED = False


def register_configs() -> None:
    """Register the structured schemas with Hydra's ConfigStore (idempotent).

    This enables Hydra CLI composition and schema validation for the typed
    config tree. The in-process ``iterparams`` sweep is unaffected.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_experiment", node=schema.ExperimentConfig)
    cs.store(group="dataset", name="base_dataset", node=schema.DataConfig)
    cs.store(group="training", name="base_training", node=schema.TrainConfig)
    cs.store(group="experiment", name="base_runtime", node=schema.RuntimeConfig)
    _REGISTERED = True


def _resolve_dotted(target: str) -> Any:
    module_name, _, attr = target.rpartition(".")
    if not module_name:
        raise ValueError(f"_target_ must be a fully-qualified path, got {target!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def instantiate_target(spec: Any) -> Any:
    """Resolve a value that may be a Hydra ``_target_`` spec or a live object.

    - ``{"_target_": "pkg.mod.Cls", **kwargs}`` -> ``Cls(**kwargs)``
    - ``{"_target_": "pkg.mod.Cls", "_partial_": True, ...}`` -> the class itself
      (kwargs ignored), useful when the framework constructs the object later.
    - any other value (including a class or instance) is returned unchanged.
    """
    if isinstance(spec, dict) and "_target_" in spec:
        obj = _resolve_dotted(spec["_target_"])
        if spec.get("_partial_", False):
            return obj
        kwargs = {k: v for k, v in spec.items() if k not in ("_target_", "_partial_")}
        return obj(**kwargs) if kwargs else obj
    return spec
