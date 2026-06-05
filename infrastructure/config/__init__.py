"""Typed configuration layer for KF_RNN experiments.

This package introduces dataclass-backed (OmegaConf-compatible) schemas that
describe the experiment hyperparameter tree, plus a bridge to/from the
``argparse.Namespace`` representation that the rest of the experiment stack
still consumes. The schemas give type-checking, validation, and serialization
(via OmegaConf) while the bridge keeps the existing pipeline runnable.

Key entry points:
    - ``ExperimentConfig`` and friends in ``schema``: the typed config tree.
    - ``config_to_namespace`` / ``namespace_to_config`` in ``bridge``.
    - ``register_configs`` in ``store``: registers schemas with Hydra's
      ConfigStore for ``_target_`` instantiation and CLI composition.
"""
from infrastructure.config.schema import (
    EnvironmentShape,
    ProblemShape,
    SplitConfig,
    DataConfig,
    SamplingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    EvalConfig,
    RuntimeConfig,
    SystemConfig,
    ModelConfig,
    ExperimentConfig,
)
from infrastructure.config.bridge import (
    config_to_namespace,
    namespace_to_config,
    split_for,
)
from infrastructure.config.store import register_configs, instantiate_target
from infrastructure.config.omega import to_yaml, validate_sweep_targets


__all__ = [
    "EnvironmentShape",
    "ProblemShape",
    "SplitConfig",
    "DataConfig",
    "SamplingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainConfig",
    "EvalConfig",
    "RuntimeConfig",
    "SystemConfig",
    "ModelConfig",
    "ExperimentConfig",
    "config_to_namespace",
    "namespace_to_config",
    "split_for",
    "register_configs",
    "instantiate_target",
    "to_yaml",
    "validate_sweep_targets",
]
