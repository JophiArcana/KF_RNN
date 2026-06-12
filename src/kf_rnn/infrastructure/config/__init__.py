"""Typed configuration layer for KF_RNN experiments.

A single dataclass representation (``ExperimentConfig``) flows from authoring
(scripts construct it in Python) through sweeping (the engine copies and
overrides it per cell) to serialization (``config_to_jsonable``). Per-model
hyperparameters live on each Predictor's own nested ``Config`` dataclass (see
``kf_rnn.model.base.Observer``), so the schema here stays model-agnostic.
"""
from kf_rnn.infrastructure.config.schema import (
    SPLIT_NAMES,
    Split,
    EnvironmentShape,
    ProblemShape,
    SystemAuxiliary,
    SystemSettings,
    SystemConfig,
    DataConfig,
    SamplingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    MetricsConfig,
    RuntimeConfig,
    ExperimentConfig,
    controller_dims,
    shape_leaves,
    propagate_problem_shape,
    copy_config,
    resolve_splits,
    config_leaves,
    config_to_jsonable,
    validate_sweep_targets,
)


__all__ = [
    "SPLIT_NAMES",
    "Split",
    "EnvironmentShape",
    "ProblemShape",
    "SystemAuxiliary",
    "SystemSettings",
    "SystemConfig",
    "DataConfig",
    "SamplingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainConfig",
    "MetricsConfig",
    "RuntimeConfig",
    "ExperimentConfig",
    "controller_dims",
    "shape_leaves",
    "propagate_problem_shape",
    "copy_config",
    "resolve_splits",
    "config_leaves",
    "config_to_jsonable",
    "validate_sweep_targets",
]
