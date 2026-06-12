import dataclasses

from ecliseutils.types import ModelPair

from kf_rnn.infrastructure.config.schema import DataConfig

# from system.base import SystemGroup


__all__ = [
    "ModelPair",
    "PARAM_GROUP_FORMATTER",
    "TRAINING_DATASET_TYPES",
    "TESTING_DATASET_TYPE",
    "DATASET_SUPPORT_PARAMS",
    "INFO_FIELDS",
    "RESULT_FIELDS",
]

PARAM_GROUP_FORMATTER: str = "{0}_d({1})"
TRAINING_DATASET_TYPES: list[str] = [
    "train",
    "valid",
]
TESTING_DATASET_TYPE: str = "test"
# Dataset support hyperparameters whose values determine generated-dataset shape.
# Derived from the typed DataConfig schema (the per-split dataset fields other
# than n_systems) plus the cross-branch system.n_systems reference, so this stays
# in sync with infrastructure.config.schema.DataConfig.
DATASET_SUPPORT_PARAMS: list[str] = [
    *(f.name for f in dataclasses.fields(DataConfig) if f.name != "n_systems"),
    "system.n_systems",
]
# Per-cell field names for the struct-of-arrays info / result grids. Each name
# maps to one LabeledArray field on InfoGrid / ResultGrid.
INFO_FIELDS: tuple[str, ...] = (
    "systems",
    "system_params",
    "dataset",
)
RESULT_FIELDS: tuple[str, ...] = (
    "time",
    "output",
    "learned_kfs",
    "systems",
    "metrics",
)




