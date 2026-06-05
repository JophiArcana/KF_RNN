import dataclasses
from argparse import Namespace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure.config.schema import DataConfig

# from system.base import SystemGroup


__all__ = [
    "ModelPair",
    "TrainFunc",
    "PARAM_GROUP_FORMATTER",
    "TRAINING_DATASET_TYPES",
    "TESTING_DATASET_TYPE",
    "DATASET_SUPPORT_PARAMS",
    "INFO_DTYPE",
    "RESULT_DTYPE",
]


ModelPair = tuple[nn.Module, TensorDict]
TrainFunc = tuple[
    Callable[[
        Namespace,
        Namespace,
        ModelPair,
        Namespace
    ], tuple[torch.Tensor, dict[str, torch.Tensor],]],
    Callable[[
        Namespace,
        Namespace,
        ModelPair,
        Namespace
    ], bool],
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
INFO_DTYPE: np.dtype = np.dtype([
    ("systems", object), # ("systems", SystemGroup),
    ("system_params", object),
    ("dataset", object)
])
RESULT_DTYPE: np.dtype = np.dtype([
    ("time", float),
    ("output", TensorDict),
    ("learned_kfs", tuple),
    ("systems", object),
    ("metrics", object),
])




