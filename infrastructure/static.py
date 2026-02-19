from argparse import Namespace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

# from system.base import SystemGroup


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
DATASET_SUPPORT_PARAMS: list[str] = [
    "n_traces",
    "total_sequence_length",
    "system.n_systems"
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




