import numpy as np
from tensordict import TensorDict

from infrastructure.utils import PTR

PARAM_GROUP_FORMATTER = "{0}_d({1})"
TRAINING_DATASET_TYPES = [
    "train",
    "valid",
]
TESTING_DATASET_TYPE = "test"
DATASET_SUPPORT_PARAMS = [
    "dataset_size",
    "total_sequence_length",
    "system.n_systems"
]
INFO_DTYPE = np.dtype([
    ("systems", np.ndarray),
    ("dataset", PTR),
    ("irreducible_loss", PTR),
])
RESULT_DTYPE = np.dtype([
    ("time", float),
    ("output", TensorDict),
    ("learned_kfs", tuple),
    ("systems", object),
    ("metrics", PTR),
])




