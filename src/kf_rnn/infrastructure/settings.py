"""Project-local runtime settings for KF_RNN.

Holds the KF_RNN-specific choices (device/dtype/precision, repo paths) and
delegates the actual torch/numpy/pandas global configuration to
``ecliseutils.configure`` (the shared utilities package). Keeping the path
anchors and named constants here preserves the existing
``from kf_rnn.infrastructure.settings import DEVICE, DATA_PATH, ...`` imports.
"""

import os
from pathlib import Path

import ecliseutils


__all__ = [
    "PRECISION",
    "DEVICE",
    "DTYPE",
    "PROJECT_NAME",
    "PROJECT_PATH",
    "DATA_PATH",
    "OUTPUT_PATH",
    "DEBUG",
    "set_debug",
]


# Debug mode. Set ``KF_RNN_DEBUG=1`` (or have RuntimeConfig.debug set it before
# this module is imported) to enable expensive diagnostics such as autograd
# anomaly detection. Off by default so normal training is not slowed down.
DEBUG: bool = os.environ.get("KF_RNN_DEBUG", "").lower() in ("1", "true", "yes")

import torch

PRECISION: int = 8
_CUDA_NUM: int = 0
DEVICE: str = "cpu"  # f"cuda:{_CUDA_NUM}"
DTYPE: torch.dtype = torch.float64
PROJECT_NAME: str = "KF_RNN"
# Resolve the repo root from this file's location
# (``<root>/src/kf_rnn/infrastructure/settings.py`` -> 3 parents up). The editable
# install makes the package importable from any cwd, and these anchors make data/
# and output/ I/O cwd-independent too, so no ``os.chdir`` side effect is needed.
PROJECT_PATH: str = str(Path(__file__).resolve().parents[3])
DATA_PATH: str = os.path.join(PROJECT_PATH, "data")
OUTPUT_PATH: str = os.path.join(PROJECT_PATH, "output")

# Apply the shared runtime configuration (sets torch default device/dtype, print
# options, autograd anomaly, numpy safe globals). This also syncs
# ``ecliseutils.settings.DEVICE`` so utilities like ``torch_load`` /
# ``stack_module_arr`` resolve the correct device.
ecliseutils.configure(device=DEVICE, dtype=DTYPE, precision=PRECISION, debug=DEBUG)


def set_debug(flag: bool) -> None:
    """Toggle debug-only global side effects (e.g. autograd anomaly detection)."""
    global DEBUG
    DEBUG = bool(flag)
    ecliseutils.set_debug(DEBUG)
