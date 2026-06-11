import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


__all__ = [
    "PRECISION",
    "DEVICE",
    "DTYPE",
    "PROJECT_NAME",
    "PROJECT_PATH",
    "DATA_PATH",
    "OUTPUT_PATH",
    "DEBUG",
]


# Debug mode. Set ``KF_RNN_DEBUG=1`` (or have RuntimeConfig.debug set it before
# this module is imported) to enable expensive diagnostics such as autograd
# anomaly detection. Off by default so normal training is not slowed down.
DEBUG: bool = os.environ.get("KF_RNN_DEBUG", "").lower() in ("1", "true", "yes")


PRECISION: int = 8
np.set_printoptions(precision=PRECISION,)
pd.set_option("display.precision", PRECISION,)
torch.set_printoptions(precision=PRECISION, sci_mode=False, linewidth=400,)

_CUDA_NUM: int = 0
DEVICE: str = "cpu" # f"cuda:{_CUDA_NUM}"
DTYPE: torch.dtype = torch.float64
PROJECT_NAME: str = "KF_RNN"
# Resolve the repo root from this file's location
# (``<root>/src/kf_rnn/infrastructure/settings.py`` -> 3 parents up). The editable
# install makes the package importable from any cwd, and these anchors make data/
# and output/ I/O cwd-independent too, so no ``os.chdir`` side effect is needed.
PROJECT_PATH: str = str(Path(__file__).resolve().parents[3])
DATA_PATH: str = os.path.join(PROJECT_PATH, "data")
OUTPUT_PATH: str = os.path.join(PROJECT_PATH, "output")

# Only constrain the visible CUDA devices when actually targeting a GPU.
if DEVICE.startswith("cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_CUDA_NUM)
torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

# Anomaly detection is expensive; only enable it in debug mode.
torch.autograd.set_detect_anomaly(DEBUG)


def set_debug(flag: bool) -> None:
    """Toggle debug-only global side effects (e.g. autograd anomaly detection).

    Lets a parsed RuntimeConfig.debug drive behavior even though this module is
    imported before any config is available.
    """
    global DEBUG
    DEBUG = bool(flag)
    torch.autograd.set_detect_anomaly(DEBUG)


# SECTION: Add safe globals for torch.load
import numpy
torch.serialization.add_safe_globals([
    numpy.core.multiarray._reconstruct,
    numpy.dtype,
    numpy.ndarray,
])




