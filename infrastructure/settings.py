import os
import numpy as np
import pandas as pd
import torch


__all__ = [
    "PRECISION",
    "DEVICE",
    "DTYPE",
    "PROJECT_NAME",
    "PROJECT_PATH",
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
# Resolve the project root from this file's location (``<root>/infrastructure/settings.py``)
# rather than searching the cwd for the substring ``PROJECT_NAME``, which silently
# produced a wrong path when the cwd did not contain that substring.
PROJECT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Only constrain the visible CUDA devices when actually targeting a GPU.
if DEVICE.startswith("cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_CUDA_NUM)
torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

# Many scripts reference paths relative to the project root (e.g. ``data/...``), so
# change into it on import by default. Set ``KF_RNN_NO_CHDIR=1`` to opt out when the
# caller manages its own working directory.
if os.environ.get("KF_RNN_NO_CHDIR", "").lower() not in ("1", "true", "yes") and os.path.isdir(PROJECT_PATH):
    os.chdir(PROJECT_PATH)

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




