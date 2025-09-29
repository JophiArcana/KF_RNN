import os
import numpy as np
import pandas as pd
import torch


PRECISION: int = 8
np.set_printoptions(precision=PRECISION,)
pd.set_option("display.precision", PRECISION,)
torch.set_printoptions(precision=PRECISION, sci_mode=False, linewidth=400,)

_CUDA_NUM: int = 0
DEVICE: str = f"cuda:{_CUDA_NUM}"
DTYPE: torch.dtype = torch.float32
PROJECT_NAME: str = "KF_RNN"
PROJECT_PATH: str = os.getcwd()[:os.getcwd().find(PROJECT_NAME)] + PROJECT_NAME

os.environ["CUDA_VISIBLE_DEVICES"] = str(_CUDA_NUM)
torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)
os.chdir(PROJECT_PATH)

torch.autograd.set_detect_anomaly(True)


# SECTION: Add safe globals for torch.load
import dimarray
import numpy
torch.serialization.add_safe_globals([
    dimarray.core.dimarraycls.DimArray,
    numpy.core.multiarray._reconstruct,
    numpy.dtype,
    numpy.ndarray,
])




