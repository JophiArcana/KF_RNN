import os
import torch


torch.set_printoptions(sci_mode=False, linewidth=400)

DEVICE: str = "cuda:0"
DTYPE: torch.dtype = torch.float32
PROJECT_NAME: str = "KF_RNN"
PROJECT_PATH: str = os.getcwd()[:os.getcwd().find(PROJECT_NAME)] + PROJECT_NAME

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)
os.chdir(PROJECT_PATH)


# SECTION: Add safe globals for torch.load
import dimarray
import numpy
torch.serialization.add_safe_globals([
    dimarray.core.dimarraycls.DimArray,
    numpy.core.multiarray._reconstruct,
    numpy.dtype,
    numpy.ndarray,
])




