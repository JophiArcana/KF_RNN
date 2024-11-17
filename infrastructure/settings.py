import os
import torch


torch.set_printoptions(sci_mode=False, linewidth=400)

DEVICE: str = "cpu"
DTYPE: torch.dtype = torch.float64
PROJECT_NAME: str = "KF_RNN"
PROJECT_PATH: str = os.getcwd()[:os.getcwd().find(PROJECT_NAME)] + PROJECT_NAME

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)
os.chdir(PROJECT_PATH)




