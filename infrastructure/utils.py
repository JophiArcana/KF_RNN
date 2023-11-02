
import torch
from argparse import Namespace
from typing import *

import numpy as np


def sample_stable_state_matrix(d: int) -> torch.DoubleTensor:
    M = torch.DoubleTensor([[2.]])
    scale = 1
    while torch.max(torch.norm(torch.linalg.eig(M)[0])) > 1:
        M = scale * torch.randn(d, d)
        scale *= 0.99
    return M


def color(z: float, scale: float=120.) -> np.ndarray:
    k = 2 * np.pi * z / scale
    return (1 + np.asarray([np.sin(k), np.sin(k + 2 * np.pi / 3), np.sin(k + 4 * np.pi / 3)], dtype=float)) / 2


def toJSON(n: Namespace):
    d = dict(vars(n))
    for k, v in d.items():
        if type(v) == Namespace:
            d[k] = toJSON(v)
    return d




