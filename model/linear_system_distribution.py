from argparse import Namespace
from typing import *

import numpy as np
import torch

from infrastructure import utils
from model.linear_system import LinearSystem


class LinearSystemDistribution(object):
    def __init__(self, sample_func: Callable[[Namespace], Dict[str, torch.Tensor]]):
        self.sample_func = sample_func

    def sample(self, SHP: Namespace, shape: Sequence[int]) -> np.ndarray[LinearSystem]:
        return utils.multi_map(
            lambda _: LinearSystem(self.sample_func(SHP), SHP.input_enabled),
            np.empty(shape), dtype=LinearSystem
        )




