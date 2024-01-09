import torch
import torch.nn as nn
from argparse import Namespace
from typing import *

from model.sequential_kf import SequentialKF
from infrastructure import utils


class RnnKF(SequentialKF):
    def __init__(self, modelArgs: Namespace, **initialization: Dict[str, torch.Tensor | nn.Parameter]):
        super().__init__()

        self.S_D = modelArgs.S_D
        self.I_D = modelArgs.I_D
        self.O_D = modelArgs.O_D

        self.F: nn.Parameter = nn.Parameter(initialization.get('F', utils.sample_stable_state_matrix(self.S_D)))
        self.B: nn.Parameter = nn.Parameter(initialization.get('B', torch.zeros(self.S_D, self.I_D)), requires_grad=modelArgs.input_enabled)
        self.H: nn.Parameter = nn.Parameter(initialization.get('H', torch.zeros(self.O_D, self.S_D)))
        self.K: nn.Parameter = nn.Parameter(initialization.get('K', torch.zeros(self.S_D, self.O_D)))




