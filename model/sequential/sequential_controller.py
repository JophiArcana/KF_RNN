from argparse import Namespace
from typing import *

import torch
import torch.nn as nn

from model.base.controller import Controller
from model.sequential.sequential_predictor import SequentialPredictor


class SequentialController(Controller, SequentialPredictor):
    def __init__(self, modelArgs: Namespace, **initialization: Dict[str, torch.Tensor | nn.Parameter]):
        Controller.__init__(self, modelArgs)
        SequentialPredictor.__init__(self, modelArgs)

    def forward(self, trace: Dict[str, torch.Tensor], mode: str = None) -> Dict[str, torch.Tensor]:
        state_estimation, inputs, observations = self.extract(trace, self.S_D)
        result = self.forward_with_initial(state_estimation, inputs, observations, mode)
        result["input_estimation"] = torch.cat([
            state_estimation[:, None],
            result["state_estimation"][:, :-1]
        ], dim=1) @ -self.L.mT
        return result




