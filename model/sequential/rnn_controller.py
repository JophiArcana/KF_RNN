from argparse import Namespace
from typing import *

import torch
import torch.nn as nn

from model.sequential.base import SequentialController
from model.sequential.rnn_predictor import RnnPredictor, RnnPredictorAnalytical


class RnnController(SequentialController, RnnPredictor):
    def __init__(self, modelArgs: Namespace, **initialization: Dict[str, torch.Tensor | nn.Parameter]):
        SequentialController.__init__(self, modelArgs)
        RnnPredictor.__init__(self, modelArgs, **initialization)

        self.L = nn.Parameter(initialization.get("L", torch.zeros((self.I_D, self.S_D))))


class RnnControllerAnalytical(RnnController, RnnPredictorAnalytical):
    def __init__(self, modelArgs: Namespace, **initialization: Dict[str, torch.Tensor | nn.Parameter]):
        RnnController.__init__(self, modelArgs, **initialization)


class RnnControllerPretrainAnalytical(RnnControllerAnalytical):
    pass




