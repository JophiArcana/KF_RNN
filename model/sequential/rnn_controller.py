from argparse import Namespace
from typing import *

import torch
import torch.nn as nn

from model.sequential.base import SequentialController
from model.sequential.rnn_predictor import RnnPredictor, RnnPredictorAnalytical, RnnPredictorPretrainAnalytical


class RnnController(SequentialController, RnnPredictor):
    def __init__(self, modelArgs: Namespace, **initialization: Dict[str, torch.Tensor | nn.Parameter]):
        SequentialController.__init__(self, modelArgs)
        RnnPredictor.__init__(self, modelArgs, **initialization)

        self.L = nn.ParameterDict({
            k: nn.Parameter(torch.zeros((d, self.S_D)))
            for k, d in vars(self.problem_shape.controller).items()
        })

class RnnControllerAnalytical(RnnController, RnnPredictorAnalytical):
    def __init__(self, modelArgs: Namespace, **initialization: Dict[str, torch.Tensor | nn.Parameter]):
        RnnController.__init__(self, modelArgs, **initialization)


class RnnControllerPretrainAnalytical(RnnControllerAnalytical, RnnPredictorPretrainAnalytical):
    pass




