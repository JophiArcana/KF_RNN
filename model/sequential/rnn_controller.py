from argparse import Namespace

import torch
import torch.nn as nn

from infrastructure import utils
from model.sequential.base import SequentialController
from model.sequential.rnn_predictor import RnnPredictor, RnnKalmanPredictor, RnnKalmanInitializedPredictor


class RnnController(SequentialController, RnnPredictor):
    def __init__(self, modelArgs: Namespace, **initialization: torch.Tensor | nn.Parameter):
        SequentialController.__init__(self, modelArgs)
        RnnPredictor.__init__(self, modelArgs, **initialization)

        self.L = nn.ParameterDict({
            k: nn.Parameter(utils.rgetitem(initialization, f"L.{k}", torch.zeros((d, self.S_D,))))
            for k, d in vars(self.problem_shape.controller).items()
        })

class RnnKalmanController(RnnController, RnnKalmanPredictor):
    def __init__(self, modelArgs: Namespace, **initialization: torch.Tensor | nn.Parameter):
        RnnController.__init__(self, modelArgs, **initialization)


class RnnKalmanInitializedControllerAnalytical(RnnKalmanController, RnnKalmanInitializedPredictor):
    pass




