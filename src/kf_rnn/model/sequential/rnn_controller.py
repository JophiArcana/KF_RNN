
import torch
import torch.nn as nn

import ecliseutils as eu
from kf_rnn.model.sequential.base import SequentialController
from kf_rnn.model.sequential.rnn_predictor import RnnPredictor, RnnKalmanPredictor, RnnKalmanInitializedPredictor
from kf_rnn.infrastructure.config.schema import controller_dims


class RnnController(SequentialController, RnnPredictor):
    def __init__(self, modelArgs: "RnnController.Config", **initialization: torch.Tensor | nn.Parameter):
        SequentialController.__init__(self, modelArgs)
        RnnPredictor.__init__(self, modelArgs, **initialization)

        self.L = nn.ParameterDict({
            k: nn.Parameter(eu.rgetitem(initialization, f"L.{k}", torch.zeros((d, self.S_D,))))
            for k, d in controller_dims(self.problem_shape).items()
        })

class RnnKalmanController(RnnController, RnnKalmanPredictor):
    def __init__(self, modelArgs: "RnnKalmanController.Config", **initialization: torch.Tensor | nn.Parameter):
        RnnController.__init__(self, modelArgs, **initialization)


class RnnKalmanInitializedControllerAnalytical(RnnKalmanController, RnnKalmanInitializedPredictor):
    pass




