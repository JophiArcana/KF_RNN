import math
from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure.experiment.training import TrainFunc
from model.base import Predictor
from model.sequential.base import SequentialPredictor


class RnnPredictor(SequentialPredictor):
    def __init__(self, modelArgs: Namespace, **initialization: Dict[str, torch.Tensor | nn.Parameter]):
        SequentialPredictor.__init__(self, modelArgs)
        self.S_D = modelArgs.S_D

        self.F = nn.Parameter(initialization.get("F", (1 - 1e-6) * torch.eye(self.S_D)))
        self.B = nn.ParameterDict({
            k: nn.Parameter(torch.zeros((self.S_D, d)))
            for k, d in vars(self.problem_shape.controller).items()
        })
        self.H = nn.Parameter(initialization.get('H', torch.zeros((self.O_D, self.S_D))))
        self.K = nn.Parameter(initialization.get('K', torch.zeros((self.S_D, self.O_D))))


class RnnPredictorAnalytical(RnnPredictor):
    @classmethod
    def train_analytical(cls,
                         exclusive: Namespace,
                         ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                         cache: Namespace
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool]:
        assert exclusive.n_train_systems == 1, f"This model cannot be initialized when the number of training systems is greater than 1."
        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs,
            lambda exclusive_: ({
                **exclusive_.train_info.systems.td().get("environment", {}),
                **exclusive_.train_info.systems.td().get("controller", {})
            }, Predictor.evaluate_run(
                exclusive_.train_info.dataset.obj["environment", "target_observation_estimation"],
                exclusive_.train_info.dataset.obj, ("environment", "observation")
            ).squeeze(-1)), cache
        )

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return RnnPredictorAnalytical.train_analytical,


class RnnPredictorPretrainAnalytical(RnnPredictorAnalytical):
    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return RnnPredictorAnalytical.train_analytical, default_train_func


class RnnPredictorDiagonalState(SequentialPredictor):
    def __init__(self, modelArgs: Namespace, **initialization: Dict[str, torch.Tensor | nn.Parameter]):
        SequentialPredictor.__init__(self, modelArgs)
        self.S_D = modelArgs.S_D

        self._C = nn.Parameter(initialization.get("F", torch.tensor([
            math.comb(self.S_D, k) * ((-1 + 0.) ** k)
            for k in range(self.S_D, 0, -1)
        ])))
        self.B = nn.ParameterDict({
            k: nn.Parameter(torch.zeros((self.S_D, d)))
            for k, d in vars(self.problem_shape.controller).items()
        })
        self.H = nn.Parameter(initialization.get('H', torch.zeros((self.O_D, self.S_D))))
        self.K = nn.Parameter(initialization.get('K', torch.zeros((self.S_D, self.O_D))))

    @property
    def F(self):
        companion_matrix = torch.diag(torch.ones((self.S_D - 1,)), diagonal=-1)
        companion_matrix[:, -1] = -self._C


        print(companion_matrix)
        return torch.diag(torch.linalg.eigvals(companion_matrix))





