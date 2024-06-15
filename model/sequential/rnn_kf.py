from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure.experiment.training import TrainFunc
from model.kf import KF
from model.sequential import SequentialKF


class RnnKF(SequentialKF):
    def __init__(self, modelArgs: Namespace, **initialization: Dict[str, torch.Tensor | nn.Parameter]):
        super().__init__(modelArgs)
        self.S_D = modelArgs.S_D

        self.F = nn.Parameter(initialization.get("F", 1.0 * torch.eye(self.S_D)))
        if self.input_enabled:
            self.B = nn.Parameter(initialization.get("B", torch.zeros(self.S_D, self.I_D)))
        else:
            self.register_buffer("B", torch.zeros((self.S_D, self.I_D)))
        self.H = nn.Parameter(initialization.get('H', torch.zeros(self.O_D, self.S_D)))
        self.K = nn.Parameter(initialization.get('K', torch.zeros(self.S_D, self.O_D)))


class RnnKFAnalytical(RnnKF):
    @classmethod
    def train_analytical(cls,
                         exclusive: Namespace,
                         ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                         cache: Namespace
    ) -> Tuple[torch.Tensor, bool]:
        assert exclusive.n_train_systems == 1, f"This model cannot be initialized when the number of training systems is greater than 1."
        return KF._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs,
            lambda exclusive_: (
                exclusive_.train_info.systems.td(),
                KF.evaluate_run(
                    exclusive_.train_info.dataset.obj["target"],
                    exclusive_.train_info.dataset.obj["observation"],
                    mask=exclusive_.train_mask
                ).squeeze(-1)
            ), cache
        )

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return RnnKFAnalytical.train_analytical,


class RnnKFPretrainAnalytical(RnnKFAnalytical):
    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return RnnKFAnalytical.train_analytical, default_train_func




