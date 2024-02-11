import torch
import torch.nn as nn
import torch.optim as optim
from argparse import Namespace
from typing import *

from model.kf import KF
from model.sequential_kf import SequentialKF
from infrastructure.train import TrainFunc


class RnnKF(SequentialKF):
    def __init__(self, modelArgs: Namespace, **initialization: Dict[str, torch.Tensor | nn.Parameter]):
        super().__init__()

        self.S_D = modelArgs.S_D
        self.I_D = modelArgs.I_D
        self.O_D = modelArgs.O_D
        self.input_enabled = modelArgs.input_enabled

        self.F: nn.Parameter = nn.Parameter(initialization.get('F', torch.eye(self.S_D)))
        self.B: nn.Parameter = nn.Parameter(initialization.get('B', torch.zeros(self.S_D, self.I_D)), requires_grad=modelArgs.input_enabled)
        self.H: nn.Parameter = nn.Parameter(initialization.get('H', torch.zeros(self.O_D, self.S_D)))
        self.K: nn.Parameter = nn.Parameter(initialization.get('K', torch.zeros(self.S_D, self.O_D)))


class RnnKFPretrainAnalytical(RnnKF):
    @classmethod
    def train_analytical(cls,
                         shared: Dict[str, Any],
                         exclusive: Dict[str, Any],
                         flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
                         optimizer: optim.Optimizer
    ) -> Tuple[torch.Tensor, bool]:
        return KF._train_with_initialization_and_error(
            shared, exclusive, flattened_ensembled_learned_kfs,
            lambda shared_, exclusive_: (
                torch.func.stack_module_state(shared_['system'])[0],
                KF.evaluate_run(exclusive_['training_dataset']['target'], exclusive_['training_dataset']['observation'], mask=exclusive_['training_mask'])
            )
        )

    @classmethod
    def train_override(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return RnnKFPretrainAnalytical.train_analytical, default_train_func

    def __init__(self, modelArgs: Namespace):
        super().__init__(modelArgs)
        self._initialization_error: torch.Tensor = None




