from argparse import Namespace
from typing import *

import numpy as np
import torch
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.experiment.training import TrainFunc
from model.base import Predictor


class LeastSquaresPredictor(Predictor):
    @classmethod
    def train_least_squares(cls,
                            exclusive: Namespace,
                            ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                            cache: Namespace
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool]:
        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs,
            cls.vmap_train_least_squares, cache
        )

    @classmethod
    def vmap_train_least_squares(cls, exclusive_: Namespace) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return utils.multi_vmap(
            exclusive_.reference_module._least_squares_initialization, 2,
            randomness="different",
        )(exclusive_.train_info.dataset.obj.to_dict())

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return cls.train_least_squares,

    def __init__(self, modelArgs: Namespace):
        self.ridge = getattr(modelArgs, "ridge", 0.)

    def _least_squares_initialization(self, trace: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError()




