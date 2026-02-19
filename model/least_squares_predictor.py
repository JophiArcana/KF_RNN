from argparse import Namespace
from typing import Sequence

import numpy as np
import torch
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.static import ModelPair
from infrastructure.static import TrainFunc
from model.base import Predictor


class LeastSquaresPredictor(Predictor):
    @classmethod
    def train_least_squares(cls,
                            THP: Namespace,
                            exclusive: Namespace,
                            model_pair: ModelPair,
                            cache: Namespace
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reference_module, stacked_modules = model_pair
        return Predictor._train_with_initialization_and_error(
            exclusive, stacked_modules,
            reference_module.vmap_train_least_squares, cache,
        )

    def vmap_train_least_squares(self, exclusive_: Namespace) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return utils.multi_vmap(
            self._least_squares_initialization, 2,
            randomness="different",
        )(exclusive_.train_info.dataset.obj.to_dict())

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return (cls.train_least_squares, Predictor.terminate_with_initialization_and_error,),

    def __init__(self, modelArgs: Namespace):
        self.ridge = getattr(modelArgs, "ridge", 0.)

    def _least_squares_initialization(self, trace: dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError()




