from argparse import Namespace
from typing import *

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool]:
        reference_module, ensembled_learned_kfs = model_pair
        return Predictor._train_with_initialization_and_error(
            exclusive, ensembled_learned_kfs,
            reference_module.vmap_train_least_squares, cache,
        )

    def vmap_train_least_squares(self, exclusive_: Namespace) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return utils.multi_vmap(
            self._least_squares_initialization, 2,
            randomness="different",
        )(exclusive_.train_info.dataset.obj.to_dict())

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return (cls.train_least_squares, Predictor.terminate_with_initialization_and_error,),

    def __init__(self, modelArgs: Namespace):
        self.ridge = getattr(modelArgs, "ridge", 0.)

    def _least_squares_initialization(self, trace: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError()




