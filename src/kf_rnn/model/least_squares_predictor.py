from argparse import Namespace
from typing import Sequence

import numpy as np
import torch
from tensordict import TensorDict

from kf_rnn.infrastructure import utils
from kf_rnn.model.base import Predictor


class LeastSquaresPredictor(Predictor):
    def vmap_train_least_squares(self, exclusive_: Namespace) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return utils.multi_vmap(
            self._least_squares_initialization, 2,
            randomness="different",
        )(exclusive_.train_info.dataset.to_dict())

    def training_recipe(self) -> Sequence[str]:
        return ["least_squares_init"]

    def __init__(self, modelArgs: Namespace):
        self.ridge = getattr(modelArgs, "ridge", 0.)

    def _least_squares_initialization(self, trace: dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError()




