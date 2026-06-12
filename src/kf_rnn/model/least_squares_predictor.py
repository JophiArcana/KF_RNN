from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.model.base import Predictor


class LeastSquaresPredictor(Predictor):
    @dataclass
    class Config(Predictor.Config):
        ridge: float = 0.0

    def vmap_train_least_squares(self, exclusive_: SimpleNamespace) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return eu.multi_vmap(
            self._least_squares_initialization, 2,
            randomness="different",
        )(exclusive_.train_info.dataset.to_dict())

    def training_recipe(self) -> Sequence[str]:
        return ["least_squares_init"]

    def __init__(self, modelArgs: "LeastSquaresPredictor.Config"):
        self.ridge = getattr(modelArgs, "ridge", 0.)

    def _least_squares_initialization(self, trace: dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError()




