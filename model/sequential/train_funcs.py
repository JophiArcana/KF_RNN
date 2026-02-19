from argparse import Namespace
from typing import Sequence

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from tensordict import TensorDict, NonTensorData

from infrastructure import utils
from infrastructure.static import ModelPair, TrainFunc
from model.base import Predictor


def train_analytical(THP: Namespace,
                     exclusive: Namespace,
                     model_pair: ModelPair,
                     cache: Namespace
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert exclusive.n_train_systems == 1, f"This model cannot be initialized when the number of training systems is greater than 1."
    return Predictor._train_with_initialization_and_error(
        exclusive, model_pair[1],
        lambda stacked_modules_, exclusive_: ({
            **exclusive_.train_info.systems.td().get("environment", {}),
            **exclusive_.train_info.systems.td().get("controller", {})
        }, Predictor.evaluate_run(
            exclusive_.train_info.dataset.obj["environment", "target_observation_estimation"],
            exclusive_.train_info.dataset.obj, ("environment", "observation")
        ).squeeze(-1)), cache
    )








