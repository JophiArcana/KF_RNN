from typing import Any, Callable

import torch
from tensordict import TensorDict

from model.base import Predictor


ModelPair = tuple[Predictor, TensorDict,]
LossFn = Callable[[ModelPair, TensorDict, dict[str, Any],], torch.Tensor]


def augment_loss_fn_with_noiseless(base_loss_fn: Callable[[ModelPair, TensorDict, dict[str, Any], torch.Tensor,], torch.Tensor]) -> LossFn:
    def augmented_loss_fn(model_pair: ModelPair, dataset: TensorDict, kwargs: dict[str, Any]) -> torch.Tensor:
        if kwargs["noiseless"]:
            target = dataset["environment", "noiseless_observation"]
            
        else:
            target = dataset["environment", "observation"]
            return base_loss_fn(model_pair, dataset, kwargs, target)

