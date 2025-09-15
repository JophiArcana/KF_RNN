import collections
from typing import Any, Callable

import torch
from tensordict import TensorDict

from model.base import Predictor


ModelPair = tuple[Predictor, TensorDict,]
SimpleLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
LossFn = Callable[[TensorDict, TensorDict, dict[str, Any],], torch.Tensor]

SIMPLE_LOSS_DICT: dict[str, SimpleLossFn] = collections.OrderedDict()
def add_to_simple_losses(loss_fn: SimpleLossFn, name: str) -> None:
    SIMPLE_LOSS_DICT[name] = loss_fn

LOSS_DICT: dict[str, LossFn] = collections.OrderedDict()
def add_to_losses(loss_fn: LossFn, name: str) -> None:
    LOSS_DICT[name] = loss_fn


# SECTION: Standard MSE loss
def _mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
    return torch.norm(output - target, dim=-1) ** 2     # [B... x N x B x L]
    # mask = dataset.get("mask", torch.full(dataset.shape[-2:], True))
    # return torch.sum(losses * mask, dim=[-2, -1,]) / torch.sum(mask, dim=[-2, -1,])

def default_mse_loss(result: TensorDict, dataset: TensorDict, kwargs: dict[str, Any]) -> torch.Tensor:
    obs_key = ("environment", "observation",)
    return _mse_loss(result[obs_key], dataset[obs_key]) + kwargs["control_coefficient"] * sum([
        _mse_loss(v, dataset["controller", k])
        for k, v in result["controller"].items()
    ])

def _get_mse_loss_fn_with_output_and_target_key(output_key: tuple[str, ...], target_key: tuple[str, ...]) -> LossFn:
    def loss_fn(result: TensorDict, dataset: TensorDict, kwargs: dict[str, Any]) -> torch.Tensor:
        return _mse_loss(result[output_key], dataset[target_key])
    return loss_fn


# SECTION: Finite difference MSE loss
def _finite_difference_mse_loss(output: torch.Tensor, target: torch.Tensor, dataset: TensorDict) -> torch.Tensor:
    target = target.clone()
    target[..., 1:, :] = target[..., 1:, :] - dataset["environment", "observation"][..., :-1, :]
    return _mse_loss(output, target)

def _get_finite_difference_mse_loss_fn_with_output_and_target_key(output_key: tuple[str, ...], target_key: tuple[str, ...]) -> LossFn:
    def loss_fn(result: TensorDict, dataset: TensorDict, kwargs: dict[str, Any]) -> torch.Tensor:
        return _finite_difference_mse_loss(result[output_key], dataset[target_key], dataset)
    return loss_fn




# def augment_loss_fn_with_noiseless(base_loss_fn: Callable[[ModelPair, TensorDict, dict[str, Any], torch.Tensor,], torch.Tensor]) -> LossFn:
#     def augmented_loss_fn(model_pair: ModelPair, dataset: TensorDict, kwargs: dict[str, Any]) -> torch.Tensor:
#         if kwargs["noiseless"]:
#             target = dataset["environment", "noiseless_observation"]
            
#         else:
#             target = dataset["environment", "observation"]
#             return base_loss_fn(model_pair, dataset, kwargs, target)

add_to_simple_losses(_mse_loss, "mse")

add_to_losses(default_mse_loss, "mse")
add_to_losses(_get_finite_difference_mse_loss_fn_with_output_and_target_key(("environment", "observation",), ("environment", "observation",)), "fd_mse")




