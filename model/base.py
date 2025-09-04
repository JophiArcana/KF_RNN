import gc
from argparse import Namespace
from collections import OrderedDict
from types import MappingProxyType
from typing import *

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils


class Observer(nn.Module):
    def __init__(self, modelArgs: Namespace):
        super().__init__()
        self.problem_shape = modelArgs.problem_shape
        self.O_D: int = self.problem_shape.environment.observation


class Predictor(Observer):
    @classmethod
    def impulse(cls,
                kf_arr: np.ndarray[nn.Module],
                length: int
    ) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    def run(cls,
            model_pair: "utils.ModelPair",
            dataset: TensorDict[str, torch.Tensor],
            kwargs: Dict[str, Any] = MappingProxyType(dict()),
            split_size: int = 1 << 20,
    ) -> TensorDict[str, torch.Tensor]:
        ensembled_kfs = model_pair[1]
        n = ensembled_kfs.ndim
        L = dataset.shape[-1]
        
        # assert d == 3, f"Expected three batch dimensions (n_systems, n_traces, sequence_length) in the dataset but got shape {dataset.shape[ensembled_kfs.ndim:]}"
        _dataset = dataset.reshape((*ensembled_kfs.shape, -1, L))
        numel = sum(v.numel() for _, v in _dataset.items())

        _result_list, n_chunks = [], utils.ceildiv(numel, split_size)
        for chunk_indices in torch.chunk(torch.arange(_dataset.shape[-2]), chunks=n_chunks, dim=0):
            _dataset_slice = _dataset.reshape(-1, *_dataset.shape[-2:])[:, chunk_indices].view(*ensembled_kfs.shape, -1, L)
            _result_list.append(TensorDict.from_dict(utils.run_module_arr(
                model_pair,
                _dataset_slice,
                kwargs,
            ), batch_size=_dataset_slice.shape))
            utils.empty_cache()

        return TensorDict.cat(_result_list, dim=n).view(dataset.shape)

    @classmethod
    def gradient(cls,
                 model_pair: "utils.ModelPair",
                 dataset: TensorDict[str, torch.Tensor],
                 kwargs: Dict[str, Any] = MappingProxyType(dict()),
                 split_size: int = 1 << 20
    ) -> TensorDict[str, torch.Tensor]:
        ensembled_kfs = model_pair[1]
        n = ensembled_kfs.ndim
        L = dataset.shape[-1]

        # assert d == 3, f"Expected three batch dimensions (n_systems, n_traces, sequence_length) in the dataset but got shape {dataset.shape[ensembled_kfs.ndim:]}"
        _dataset = dataset.reshape(*ensembled_kfs.shape, -1, L)
        numel = sum(v.numel() for _, v in _dataset.items())

        _result_list, n_chunks = [], utils.ceildiv(numel, split_size)
        for chunk_indices in torch.chunk(torch.arange(_dataset.shape[-2]), chunks=n_chunks, dim=0):
            _dataset_slice = _dataset.view(-1, *_dataset.shape[-2:])[:, chunk_indices].view(*ensembled_kfs.shape, -1, L)
            _dataset_slice = TensorDict.from_dict(_dataset_slice, batch_size=_dataset_slice.shape)

            out = Predictor.run(model_pair, _dataset_slice)[..., -1]["environment", "observation"].norm() ** 2
            params = OrderedDict({k: v for k, v in _dataset_slice.items() if v.requires_grad}) 
            _result_list.append(TensorDict(dict(zip(
                params.keys(),
                torch.autograd.grad(out, (*params.values(),), allow_unused=True)
            )), batch_size=_dataset_slice.shape))
        return TensorDict.cat(_result_list, dim=n).view(dataset.shape)

    @classmethod
    def evaluate_run(cls,
                     result: torch.Tensor | float,                          # [B... x N x B x L x ...]
                     target_dict: TensorDict[str, torch.Tensor],            # [B... x N x B x L x ...]
                     target_key: Tuple[str, ...],
                     batch_mean: bool = True,
    ) -> torch.Tensor:
        losses = torch.norm(result - target_dict[target_key], dim=-1) ** 2  # [B... x N x B x L]
        mask = target_dict.get("mask", torch.full(target_dict.shape[-1:], True))
        result_ = torch.sum(losses * mask, dim=-1) / torch.sum(mask, dim=-1)
        return result_.mean(dim=-1) if batch_mean else result_

    @classmethod
    def compute_losses(
        cls,
        result: TensorDict[str, torch.Tensor],
        dataset: TensorDict[str, torch.Tensor],
        THP: Namespace,
    ) -> torch.Tensor:
        return Predictor.evaluate_run(
            result["environment", "observation"],
            dataset, ("environment", "observation")
        )

    @classmethod
    def clone_parameter_state(cls, model_pair: "utils.ModelPair") -> "utils.ModelPair":
        reference_module, ensembled_learned_kfs = model_pair
        reset_ensembled_learned_kfs = TensorDict({}, batch_size=ensembled_learned_kfs.batch_size)
        for k, v in utils.td_items(ensembled_learned_kfs).items():
            t = utils.rgetattr(reference_module, k)
            k = (*k.split("."),)
            if isinstance(t, nn.Parameter):
                reset_ensembled_learned_kfs[k] = nn.Parameter(v.clone(), requires_grad=t.requires_grad)
            else:
                reset_ensembled_learned_kfs[k] = torch.Tensor(v.clone())
        return (reference_module, reset_ensembled_learned_kfs,)

    @classmethod
    def terminate_with_initialization_and_error(
        cls,
        THP: Namespace,
        exclusive: Namespace,
        model_pair: "utils.ModelPair",
        cache: Namespace,
    ) -> bool:
        return getattr(cache, "done", False)

    @classmethod
    def _train_with_initialization_and_error(cls,
                                             exclusive: Namespace,
                                             ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                                             initialization_func: Callable[[
                                                 Namespace
                                             ], Tuple[Dict[str, Any], torch.Tensor]],
                                             cache: Namespace
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not hasattr(cache, "initialization_error"):
            initialization, error_ = initialization_func(exclusive)
            for k, v in ensembled_learned_kfs.items(include_nested=True, leaves_only=True):
                ensembled_learned_kfs[k] = utils.rgetitem(initialization, k if isinstance(k, str) else ".".join(k)).expand_as(v)
            cache.initialization_error = error_.expand(ensembled_learned_kfs.shape)
            error = Predictor.evaluate_run(0, exclusive.train_info.dataset.obj, ("environment", "observation")).mean(dim=-1)
        else:
            cache.done = True
            error = cache.initialization_error
        return error[None], {}

    """ forward
        :parameter {
            'state': [B x S_D],
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns {
            'state_estimation': [B x L x S_D],              (Optional)
            'observation_estimation': [B x L x O_D],
            'state_covariance': [B x L x S_D x S_D],        (Optional)
            'observation_covariance': [B x L x O_D x O_D]   (Optional)
        }
    """
    def forward(self, trace: Dict[str, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    @classmethod
    def trace_to_td(cls, trace: Dict[str, Dict[str, torch.Tensor]]) -> TensorDict[str, torch.Tensor]:
        return TensorDict.from_dict(trace, batch_size=trace["environment"]["observation"].shape[:-1])

    @classmethod
    def train_func_list(cls, default_train_func: Tuple[Any, Any]) -> Sequence[Tuple[Any, Any]]:
        return default_train_func,

    @classmethod
    def analytical_error(cls,
                         kfs: TensorDict[str, torch.Tensor],                # [B... x ...]
                         sg_td: TensorDict[str, torch.Tensor]               # [B... x ...]
    ) -> TensorDict[str, torch.Tensor]:                                     # [B... x ...]
        return cls._analytical_error_and_cache(kfs, sg_td)[0]

    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict[str, torch.Tensor],     # [B... x ...]
                                    sg_td: TensorDict[str, torch.Tensor]    # [B... x ...]
    ) -> Tuple[TensorDict[str, torch.Tensor], Namespace]:                   # [B...]
        raise NotImplementedError(f"Analytical error does not exist for model {cls}")


# TODO: This is more semantically correct
# class Controller(Observer):
class Controller(Predictor):
    @classmethod
    def compute_losses(
        cls,
        result: TensorDict[str, torch.Tensor],
        dataset: TensorDict[str, torch.Tensor],
        THP: Namespace,
    ) -> torch.Tensor:
        observation_losses = Predictor.evaluate_run(
            result["environment", "observation"],
            dataset, ("environment", "observation")
        )
        action_losses = sum([
            Predictor.evaluate_run(v, dataset, ("controller", k))
            for k, v in result["controller"].items()
        ])
        return observation_losses + THP.control_coefficient * action_losses




