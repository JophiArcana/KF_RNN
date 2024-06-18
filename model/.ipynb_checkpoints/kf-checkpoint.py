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


class KF(nn.Module):
    @classmethod
    def impulse(cls,
                kf_arr: np.ndarray[nn.Module],
                length: int
    ) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    def run(cls,
            reference_module: nn.Module,
            ensembled_kfs: TensorDict[str, torch.Tensor],
            dataset: TensorDict[str, torch.Tensor],
            kwargs: Dict[str, Any] = MappingProxyType(dict()),
            split_size: int = 1 << 20
    ) -> torch.Tensor:
        n = ensembled_kfs.ndim
        L = dataset.shape[-1]

        # assert d == 3, f"Expected three batch dimensions (n_systems, dataset_size, sequence_length) in the dataset but got shape {dataset.shape[ensembled_kfs.ndim:]}"
        _dataset = dataset.view(*ensembled_kfs.shape, -1, L)
        _dataset_size = sum(v.numel() for _, v in _dataset.items())

        splits = torch.round(_dataset.shape[n] * torch.linspace(0, 1, (_dataset_size + split_size - 1) // split_size + 1)).to(torch.int)
        splits = torch.tensor(sorted(set(splits.tolist())))

        _result_list = []
        for lo, hi in zip(splits[:-1], splits[1:]):
            _result_list.append(utils.run_module_arr(
                reference_module,
                ensembled_kfs,
                _dataset.view(-1, *_dataset.shape[-2:])[:, lo:hi].view(*ensembled_kfs.shape, hi - lo, L),
                kwargs
            )["observation_estimation"])

            torch.cuda.empty_cache()
            gc.collect()

        _result = torch.cat(_result_list, dim=n)
        return _result.view_as(dataset["observation"])

    @classmethod
    def gradient(cls,
                 reference_module: nn.Module,
                 ensembled_kfs: TensorDict[str, torch.Tensor],
                 dataset: TensorDict[str, torch.Tensor],
                 kwargs: Dict[str, Any] = MappingProxyType(dict()),
                 split_size: int = 1 << 20
    ) -> TensorDict[str, torch.Tensor]:
        n = ensembled_kfs.ndim
        L = dataset.shape[-1]

        # assert d == 3, f"Expected three batch dimensions (n_systems, dataset_size, sequence_length) in the dataset but got shape {dataset.shape[ensembled_kfs.ndim:]}"
        _dataset = dataset.view(*ensembled_kfs.shape, -1, L)
        _dataset_size = sum(v.numel() for _, v in _dataset.items())

        splits = torch.round(_dataset.shape[n] * torch.linspace(0, 1, (_dataset_size + split_size - 1) // split_size + 1)).to(torch.int)
        splits = torch.tensor(sorted(set(splits.tolist())))

        _result_list = []
        for lo, hi in zip(splits[:-1], splits[1:]):
            _dataset_slice = _dataset.view(-1, *_dataset.shape[-2:])[:, lo:hi].view(*ensembled_kfs.shape, hi - lo, L)
            _dataset_slice = TensorDict.from_dict(_dataset_slice, batch_size=_dataset_slice.shape)

            out = utils.run_module_arr(reference_module, ensembled_kfs, _dataset_slice, kwargs)["observation_estimation"][:, -1].norm() ** 2
            params = OrderedDict({k: v for k, v in _dataset_slice.items() if v.requires_grad}) 
            _result_list.append(TensorDict(dict(zip(
                params.keys(),
                torch.autograd.grad(out, (*params.values(),), allow_unused=True)
            )), batch_size=_dataset_slice.shape))

            torch.cuda.empty_cache()
            gc.collect()

        _result = torch.cat(_result_list, dim=n)
        return _result.view(dataset.shape)

    @classmethod
    def evaluate_run(cls,
                     result: torch.Tensor | float,
                     target: torch.Tensor | float,
                     mask: torch.Tensor = None,
                     batch_mean: bool = True
    ) -> torch.Tensor:
        losses = torch.norm(result - target, dim=-1) ** 2
        if mask is not None:
            # assert mask.ndim <= 2, f"mask.ndim must be less than or equal to 2 but got {mask.ndim}."
            mask = mask.expand(losses.shape[-mask.ndim:])
            result_ = torch.sum(losses * mask, dim=-1) / torch.sum(mask, dim=-1)
        else:
            result_ = torch.mean(losses, dim=-1)
        return result_.mean(-1) if batch_mean else result_

    @classmethod
    def clone_parameter_state(cls,
                              reference_module: nn.Module,
                              ensembled_learned_kfs: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        reset_ensembled_learned_kfs = TensorDict({}, batch_size=ensembled_learned_kfs.batch_size)
        for k, v in ensembled_learned_kfs.items():
            t = utils.rgetattr(reference_module, k)
            if isinstance(t, nn.Parameter):
                reset_ensembled_learned_kfs[k] = nn.Parameter(v.clone(), requires_grad=t.requires_grad)
            else:
                reset_ensembled_learned_kfs[k] = torch.Tensor(v.clone())
        return reset_ensembled_learned_kfs

    @classmethod
    def _train_with_initialization_and_error(cls,
                                             exclusive: Namespace,
                                             ensembled_learned_kfs: TensorDict[str, torch.Tensor],
                                             initialization_func: Callable[[
                                                 Namespace
                                             ], Tuple[Dict[str, torch.Tensor], torch.Tensor]],
                                             cache: Namespace
    ) -> Tuple[torch.Tensor, bool]:
        def terminate_condition() -> bool:
            return getattr(cache, "done", False)
        assert not terminate_condition()

        if not hasattr(cache, "initialization_error"):
            initialization, error_ = initialization_func(exclusive)
            for k, v in ensembled_learned_kfs.items():
                ensembled_learned_kfs[k] = initialization[k].expand_as(v)
            cache.initialization_error = error_.expand(ensembled_learned_kfs.shape)
            error = KF.evaluate_run(0, exclusive.train_info.dataset.obj["observation"], mask=exclusive.train_mask).mean(-1)
        else:
            cache.done = True
            error = cache.initialization_error
        cache.t += 1
        return error[None], terminate_condition()

    def __init__(self, modelArgs: Namespace):
        super().__init__()
        self.I_D: int = modelArgs.I_D
        self.O_D: int = modelArgs.O_D
        self.input_enabled: bool = modelArgs.input_enabled

    def extract(self, trace: Dict[str, torch.Tensor], S_D: int) -> Sequence[torch.Tensor]:
        inputs, observations = trace['input'], trace['observation']
        B, L = observations.shape[:2]
        state = (torch.randn if self.training else torch.zeros)((B, S_D))
        return state, inputs, observations

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
    def forward(self, trace: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    @classmethod
    def train_func_list(cls, default_train_func: Any) -> Sequence[Any]:
        return default_train_func,

    @classmethod
    def analytical_error(cls,
                         kfs: TensorDict[str, torch.Tensor],    # [B... x ...]
                         sg_td: TensorDict[str, torch.Tensor]   # [B... x ...]
    ) -> torch.Tensor:                                          # [B...]
        raise NotImplementedError(f"Analytical error does not exist for model {cls}")

    @classmethod
    def to_sequential_batch(cls, kfs: TensorDict[str, torch.Tensor], input_enabled: bool) -> TensorDict[str, torch.Tensor]:
        raise NotImplementedError(f"Model {cls} not convertible to sequential model")




