"""Encapsulation of the vmapped model ensemble.

Historically the codebase passed around a bare ``ModelPair`` tuple
``(reference_module, stacked_params_td)`` and reimplemented the reshape /
chunk-by-numel / vmap plumbing in several places. ``EnsembleModule`` centralizes
that logic behind a small object. The underlying ``(reference_module,
stacked_params)`` tuple remains the on-the-wire representation (it is what gets
stored in result records and checkpoints), so ``EnsembleModule`` is a thin,
allocation-cheap wrapper that can be created from / converted to a pair.
"""
from __future__ import annotations

from collections import OrderedDict
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.static import ModelPair


# Default run-chunking threshold (in element count). Mirrors RuntimeConfig.split_size;
# callers with access to the config (e.g. the training loop) should pass that value.
DEFAULT_SPLIT_SIZE: int = 1 << 20


class EnsembleModule:
    DEFAULT_SPLIT_SIZE: int = DEFAULT_SPLIT_SIZE

    def __init__(self, reference_module: nn.Module, stacked_params: TensorDict):
        self.reference_module = reference_module
        self.stacked_params = stacked_params

    # SECTION: Construction / conversion
    @classmethod
    def from_pair(cls, model_pair: ModelPair) -> "EnsembleModule":
        return cls(model_pair[0], model_pair[1])

    @classmethod
    def from_module_array(cls, module_arr: np.ndarray) -> "EnsembleModule":
        return cls.from_pair(utils.stack_module_arr(module_arr))

    @property
    def pair(self) -> ModelPair:
        return (self.reference_module, self.stacked_params)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.stacked_params.shape

    @property
    def ndim(self) -> int:
        return self.stacked_params.ndim

    def parameters(self) -> list[nn.Parameter]:
        return [v for v in self.stacked_params.values() if isinstance(v, nn.Parameter)]

    # SECTION: Core execution
    def _chunked_apply(self, dataset: TensorDict, per_chunk, split_size: int) -> TensorDict:
        """Shared reshape -> chunk-by-numel -> per-chunk -> concat skeleton used by
        both ``run`` and ``gradient`` (previously duplicated)."""
        shape, n, L = self.shape, self.ndim, dataset.shape[-1]
        _dataset = dataset.reshape((*shape, -1, L))
        numel = sum(v.numel() for _, v in _dataset.items())

        # numel (or the trace dimension) can be 0 for an empty dataset; torch.chunk requires chunks >= 1.
        results, n_chunks = [], max(1, utils.ceildiv(numel, split_size))
        for chunk_indices in torch.chunk(torch.arange(_dataset.shape[-2]), chunks=n_chunks, dim=0):
            slice_td = _dataset.reshape(-1, *_dataset.shape[-2:])[:, chunk_indices].view(*shape, -1, L)
            results.append(per_chunk(slice_td))
        return TensorDict.cat(results, dim=n).view(dataset.shape)

    def run(
            self,
            dataset: TensorDict,
            kwargs: dict[str, Any] = MappingProxyType(dict()),
            split_size: int = DEFAULT_SPLIT_SIZE,
    ) -> TensorDict:
        def per_chunk(slice_td: TensorDict) -> TensorDict:
            out = TensorDict(utils.run_module_arr(self.pair, slice_td, kwargs), batch_size=slice_td.shape)
            utils.empty_cache()
            return out
        return self._chunked_apply(dataset, per_chunk, split_size)

    def gradient(
            self,
            dataset: TensorDict,
            kwargs: dict[str, Any] = MappingProxyType(dict()),
            split_size: int = DEFAULT_SPLIT_SIZE,
    ) -> TensorDict:
        def per_chunk(slice_td: TensorDict) -> TensorDict:
            slice_td = TensorDict.from_dict(slice_td, batch_size=slice_td.shape)
            out = self.run(slice_td)[..., -1]["environment", "observation"].norm() ** 2
            params = OrderedDict({k: v for k, v in slice_td.items() if v.requires_grad})
            return TensorDict(dict(zip(
                params.keys(),
                torch.autograd.grad(out, (*params.values(),), allow_unused=True),
            )), batch_size=slice_td.shape)
        return self._chunked_apply(dataset, per_chunk, split_size)

    def clone(self) -> "EnsembleModule":
        """Return a copy with freshly-cloned parameter tensors (preserving
        ``requires_grad``), used to reset trainable state between stages."""
        reset = TensorDict({}, batch_size=self.stacked_params.batch_size)
        for k, v in utils.td_items(self.stacked_params).items():
            t = utils.rgetattr(self.reference_module, k)
            key = (*k.split("."),)
            if isinstance(t, nn.Parameter):
                reset[key] = nn.Parameter(v.clone(), requires_grad=t.requires_grad)
            else:
                reset[key] = torch.Tensor(v.clone())
        return EnsembleModule(self.reference_module, reset)
