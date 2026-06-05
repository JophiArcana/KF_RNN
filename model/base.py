import gc
from argparse import Namespace
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.ensemble import EnsembleModule


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
            dataset: TensorDict,
            kwargs: dict[str, Any] = MappingProxyType(dict()),
            split_size: int = EnsembleModule.DEFAULT_SPLIT_SIZE,
    ) -> TensorDict:
        # Thin shim: the ensemble reshape/chunk/vmap plumbing lives in EnsembleModule.
        return EnsembleModule.from_pair(model_pair).run(dataset, kwargs, split_size)

    @classmethod
    def gradient(cls,
                 model_pair: "utils.ModelPair",
                 dataset: TensorDict,
                 kwargs: dict[str, Any] = MappingProxyType(dict()),
                 split_size: int = EnsembleModule.DEFAULT_SPLIT_SIZE,
    ) -> TensorDict:
        return EnsembleModule.from_pair(model_pair).gradient(dataset, kwargs, split_size)

    @classmethod
    def evaluate_run(cls,
                     result: torch.Tensor | float,                          # [B... x N x B x L x ...]
                     target_dict: TensorDict,            # [B... x N x B x L x ...]
                     target_key: tuple[str, ...],
                     batch_mean: bool = True,
    ) -> torch.Tensor:
        losses = torch.norm(result - target_dict[target_key], dim=-1) ** 2  # [B... x N x B x L]
        mask = target_dict.get("mask", torch.full(target_dict.shape[-1:], True))
        result_ = torch.sum(losses * mask, dim=-1) / torch.sum(mask, dim=-1)
        return result_.mean(dim=-1) if batch_mean else result_

    @classmethod
    def clone_parameter_state(cls, model_pair: "utils.ModelPair") -> "utils.ModelPair":
        return EnsembleModule.from_pair(model_pair).clone().pair

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
    def forward(self, trace: dict[str, dict[str, torch.Tensor]], **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        raise NotImplementedError()

    @classmethod
    def trace_to_td(cls, trace: dict[str, dict[str, torch.Tensor]]) -> TensorDict:
        return TensorDict.from_dict(trace, batch_size=trace["environment"]["observation"].shape[:-1])

    def training_recipe(self) -> Sequence[Any]:
        """Data-driven training recipe: an ordered sequence of stage specs.

        A spec is a stage-registry name (e.g. ``"sgd"``) or a ``TrainingStage``
        instance. The engine resolves names via ``STAGE_REGISTRY`` and validates
        them against this model. Default: a single SGD stage.
        """
        return ["sgd"]

    @classmethod
    def analytical_error(cls,
                         kfs: TensorDict,                # [B... x ...]
                         sg_td: TensorDict               # [B... x ...]
    ) -> TensorDict:                                     # [B... x ...]
        return cls._analytical_error_and_cache(kfs, sg_td)[0]

    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict,     # [B... x ...]
                                    sg_td: TensorDict    # [B... x ...]
    ) -> tuple[TensorDict, Namespace]:                   # [B...]
        raise NotImplementedError(f"Analytical error does not exist for model {cls}")


# A Controller is an Observer (it emits observations/actions); concrete controllers gain
# Predictor machinery by also inheriting a Predictor subclass (e.g. SequentialController,
# TransformerController), so this base is intentionally the lighter-weight Observer.
class Controller(Observer):
    @classmethod
    def compute_losses(
        cls,
        result: TensorDict,
        dataset: TensorDict,
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




