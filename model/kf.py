from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase

from infrastructure.settings import device
from infrastructure import utils


class KF(nn.Module):
    @classmethod
    def evaluate_run(cls,
                     result: torch.Tensor | float,
                     target: torch.Tensor | float,
                     mask: torch.Tensor = None,
                     batch_mean: bool = True
    ) -> torch.Tensor:
        losses = torch.norm(result - target, dim=-1) ** 2
        if mask is not None:
            mask = mask.expand(losses.shape[-3:])
            result_ = torch.sum(losses * mask, dim=-1) / torch.sum(mask, dim=-1)
        else:
            result_ = torch.mean(losses, dim=-1)
        return result_.mean(-1) if batch_mean else result_

    @classmethod
    def _train_with_initialization_and_error(cls,
                                             shared: Dict[str, Any],
                                             exclusive: Dict[str, Any],
                                             flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
                                             initialization_func: Callable[[
                                                 Dict[str, Any],
                                                 Dict[str, Any]
                                             ], Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> Tuple[torch.Tensor, bool]:
        done_override = (base_model := exclusive['base_model'])._initialization_error is not None
        if not done_override:
            initialization, error = initialization_func(shared, exclusive)
            for k, v in flattened_ensembled_learned_kfs.items():
                v.data = initialization[k].expand_as(v).clone()
            base_model._initialization_error = error
            error = KF.evaluate_run(0, exclusive['training_dataset']['observation'], mask=exclusive['training_mask'])
        else:
            error = base_model._initialization_error
        return error[:, None], done_override

    def __init__(self):
        super().__init__()
        self.input_enabled = None

    def extract(self, trace: Dict[str, torch.Tensor], S_D: int) -> Sequence[torch.Tensor]:
        inputs, observations = trace['input'], trace['observation']
        B, L = observations.shape[:2]
        if self.training:
            state = torch.randn((B, S_D), device=device)
        else:
            state = torch.zeros((B, S_D), device=device)
        return state, inputs, observations

    def run(self,
            dataset: TensorDict[str, torch.Tensor] | TensorDictBase[str, torch.Tensor],
            flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
            **kwargs
    ) -> torch.Tensor:
        return utils.run_stacked_modules(
            base_module=self,
            stacked_modules=flattened_ensembled_learned_kfs,
            args=dict(dataset),
            kwargs=kwargs
        )['observation_estimation']

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
    def analytical_error(cls,
                         kfs: TensorDict[str, torch.Tensor],    # [B... x ...]
                         systems: TensorDict[str, torch.Tensor] # [B... x ...]
    ) -> torch.Tensor:                                          # [B...]
        raise NotImplementedError(f"Analytical error does not exist for model {cls}")

    @classmethod
    def to_sequential_batch(cls, kfs: TensorDict[str, torch.Tensor], input_enabled: bool) -> TensorDict[str, torch.Tensor]:
        raise NotImplementedError(f"Model {cls} not convertible to sequential model")

    def to_sequential(self) -> nn.Module:
        raise NotImplementedError(f"Model {type(self)} not convertible with sequential model")




