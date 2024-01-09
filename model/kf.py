from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase

from infrastructure.settings import dev_type
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

    def __init__(self):
        super().__init__()
        self.input_enabled = None

    def extract(self, trace: Dict[str, torch.Tensor], S_D: int) -> Sequence[torch.Tensor]:
        inputs, observations = trace['input'], trace['observation']
        B, L = observations.shape[:2]
        if self.training:
            state = torch.randn((B, S_D), device=dev_type)
        else:
            state = torch.zeros((B, S_D), device=dev_type)
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




