import torch
import torch.nn as nn
from typing import *


class KF(nn.Module):
    def extract(self, trace: Dict[str, torch.Tensor], S_D: int) -> Sequence[torch.Tensor]:
        inputs, observations = trace['input'], trace['observation']
        B, L = observations.shape[:2]
        if self.training:
            state = torch.randn((B, S_D), device=inputs.device)
        else:
            state = torch.zeros((B, S_D), device=inputs.device)
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




