import torch
import torch.nn as nn
import torch.nn.functional as Fn
from argparse import Namespace
from typing import *

from model.kf import KF


class CnnKF(KF):
    def __init__(self, modelArgs: Namespace):
        super().__init__()

        self.I_D = modelArgs.I_D
        self.O_D = modelArgs.O_D
        self.ir_length = modelArgs.ir_length
        self.input_enabled = modelArgs.input_enabled

        self.input_IR = nn.Parameter(torch.zeros(modelArgs.I_D, self.ir_length, modelArgs.O_D), requires_grad=self.input_enabled)
        self.observation_IR = nn.Parameter(torch.zeros(modelArgs.O_D, self.ir_length, modelArgs.O_D))

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns None
    """
    def _initialize_least_squares(self, trace: Dict[str, torch.Tensor]) -> None:
        device = self.observation_IR.device
        state, inputs, observations = KF.extract(trace, 0)
        B, L = inputs.shape[:2]

        indices = torch.arange(L, device=device)[:, None] - torch.arange(self.ir_length, device=device)
        padded_observations = torch.cat([
            torch.zeros((B, 1, self.O_D), device=device),
            observations[:, :-1],
            torch.zeros((B, self.ir_length - 1, self.O_D), device=device)
        ], dim=1)
        X_observation = padded_observations[:, indices]                                                 # [B x L x R x O_D]

        if self.input_enabled:
            padded_inputs = torch.cat([
                inputs,
                torch.zeros((B, self.ir_length - 1, self.I_D), device=device)
            ], dim=1)
            X_input = padded_inputs[:, indices]                                                         # [B x L x R x I_D], [B x L x R x O_D]
            flattened_X = torch.cat([X_input, X_observation], dim=-1).view((B * L, -1))         # [BL x R(I_D + O_D)]
        else:
            flattened_X = X_observation.view((B * L, -1))                                               # [BL x RO_D]

        flattened_w = torch.linalg.pinv(flattened_X) @ observations.view((B * L, -1))
        w = flattened_w.unflatten(0, (self.ir_length, -1)).transpose(0, 1)                              # [? x R x O_D]
        """ print(torch.norm(flattened_X @ flattened_w - observations.view((B * L, -1))) ** 2) """

        if self.input_enabled:
            self.input_IR.data, self.observation_IR.data = w[:self.I_D], w[self.I_D:]
        else:
            self.observation_IR.data = w

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns {
            'observation_estimation': [B x L x O_D]
        }
    """
    def forward(self, trace: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state, inputs, observations = KF.extract(trace, 0)
        B, L = inputs.shape[:2]

        result = Fn.conv2d(
            self.observation_IR,
            observations[:, :L].transpose(-2, -1).unsqueeze(-1).flip(-2),
            padding=(L, 0)
        )[:, :L] + Fn.conv2d(
            self.input_IR,
            inputs[:, :L].transpose(-2, -1).unsqueeze(-1).flip(-2),
            padding=(L - 1, 0)
        )[:, :L]

        return {
            'observation_estimation': result
        }




