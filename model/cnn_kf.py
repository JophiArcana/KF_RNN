import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
from argparse import Namespace
from typing import *

from model.kf import KF
from infrastructure.train import run, evaluate_run


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
        :returns {
            'observation_estimation': [B x L x O_D]
        }
    """
    def forward(self, trace: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state, inputs, observations = self.extract(trace, 0)
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


class CnnKFLeastSquares(CnnKF):
    @classmethod
    def train_override(cls,
                       hp: Namespace,
                       exclusive: Dict[str, Any],
                       flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
                       optimizer: optim.Optimizer,
                       dev_type: str
    ) -> torch.Tensor:
        with torch.set_grad_enabled(False):
            train_result = run(exclusive['training_dataset'], exclusive['base_model'], flattened_ensembled_learned_kfs)
        losses = evaluate_run(train_result, exclusive['training_dataset']['observation'], exclusive['training_mask'])

        if not (base_model := exclusive['base_model'])._initialized:
            initialization = torch.vmap(CnnKFLeastSquares._least_squares_initialization, (None, 0, None))(
                hp.model,
                dict(exclusive['training_dataset']),
                dev_type
            )
            for k, v in flattened_ensembled_learned_kfs.items():
                v.data = initialization[k]
            base_model._initialized = True

        return losses[:, None]

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns None
    """
    @classmethod
    def _least_squares_initialization(cls,
                                      mhp: Namespace,
                                      trace: Dict[str, torch.Tensor],
                                      dev_type: str
    ) -> Dict[str, torch.Tensor]:
        inputs, observations = trace['input'], trace['observation']
        B, L = inputs.shape[:2]

        indices = torch.arange(L, device=dev_type)[:, None] - torch.arange(mhp.ir_length, device=dev_type)
        padded_observations = torch.cat([
            torch.zeros((B, 1, mhp.O_D), device=dev_type),
            observations[:, :-1],
            torch.zeros((B, mhp.ir_length - 1, mhp.O_D), device=dev_type)
        ], dim=1)
        X_observation = padded_observations[:, indices]                                                 # [B x L x R x O_D]

        if mhp.input_enabled:
            padded_inputs = torch.cat([
                inputs,
                torch.zeros((B, mhp.ir_length - 1, mhp.I_D), device=dev_type)
            ], dim=1)
            X_input = padded_inputs[:, indices]                                                         # [B x L x R x I_D], [B x L x R x O_D]
            flattened_X = torch.cat([X_input, X_observation], dim=-1).view((B * L, -1))         # [BL x R(I_D + O_D)]
        else:
            flattened_X = X_observation.view((B * L, -1))                                               # [BL x RO_D]

        flattened_w = torch.linalg.pinv(flattened_X) @ observations.view((B * L, -1))                   # [R? x O_D]
        w = flattened_w.unflatten(0, (mhp.ir_length, -1)).transpose(0, 1)                               # [? x R x O_D]

        return {
            'input_IR': w[:mhp.I_D] if mhp.input_enabled else torch.zeros((mhp.I_D, mhp.ir_length, mhp.O_D), device=dev_type),
            'observation_IR': w[mhp.I_D:] if mhp.input_enabled else w
        }

    def __init__(self, modelArgs: Namespace):
        super().__init__(modelArgs)
        self._initialized: bool = False




