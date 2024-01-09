import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
from argparse import Namespace
from typing import *

from model.kf import KF
from infrastructure.train import TrainFunc
from infrastructure.settings import dev_type


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
    def forward(self, trace: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
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

        return {'observation_estimation': result}


class CnnKFLeastSquares(CnnKF):
    @classmethod
    def train_least_squares(cls,
                            hp: Namespace,
                            exclusive: Dict[str, Any],
                            flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
                            optimizer: optim.Optimizer
    ) -> Tuple[torch.Tensor, bool]:
        done_override = (base_model := exclusive['base_model'])._initialization_error is not None
        if not done_override:
            initialization, error = torch.vmap(CnnKFLeastSquares._least_squares_initialization, (None, 0))(
                base_model,
                dict(exclusive['training_dataset'])
            )
            for k, v in flattened_ensembled_learned_kfs.items():
                v.data = initialization[k]
            base_model._initialization_error = error

            y = exclusive['training_dataset']['observation'].flatten(1, 2)
            error = (y.norm(dim=(1, 2)) ** 2) / y.shape[1]
        else:
            error = base_model._initialization_error
        return error[:, None], done_override

    @classmethod
    def train_override(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return (CnnKFLeastSquares.train_least_squares,)

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns None
    """
    def _least_squares_initialization(self, trace: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        inputs, observations = trace['input'], trace['observation']
        B, L = inputs.shape[:2]

        # DONE: Implement online least squares for memory efficiency
        split = [0]
        while split[-1] != L:
            split.append(min(split[-1] + min(L, 1 << 16), L))

        padded_observations = torch.cat([                                                           # [B x L x O_D]
            torch.zeros((B, 1, self.O_D), device=dev_type),
            observations[:, :-1],
            torch.zeros((B, 1, self.O_D), device=dev_type)
        ], dim=1)
        padded_inputs = torch.cat([                                                                 # [B x L x I_D]
            inputs,
            torch.zeros((B, 1, self.I_D), device=dev_type)
        ], dim=1)

        k = self.I_D if self.input_enabled else 0
        XTX = torch.zeros((r_ := self.ir_length * (k + self.O_D), r_), device=dev_type)                     # [R? x R?]
        XTy = torch.zeros((r_, self.O_D), device=dev_type)                                                  # [R? x O_D]
        yTy = torch.zeros((self.O_D, self.O_D), device=dev_type)                                            # [O_D x O_D]

        for i in range(len(split) - 1):
            lo, hi = split[i], split[i + 1]
            l = hi - lo

            indices = (torch.arange(lo, hi, device=dev_type)[:, None] - torch.arange(self.ir_length, device=dev_type)).clamp_min(-1)

            X_observation = padded_observations[:, indices]                                                 # [B x l x R x O_D]
            if self.input_enabled:
                X_input = padded_inputs[:, indices]                                                         # [B x l x R x I_D]
                flattened_X = torch.cat([X_input, X_observation], dim=-1).view((B * l, -1))                 # [Bl x R(I_D + O_D)]
            else:
                flattened_X = X_observation.view((B * l, -1))                                               # [Bl x RO_D]
            flattened_observations = observations[:, lo:hi].view((B * l, self.O_D))                         # [Bl x O_D]

            XTX = XTX + (flattened_X.T @ flattened_X)
            XTy = XTy + (flattened_X.T @ flattened_observations)
            yTy = yTy + (flattened_observations.T @ flattened_observations)

            torch.cuda.empty_cache()

        XTX_lI_inv = torch.linalg.inv(XTX + self.ridge * torch.eye(r_, device=dev_type))                    # [R? x R?]
        flattened_w = XTX_lI_inv @ XTy
        w = flattened_w.unflatten(0, (self.ir_length, -1)).transpose(0, 1)                                  # [? x R x O_D]

        error = torch.trace(yTy + XTy.T @ (XTX_lI_inv @ XTX @ XTX_lI_inv - 2 * XTX_lI_inv) @ XTy) / (B * L)
        return {
            'input_IR': w[:self.I_D] if self.input_enabled else torch.zeros((self.I_D, self.ir_length, self.O_D), device=dev_type),
            'observation_IR': w[self.I_D:] if self.input_enabled else w
        }, error

    def __init__(self, modelArgs: Namespace):
        super().__init__(modelArgs)
        self.ridge = getattr(modelArgs, 'ridge', 0.)
        self._initialization_error: torch.Tensor = None


class CnnKFPretrainLeastSquares(CnnKFLeastSquares):
    @classmethod
    def train_override(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return CnnKFLeastSquares.train_least_squares, default_train_func




