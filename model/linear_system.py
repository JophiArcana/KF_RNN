from argparse import Namespace
from typing import *

from infrastructure.settings import DTYPE, DEVICE
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.discrete_are import Riccati


class LinearSystem(nn.Module):
    @classmethod
    def sample_stable_system(cls, shp: Namespace, **kwargs: Dict[str, torch.Tensor]) -> nn.Module:
        params = {
            'F': utils.sample_stable_state_matrix(shp.S_D),
            'B': torch.randn(shp.S_D, shp.I_D),
            'H': torch.randn(shp.O_D, shp.S_D),
            'sqrt_S_W': torch.randn(shp.S_D, shp.S_D) / shp.SNR,
            'sqrt_S_V': torch.randn(shp.O_D, shp.O_D) / shp.SNR
        }
        params.update(kwargs)
        return LinearSystem(params, shp.input_enabled)

    @classmethod
    def generate_dataset(cls,
                         systems_arr: np.ndarray[nn.Module],    # [N...]
                         batch_size: int,                       # B
                         seq_length: int                        # L
    ) -> TensorDict[str, torch.Tensor]:
        base_module: LinearSystem = systems_arr.ravel()[0]

        S_state_inf = utils.stack_tensor_arr(utils.multi_map(
            lambda sys: sys.S_state_inf, systems_arr, dtype=torch.Tensor
        ))                                                      # [N... x S_D x S_D]

        L, V = torch.linalg.eig(S_state_inf)
        sqrt_S_state_inf = (V @ torch.diag_embed(L ** 0.5) @ torch.linalg.inv(V)).real

        states = torch.randn((*systems_arr.shape, batch_size, base_module.S_D)) @ sqrt_S_state_inf.mT   # [N... x B x S_D]
        return LinearSystem.continue_dataset(systems_arr, states, seq_length)

    @classmethod
    def continue_dataset(cls,
                         systems_arr: np.ndarray[nn.Module],    # [N...]
                         states: torch.Tensor,                  # [N... x B x S_D]
                         seq_length: int                        # L
    ) -> TensorDict[str, torch.Tensor]:
        base_module: LinearSystem = systems_arr.ravel()[0]
        batch_size = states.shape[-2]

        inputs = torch.randn((*systems_arr.shape, batch_size, seq_length, base_module.I_D))             # [N x B x L x I_D]
        with torch.set_grad_enabled(False):
            observations = utils.run_module_arr(
                *utils.stack_module_arr(systems_arr),
                args=(states, inputs)
            )['observation']

        return TensorDict({
            'input': inputs,
            'observation': observations
        }, batch_size=(*systems_arr.shape, batch_size, seq_length))

    def __init__(self, params: Dict[str, torch.Tensor], input_enabled: bool):
        super().__init__()
        self.eval()

        F, B, H, sqrt_S_W, sqrt_S_V = map(params.__getitem__, ('F', 'B', 'H', 'sqrt_S_W', 'sqrt_S_V'))
        self.input_enabled = input_enabled
        self.S_D = F.shape[0]                                                       # State dimension
        self.I_D = B.shape[1]                                                       # Input dimension
        self.O_D = H.shape[0]                                                       # Observation dimension

        if not torch.all(torch.linalg.eig(F)[0].abs() < 1):
            raise RuntimeError(f"Eigenvalues of F matrix {F.clone().detach()} are unstable")
        self.F = nn.Parameter(F.clone())                                            # S_D x S_D
        if self.input_enabled:                                                      # S_D x I_D
            self.B = nn.Parameter(B.clone())
        else:
            self.B = nn.Parameter(torch.zeros((self.S_D, self.I_D)), requires_grad=False)
        self.H = nn.Parameter(H.clone())                                            # O_D x S_D
        self.sqrt_S_W = nn.Parameter(sqrt_S_W.clone())                              # S_D x S_D
        self.sqrt_S_V = nn.Parameter(sqrt_S_V.clone())                              # O_D x O_D

        self.S_W = self.sqrt_S_W @ self.sqrt_S_W.mT
        self.S_V = self.sqrt_S_V @ self.sqrt_S_V.mT

        L, V = torch.linalg.eig(self.F)
        Vinv = torch.linalg.inv(V)
        self.S_state_inf = (V @ (
            (Vinv @ torch.complex(self.S_W, torch.zeros_like(self.S_W)) @ Vinv.mT) / (1 - L * L[:, None])
        ) @ V.mT).real

        S_state_inf_intermediate = Riccati.apply(self.F.mT, self.H.mT, self.S_W, self.S_V)
        self.S_observation_inf = self.H @ S_state_inf_intermediate @ self.H.mT + self.S_V
        self.K = S_state_inf_intermediate @ self.H.mT @ torch.inverse(self.S_observation_inf)
        self._M = torch.eye(self.S_D) - self.K @ self.H

    def forward(self,
                state: torch.Tensor,    # [B x S_D],
                inputs: torch.Tensor    # [B x T x I_D]
    ) -> Dict[str, torch.Tensor]:       # [B x T x S_D], [B x T x O_D]
        B, T, _ = inputs.shape
        W = torch.randn((B, T, self.S_D)) @ self.sqrt_S_W.mT
        V = torch.randn((B, T, self.O_D)) @ self.sqrt_S_V.mT

        states, observations = [], []
        for i in range(inputs.shape[1]):
            state = state @ self.F.mT + inputs[:, i] @ self.B.mT + W[:, i]

            states.append(state)
            observations.append(state @ self.H.mT + V[:, i])

        return {
            'state': torch.stack(states, dim=1),
            'observation': torch.stack(observations, dim=1)
        }




