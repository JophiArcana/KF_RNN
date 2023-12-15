import torch
import torch.nn as nn
import scipy as sc
from argparse import Namespace
from typing import *

from infrastructure import utils


class LinearSystem(nn.Module):
    @classmethod
    def sample_stable_system(cls, modelArgs: Namespace, base_system: nn.Module = None, **kwargs) -> nn.Module:
        F = kwargs.get('F', utils.sample_stable_state_matrix(modelArgs.S_D) if base_system is None else base_system.F)
        B = kwargs.get('B', torch.randn(modelArgs.S_D, modelArgs.I_D) if base_system is None else base_system.B)
        H = kwargs.get('H', torch.randn(modelArgs.O_D, modelArgs.S_D) if base_system is None else base_system.H)
        sqrt_S_W = kwargs.get('sqrt_S_W', torch.randn(modelArgs.S_D, modelArgs.S_D) / modelArgs.SNR if base_system is None else base_system.sqrt_S_W)
        sqrt_S_V = kwargs.get('sqrt_S_V', torch.randn(modelArgs.O_D, modelArgs.O_D) / modelArgs.SNR if base_system is None else base_system.sqrt_S_V)

        return LinearSystem(F, B, H, sqrt_S_W, sqrt_S_V)

    def __init__(self,
                 F: torch.Tensor,
                 B: torch.Tensor,
                 H: torch.Tensor,
                 sqrt_S_W: torch.Tensor,
                 sqrt_S_V: torch.Tensor
    ):
        super().__init__()

        self.S_D = F.shape[0]                                                       # State dimension
        self.I_D = B.shape[1]                                                       # Input dimension
        self.O_D = H.shape[0]                                                       # Observation dimension

        self.F = nn.Parameter(F, requires_grad=False)                               # S_D x S_D
        self.B = nn.Parameter(B, requires_grad=False)                               # S_D x I_D
        self.H = nn.Parameter(H, requires_grad=False)                               # O_D x S_D
        self.sqrt_S_W = nn.Parameter(sqrt_S_W, requires_grad=False)                 # S_D x S_D
        self.sqrt_S_V = nn.Parameter(sqrt_S_V, requires_grad=False)                 # O_D x O_D

        self.S_W: nn.Parameter = nn.Parameter(self.sqrt_S_W @ self.sqrt_S_W.T, requires_grad=False)
        self.S_V: nn.Parameter = nn.Parameter(self.sqrt_S_V @ self.sqrt_S_V.T, requires_grad=False)

        S_state_inf_intermediate = torch.Tensor(
            sc.linalg.solve_discrete_are(self.F.T, self.H.T, self.S_W, self.S_V)
        )
        self.S_observation_inf = nn.Parameter(self.H @ S_state_inf_intermediate @ self.H.T + self.S_V, requires_grad=False)
        self.K = nn.Parameter(S_state_inf_intermediate @ self.H.T @ torch.inverse(self.S_observation_inf), requires_grad=False)

    def forward(self,
                state: torch.Tensor,            # [B x S_D]
                inputs: torch.Tensor            # [B x T x I_D]
                ) -> Dict[str, torch.Tensor]:   # [B x T x S_D], [B x T x O_D]
        device = self.F.device

        B, T, _ = inputs.shape
        W = (torch.randn(B * T, self.S_D, device=device) @ self.sqrt_S_W.T).view(B, T, self.S_D)
        V = (torch.randn(B * T, self.O_D, device=device) @ self.sqrt_S_V.T).view(B, T, self.O_D)

        states, observations = [], []
        for i in range(inputs.shape[1]):
            states.append(state := (state @ self.F.T + inputs[:, i] @ self.B.T + W[:, i]))
            observations.append(state @ self.H.T + V[:, i])

        return {
            'state': torch.stack(states, dim=1),
            'observation': torch.stack(observations, dim=1)
        }




