import torch
import torch.nn as nn
from tensordict import TensorDict
import scipy as sc
from argparse import Namespace
from typing import *

from infrastructure import utils
from infrastructure.settings import dev_type


class LinearSystem(nn.Module):
    @classmethod
    def sample_stable_system(cls, shp: Namespace, **kwargs) -> nn.Module:
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
                         systems: List[nn.Module],
                         batch_size: int,
                         seq_length: int
    ) -> TensorDict[str, torch.Tensor]:
        state = torch.randn(len(systems), batch_size, systems[0].S_D, device=dev_type)
        inputs = torch.randn(len(systems), batch_size, seq_length, systems[0].I_D, device=dev_type)

        with torch.set_grad_enabled(False):
            observations = utils.run_stacked_modules(
                base_module=systems[0],
                stacked_modules=torch.func.stack_module_state(systems)[0],
                args=(state, inputs),
                kwargs=dict()
            )['observation']

        return TensorDict({
            'input': inputs,
            'observation': observations
        }, batch_size=(len(systems), batch_size, seq_length), device=dev_type)

    def __init__(self, params: Dict[str, torch.Tensor], input_enabled: bool):
        super().__init__()
        self.eval()

        F, B, H, sqrt_S_W, sqrt_S_V = map(params.__getitem__, ('F', 'B', 'H', 'sqrt_S_W', 'sqrt_S_V'))
        self.input_enabled = input_enabled
        self.S_D = F.shape[0]                                                       # State dimension
        self.I_D = B.shape[1]                                                       # Input dimension
        self.O_D = H.shape[0]                                                       # Observation dimension

        self.F = nn.Parameter(F, requires_grad=False)                               # S_D x S_D
        if self.input_enabled:                                                      # S_D x I_D
            self.B = nn.Parameter(B, requires_grad=False)
        else:
            self.B = nn.Parameter(torch.zeros(self.S_D, self.I_D), requires_grad=False)
        self.H = nn.Parameter(H, requires_grad=False)                               # O_D x S_D
        self.sqrt_S_W = nn.Parameter(sqrt_S_W, requires_grad=False)                 # S_D x S_D
        self.sqrt_S_V = nn.Parameter(sqrt_S_V, requires_grad=False)                 # O_D x O_D

        self.S_W: nn.Parameter = nn.Parameter(self.sqrt_S_W @ self.sqrt_S_W.T, requires_grad=False)
        self.S_V: nn.Parameter = nn.Parameter(self.sqrt_S_V @ self.sqrt_S_V.T, requires_grad=False)

        S_state_inf_intermediate = torch.Tensor(sc.linalg.solve_discrete_are(self.F.T, self.H.T, self.S_W, self.S_V))
        self.S_observation_inf = nn.Parameter(self.H @ S_state_inf_intermediate @ self.H.T + self.S_V, requires_grad=False)
        self.K = nn.Parameter(S_state_inf_intermediate @ self.H.T @ torch.inverse(self.S_observation_inf), requires_grad=False)

    def forward(self,
                state: torch.Tensor,            # [B x S_D]
                inputs: torch.Tensor            # [B x T x I_D]
    ) -> Dict[str, torch.Tensor]:               # [B x T x S_D], [B x T x O_D]
        B, T, _ = inputs.shape
        W = (torch.randn(B * T, self.S_D, device=dev_type) @ self.sqrt_S_W.T).view(B, T, self.S_D)
        V = (torch.randn(B * T, self.O_D, device=dev_type) @ self.sqrt_S_V.T).view(B, T, self.O_D)

        states, observations = [], []
        for i in range(inputs.shape[1]):
            states.append(state := state @ self.F.T + inputs[:, i] @ self.B.T + W[:, i])
            observations.append(state @ self.H.T + V[:, i])

        return {
            'state': torch.stack(states, dim=1),
            'observation': torch.stack(observations, dim=1)
        }




