from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.discrete_are import solve_discrete_are
from system.core import SystemGroup, SystemDistribution


class LinearSystemGroup(SystemGroup):
    class Distribution(SystemDistribution):
        def __init__(self):
            SystemDistribution.__init__(self, LinearSystemGroup)

    @classmethod
    def sample_stable_systems(cls, shp: Namespace, batch_size: Tuple[int, ...], **kwargs: torch.Tensor):
        params = {
            "F": utils.sample_stable_state_matrix(shp.S_D, batch_size),
            "B": torch.randn((*batch_size, shp.S_D, shp.I_D)),
            "H": torch.randn((*batch_size, shp.O_D, shp.S_D)),
            "sqrt_S_W": torch.randn((*batch_size, shp.S_D, shp.S_D)) / shp.SNR,
            "sqrt_S_V": torch.randn((*batch_size, shp.O_D, shp.O_D)) / shp.SNR
        }
        for k, v in params.items():
            if k in kwargs:
                params[k] = kwargs[k].expand_as(v)
        return LinearSystemGroup(params, shp.input_enabled)

    def __init__(self, params: Dict[str, torch.Tensor], input_enabled: bool):
        SystemGroup.__init__(self, input_enabled)

        F, B, H, sqrt_S_W, sqrt_S_V = map(params.__getitem__, ("F", "B", "H", "sqrt_S_W", "sqrt_S_V"))

        self.group_shape = F.shape[:-2]
        self.S_D = F.shape[-1]                                                      # State dimension
        self.I_D = B.shape[-1]                                                      # Input dimension
        self.O_D = H.shape[-2]                                                      # Observation dimension

        if not torch.all(torch.linalg.eigvals(F).abs() < 1):
            raise RuntimeError(f"Eigenvalues of F matrix {F.clone().detach()} are unstable")
        self.F = nn.Parameter(F.clone())                                            # [N... x S_D x S_D]
        if self.input_enabled:                                                      # [N... x S_D x I_D]
            self.B = nn.Parameter(B.clone())
        else:
            self.register_buffer("B", torch.zeros_like(B))
        self.H = nn.Parameter(H.clone())                                            # [N... x O_D x S_D]
        self.sqrt_S_W = nn.Parameter(sqrt_S_W.clone())                              # [N... x S_D x S_D]
        self.sqrt_S_V = nn.Parameter(sqrt_S_V.clone())                              # [N... x O_D x O_D]

        self.register_buffer("S_W", self.sqrt_S_W @ self.sqrt_S_W.mT)               # [N... x S_D x S_D]
        self.register_buffer("S_V", self.sqrt_S_V @ self.sqrt_S_V.mT)               # [N... x O_D x O_D]

        L, V = torch.linalg.eig(self.F)                                             # [N... x S_D], [N... x S_D x S_D]
        Vinv = torch.inverse(V)                                                     # [N... x S_D x S_D]
        self.register_buffer("S_state_inf", (V @ (
            (Vinv @ torch.complex(self.S_W, torch.zeros_like(self.S_W)) @ Vinv.mT) / (1 - L.unsqueeze(-1) * L.unsqueeze(-2))
        ) @ V.mT).real)                                                                                             # [N... x S_D x S_D]
        self.register_buffer("S_observation_inf", self.H @ self.S_state_inf @ self.H.mT + self.S_V)                 # [N... x S_D x S_D]

        S_state_inf_intermediate = solve_discrete_are(self.F.mT, self.H.mT, self.S_W, self.S_V)                     # [N... x S_D x S_D]
        self.register_buffer("S_prediction_err_inf", self.H @ S_state_inf_intermediate @ self.H.mT + self.S_V)      # [N... x O_D x O_D]
        self.register_buffer("K", S_state_inf_intermediate @ self.H.mT @ torch.inverse(self.S_prediction_err_inf))  # [N... x S_D x O_D]
        self.register_buffer("irreducible_loss", utils.batch_trace(self.S_prediction_err_inf))                      # [N...]

    def supply_input(self,
                     state_estimation: torch.Tensor             # [N... x B x S_D]
    ) -> torch.Tensor:
        B = state_estimation.shape[-2]
        return torch.randn((*self.group_shape, B, self.I_D))    # [N... x B x I_D]

    def generate_dataset(self, batch_size: int, sequence_length: int) -> TensorDict[str, torch.Tensor]:
        B, L = batch_size, sequence_length

        W = torch.randn((L, *self.group_shape, B, self.S_D)) @ self.sqrt_S_W.mT                     # [L x N... x B x S_D]
        V = torch.randn((L, *self.group_shape, B, self.O_D)) @ self.sqrt_S_V.mT                     # [L x N... x B x O_D]

        state = torch.randn((*self.group_shape, B, self.S_D)) @ utils.sqrtm(self.S_state_inf).mT    # [N... x B x S_D]
        state_estimation = torch.zeros((*self.group_shape, B, self.S_D))                            # [N... x B x S_D]

        states, inputs, observations, targets = [], [], [], []
        with torch.set_grad_enabled(False):
            for W_, V_ in zip(W, V):
                inputs.append(input_ := self.supply_input(state_estimation))
                states.append(state := state @ self.F.mT + input_ @ self.B.mT + W_)                     # [N... x B x S_D]
                state_estimation = state_estimation @ self.F.mT + input_ @ self.B.mT                    # [N... x B x S_D]

                targets.append(target_ := state_estimation @ self.H.mT)                                 # [N... x B x O_D]
                observations.append(observation_ := state @ self.H.mT + V_)                             # [N... x B x O_D]
                state_estimation = state_estimation + (observation_ - target_) @ self.K.mT              # [N... x B x S_D]

        return TensorDict({
            "input": torch.stack(inputs, dim=-2),                                                   # [N... x B x L x I_D]
            "observation": torch.stack(observations, dim=-2),                                       # [N... x B x L x O_D]
            "target": torch.stack(targets, dim=-2)                                                  # [N... x B x L x O_D]
        }, batch_size=(*self.group_shape, B, L))

    def forward(self,
                state: torch.Tensor,            # [N... x B x S_D]
                inputs: torch.Tensor            # [N... x B x L x I_D]
    ) -> TensorDict[str, torch.Tensor]:         # [N... x B x L x S_D], [N... x B x L x O_D]
        B, L = inputs.shape[-3:-1]
        W = torch.randn((L, *self.group_shape, B, self.S_D)) @ self.sqrt_S_W.mT     # [L x N... x B x S_D]
        V = torch.randn((L, *self.group_shape, B, self.O_D)) @ self.sqrt_S_V.mT     # [L x N... x B x O_D]

        states, observations = [], []
        inputs = inputs.permute(-2, *range(len(self.group_shape) + 1), -1)          # [L x N... x B x I_D]
        for W_, V_, inputs_ in zip(W, V, inputs):
            state = state @ self.F.mT + inputs_ @ self.B.mT + W_

            states.append(state)
            observations.append(state @ self.H.mT + V_)

        return TensorDict({
            "state": torch.stack(states, dim=-2),                                   # [N... x B x L x S_D]
            "observation": torch.stack(observations, dim=-2)                        # [N... x B x L x O_D]
        }, batch_size=(*self.group_shape, B, L))


class MOPDistribution(LinearSystemGroup.Distribution):
    def __init__(self, F_mode: str, H_mode: str, W_std: float, V_std: float) -> None:
        LinearSystemGroup.Distribution.__init__(self)
        assert F_mode in ("gaussian", "uniform"), f"F_mode must be one of (gaussian, uniform) but got {F_mode}."
        self.F_mode = F_mode

        assert H_mode in ("gaussian", "uniform"), f"H_mode must be one of (gaussian, uniform) but got {H_mode}."
        self.H_mode = H_mode

        self.W_std = W_std
        self.V_std = V_std

    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        if self.F_mode == "gaussian":
            F = torch.randn((*shape, SHP.S_D, SHP.S_D))
        else:
            F = torch.zeros((*shape, SHP.S_D, SHP.S_D)).uniform_(-1., 1.)
        F *= (0.95 / torch.linalg.eigvals(F).abs().max(dim=-1).values.unsqueeze(-1).unsqueeze(-2))

        B = torch.randn((*shape, SHP.S_D, SHP.I_D)) / (3 ** 0.5)

        if self.H_mode == "gaussian":
            H = torch.randn((*shape, SHP.O_D, SHP.S_D)) / (3 ** 0.5)
        else:
            H = torch.zeros((*shape, SHP.O_D, SHP.S_D)).uniform_(-1., 1.)

        sqrt_S_W = (torch.eye(SHP.S_D) * self.W_std).expand(*shape, SHP.S_D, SHP.S_D)
        sqrt_S_V = (torch.eye(SHP.O_D) * self.V_std).expand(*shape, SHP.O_D, SHP.O_D)

        return {"F": F, "B": B, "H": H, "sqrt_S_W": sqrt_S_W, "sqrt_S_V": sqrt_S_V}




