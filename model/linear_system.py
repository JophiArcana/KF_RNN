from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.discrete_are import solve_discrete_are


class LinearSystemGroup(nn.Module):
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
        super().__init__()
        self.eval()

        F, B, H, sqrt_S_W, sqrt_S_V = map(params.__getitem__, ("F", "B", "H", "sqrt_S_W", "sqrt_S_V"))
        self.input_enabled = input_enabled

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

        S_state_inf_intermediate = solve_discrete_are(self.F.mT, self.H.mT, self.S_W, self.S_V)                     # [N... x S_D x S_D]
        self.register_buffer("S_observation_inf", self.H @ S_state_inf_intermediate @ self.H.mT + self.S_V)         # [N... x O_D x O_D]
        self.register_buffer("K", S_state_inf_intermediate @ self.H.mT @ torch.inverse(self.S_observation_inf))     # [N... x S_D x O_D]

    def generate_dataset(self,
                         batch_size: int,       # B
                         seq_length: int        # L
    ) -> TensorDict[str, torch.Tensor]:
        L, V = torch.linalg.eig(self.S_state_inf)
        sqrt_S_state_inf = (V @ torch.diag_embed(L ** 0.5) @ torch.inverse(V)).real

        states = torch.randn((*self.group_shape, batch_size, self.S_D)) @ sqrt_S_state_inf.mT   # [N... x B x S_D]
        return self.continue_dataset(states, seq_length)

    def td(self) -> TensorDict[str, torch.Tensor]:
        return TensorDict({
            **dict(self.named_parameters()),
            **dict(self.named_buffers())
        }, batch_size=self.group_shape)

    def continue_dataset(self,
                         states: torch.Tensor,  # [N... x B x S_D]
                         seq_length: int        # L
    ) -> TensorDict[str, torch.Tensor]:
        B = states.shape[-2]
        inputs = torch.randn((*self.group_shape, B, seq_length, self.I_D))  # [N... x B x L x I_D]
        with torch.set_grad_enabled(False):
            observations = self(states, inputs)["observation"]              # [N... x B x L x O_D]

        return TensorDict({
            "input": inputs,
            "observation": observations
        }, batch_size=(*self.group_shape, B, seq_length))

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


class AnalyticalKFGroup(nn.Module):
    def __init__(self, lsg: LinearSystemGroup):
        super().__init__()
        self.eval()

        self.group_shape = lsg.group_shape
        self.S_D, self.I_D, self.O_D = lsg.S_D, lsg.I_D, lsg.O_D
        self.input_enabled = lsg.input_enabled

        for k, v in lsg.named_parameters():
            self.register_parameter(k, v)
        for k, v in lsg.named_buffers():
            self.register_buffer(k, v)

    def add_targets(self,
                    dataset: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        with torch.set_grad_enabled(False):
            dataset["target"] = self(dataset, steady_state=True)["observation_estimation"]
        return dataset

    def forward(self, trace: TensorDict[str, torch.Tensor], steady_state: bool = False) -> TensorDict[str, torch.Tensor]:
        inputs, observations = trace["input"], trace["observation"]                                     # [N... x B x L x I_D], [N... x B x L x O_D]
        B, L = inputs.shape[-3:-1]
        state = torch.zeros((*self.group_shape, B, self.S_D))                                           # [N... x B x S_D]


        state_estimation = state                                                                        # [N... x B x S_D]
        state_covariance = torch.zeros((*self.group_shape, self.S_D, self.S_D))                         # [N... x S_D x S_D]

        state_estimations, observation_estimations = [], []
        state_covariances, observation_covariances = [], []


        inputs = inputs.permute(-2, *range(len(self.group_shape) + 1), -1)                              # [L x N... x B x I_D]
        observations = observations.permute(-2, *range(len(self.group_shape) + 1), -1)                  # [L x N... x B x O_D]
        for inputs_, observations_ in zip(inputs, observations):
            # Prediction
            state_estimation = state_estimation @ self.F.mT + inputs_ @ self.B.mT                       # [N... x B x S_D]
            state_covariance = self.F @ state_covariance @ self.F.mT + self.S_W                         # [N... x S_D x S_D]

            observation_estimations.append(
                observation_estimation := state_estimation @ self.H.mT                                  # [N... x B x O_D]
            )
            observation_covariances.append(
                observation_covariance := self.H @ state_covariance @ self.H.mT + self.S_V              # [N... x O_D x O_D]
            )

            if steady_state:                                                                            # [N... x S_D x O_D]
                K = self.K
            else:
                K = state_covariance @ self.H.mT @ torch.inverse(observation_covariance)                # [N... x S_D x O_D]
            state_covariances.append(
                state_covariance := (torch.eye(self.S_D) - K @ self.H) @ state_covariance               # [N... x S_D x S_D]
            )

            # Innovation
            state_estimations.append(
                state_estimation := state_estimation + (observations_ - observation_estimation) @ K.mT  # [N... x B x S_D]
            )

        return TensorDict({
            "state_estimation": torch.stack(state_estimations, dim=-2),                                 # [N... x B x L x S_D]
            "state_covariance": torch.stack(state_covariances, dim=-3).unsqueeze(-4).expand(*self.group_shape, B, -1, -1, -1),              # [N... x B x L x S_D x S_D]
            "observation_estimation": torch.stack(observation_estimations, dim=-2),                     # [N... x B x L x O_D]
            "observation_covariance": torch.stack(observation_covariances, dim=-3).unsqueeze(-4).expand(*self.group_shape, B, -1, -1, -1)   # [N... x B x L x O_D x O_D]
        }, batch_size=(*self.group_shape, B, L))




