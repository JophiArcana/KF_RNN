from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.discrete_are import solve_discrete_are
from system.simple.base import SystemGroup, SystemDistribution


class LTISystem(SystemGroup):
    class Distribution(SystemDistribution):
        def __init__(self):
            SystemDistribution.__init__(self, LTISystem)

    def __init__(self, problem_shape: Namespace, params: TensorDict[str, torch.tensor]):
        SystemGroup.__init__(self, problem_shape, params.shape)
        for param_name in ("F", "H", "sqrt_S_W", "sqrt_S_V"):
            self.register_parameter(param_name, nn.Parameter(params[param_name]))
        self.B = nn.ParameterDict(params.get("B", {}))

        # SECTION: Define system group dimensions
        self.S_D = self.F.shape[-1]                                                 # State dimension
        self.O_D = self.H.shape[-2]

        # SECTION: Compute all the system matrices
        if not torch.all(torch.linalg.eigvals(self.F).abs() < 1):
            raise RuntimeError(f"Eigenvalues of F matrix {self.F.clone().detach()} are unstable.")

        self.register_buffer("S_W", self.sqrt_S_W @ self.sqrt_S_W.mT)                                               # [N... x S_D x S_D]
        self.register_buffer("S_V", self.sqrt_S_V @ self.sqrt_S_V.mT)                                               # [N... x O_D x O_D]

        L, V = torch.linalg.eig(self.F)                                                                             # [N... x S_D], [N... x S_D x S_D]
        Vinv = torch.inverse(V)                                                                                     # [N... x S_D x S_D]
        self.register_buffer("S_state_inf", (V @ (
            (Vinv @ torch.complex(self.S_W, torch.zeros_like(self.S_W)) @ Vinv.mT) / (1 - L.unsqueeze(-1) * L.unsqueeze(-2))
        ) @ V.mT).real)                                                                                             # [N... x S_D x S_D]
        self.register_buffer("S_observation_inf", self.H @ self.S_state_inf @ self.H.mT + self.S_V)                 # [N... x S_D x S_D]

        S_state_inf_intermediate = solve_discrete_are(self.F.mT, self.H.mT, self.S_W, self.S_V)                     # [N... x S_D x S_D]
        self.register_buffer("S_prediction_err_inf", self.H @ S_state_inf_intermediate @ self.H.mT + self.S_V)      # [N... x O_D x O_D]
        self.register_buffer("K", S_state_inf_intermediate @ self.H.mT @ torch.inverse(self.S_prediction_err_inf))  # [N... x S_D x O_D]
        self.register_buffer("zero_predictor_loss", utils.batch_trace(self.S_observation_inf))                      # [N...]
        self.register_buffer("irreducible_loss", utils.batch_trace(self.S_prediction_err_inf))

    def sample_initial_state(self,
                             batch_size: int                # B
    ) -> TensorDict[str, torch.Tensor]:                     # [N... x B x ...]
        state = torch.randn((*self.group_shape, batch_size, self.S_D)) @ utils.sqrtm(self.S_state_inf).mT           # [N... x B x S_D]
        w = torch.randn((*self.group_shape, batch_size, self.S_D)) @ self.sqrt_S_W.mT                               # [N... x B x S_D]
        v = torch.randn((*self.group_shape, batch_size, self.O_D)) @ self.sqrt_S_V.mT                               # [N... x B x O_D]
        observation = state @ self.H.mT + v                                                                         # [N... x B x O_D]

        target_state_estimation = torch.zeros_like(state)                                                           # [N... x B x S_D]
        target_observation_estimation = torch.zeros_like(observation)                                               # [N... x B x O_D]

        return TensorDict({
            "state": state,
            "observation": observation,
            # "w": w, "v": v,
            "target_state_estimation": target_state_estimation,
            "target_observation_estimation": target_observation_estimation
        }, batch_size=(*self.group_shape, batch_size))

    def step(self,
             state: TensorDict[str, torch.Tensor],          # [C... x N... x B x ...]
             action: TensorDict[str, torch.Tensor]          # [C... x N... x B x ...]
    ) -> TensorDict[str, torch.Tensor]:                     # [C... x N... x B x ...]
        batch_size = state.shape[-1]

        w = torch.randn((*self.group_shape, batch_size, self.S_D)) @ self.sqrt_S_W.mT                               # [N... x B x S_D]
        v = torch.randn((*self.group_shape, batch_size, self.O_D)) @ self.sqrt_S_V.mT                               # [N... x B x O_D]

        x_, target_xh_ = state["state"], state["target_state_estimation"]                                           # [C... x N... x B x S_D]
        y_, target_yh_ = state["observation"], state["target_observation_estimation"]                               # [C... x N... x B x O_D]

        u = sum(ac @ self.B[ac_name].mT for ac_name, ac in action.items())

        x = x_ @ self.F.mT + u + w
        y = x @ self.H.mT + v

        target_xh = target_xh_ + u                                                                                  # [C... x N... x B x S_D]
        target_yh = target_xh @ self.H.mT                                                                           # [C... x N... x B x O_D]
        target_xh = target_xh + (y_ - target_yh_) @ self.K.mT                                                       # [C... x N... x B x S_D]

        return TensorDict({
            "state": x,
            "observation": y,
            # "w": w, "v": v,
            "target_state_estimation": target_xh,
            "target_observation_estimation": target_yh
        }, batch_size=x.shape[:-1])


class MOPDistribution(LTISystem.Distribution):
    def __init__(self, F_mode: str, H_mode: str, W_std: float, V_std: float) -> None:
        LTISystem.Distribution.__init__(self)
        assert F_mode in ("gaussian", "uniform"), f"F_mode must be one of (gaussian, uniform) but got {F_mode}."
        self.F_mode = F_mode

        assert H_mode in ("gaussian", "uniform"), f"H_mode must be one of (gaussian, uniform) but got {H_mode}."
        self.H_mode = H_mode

        self.W_std, self.V_std = W_std, V_std

    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> TensorDict[str, torch.Tensor]:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation
        if self.F_mode == "gaussian":
            F = torch.randn((*shape, S_D, S_D))
        else:
            F = torch.zeros((*shape, S_D, S_D)).uniform_(-1., 1.)
        F *= (0.95 / torch.linalg.eigvals(F).abs().max(dim=-1).values.unsqueeze(-1).unsqueeze(-2))

        B = TensorDict({
            k: torch.randn((*shape, S_D, I_D)) / (3 ** 0.5)
            for k, I_D in vars(SHP.problem_shape.controller).items()
        }, batch_size=(*shape, S_D))

        if self.H_mode == "gaussian":
            H = torch.randn((*shape, O_D, S_D)) / (3 ** 0.5)
        else:
            H = torch.zeros((*shape, O_D, S_D)).uniform_(-1., 1.)

        sqrt_S_W = (torch.eye(S_D) * self.W_std).expand(*shape, S_D, S_D)
        sqrt_S_V = (torch.eye(O_D) * self.V_std).expand(*shape, O_D, O_D)

        return TensorDict({
            "F": F, "B": B, "H": H, "sqrt_S_W": sqrt_S_W, "sqrt_S_V": sqrt_S_V
        }, batch_size=shape)




