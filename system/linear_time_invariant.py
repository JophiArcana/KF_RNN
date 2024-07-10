from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.discrete_are import solve_discrete_are
from system.base import SystemGroup, SystemDistribution
from system.controller import LinearControllerGroup
from system.environment import LTIEnvironment
from model.sequential import SequentialPredictor


class LQGController(LinearControllerGroup):
    def __init__(self, problem_shape: Namespace, params: TensorDict[str, torch.Tensor], control_noise_std: float):
        LinearControllerGroup.__init__(self, problem_shape, params.shape)

        self.Q = nn.ParameterDict({
            k: params["controller", "Q", k]
            for k in vars(self.problem_shape.controller)
        })
        self.R = nn.ParameterDict({
            k: params["controller", "R", k]
            for k in vars(self.problem_shape.controller)
        })

        for k in vars(self.problem_shape.controller):
            F, B, Q, R = params["environment", "F"], params["environment", "B", k], self.Q[k], self.R[k]
            S = solve_discrete_are(F, B, Q, R)
            self.L.register_buffer(k, torch.inverse(B.mT @ S @ B + R) @ B.mT @ S @ F)

        self.control_noise_std = control_noise_std

    def act(self,
            history: TensorDict[str, torch.Tensor]  # [N... x B x L x ...]
    ) -> TensorDict[str, torch.Tensor]:
        return LinearControllerGroup.act(self, history).apply(
            lambda t: t + (self.control_noise_std * t.norm() / (t.numel() ** 0.5)) * torch.randn_like(t)
        )


class LTISystem(SystemGroup):
    class Distribution(SystemDistribution):
        def __init__(self):
            SystemDistribution.__init__(self, LTISystem)

    def __init__(self, problem_shape: Namespace, params: TensorDict[str, torch.tensor], control_noise_std: float = 0.0):
        # SECTION: Set up controller
        SystemGroup.__init__(self,
                             problem_shape,
                             LTIEnvironment(problem_shape, params["environment"]),
                             LQGController(problem_shape, params, control_noise_std)
        )

        # SECTION: Set up the effective system that produces the same distribution of data, but without controls.
        F = self.environment.F
        H, K = self.environment.H, self.environment.K
        I, zeros = torch.eye(self.environment.S_D), torch.zeros((self.environment.S_D, self.environment.S_D))

        KH = self.environment.K @ H
        BL = zeros + sum(
            self.environment.B[k] @ getattr(self.controller.L, k)
            for k in vars(self.problem_shape.controller)
        )
        F_BL, I_KH = F - BL, I - KH
        conj_S_V = K @ self.environment.S_V @ K.mT

        self.effective = LTIEnvironment(
            Namespace(
                environment=self.problem_shape.environment,
                controller=Namespace()
            ), TensorDict({
                "F": torch.cat([
                    torch.cat([F_BL + BL @ I_KH, -BL @ I_KH], dim=-1),
                    F_BL @ torch.cat([KH, I_KH], dim=-1)
                ], dim=-2),
                "H": torch.cat([H, torch.zeros_like(H)], dim=-1),
                "sqrt_S_W": utils.sqrtm(torch.cat([
                    torch.cat([self.environment.S_W + BL @ conj_S_V @ BL.mT, -BL @ conj_S_V @ F_BL.mT], dim=-1),
                    torch.cat([F_BL @ conj_S_V @ -BL.mT, F_BL @ conj_S_V @ F_BL.mT], dim=-1)
                ], dim=-2)),
                "sqrt_S_V": self.environment.sqrt_S_V.clone()
            }, batch_size=self.group_shape)
        )
        effective_L = nn.Module()
        for k in vars(self.problem_shape.controller):
            L = getattr(self.controller.L, k)
            effective_L.register_buffer(k, L @ torch.cat([KH, I_KH], dim=-1))
        self.effective.register_module("L", effective_L)

        self.register_buffer("irreducible_loss", SequentialPredictor.analytical_error(
            TensorDict.from_dict({
                **self.environment.td(),
                **self.controller.td(),
            }, batch_size=self.group_shape),
            self.td()
        ))
        self.register_buffer("zero_predictor_loss", utils.batch_trace(
            self.effective.H @ self.effective.S_state_inf @ self.effective.H.mT + self.effective.S_V
        ))


class MOPDistribution(LTISystem.Distribution):
    def __init__(self,
                 F_mode: str,
                 H_mode: str,
                 W_std: float,
                 V_std: float,
                 B_scale: float = 1.0,
                 Q_scale: float = 0.1,
                 R_scale: float = 1.0
    ) -> None:
        LTISystem.Distribution.__init__(self)
        assert F_mode in ("gaussian", "uniform"), f"F_mode must be one of (gaussian, uniform) but got {F_mode}."
        self.F_mode = F_mode

        assert H_mode in ("gaussian", "uniform"), f"H_mode must be one of (gaussian, uniform) but got {H_mode}."
        self.H_mode = H_mode

        self.W_std, self.V_std = W_std, V_std
        self.B_scale, self.Q_scale, self.R_scale = B_scale, Q_scale, R_scale

    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> TensorDict[str, torch.Tensor]:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation
        if self.F_mode == "gaussian":
            F = torch.randn((*shape, S_D, S_D))
        else:
            F = torch.zeros((*shape, S_D, S_D)).uniform_(-1., 1.)
        F *= (0.95 / torch.linalg.eigvals(F).abs().max(dim=-1).values.unsqueeze(-1).unsqueeze(-2))

        B = TensorDict({
            k: self.B_scale * torch.randn((*shape, S_D, I_D)) / (3 ** 0.5)
            for k, I_D in vars(SHP.problem_shape.controller).items()
        }, batch_size=(*shape, S_D))

        if self.H_mode == "gaussian":
            H = torch.randn((*shape, O_D, S_D)) / (3 ** 0.5)
        else:
            H = torch.zeros((*shape, O_D, S_D)).uniform_(-1., 1.)

        sqrt_S_W = (torch.eye(S_D) * self.W_std).expand(*shape, S_D, S_D)
        sqrt_S_V = (torch.eye(O_D) * self.V_std).expand(*shape, O_D, O_D)

        to_psd = lambda M: utils.sqrtm(M @ M.mT)
        Q = TensorDict({
            k: torch.randn((*shape, SHP.S_D, SHP.S_D)) * self.Q_scale
            for k, d in vars(SHP.problem_shape.controller).items()
        }, batch_size=shape).apply(to_psd)
        R = TensorDict({
            k: torch.randn((*shape, d, d)) * self.R_scale
            for k, d in vars(SHP.problem_shape.controller).items()
        }, batch_size=shape).apply(to_psd)

        return TensorDict.from_dict({
            "environment": {"F": F, "B": B, "H": H, "sqrt_S_W": sqrt_S_W, "sqrt_S_V": sqrt_S_V},
            "controller": {"Q": Q, "R": R}
        }, batch_size=shape).apply(nn.Parameter)




