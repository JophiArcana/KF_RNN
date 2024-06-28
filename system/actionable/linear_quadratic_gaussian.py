from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.discrete_are import solve_discrete_are
from system.controller import LinearControllerGroup
from system.simple.base import SystemDistribution
from system.simple.linear_time_invariant import LTISystem, MOPDistribution
from system.actionable.base import ActionableSystemGroup


class LQGController(LinearControllerGroup):
    def __init__(self, system: LTISystem, params: TensorDict[str, torch.Tensor], control_noise_std: float):
        LinearControllerGroup.__init__(self, system.problem_shape, params.shape)

        self.Q = nn.ParameterDict(params["Q"].items())
        self.R = nn.ParameterDict(params["R"].items())

        S = TensorDict({
            k: solve_discrete_are(system.F, system.B[k], self.Q[k], self.R[k])
            for k in vars(self.problem_shape.controller)
        }, batch_size=self.group_shape)
        for k in vars(self.problem_shape.controller):
            self.L.register_buffer(k, torch.inverse(system.B[k].mT @ S[k] @ system.B[k] + self.R[k]) @ system.B[k].mT @ S[k] @ system.F)

        self.control_noise_std = control_noise_std

    def forward(self,
                history: TensorDict[str, torch.Tensor]      # [N... x B x L x ...]
    ) -> TensorDict[str, torch.Tensor]:
        return LinearControllerGroup.forward(self, history).apply(
            lambda t: t + self.control_noise_std * torch.randn_like(t)
        )

class LQGSystem(ActionableSystemGroup, LTISystem):
    class Distribution(SystemDistribution):
        def __init__(self):
            SystemDistribution.__init__(self, LQGSystem)

    def __init__(self, problem_shape: Namespace, params: TensorDict[str, torch.Tensor], control_noise_std: float = 0.0):
        LTISystem.__init__(self, problem_shape, params)
        ActionableSystemGroup.set_controller(self, LQGController(self, params["controller"], control_noise_std))


class LQGDistribution(LQGSystem.Distribution, MOPDistribution):
    def __init__(self,
                 F_mode: str,
                 H_mode: str,
                 W_std: float,
                 V_std: float,
                 Q_scale: float,
                 R_scale: float
    ) -> None:
        MOPDistribution.__init__(self, F_mode, H_mode, W_std, V_std)
        LQGSystem.Distribution.__init__(self)

        self.Q_scale = Q_scale
        self.R_scale = R_scale

    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> TensorDict[str, torch.Tensor]:
        result = MOPDistribution.sample_parameters(self, SHP, shape)

        Q_ = TensorDict({
            k: torch.randn((*shape, SHP.S_D, SHP.S_D)) * self.Q_scale
            for k, d in vars(SHP.problem_shape.controller).items()
        }, batch_size=shape)
        R_ = TensorDict({
            k: torch.randn((*shape, d, d)) * self.R_scale
            for k, d in vars(SHP.problem_shape.controller).items()
        }, batch_size=shape)

        result["controller"] = TensorDict({
            "Q": Q_, "R": R_
        }, batch_size=shape).apply(lambda M: utils.sqrtm(M @ M.mT))
        return result




