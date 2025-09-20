from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.discrete_are import solve_discrete_are
from system.module_group import ModuleGroup


class EnvironmentGroup(ModuleGroup):
    def __init__(self, problem_shape: Namespace, group_shape: Tuple[int, ...]):
        ModuleGroup.__init__(self, group_shape)
        self.problem_shape = problem_shape

    def sample_initial_state(
        self,
        batch_size: int                             # B
    ) -> TensorDict:             # [N... x B x ...]
        raise NotImplementedError()

    def step(
        self,
        state: TensorDict,       # [N... x B x ...]
        action: TensorDict       # [N... x B x ...]
    ) -> TensorDict:             # [N... x B x ...]
        raise NotImplementedError()


class LTIEnvironment(EnvironmentGroup):
    def __init__(self, problem_shape: Namespace, params: TensorDict, initial_state_scale: float, settings: Namespace,):
        EnvironmentGroup.__init__(self, problem_shape, params.shape)

        for param_name in ("F", "H", "sqrt_S_W", "sqrt_S_V",):
            if isinstance(param := params[param_name], nn.Parameter):
                self.register_parameter(param_name, param)
            else:
                self.register_buffer(param_name, param)

        self.B = nn.ParameterDict({
            k: params["B", k]
            for k in vars(self.problem_shape.controller)
        })

        # if not torch.all(torch.linalg.eigvals(self.F).abs() < 1 + 1e-5):
        #     raise RuntimeError(f"Eigenvalues of F matrix {self.F.clone().detach()} are unstable.")

        # SECTION: Define system group dimensions
        self.S_D = self.F.shape[-1]                 # State dimension
        self.O_D = self.H.shape[-2]

        # SECTION: Compute all the system matrices
        self.register_buffer("S_W", self.sqrt_S_W @ self.sqrt_S_W.mT)                                               # [N... x S_D x S_D]
        self.register_buffer("S_V", self.sqrt_S_V @ self.sqrt_S_V.mT)                                               # [N... x O_D x O_D]

        L, V = torch.linalg.eig(self.F)                                                                             # [N... x S_D], [N... x S_D x S_D]
        Vinv = torch.inverse(V)                                                                                     # [N... x S_D x S_D]
        self.register_buffer("S_state_inf", (V @ (
            (Vinv @ torch.complex(self.S_W, torch.zeros_like(self.S_W)) @ Vinv.mT) / (1 - L.unsqueeze(-1) * L.unsqueeze(-2))
        ) @ V.mT).real)                                                                                             # [N... x S_D x S_D]

        self.include_analytical: bool = getattr(settings, "include_analytical", True)
        if self.include_analytical:
            S_state_inf_intermediate = solve_discrete_are(self.F.mT, self.H.mT, self.S_W, self.S_V)                     # [N... x S_D x S_D]
            self.register_buffer("S_prediction_err_inf", self.H @ S_state_inf_intermediate @ self.H.mT + self.S_V)      # [N... x O_D x O_D]
            self.register_buffer("K", S_state_inf_intermediate @ self.H.mT @ torch.inverse(self.S_prediction_err_inf))  # [N... x S_D x O_D]
            self.register_buffer("irreducible_loss", utils.batch_trace(self.S_prediction_err_inf))                      # [N...]

        self.initial_state_scale = initial_state_scale

    def sample_initial_state(
        self,
        batch_size: int                             # B
    ) -> TensorDict:             # [N... x B x ...]
        w = torch.randn((*self.group_shape, batch_size, self.S_D)) @ self.sqrt_S_W.mT                               # [N... x B x S_D]
        v = torch.randn((*self.group_shape, batch_size, self.O_D)) @ self.sqrt_S_V.mT                               # [N... x B x O_D]

        x = self.initial_state_scale * torch.randn((*self.group_shape, batch_size, self.S_D), requires_grad=True) @ utils.sqrtm(self.S_state_inf).mT    # [N... x B x S_D]
        y = x @ self.H.mT + v
        noiseless_y = torch.zeros_like(y)

        if self.include_analytical:
            target = {
                "target_observation_estimation": torch.zeros_like(y),                                               # [C... x N... x B x O_D]
                "target_state_estimation": y @ self.K.mT,                                                           # [C... x N... x B x S_D]
            }
        else:
            target = {}

        return TensorDict({
            "state": x,
            "observation": y,
            "noiseless_observation": noiseless_y,
            # "w": w, "v": v,
            **target,
        }, batch_size=(*self.group_shape, batch_size))

    def step(
        self,
        state: TensorDict,       # [C... x N... x B x ...]
        action: TensorDict       # [C... x N... x B x ...]
    ) -> TensorDict:             # [C... x N... x B x ...]
        batch_size = state.shape[-1]

        w = torch.randn((*self.group_shape, batch_size, self.S_D)) @ self.sqrt_S_W.mT                               # [N... x B x S_D]
        v = torch.randn((*self.group_shape, batch_size, self.O_D)) @ self.sqrt_S_V.mT                               # [N... x B x O_D]

        x_ = state["state"]                                                                                         # [C... x N... x B x S_D]
        u = sum(action[ac_name] @ self.B[ac_name].mT for ac_name in vars(self.problem_shape.controller))            # [C... x N... x B x S_D]

        x = x_ @ self.F.mT + u + w
        y = x @ self.H.mT + v
        noiseless_y = (x_ @ self.F.mT + u) @ self.H.mT

        if self.include_analytical:
            target_xh_ = state["target_state_estimation"]                                                           # [C... x N... x B x S_D]
            target_xh = target_xh_ @ self.F.mT + u                                                                  # [C... x N... x B x S_D]
            target_yh = target_xh @ self.H.mT                                                                       # [C... x N... x B x O_D]
            target_xh = target_xh + (y - target_yh) @ self.K.mT                                                     # [C... x N... x B x S_D]
            target = {
                "target_state_estimation": target_xh,
                "target_observation_estimation": target_yh,
            }
        else:
            target = {}

        return TensorDict({
            "state": x,
            "observation": y,
            "noiseless_observation": noiseless_y,
            # "w": w, "v": v,
            **target,
        }, batch_size=x.shape[:-1])


class LTIZeroNoiseEnvironment(LTIEnvironment):
    def __init__(self, problem_shape: Namespace, params: TensorDict, initial_state_scale: float, settings: Namespace,):
        nn.Module.__init__(self)
        for param_name in ("F", "H", "sqrt_S_W", "sqrt_S_V",):
            if isinstance(param := params[param_name], nn.Parameter):
                self.register_parameter(param_name, param)
            else:
                self.register_buffer(param_name, param)

        assert torch.norm(self.sqrt_S_W) == 0, f"Norm of W in noiseless environment should be 0 but got {torch.norm(self.sqrt_S_W)}."
        assert torch.norm(self.sqrt_S_V) == 0, f"Norm of V in noiseless environment should be 0 but got {torch.norm(self.sqrt_S_V)}."
        assert not getattr(settings, "include_analytical", True), "Cannot include analysis for zero noise environments."
        
        LTIEnvironment.__init__(self, problem_shape, params, initial_state_scale, settings)


    def sample_initial_state(
        self,
        batch_size: int                             # B
    ) -> TensorDict:             # [N... x B x ...]
        x = (self.initial_state_scale / self.S_D ** 0.5) * torch.randn((*self.group_shape, batch_size, self.S_D,), requires_grad=True)   # [N... x B x S_D]
        
        y = x @ self.H.mT
        noiseless_y = torch.zeros_like(y)

        if self.include_analytical:
            target = {
                "target_observation_estimation": torch.zeros_like(y),                                               # [C... x N... x B x O_D]
                "target_state_estimation": y @ self.K.mT,                                                           # [C... x N... x B x S_D]
            }
        else:
            target = {}

        return TensorDict({
            "state": x,
            "observation": y,
            "noiseless_observation": noiseless_y,
            **target,
        }, batch_size=(*self.group_shape, batch_size))




