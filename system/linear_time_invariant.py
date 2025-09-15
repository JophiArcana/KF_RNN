from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.discrete_are import solve_discrete_are
from model.zero_predictor import ZeroController
from model.copy_predictor import CopyPredictor
from system.base import SystemGroup, SystemDistribution
from system.controller import LinearControllerGroup
from system.environment import LTIEnvironment, LTIZeroNoiseEnvironment


class LQGController(LinearControllerGroup):
    def __init__(self, problem_shape: Namespace, params: TensorDict, control_noise_std: float):
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
            history: TensorDict  # [N... x B x L x ...]
    ) -> TensorDict:
        return LinearControllerGroup.act(self, history).apply(
            lambda t: t + (self.control_noise_std * t.norm() / (t.numel() ** 0.5)) * torch.randn_like(t)
        )


class LTISystem(SystemGroup):
    class Distribution(SystemDistribution):
        def __init__(self):
            SystemDistribution.__init__(self, LTISystem)

    def __init__(self, hyperparameters: Namespace, params: TensorDict):
        # SECTION: Set up controller
        problem_shape = hyperparameters.problem_shape
        auxiliary = hyperparameters.auxiliary
        settings = hyperparameters.settings
        SystemGroup.__init__(
            self,
            hyperparameters,
            LTIEnvironment(problem_shape, params["environment"], getattr(auxiliary, "initial_state_scale", 1.0), settings,),
            LQGController(problem_shape, params, getattr(auxiliary, "control_noise_std", 0.0)),
        )

        # SECTION: Set up the effective system that produces the same distribution of data, but without controls.
        if getattr(settings, "include_analytical", True):
            self.setup_analytical()
    
    def setup_analytical(self):
        F = self.environment.F
        H = self.environment.H
        K = self.environment.K
        I, zeros = torch.eye(self.environment.S_D), torch.zeros((self.environment.S_D, self.environment.S_D))

        KH = K @ H
        BL = zeros + sum(
            self.environment.B[k] @ getattr(self.controller.L, k)
            for k in vars(self.hyperparameters.problem_shape.controller)
        )
        F_BL, I_KH = F - BL, I - KH

        # SECTION: Compute effective transition, observation, and control matrices of LQG system
        self.register_buffer("F_augmented", torch.cat([
            torch.cat([F_BL + BL @ I_KH, -BL @ I_KH,], dim=-1),
            F_BL @ torch.cat([KH, I_KH,], dim=-1)
        ], dim=-2))
        self.register_buffer("H_augmented", torch.cat([H, torch.zeros_like(H)], dim=-1))
        L_augmented = nn.Module()
        for k in vars(self.hyperparameters.problem_shape.controller):
            L = getattr(self.controller.L, k)
            L_augmented.register_buffer(k, L @ torch.cat([KH, I_KH,], dim=-1))
        self.register_module("L_augmented", L_augmented)

        # SECTION: Register irreducible loss
        zero_predictor_loss = ZeroController.analytical_error(None, self.td())
        self.register_module("zero_predictor_loss", utils.buffer_dict(zero_predictor_loss))

        irreducible_loss = TensorDict.from_dict({
            "environment": {"observation": self.environment.irreducible_loss.clone()},
            "controller": zero_predictor_loss["controller"].apply(torch.zeros_like)
        }, batch_size=self.group_shape)
        self.register_module("irreducible_loss", utils.buffer_dict(irreducible_loss))

        assert len(vars(self.environment.problem_shape.controller)) == 0
        copy_predictor_loss = CopyPredictor.analytical_error(None, self.td())
        self.register_module("copy_predictor_loss", utils.buffer_dict(copy_predictor_loss))


class LTIZeroNoiseSystem(LTISystem):
    class Distribution(SystemDistribution):
        def __init__(self):
            SystemDistribution.__init__(self, LTIZeroNoiseSystem)

    def __init__(self, hyperparameters: Namespace, params: TensorDict):
        # SECTION: Set up controller
        problem_shape = hyperparameters.problem_shape
        auxiliary = hyperparameters.auxiliary
        settings = hyperparameters.settings
        SystemGroup.__init__(
            self,
            hyperparameters,
            LTIZeroNoiseEnvironment(problem_shape, params["environment"], getattr(auxiliary, "initial_state_scale", 1.0), settings,),
            LQGController(problem_shape, params, getattr(auxiliary, "control_noise_std", 0.0)),
        )


class MOPDistribution(LTISystem.Distribution):
    def __init__(
            self,
            F_mode: str,
            H_mode: str,
            W_std: float,
            V_std: float,
            B_scale: float = 1.0,
            Q_scale: float = 0.1,
            R_scale: float = 1.0,
    ) -> None:
        LTISystem.Distribution.__init__(self)
        self.F_mode = F_mode
        self.H_mode = H_mode

        self.W_std, self.V_std = W_std, V_std
        self.B_scale, self.Q_scale, self.R_scale = B_scale, Q_scale, R_scale

    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> TensorDict:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation

        match self.F_mode:
            case "gaussian":
                F = torch.randn((*shape, S_D, S_D,))
            case "uniform":
                F = torch.zeros((*shape, S_D, S_D,)).uniform_(-1., 1.)
            case _:
                raise ValueError(self.F_mode)
        F *= (0.95 / torch.linalg.eigvals(F).abs().max(dim=-1).values[..., None, None])
        B = TensorDict({
            k: self.B_scale * torch.randn((*shape, S_D, I_D)) / (3 ** 0.5)
            for k, I_D in vars(SHP.problem_shape.controller).items()
        }, batch_size=(*shape, S_D))

        match self.H_mode:
            case "gaussian":
                H = torch.randn((*shape, O_D, S_D,)) / (3 ** 0.5)
            case "uniform":
                H = torch.zeros((*shape, O_D, S_D,)).uniform_(-1., 1.)
            case _:
                raise ValueError(self.H_mode)

        sqrt_S_W = (torch.eye(S_D) * self.W_std).expand((*shape, S_D, S_D,))
        sqrt_S_V = (torch.eye(O_D) * self.V_std).expand((*shape, O_D, O_D,))

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
            "controller": {"Q": Q, "R": R},
        }, batch_size=shape).apply(nn.Parameter)


class OrthonormalDistribution(LTIZeroNoiseSystem.Distribution):
    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> TensorDict:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation
        assert S_D == O_D, "Orthonormal system setting is assumed to be fully observed."
        assert len(vars(SHP.problem_shape.controller).items()) == 0, "Orthonormal system setting is assumed to have no controls."

        _Z = torch.randn((*shape, S_D, S_D,))
        _Q, _R = torch.linalg.qr(_Z)
        F = _Q * torch.sgn(torch.diagonal(_R, dim1=-2, dim2=-1))[..., None]

        B = TensorDict({}, batch_size=(*shape, S_D,))

        H = torch.eye(S_D).expand((*shape, S_D, O_D,))

        sqrt_S_W = torch.zeros((*shape, S_D, S_D,))
        sqrt_S_V = torch.zeros((*shape, O_D, O_D,))

        Q = TensorDict({}, batch_size=(*shape, S_D, S_D,))
        R = TensorDict({}, batch_size=shape)

        return TensorDict.from_dict({
            "environment": {"F": F, "B": B, "H": H, "sqrt_S_W": sqrt_S_W, "sqrt_S_V": sqrt_S_V},
            "controller": {"Q": Q, "R": R},
        }, batch_size=shape).apply(nn.Parameter)


class ContinuousDistribution(LTISystem.Distribution):
    def __init__(
            self,
            F_mode: str,
            H_mode: str,
            eps: float,
            W_std: float,
            V_std: float,
    ) -> None:
        LTISystem.Distribution.__init__(self)
        self.F_mode = F_mode
        self.H_mode = H_mode

        self.eps = eps
        self.W_std, self.V_std = W_std, V_std

    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> TensorDict:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation

        match self.F_mode:
            case "gaussian":
                F = torch.randn((*shape, S_D, S_D))
            case "uniform":
                F = torch.zeros((*shape, S_D, S_D)).uniform_(-1., 1.)
            case _:
                raise ValueError(self.F_mode)
        F /= torch.linalg.eigvals(F).abs().max(dim=-1).values[..., None, None]
        F = (1 - 2 * self.eps) * torch.eye(S_D) - self.eps * F

        B = TensorDict({}, batch_size=(*shape, S_D))

        match self.H_mode:
            case "gaussian":
                H = torch.randn((*shape, O_D, S_D)) / (3 ** 0.5)
            case "uniform":
                H = torch.zeros((*shape, O_D, S_D)).uniform_(-1., 1.)
            case _:
                raise ValueError(self.H_mode)

        sqrt_S_W = (torch.eye(S_D) * self.W_std * self.eps).expand((*shape, S_D, S_D,))
        sqrt_S_V = (torch.eye(O_D) * self.V_std * self.eps).expand((*shape, O_D, O_D,))

        Q = TensorDict({}, batch_size=(*shape, S_D, S_D))
        R = TensorDict({}, batch_size=shape)

        return TensorDict.from_dict({
            "environment": {"F": F, "B": B, "H": H, "sqrt_S_W": sqrt_S_W, "sqrt_S_V": sqrt_S_V},
            "controller": {"Q": Q, "R": R},
        }, batch_size=shape).apply(nn.Parameter)


class ContinuousNoiselessDistribution(LTIZeroNoiseSystem.Distribution):
    def __init__(
            self,
            F_mode: str,
            H_mode: str,
            eps: float,
    ) -> None:
        LTIZeroNoiseSystem.Distribution.__init__(self)
        self.F_mode = F_mode
        self.H_mode = H_mode
        self.eps = eps

    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> TensorDict:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation
        assert len(vars(SHP.problem_shape.controller).items()) == 0, "Orthonormal system setting is assumed to have no controls."

        match self.F_mode:
            case "gaussian":
                F = torch.randn((*shape, S_D, S_D))
            case "uniform":
                F = torch.zeros((*shape, S_D, S_D)).uniform_(-1., 1.)
            case _:
                raise ValueError(self.F_mode)
        F /= torch.linalg.eigvals(F).abs().max(dim=-1).values[..., None, None]
        F = (1 - 2 * self.eps) * torch.eye(S_D) - self.eps * F

        B = TensorDict({}, batch_size=(*shape, S_D))

        match self.H_mode:
            case "gaussian":
                H = torch.randn((*shape, O_D, S_D)) / (3 ** 0.5)
            case "uniform":
                H = torch.zeros((*shape, O_D, S_D)).uniform_(-1., 1.)
            case _:
                raise ValueError(self.H_mode)

        sqrt_S_W = torch.zeros((*shape, S_D, S_D,))
        sqrt_S_V = torch.zeros((*shape, O_D, O_D,))

        Q = TensorDict({}, batch_size=(*shape, S_D, S_D))
        R = TensorDict({}, batch_size=shape)

        return TensorDict.from_dict({
            "environment": {"F": F, "B": B, "H": H, "sqrt_S_W": sqrt_S_W, "sqrt_S_V": sqrt_S_V},
            "controller": {"Q": Q, "R": R},
        }, batch_size=shape).apply(nn.Parameter)




