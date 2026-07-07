
import einops
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

import ecliseutils as eu
from ecliseutils.are import solve_discrete_are
from kf_rnn.model.zero_predictor import ZeroController
from kf_rnn.model.copy_predictor import CopyPredictor
from kf_rnn.system.base import SystemGroup, SystemDistribution
from kf_rnn.system.controller import LinearControllerGroup
from kf_rnn.system.environment import LTIEnvironment, LTIZeroNoiseEnvironment
from kf_rnn.infrastructure.config.schema import ProblemShape, SystemConfig, controller_dims, shape_leaves


class LQGController(LinearControllerGroup):
    def __init__(self, problem_shape: ProblemShape, params: TensorDict, control_noise_std: float):
        LinearControllerGroup.__init__(self, problem_shape, params.shape)

        self.Q = nn.ParameterDict({
            k: params["controller", "Q", k]
            for k in controller_dims(self.problem_shape)
        })
        self.R = nn.ParameterDict({
            k: params["controller", "R", k]
            for k in controller_dims(self.problem_shape)
        })

        for k in controller_dims(self.problem_shape):
            F, B, Q, R = params["environment", "F"], params["environment", "B", k], self.Q[k], self.R[k]
            S = solve_discrete_are(F, B, Q, R)
            # L = (B^T S B + R)^{-1} B^T S F; solve against the SPD control-cost matrix.
            self.L.register_buffer(k, torch.linalg.solve(B.mT @ S @ B + R, B.mT @ S @ F))

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

    def __init__(self, hyperparameters: SystemConfig, params: TensorDict):
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
            for k in controller_dims(self.hyperparameters.problem_shape)
        )
        F_BL, I_KH = F - BL, I - KH

        # SECTION: Compute effective transition, observation, and control matrices of LQG system
        self.register_buffer("F_augmented", torch.cat([
            torch.cat([F_BL + BL @ I_KH, -BL @ I_KH,], dim=-1),
            F_BL @ torch.cat([KH, I_KH,], dim=-1)
        ], dim=-2))
        self.register_buffer("H_augmented", torch.cat([H, torch.zeros_like(H)], dim=-1))
        L_augmented = nn.Module()
        for k in controller_dims(self.hyperparameters.problem_shape):
            L = getattr(self.controller.L, k)
            L_augmented.register_buffer(k, L @ torch.cat([KH, I_KH,], dim=-1))
        self.register_module("L_augmented", L_augmented)

        # SECTION: Register irreducible loss
        zero_predictor_loss = ZeroController.analytical_error(None, self.td())
        self.register_module("zero_predictor_loss", eu.buffer_dict(zero_predictor_loss))

        assert len(self.hyperparameters.problem_shape.controller) == 0
        copy_predictor_loss = CopyPredictor.analytical_error(None, self.td())
        self.register_module("copy_predictor_loss", eu.buffer_dict(copy_predictor_loss))

        irreducible_loss = TensorDict({
            (*k.split("."),): torch.zeros(self.group_shape)
            for k in shape_leaves(self.hyperparameters.problem_shape)
        }, batch_size=self.group_shape)
        irreducible_loss["environment", "observation"] = self.environment.irreducible_loss.clone()
        self.register_module("irreducible_loss", eu.buffer_dict(irreducible_loss))

    def generate_dataset(self, batch_size: int, sequence_length: int) -> TensorDict:
        """Fast path for the linear (LQE / optimal-LQR) rollout.

        When the closed loop is a plain linear-Gaussian recursion -- the built-in
        ``LQGController`` with no exploratory control noise and the analytical
        Kalman targets available -- the whole trajectory (states, observations and
        the propagated Kalman targets) is ``z_t = A z_{t-1} + b_t``. We sample it
        in one batched parallel scan instead of the generic per-step Python loop in
        :meth:`SystemGroup.generate_dataset`, which is what makes the abstraction
        slow. Any other controller (e.g. a learned ``NNControllerGroup``) or
        configuration falls back to the generic loop.
        """
        controller = self.controller
        if (
            getattr(self.environment, "include_analytical", False)
            and isinstance(controller, LQGController)
            and float(getattr(controller, "control_noise_std", 0.0)) == 0.0
        ):
            return self._fast_generate_dataset(batch_size, sequence_length)
        return SystemGroup.generate_dataset(self, batch_size, sequence_length)

    def _fast_generate_dataset(self, batch_size: int, sequence_length: int) -> TensorDict:
        env = self.environment
        cdims = controller_dims(self.hyperparameters.problem_shape)

        F, H, K = env.F, env.H, env.K                                              # [N... x S x S], [N... x O x S], [N... x S x O]
        group_shape = F.shape[:-2]
        S_D, O_D = env.S_D, env.O_D
        B, L = batch_size, sequence_length
        kw = dict(device=F.device, dtype=F.dtype)

        # SECTION: Augmented transition for z = [x; xhat] (true state, Kalman estimate)
        #   x_t    = F x_{t-1} - BL xhat_{t-1} + w_t
        #   xhat_t = KHF x_{t-1} + (F - BL - KHF) xhat_{t-1} + (KH w_t + K v_t)
        KH = K @ H                                                                 # [N... x S x S]
        BL = torch.zeros_like(F) + sum(
            env.B[k] @ getattr(self.controller.L, k) for k in cdims
        )                                                                          # [N... x S x S]
        FBL = F - BL                                                               # [N... x S x S]
        KHF = KH @ F                                                               # [N... x S x S]
        A = torch.cat([
            torch.cat([F, -BL], dim=-1),
            torch.cat([KHF, FBL - KHF], dim=-1),
        ], dim=-2)                                                                 # [N... x 2S x 2S]

        def apply_mat(t: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
            # Apply ``M`` ([N... x p x q]) to the last axis of ``t`` ([N... x B x T x q]),
            # broadcasting over the trace (B) axis; returns [N... x B x T x p].
            return t @ M.reshape(*M.shape[:-2], 1, *M.shape[-2:]).mT

        # SECTION: Initial timestep (t = 0), mirroring LTIEnvironment.sample_initial_state
        sqrt_S_state_inf = eu.sqrtm(env.S_state_inf)                              # [N... x S x S]
        x0 = env.initial_state_scale * (torch.randn((*group_shape, B, S_D), **kw) @ sqrt_S_state_inf.mT)
        v0 = torch.randn((*group_shape, B, O_D), **kw) @ env.sqrt_S_V.mT          # [N... x B x O]
        y0 = x0 @ H.mT + v0                                                        # [N... x B x O]
        xhat0 = y0 @ K.mT                                                          # [N... x B x S]
        z0 = torch.cat([x0, xhat0], dim=-1)                                        # [N... x B x 2S]

        if L > 1:
            # SECTION: Driving noise for t = 1 .. L-1 and the parallel scan
            W = apply_mat(torch.randn((*group_shape, B, L - 1, S_D), **kw), env.sqrt_S_W)    # [N... x B x L-1 x S]
            Vn = apply_mat(torch.randn((*group_shape, B, L - 1, O_D), **kw), env.sqrt_S_V)   # [N... x B x L-1 x O]
            b = torch.cat([W, apply_mat(W, KH) + apply_mat(Vn, K)], dim=-1)        # [N... x B x L-1 x 2S]

            B_input = torch.cat([z0[..., None, :], b], dim=-2)                     # [N... x B x L x 2S]
            z_rest = eu.dense_linear_scan(A[..., None, :, :], B_input)             # [N... x B x L-1 x 2S]
            z = torch.cat([z0[..., None, :], z_rest], dim=-2)                      # [N... x B x L x 2S]
            v_all = torch.cat([v0[..., None, :], Vn], dim=-2)                      # [N... x B x L x O]
        else:
            z = z0[..., None, :]                                                   # [N... x B x 1 x 2S]
            v_all = v0[..., None, :]                                               # [N... x B x 1 x O]

        x, xhat = z[..., :S_D], z[..., S_D:]                                       # [N... x B x L x S] each
        observation = apply_mat(x, H) + v_all                                      # [N... x B x L x O]

        # SECTION: Linear readouts that are zero at t = 0 (no prediction precedes it)
        zeros_O = torch.zeros((*group_shape, B, 1, O_D), **kw)
        if L > 1:
            # noiseless_observation_t = H (F x_{t-1} - BL xhat_{t-1}) = H (x_t - w_t)
            noiseless_observation = torch.cat([zeros_O, apply_mat(x[..., 1:, :] - W, H)], dim=-2)
            # target_observation_estimation_t = H (F - BL) xhat_{t-1}
            target_obs_est = torch.cat([zeros_O, apply_mat(apply_mat(xhat[..., :-1, :], FBL), H)], dim=-2)
        else:
            noiseless_observation = zeros_O
            target_obs_est = zeros_O

        batch = (*group_shape, B, L)
        controller_td = {}
        for k, I_D in cdims.items():
            zeros_I = torch.zeros((*group_shape, B, 1, I_D), **kw)
            if L > 1:
                # controller action_t = -L_k xhat_{t-1}
                act = -apply_mat(xhat[..., :-1, :], getattr(self.controller.L, k))
                controller_td[k] = torch.cat([zeros_I, act], dim=-2)
            else:
                controller_td[k] = zeros_I

        return TensorDict({
            "environment": TensorDict({
                "state": x,
                "observation": observation,
                "noiseless_observation": noiseless_observation,
                "target_state_estimation": xhat,
                "target_observation_estimation": target_obs_est,
            }, batch_size=batch),
            "controller": TensorDict(controller_td, batch_size=batch),
        }, batch_size=batch)


class LTIZeroNoiseSystem(LTISystem):
    class Distribution(SystemDistribution):
        def __init__(self):
            SystemDistribution.__init__(self, LTIZeroNoiseSystem)

    def __init__(self, hyperparameters: SystemConfig, params: TensorDict):
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


def _sample_by_mode(
        mode: str,
        size: tuple[int, ...],
        gaussian_scale: float = 1.0,
) -> torch.Tensor:
    """Sample a matrix of the given ``size`` according to a "gaussian" or "uniform" mode."""
    match mode:
        case "gaussian":
            return torch.randn(size) * gaussian_scale
        case "uniform":
            return torch.zeros(size).uniform_(-1., 1.)
        case _:
            raise ValueError(mode)


def _empty_controller_params(
        shape: tuple[int, ...],
        S_D: int,
) -> tuple[TensorDict, TensorDict, TensorDict]:
    """Empty (B, Q, R) parameter containers for control-free system settings."""
    B = TensorDict({}, batch_size=(*shape, S_D))
    Q = TensorDict({}, batch_size=(*shape, S_D, S_D))
    R = TensorDict({}, batch_size=shape)
    return B, Q, R


def _pack_lti_params(
        shape: tuple[int, ...],
        F: torch.Tensor,
        B: TensorDict,
        H: torch.Tensor,
        sqrt_S_W: torch.Tensor,
        sqrt_S_V: torch.Tensor,
        Q: TensorDict,
        R: TensorDict,
) -> TensorDict:
    """Pack sampled LTI matrices into the canonical parameter TensorDict of nn.Parameters."""
    return TensorDict.from_dict({
        "environment": {"F": F, "B": B, "H": H, "sqrt_S_W": sqrt_S_W, "sqrt_S_V": sqrt_S_V},
        "controller": {"Q": Q, "R": R},
    }, batch_size=shape).apply(nn.Parameter)


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

    def sample_parameters(self, SHP: SystemConfig, shape: tuple[int, ...]) -> TensorDict:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation

        F = _sample_by_mode(self.F_mode, (*shape, S_D, S_D,))
        F *= (0.95 / torch.linalg.eigvals(F).abs().max(dim=-1).values[..., None, None])
        B = TensorDict({
            k: self.B_scale * torch.randn((*shape, S_D, I_D)) / (3 ** 0.5)
            for k, I_D in controller_dims(SHP.problem_shape).items()
        }, batch_size=(*shape, S_D))

        H = _sample_by_mode(self.H_mode, (*shape, O_D, S_D,), gaussian_scale=1 / (3 ** 0.5))

        sqrt_S_W = (torch.eye(S_D) * self.W_std).expand((*shape, S_D, S_D,))
        sqrt_S_V = (torch.eye(O_D) * self.V_std).expand((*shape, O_D, O_D,))

        to_psd = lambda M: eu.sqrtm(M @ M.mT)
        Q = TensorDict({
            k: torch.randn((*shape, SHP.S_D, SHP.S_D)) * self.Q_scale
            for k, d in controller_dims(SHP.problem_shape).items()
        }, batch_size=shape).apply(to_psd)
        R = TensorDict({
            k: torch.randn((*shape, d, d)) * self.R_scale
            for k, d in controller_dims(SHP.problem_shape).items()
        }, batch_size=shape).apply(to_psd)

        return _pack_lti_params(shape, F, B, H, sqrt_S_W, sqrt_S_V, Q, R)


class OrthonormalDistribution(LTIZeroNoiseSystem.Distribution):
    def sample_parameters(self, SHP: SystemConfig, shape: tuple[int, ...]) -> TensorDict:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation
        assert S_D == O_D, "Orthonormal system setting is assumed to be fully observed."
        assert len(controller_dims(SHP.problem_shape).items()) == 0, "Orthonormal system setting is assumed to have no controls."

        _Z = torch.randn((*shape, S_D, S_D,))
        _Q, _R = torch.linalg.qr(_Z)
        F = _Q * torch.sgn(torch.diagonal(_R, dim1=-2, dim2=-1))[..., None]

        H = torch.eye(S_D).expand((*shape, S_D, O_D,))

        sqrt_S_W = torch.zeros((*shape, S_D, S_D,))
        sqrt_S_V = torch.zeros((*shape, O_D, O_D,))

        B, Q, R = _empty_controller_params(shape, S_D)

        return _pack_lti_params(shape, F, B, H, sqrt_S_W, sqrt_S_V, Q, R)


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

    def sample_parameters(self, SHP: SystemConfig, shape: tuple[int, ...]) -> TensorDict:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation

        F = _sample_by_mode(self.F_mode, (*shape, S_D, S_D))
        F /= torch.linalg.eigvals(F).abs().max(dim=-1).values[..., None, None]
        F = (1 - 2 * self.eps) * torch.eye(S_D) - self.eps * F

        H = _sample_by_mode(self.H_mode, (*shape, O_D, S_D), gaussian_scale=1 / (3 ** 0.5))

        sqrt_S_W = (torch.eye(S_D) * self.W_std * self.eps).expand((*shape, S_D, S_D,))
        sqrt_S_V = (torch.eye(O_D) * self.V_std * self.eps).expand((*shape, O_D, O_D,))

        B, Q, R = _empty_controller_params(shape, S_D)

        return _pack_lti_params(shape, F, B, H, sqrt_S_W, sqrt_S_V, Q, R)


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

    def sample_parameters(self, SHP: SystemConfig, shape: tuple[int, ...]) -> TensorDict:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation
        assert len(controller_dims(SHP.problem_shape).items()) == 0, "Continuous noiseless system setting is assumed to have no controls."

        F = _sample_by_mode(self.F_mode, (*shape, S_D, S_D))
        F /= torch.linalg.eigvals(F).abs().max(dim=-1).values[..., None, None]
        # F = (1 - 2 * self.eps) * torch.eye(S_D) - self.eps * F
        F = torch.eye(S_D) - self.eps * F

        H = _sample_by_mode(self.H_mode, (*shape, O_D, S_D), gaussian_scale=1 / (3 ** 0.5))

        sqrt_S_W = torch.zeros((*shape, S_D, S_D,))
        sqrt_S_V = torch.zeros((*shape, O_D, O_D,))

        B, Q, R = _empty_controller_params(shape, S_D)

        return _pack_lti_params(shape, F, B, H, sqrt_S_W, sqrt_S_V, Q, R)


class PeriodicDistribution(LTIZeroNoiseSystem.Distribution):
    def __init__(
            self,
            periods: list[int],
            deterministic: bool,
    ) -> None:
        LTIZeroNoiseSystem.Distribution.__init__(self)
        self.periods = torch.tensor(periods)
        self.deterministic = deterministic

    def sample_parameters(self, SHP: SystemConfig, shape: tuple[int, ...]) -> TensorDict:
        S_D, O_D = SHP.S_D, SHP.problem_shape.environment.observation
        assert S_D == 2 * O_D, "State dimension is expected to be twice observation dimension."
        assert len(controller_dims(SHP.problem_shape).items()) == 0, "Periodic system setting is assumed to have no controls."

        if not self.deterministic:
            period_idx = torch.randint(0, len(self.periods), (*shape, O_D,))
        else:
            period_idx = (torch.arange(np.prod(shape).item() * O_D) % len(self.periods)).view((*shape, O_D,))
        period = self.periods[period_idx]

        frequency = (2 * torch.pi) / period
        sign = torch.where(torch.rand_like(frequency) > 0.5, 1.0, -1.0)
        frequency = frequency * sign

        cos, sin = torch.diag_embed(torch.cos(frequency)), torch.diag_embed(torch.sin(frequency))
        _F = torch.stack((
            torch.stack((cos, -sin), dim=-1),
            torch.stack((sin, cos), dim=-1),
        ), dim=-2)
        F = einops.rearrange(_F, "... n1 n2 t1 t2 -> ... (n1 t1) (n2 t2)")

        B = TensorDict({}, batch_size=(*shape, S_D,))

        _H = torch.zeros((O_D, S_D,))
        _H[range(O_D), range(0, S_D, 2)] = 1.0
        H = _H.expand((*shape, O_D, S_D,))

        sqrt_S_W = torch.zeros((*shape, S_D, S_D,))
        sqrt_S_V = torch.zeros((*shape, O_D, O_D,))

        _, Q, R = _empty_controller_params(shape, S_D)

        return _pack_lti_params(shape, F, B, H, sqrt_S_W, sqrt_S_V, Q, R)




