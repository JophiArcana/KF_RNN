import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
from argparse import Namespace
from typing import *

from tensordict import TensorDict

from model.kf import KF
from model.rnn_kf import RnnKF
from infrastructure import utils
from infrastructure.train import TrainFunc
from infrastructure.settings import device


class CnnKF(KF):
    def __init__(self, modelArgs: Namespace):
        super().__init__()

        self.I_D = modelArgs.I_D
        self.O_D = modelArgs.O_D
        self.ir_length = modelArgs.ir_length
        self.input_enabled = modelArgs.input_enabled

        self.input_IR = nn.Parameter(torch.zeros(modelArgs.I_D, self.ir_length, modelArgs.O_D), requires_grad=self.input_enabled)   # [I_D x R x O_D]
        self.observation_IR = nn.Parameter(torch.zeros(modelArgs.O_D, self.ir_length, modelArgs.O_D))                               # [O_D x R x O_D]

    @classmethod
    def analytical_error(cls,
                         kfs: TensorDict[str, torch.Tensor],    # [B... x ...]
                         systems: TensorDict[str, torch.Tensor] # [B... x ...]
    ) -> torch.Tensor:                                          # [B...]
        # Variable definition
        def extract_var(d: TensorDict[str, torch.Tensor], k: str):
            re = torch.Tensor(d[k])
            return torch.complex(re, torch.zeros_like(re))

        P = extract_var(kfs, 'input_IR')                                                        # [B... x I_D x R x O_D]
        P = P.permute(*range(P.ndim - 3), -2, -1, -3)                                           # [B... x R x O_D x I_D]

        Q = extract_var(kfs, 'observation_IR')                                                  # [B... x O_D x R x O_D]
        Q = Q.permute(*range(P.ndim - 3), -2, -1, -3)                                           # [B... x R x O_D x O_D]

        F = extract_var(systems, 'F')                                                           # [B... x S_D x S_D]
        B = extract_var(systems, 'B')                                                           # [B... x S_D x I_D]
        H = extract_var(systems, 'H')                                                           # [B... x O_D x S_D]
        sqrt_S_W = extract_var(systems, 'sqrt_S_W')                                             # [B... x S_D x S_D]
        sqrt_S_V = extract_var(systems, 'sqrt_S_V')                                             # [B... x O_D x O_D]

        S_D, I_D, O_D = F.shape[-1], B.shape[-1], H.shape[-2]
        R = P.shape[-3]

        L, V = torch.linalg.eig(F)                                                              # [B... x S_D], [B... x S_D x S_D]
        Vinv = torch.linalg.inv(V)                                                              # [B... x S_D x S_D]

        Hs = H @ V                                                                              # [B... x O_D x S_D]
        Bs = Vinv @ B                                                                           # [B... x S_D x I_D]
        sqrt_S_Ws = Vinv @ sqrt_S_W                                                             # [B... x S_D x S_D]

        # State evolution noise error
        # Highlight
        ws_current_err = (Hs @ sqrt_S_Ws).norm(dim=(-2, -1)) ** 2                               # [B...]

        L_pow_series = L.unsqueeze(-2) ** torch.arange(1, R + 1, device=device)[:, None]      # [B... x R x S_D]
        L_pow_series_inv = 1. / L_pow_series                                                    # [B... x R x S_D]

        QlHsLl = (Q @ Hs.unsqueeze(-3)) * L_pow_series_inv.unsqueeze(-2)                        # [B... x R x O_D x S_D]
        Hs_cumQlHsLl = Hs.unsqueeze(-3) - torch.cumsum(QlHsLl, dim=-3)                          # [B... x R x O_D x S_D]
        Hs_cumQlHsLl_Lk = Hs_cumQlHsLl * L_pow_series.unsqueeze(-2)                             # [B... x R x O_D x S_D]

        # Highlight
        ws_recent_err = (Hs_cumQlHsLl_Lk @ sqrt_S_Ws.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2   # [B...]

        Hs_cumQlHsLl_R = Hs_cumQlHsLl.index_select(-3, torch.tensor([R - 1], device=device)).squeeze(-3) # [B... x O_D x S_D]
        cll = L.unsqueeze(-1) * L.unsqueeze(-2)                                                 # [B... x S_D x S_D]

        # Highlight
        _ws_geometric = (Hs_cumQlHsLl_R.mT @ Hs_cumQlHsLl_R) * ((cll ** (R + 1)) / (1 - cll))   # [B... x S_D x S_D]
        ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ _ws_geometric @ sqrt_S_Ws)          # [B...]

        # Observation noise error
        # Highlight
        v_current_err = sqrt_S_V.norm(dim=(-2, -1)) ** 2                                        # [B...]

        # Highlight
        v_recent_err = (Q @ sqrt_S_V.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2           # [B...]

        err = ws_current_err + ws_recent_err + ws_geometric_err + v_current_err + v_recent_err  # [B...]
        return err.real

    @classmethod
    def to_sequential_batch(cls, kfs: TensorDict[str, torch.Tensor], input_enabled: bool) -> TensorDict[str, torch.Tensor]:
        input_IR, observation_IR = torch.Tensor(kfs['input_IR']), torch.Tensor(kfs['observation_IR'])               # [... x I_D x R x O_D], [... x O_D x R x O_D]
        I_D, R, O_D = input_IR.shape[-3:]

        S_D = R * ((I_D + O_D) if input_enabled else O_D)

        def expand_right(t: torch.Tensor) -> torch.Tensor:
            return t.view(*t.shape, *(1 for _ in range(kfs.ndim))).expand(*t.shape, *kfs.shape)

        permuted_input_IR = input_IR.permute(-3, -2, -1, *range(kfs.ndim))                                          # [I_D x R x O_D x ...]
        permuted_observation_IR = observation_IR.permute(-3, -2, -1, *range(kfs.ndim))                              # [O_D x R x O_D x ...]

        # DONE: Construct F matrix
        F00 = expand_right(torch.diag_embed(torch.ones(((R - 1) * O_D,), device=device), offset=-O_D)).clone()    # [RO_D x RO_D x ...]
        F00[:O_D] = permuted_observation_IR.transpose(0, 2).flatten(1, 2)

        if input_enabled:
            F01 = torch.zeros((R * O_D, R * I_D, *kfs.shape), device=device)                                      # [RO_D x RI_D x ...]
            F01[:O_D, :-I_D] = permuted_input_IR[:, 1:].transpose(0, 2).flatten(1, 2)

            F10 = torch.zeros((R * I_D, R * O_D, *kfs.shape), device=device)                                      # [RI_D x RO_D x ...]
            F11 = expand_right(torch.diag_embed(torch.ones(((R - 1) * I_D,), device=device), offset=-I_D))        # [RI_D x RI_D x ...]

            F = torch.cat([
                torch.cat([F00, F01], dim=1),
                torch.cat([F10, F11], dim=1)
            ], dim=0).permute(*range(2, kfs.ndim + 2), 0, 1)                                                        # [... x R(O_D + I_D) x R(O_D + I_D)]
        else:
            F = F00.permute(*range(2, kfs.ndim + 2), 0, 1)                                                          # [... x RO_D x RO_D]

        # DONE: Construct B matrix
        B0 = torch.cat([
            permuted_input_IR[:, 0].transpose(0, 1),                                                                # [O_D x I_D x ...]
            torch.zeros(((R - 1) * O_D, I_D, *kfs.shape), device=device),                                         # [(R - 1)O_D x I_D x ...]
        ], dim=0)                                                                                                   # [RO_D x I_D x ...]

        if input_enabled:
            B1 = torch.cat([
                expand_right(torch.eye(I_D, device=device)),                                                      # [I_D x I_D x ...]
                torch.zeros(((R - 1) * I_D, I_D, *kfs.shape), device=device)                                      # [(R - 1)I_D x I_D x ...]
            ])                                                                                                      # [RI_D x I_D x ...]
            B = torch.cat([B0, B1], dim=0).permute(*range(2, kfs.ndim + 2), 0, 1)                                   # [... x R(O_D + I_D) x I_D]
        else:
            B = B0.permute(*range(2, kfs.ndim + 2), 0, 1)                                                           # [... x RO_D x I_D]

        # DONE: Construct H matrix
        H = torch.hstack([
            torch.eye(O_D, device=device),                                                                        # [O_D x O_D]
            torch.zeros((O_D, S_D - O_D), device=device)                                                          # [O_D x ((R - 1)O_D + RI_D)] or [O_D x (R - 1)O_D]
        ]).expand(*kfs.shape, O_D, S_D)                                                                             # [... x O_D x R(O_D + I_D)] or [... x O_D x RO_D]

        # DONE: Construct K matrix
        K = H.mT                                                                                                    # [... x R(O_D + I_D) x O_D] or [... x RO_D x O_D]

        return TensorDict({'F': F, 'B': B, 'H': H, 'K': K}, batch_size=kfs.shape, device=device)

    def to_sequential(self) -> nn.Module:
        sequentialModelArgs = Namespace(
            S_D=self.ir_length * ((self.I_D + self.O_D) if self.input_enabled else self.O_D),
            I_D=self.I_D,
            O_D=self.O_D,
            input_enabled=self.input_enabled
        )

        # DONE: Construct F matrix
        F00 = torch.diag_embed(torch.ones(((self.ir_length - 1) * self.O_D,), device=device), offset=-self.O_D)   # [RO_D x RO_D]
        F00[:self.O_D] = self.observation_IR.transpose(0, 2).flatten(1, 2)

        if self.input_enabled:
            F01 = torch.zeros((self.ir_length * self.O_D, self.ir_length * self.I_D), device=device)                  # [RO_D x RI_D]
            F01[:self.O_D, :-self.I_D] = self.input_IR[:, 1:].transpose(0, 2).flatten(1, 2)

            F10 = torch.zeros((self.ir_length * self.I_D, self.ir_length * self.O_D), device=device)                  # [RI_D x RO_D]
            F11 = torch.diag_embed(torch.ones(((self.ir_length - 1) * self.I_D,), device=device), offset=-self.I_D)   # [RI_D x RI_D]

            F = torch.vstack([
                torch.hstack([F00, F01]),
                torch.hstack([F10, F11])
            ]).to_sparse_coo()                                                                                          # [R(O_D + I_D) x R(O_D + I_D)]
        else:
            F = F00.to_sparse_coo()                                                                                     # [RO_D x RO_D]

        # DONE: Construct B matrix
        B0 = torch.vstack([
            self.input_IR[:, 0].mT,                                                                                 # [O_D x I_D]
            torch.zeros(((self.ir_length - 1) * self.O_D, self.I_D), device=device),                              # [(R - 1)O_D x I_D]
        ])                                                                                                          # [RO_D x I_D]

        if self.input_enabled:
            B1 = torch.vstack([
                torch.eye(self.I_D, device=device),                                                                   # [I_D x I_D]
                torch.zeros(((self.ir_length - 1) * self.I_D, self.I_D), device=device)                               # [(R - 1)I_D x I_D]
            ])                                                                                                          # [RI_D x I_D]
            B = torch.cat([B0, B1], dim=0).to_sparse_coo()                                                              # [R(O_D + I_D) x I_D]
        else:
            B = B0.to_sparse_coo()                                                                                      # [RO_D x I_D]

        # DONE: Construct H matrix
        H = torch.hstack([
            torch.eye(self.O_D, device=device),                                                                   # [O_D x O_D]
            torch.zeros((self.O_D, sequentialModelArgs.S_D - self.O_D), device=device)                            # [O_D x ((R - 1)O_D + RI_D)] or [O_D x (R - 1)O_D]
        ]).to_sparse_coo()                                                                                          # [O_D x R(O_D + I_D)] or [O_D x RO_D]

        # DONE: Construct K matrix
        K = H.mT                                                                                                    # [R(O_D + I_D) x O_D] or [RO_D x O_D]

        return RnnKF(sequentialModelArgs, F=F, B=B, H=H, K=K)

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns {
            'observation_estimation': [B x L x O_D]
        }
    """
    def forward(self, trace: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        state, inputs, observations = self.extract(trace, 0)
        B, L = inputs.shape[:2]

        result = Fn.conv2d(
            self.observation_IR,
            observations[:, :L].transpose(-2, -1).unsqueeze(-1).flip(-2),
            padding=(L, 0)
        )[:, :L] + Fn.conv2d(
            self.input_IR,
            inputs[:, :L].transpose(-2, -1).unsqueeze(-1).flip(-2),
            padding=(L - 1, 0)
        )[:, :L]

        return {'observation_estimation': result}


class CnnKFLeastSquares(CnnKF):
    @classmethod
    def train_least_squares(cls,
                            shared: Dict[str, Any],
                            exclusive: Dict[str, Any],
                            flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
                            optimizer: optim.Optimizer
    ) -> Tuple[torch.Tensor, bool]:
        return KF._train_with_initialization_and_error(
            shared, exclusive, flattened_ensembled_learned_kfs,
            lambda shared_, exclusive_: torch.vmap(CnnKFLeastSquares._least_squares_initialization, (None, 0))(
                exclusive_['base_model'],
                dict(exclusive_['training_dataset'])
            )
        )

    @classmethod
    def train_override(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return (CnnKFLeastSquares.train_least_squares,)

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns None
    """
    def _least_squares_initialization(self, trace: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        inputs, observations = trace['input'], trace['observation']
        B, L = inputs.shape[:2]

        # DONE: Implement online least squares for memory efficiency
        split = [0]
        while split[-1] != L:
            split.append(min(split[-1] + min(L, 1 << 16), L))

        padded_observations = torch.cat([                                                           # [B x L x O_D]
            torch.zeros((B, 1, self.O_D), device=device),
            observations[:, :-1],
            torch.zeros((B, 1, self.O_D), device=device)
        ], dim=1)
        padded_inputs = torch.cat([                                                                 # [B x L x I_D]
            inputs,
            torch.zeros((B, 1, self.I_D), device=device)
        ], dim=1)

        k = self.I_D if self.input_enabled else 0
        XTX = torch.zeros((r_ := self.ir_length * (k + self.O_D), r_), device=device)                     # [R? x R?]
        XTy = torch.zeros((r_, self.O_D), device=device)                                                  # [R? x O_D]
        yTy = torch.zeros((self.O_D, self.O_D), device=device)                                            # [O_D x O_D]

        for i in range(len(split) - 1):
            lo, hi = split[i], split[i + 1]
            l = hi - lo

            indices = (torch.arange(lo, hi, device=device)[:, None] - torch.arange(self.ir_length, device=device)).clamp_min(-1)

            X_observation = padded_observations[:, indices]                                                 # [B x l x R x O_D]
            if self.input_enabled:
                X_input = padded_inputs[:, indices]                                                         # [B x l x R x I_D]
                flattened_X = torch.cat([X_input, X_observation], dim=-1).view((B * l, -1))                 # [Bl x R(I_D + O_D)]
            else:
                flattened_X = X_observation.view((B * l, -1))                                               # [Bl x RO_D]
            flattened_observations = observations[:, lo:hi].view((B * l, self.O_D))                         # [Bl x O_D]

            XTX = XTX + (flattened_X.T @ flattened_X)
            XTy = XTy + (flattened_X.T @ flattened_observations)
            yTy = yTy + (flattened_observations.T @ flattened_observations)

            torch.cuda.empty_cache()

        XTX_lI_inv = torch.linalg.inv(XTX + self.ridge * torch.eye(r_, device=device))                    # [R? x R?]
        flattened_w = XTX_lI_inv @ XTy
        w = flattened_w.unflatten(0, (self.ir_length, -1)).transpose(0, 1)                                  # [? x R x O_D]

        error = torch.trace(yTy + XTy.T @ (XTX_lI_inv @ XTX @ XTX_lI_inv - 2 * XTX_lI_inv) @ XTy) / (B * L)
        return {
            'input_IR': w[:self.I_D] if self.input_enabled else torch.zeros((self.I_D, self.ir_length, self.O_D), device=device),
            'observation_IR': w[self.I_D:] if self.input_enabled else w
        }, error

    def __init__(self, modelArgs: Namespace):
        super().__init__(modelArgs)
        self.ridge = getattr(modelArgs, 'ridge', 0.)
        self._initialization_error: torch.Tensor = None


class CnnKFPretrainLeastSquares(CnnKFLeastSquares):
    @classmethod
    def train_override(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return CnnKFLeastSquares.train_least_squares, default_train_func


class CnnKFAnalytical(CnnKF):
    @classmethod
    def train_analytical(cls,
                         shared: Dict[str, Any],
                         exclusive: Dict[str, Any],
                         flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
                         optimizer: optim.Optimizer
    ) -> Tuple[torch.Tensor, bool]:
        return KF._train_with_initialization_and_error(
            shared, exclusive, flattened_ensembled_learned_kfs,
            lambda shared_, exclusive_: torch.vmap(CnnKFAnalytical._least_squares_initialization)(torch.func.stack_module_state(shared_['system'])[0])
        )

    @classmethod
    def train_override(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return (CnnKFAnalytical.train_analytical,)

    def _analytical_initialization(self, system_state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        F, B, H, K = map(system_state_dict.__getitem__, ('F', 'B', 'H', 'K'))
        S_D = F.shape[0]

        powers = utils.pow_series(F @ (torch.eye(S_D, device=device) - K @ H), self.ir_length)    # [R x S_D x S_D]
        return {
            'input_IR': (H @ powers @ B).permute(2, 0, 1),                                          # [I_D x R x O_D]
            'observation_IR': (H @ powers @ (F @ K)).permute(2, 0, 1)                               # [O_D x R x O_D]
        }, torch.full((), torch.nan)

    def __init__(self, modelArgs: Namespace):
        super().__init__(modelArgs)
        self._initialization_error: torch.Tensor = None





