from typing import *

import torch
import torch.nn.functional as Fn
from tensordict import TensorDict

from infrastructure import utils
from model.base import Predictor


class ConvolutionalPredictor(Predictor):
    @classmethod
    def analytical_error(cls,
                         kfs: TensorDict[str, torch.Tensor],    # [B... x ...]
                         systems: TensorDict[str, torch.Tensor] # [B... x ...]
    ) -> torch.Tensor:                                          # [B...]
        # Variable definition
        P = utils.complex(kfs["input_IR"])                                                      # [B... x I_D x R x O_D]
        P = P.permute(*range(P.ndim - 3), -2, -1, -3)                                           # [B... x R x O_D x I_D]

        Q = utils.complex(kfs["observation_IR"])                                                # [B... x O_D x R x O_D]
        Q = Q.permute(*range(P.ndim - 3), -2, -1, -3)                                           # [B... x R x O_D x O_D]

        F = utils.complex(systems["F"])                                                         # [B... x S_D x S_D]
        B = utils.complex(systems["B"])                                                         # [B... x S_D x I_D]
        H = utils.complex(systems["H"])                                                         # [B... x O_D x S_D]
        sqrt_S_W = utils.complex(systems["sqrt_S_W"])                                           # [B... x S_D x S_D]
        sqrt_S_V = utils.complex(systems["sqrt_S_V"])                                           # [B... x O_D x O_D]

        R = P.shape[-3]

        L, V = torch.linalg.eig(F)                                                              # [B... x S_D], [B... x S_D x S_D]
        Vinv = torch.inverse(V)                                                                 # [B... x S_D x S_D]

        Hs = H @ V                                                                              # [B... x O_D x S_D]
        Bs = Vinv @ B                                                                           # [B... x S_D x I_D]
        sqrt_S_Ws = Vinv @ sqrt_S_W                                                             # [B... x S_D x S_D]

        # State evolution noise error
        # Highlight
        ws_current_err = (Hs @ sqrt_S_Ws).norm(dim=(-2, -1)) ** 2                               # [B...]

        L_pow_series = L.unsqueeze(-2) ** torch.arange(1, R + 1)[:, None]                       # [B... x R x S_D]
        L_pow_series_inv = 1. / L_pow_series                                                    # [B... x R x S_D]

        QlHsLl = (Q @ Hs.unsqueeze(-3)) * L_pow_series_inv.unsqueeze(-2)                        # [B... x R x O_D x S_D]
        Hs_cumQlHsLl = Hs.unsqueeze(-3) - torch.cumsum(QlHsLl, dim=-3)                          # [B... x R x O_D x S_D]
        Hs_cumQlHsLl_Lk = Hs_cumQlHsLl * L_pow_series.unsqueeze(-2)                             # [B... x R x O_D x S_D]

        # Highlight
        ws_recent_err = (Hs_cumQlHsLl_Lk @ sqrt_S_Ws.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2   # [B...]

        Hs_cumQlHsLl_R = Hs_cumQlHsLl.index_select(-3, torch.tensor([R - 1])).squeeze(-3)       # [B... x O_D x S_D]
        cll = L.unsqueeze(-1) * L.unsqueeze(-2)                                                 # [B... x S_D x S_D]

        # Highlight
        _ws_geometric = (Hs_cumQlHsLl_R.mT @ Hs_cumQlHsLl_R) * ((cll ** (R + 1)) / (1 - cll))   # [B... x S_D x S_D]
        ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ _ws_geometric @ sqrt_S_Ws)          # [B...]

        # Observation noise error
        # Highlight
        v_current_err = sqrt_S_V.norm(dim=(-2, -1)) ** 2                                        # [B...]

        # Highlight
        # TODO: Backward pass on the above one breaks when Q = 0 for some reason
        # v_recent_err = (Q @ sqrt_S_V.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2           # [B...]
        v_recent_err = utils.batch_trace(sqrt_S_V.mT @ (Q.mT @ Q).sum(dim=-3) @ sqrt_S_V)       # [B...]

        err = ws_current_err + ws_recent_err + ws_geometric_err + v_current_err + v_recent_err  # [B...]
        return err.real

    @classmethod
    def to_sequential_batch(cls, kfs: TensorDict[str, torch.Tensor], input_enabled: bool) -> TensorDict[str, torch.Tensor]:
        input_IR, observation_IR = torch.Tensor(kfs["input_IR"]), torch.Tensor(kfs["observation_IR"])       # [... x I_D x R x O_D], [... x O_D x R x O_D]
        I_D, R, O_D = input_IR.shape[-3:]

        S_D = R * ((I_D + O_D) if input_enabled else O_D)

        def expand_right(t: torch.Tensor) -> torch.Tensor:
            return t.view(*t.shape, *(1 for _ in range(kfs.ndim))).expand(*t.shape, *kfs.shape)

        permuted_input_IR = input_IR.permute(-3, -2, -1, *range(kfs.ndim))                                  # [I_D x R x O_D x ...]
        permuted_observation_IR = observation_IR.permute(-3, -2, -1, *range(kfs.ndim))                      # [O_D x R x O_D x ...]

        # DONE: Construct F matrix
        F00 = expand_right(torch.diag_embed(torch.ones(((R - 1) * O_D,)), offset=-O_D)).clone()             # [RO_D x RO_D x ...]
        F00[:O_D] = permuted_observation_IR.transpose(0, 2).flatten(1, 2)

        if input_enabled:
            F01 = torch.zeros((R * O_D, R * I_D, *kfs.shape))                                               # [RO_D x RI_D x ...]
            F01[:O_D, :-I_D] = permuted_input_IR[:, 1:].transpose(0, 2).flatten(1, 2)

            F10 = torch.zeros((R * I_D, R * O_D, *kfs.shape))                                               # [RI_D x RO_D x ...]
            F11 = expand_right(torch.diag_embed(torch.ones(((R - 1) * I_D,)), offset=-I_D))                 # [RI_D x RI_D x ...]

            F = torch.cat([
                torch.cat([F00, F01], dim=1),
                torch.cat([F10, F11], dim=1)
            ], dim=0).permute(*range(2, kfs.ndim + 2), 0, 1)                                                # [... x R(O_D + I_D) x R(O_D + I_D)]
        else:
            F = F00.permute(*range(2, kfs.ndim + 2), 0, 1)                                                  # [... x RO_D x RO_D]

        # DONE: Construct B matrix
        B0 = torch.cat([
            permuted_input_IR[:, 0].transpose(0, 1),                                                        # [O_D x I_D x ...]
            torch.zeros(((R - 1) * O_D, I_D, *kfs.shape)),                                                  # [(R - 1)O_D x I_D x ...]
        ], dim=0)                                                                                           # [RO_D x I_D x ...]

        if input_enabled:
            B1 = torch.cat([
                expand_right(torch.eye(I_D)),                                                               # [I_D x I_D x ...]
                torch.zeros(((R - 1) * I_D, I_D, *kfs.shape))                                               # [(R - 1)I_D x I_D x ...]
            ])                                                                                              # [RI_D x I_D x ...]
            B = torch.cat([B0, B1], dim=0).permute(*range(2, kfs.ndim + 2), 0, 1)                           # [... x R(O_D + I_D) x I_D]
        else:
            B = B0.permute(*range(2, kfs.ndim + 2), 0, 1)                                                   # [... x RO_D x I_D]

        # DONE: Construct H matrix
        H = torch.hstack([
            torch.eye(O_D),                                                                                 # [O_D x O_D]
            torch.zeros((O_D, S_D - O_D))                                                                   # [O_D x ((R - 1)O_D + RI_D)] or [O_D x (R - 1)O_D]
        ]).expand(*kfs.shape, O_D, S_D)                                                                     # [... x O_D x R(O_D + I_D)] or [... x O_D x RO_D]

        # DONE: Construct K matrix
        K = H.mT                                                                                            # [... x R(O_D + I_D) x O_D] or [... x RO_D x O_D]

        return TensorDict({"F": F, "B": B, "H": H, "K": K}, batch_size=kfs.shape)

    """ forward
        :parameter {
            "input": [B x L x I_D],
            "observation": [B x L x O_D]
        }
        :returns {
            "observation_estimation": [B x L x O_D]
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

        return {"observation_estimation": result}




