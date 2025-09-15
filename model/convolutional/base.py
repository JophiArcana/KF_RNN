from argparse import Namespace
from typing import *

import torch
import torch.nn.functional as Fn
from tensordict import TensorDict

from infrastructure import utils
from model.base import Predictor


class ConvolutionalPredictor(Predictor):
    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict,         # [B... x ...]
                                    systems: TensorDict,     # [B... x ...]
    ) -> Tuple[TensorDict, Namespace]:                       # [B...]
        # Variable definition
        controller_keys = systems.get(("environment", "B"), {}).keys()
        shape = utils.broadcast_shapes(kfs.shape, systems.shape)
        default_td = TensorDict({}, batch_size=shape)

        Q = utils.complex(kfs["observation_IR"])                                                        # [B... x O_D x R x O_D]
        Q = Q.permute(*range(Q.ndim - 3), -2, -1, -3)                                                   # [B... x R x O_D x O_D]

        P = utils.complex(kfs["input_IR"]) if len(controller_keys) > 0 else default_td                  # [B... x I_D x R x O_D]
        P = P.apply(lambda t: t.permute(*range(Q.ndim - 3), -2, -1, -3))                                # [B... x R x O_D x I_D]

        F = utils.complex(systems["environment", "F"])                                                  # [B... x S_D x S_D]
        K = utils.complex(systems["environment", "K"])                                                  # [B... x S_D x O_D]
        L = utils.complex(systems["controller", "L"]) if len(controller_keys) > 0 else default_td       # [B... x I_D? x S_D]
        sqrt_S_W = utils.complex(systems["environment", "sqrt_S_W"])                                    # [B... x S_D x S_D]
        sqrt_S_V = utils.complex(systems["environment", "sqrt_S_V"])                                    # [B... x O_D x O_D]

        Fa = utils.complex(systems["F_augmented"])                                                      # [B... x 2S_D x 2S_D]
        Ha = utils.complex(systems["H_augmented"])                                                      # [B... x O_D x 2S_D]
        La = utils.complex(systems["L_augmented"]) if len(controller_keys) > 0 else default_td          # [B... x I_D? x 2S_D]

        S_D, O_D = K.shape[-2:]
        R = Q.shape[-3]

        M = Fa                                                                                          # [B... x 2S_D x 2S_D]
        D, V = torch.linalg.eig(M)                                                                      # [B... x 2S_D], [B... x 2S_D x 2S_D]
        Vinv = torch.inverse(V)                                                                         # [B... x 2S_D x 2S_D]

        Has = Ha @ V                                                                                    # [B... x O_D x 2S_D]
        Las = La.apply(lambda t: t @ V)                                                                 # [B... x I_D? x 2S_D]
        sqrt_S_Ws = Vinv @ torch.cat([sqrt_S_W, torch.zeros_like(sqrt_S_W)], dim=-2)                    # [B... x 2S_D x S_D]

        # Precomputation
        Dj = D[..., None, :]                                                                            # [B... x 1 x 2S_D]
        D_pow_series = D[..., None, :] ** torch.arange(R + 1)[:, None]                                  # [B... x (R + 1) x 2S_D]
        D_pow_series_inv = 1 / D_pow_series                                                             # [B... x (R + 1) x 2S_D]

        BL = utils.complex(torch.zeros((*shape, S_D, S_D)) + sum(
            systems["environment", "B", k] @ systems["controller", "L", k]
            for k in controller_keys
        ))                                                                                              # [B... x S_D x S_D]
        Vinv_BL_F_BLK = Vinv @ torch.cat([-BL, F - BL], dim=-2) @ K                                     # [B... x 2S_D x O_D]

        Ql_PlLK = Q - sum(
            P[k] @ L[k] @ K
            for k in controller_keys
        )                                                                                               # [B... x R x O_D x O_D]
        QlHas_PlLasD_l = (Q @ Has[..., None, :, :] - sum(
            P[k] @ Las[k][..., None, :, :]
            for k in controller_keys
        )) * D_pow_series_inv[..., 1:, None, :]                                                         # [B... x R x O_D x 2S_D]
        Has_cumQlHas_PlLasD_l = Has[..., None, :, :] - torch.cumsum(torch.cat([
            torch.zeros_like(QlHas_PlLasD_l[..., -1:, :, :]),
            QlHas_PlLasD_l
        ], dim=-3), dim=-3)                                                                             # [B... x (R + 1) x O_D x 2S_D]
        Has_cumQlHas_PlLasD_lDk = Has_cumQlHas_PlLasD_l * D_pow_series[..., :, None, :]

        Has_cumQlHas_PlLasD_lDk_R = Has_cumQlHas_PlLasD_lDk[..., -1, :, :]                              # [B... x O_D x 2S_D]
        Has_cumQlHas_PlLasD_lDk = Has_cumQlHas_PlLasD_lDk[..., :-1, :, :]                               # [B... x R x O_D x 2S_D]

        inf_geometric_Has_cumQlHas_PlLasD_lDk = utils.hadamard_conjugation(
            Has_cumQlHas_PlLasD_lDk_R, Has_cumQlHas_PlLasD_lDk_R,
            Dj, Dj, torch.eye(O_D)
        )                                                                                               # [B... x 2S_D x 2S_D]

        # State evolution noise error
        # Highlight
        ws_recent_err = (torch.norm(Has_cumQlHas_PlLasD_lDk @ sqrt_S_Ws[..., None, :, :], dim=[-2, -1]) ** 2).sum(dim=-1)

        # Highlight
        ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ inf_geometric_Has_cumQlHas_PlLasD_lDk @ sqrt_S_Ws)  # [B...]

        # Observation noise error
        # Highlight
        v_current_err = torch.norm(sqrt_S_V, dim=[-2, -1]) ** 2                                         # [B...]

        # Highlight
        v_recent_err = (torch.norm((Has_cumQlHas_PlLasD_lDk @ Vinv_BL_F_BLK[..., None, :, :] - Ql_PlLK) @ sqrt_S_V[..., None, :, :], dim=[-2, -1]) ** 2).sum(dim=-1)

        # Highlight
        v_geometric_err = utils.batch_trace(sqrt_S_V.mT @ Vinv_BL_F_BLK.mT @ (
            inf_geometric_Has_cumQlHas_PlLasD_lDk
        ) @ Vinv_BL_F_BLK @ sqrt_S_V)                                                                   # [B...]

        err = torch.real(ws_recent_err + ws_geometric_err + v_current_err + v_recent_err + v_geometric_err)
        cache = Namespace()
        return TensorDict.from_dict({"environment": {"observation": err}}, batch_size=shape), cache

    def __init__(self, modelArgs: Namespace):
        Predictor.__init__(self, modelArgs)
        self.input_IR = None
        self.observation_IR = None

    """ forward
        :parameter {
            "environment": {
                "observation": [B x L x O_D]
            },
            "controller": {
                "input": [B x L x I_D]
            }
        }
        :returns {
            "environment": {
                "observation": [B x L x O_D]
            },
            "controller": {
                "input": [B x L x I_D]
            }
        }
    """
    def forward(self, trace: Dict[str, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        trace = self.trace_to_td(trace)
        actions, observations = trace["controller"], trace["environment"]["observation"]

        B, L = trace.shape
        result = Fn.conv2d(
            self.observation_IR,
            observations[:, :L].transpose(-2, -1).unsqueeze(-1).flip(-2),
            padding=(L, 0)
        )[:, :L] + sum([Fn.conv2d(
            self.input_IR[k],
            v[:, :L].transpose(-2, -1).unsqueeze(-1).flip(-2),
            padding=(L - 1, 0)
        )[:, :L] for k, v in actions.items()])
        return {
            "environment": {"observation": result},
            "controller": {}
        }




