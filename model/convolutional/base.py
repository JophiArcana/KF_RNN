from argparse import Namespace
from typing import *

import torch
import torch.nn.functional as Fn
from tensordict import TensorDict

from infrastructure import utils
from model.base import Predictor


class ConvolutionalPredictor(Predictor):
    @classmethod
    def analytical_error(cls,
                         kfs: TensorDict[str, torch.Tensor],        # [B... x ...]
                         systems: TensorDict[str, torch.Tensor],    # [B... x ...]
    ) -> torch.Tensor:                                              # [B...]
        # Variable definition
        controller_keys = kfs.get("B", {}).keys()

        Q = utils.complex(kfs["observation_IR"])                                                # [B... x O_D x R x O_D]
        Q = Q.permute(*range(Q.ndim - 3), -2, -1, -3)                                           # [B... x R x O_D x O_D]

        K = utils.complex(systems["environment", "K"])                                          # [B... x S_D x O_D]
        L = utils.complex(systems["controller", "L"])                                           # [B... x I_D? x S_D]

        Fa = utils.complex(systems["effective", "F"])                                           # [B... x 2S_D x 2S_D]
        Ha = utils.complex(systems["effective", "H"])                                           # [B... x O_D x 2S_D]
        sqrt_S_Wa = utils.complex(systems["effective", "sqrt_S_W"])                             # [B... x 2S_D x 2S_D]
        sqrt_S_Va = utils.complex(systems["effective", "sqrt_S_V"])                             # [B... x O_D x O_D]
        La = utils.complex(systems["effective", "L"])                                           # [B... x I_D? x 2S_D]

        S_D, O_D = K.shape[-2:]
        R = Q.shape[-3]

        D, V = torch.linalg.eig(Fa)                                                             # [B... x S_D], [B... x S_D x S_D]
        Vinv = torch.inverse(V)                                                                 # [B... x S_D x S_D]

        Hs = Ha @ V                                                                             # [B... x O_D x S_D]
        sqrt_S_Ws = Vinv @ sqrt_S_Wa                                                            # [B... x S_D x S_D]

        # State evolution noise error
        # Highlight
        ws_current_err = (Hs @ sqrt_S_Ws).norm(dim=(-2, -1)) ** 2                               # [B...]

        D_pow_series = D.unsqueeze(-2) ** torch.arange(1, R + 1)[:, None]                       # [B... x R x S_D]
        D_pow_series_inv = 1. / D_pow_series                                                    # [B... x R x S_D]

        QlHsDl = (Q @ Hs.unsqueeze(-3)) * D_pow_series_inv.unsqueeze(-2)                        # [B... x R x O_D x S_D]
        Hs_cumQlHsDl = Hs.unsqueeze(-3) - torch.cumsum(QlHsDl, dim=-3)                          # [B... x R x O_D x S_D]
        Hs_cumQlHsDl_Dk = Hs_cumQlHsDl * D_pow_series.unsqueeze(-2)                             # [B... x R x O_D x S_D]

        # Highlight
        ws_recent_err = (Hs_cumQlHsDl_Dk @ sqrt_S_Ws.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2   # [B...]

        Hs_cumQlHsDl_R = Hs_cumQlHsDl[..., -1, :, :]                                            # [B... x O_D x S_D]
        cll = D.unsqueeze(-1) * D.unsqueeze(-2)                                                 # [B... x S_D x S_D]

        # Highlight
        _ws_geometric = (Hs_cumQlHsDl_R.mT @ Hs_cumQlHsDl_R) * ((cll ** (R + 1)) / (1 - cll))   # [B... x S_D x S_D]
        ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ _ws_geometric @ sqrt_S_Ws)          # [B...]

        # Observation noise error
        # Highlight
        v_current_err = sqrt_S_Va.norm(dim=(-2, -1)) ** 2                                       # [B...]

        # Highlight
        # TODO: Backward pass on the above one breaks when Q = 0 for some reason
        # v_recent_err = (Q @ sqrt_S_Va.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2          # [B...]
        v_recent_err = utils.batch_trace(sqrt_S_Va.mT @ (Q.mT @ Q).sum(dim=-3) @ sqrt_S_Va)     # [B...]

        err = ws_current_err + ws_recent_err + ws_geometric_err + v_current_err + v_recent_err  # [B...]
        return err.real

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




