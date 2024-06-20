from typing import Dict, Any, Sequence

import torch
from tensordict import TensorDict

from infrastructure import utils
from model.base import Predictor


class ZeroPredictor(Predictor):
    @classmethod
    def analytical_error(cls,
                         kfs: TensorDict[str, torch.Tensor],    # [B... x ...]
                         systems: TensorDict[str, torch.Tensor] # [B... x ...]
    ) -> torch.Tensor:                                          # [B...]
        # Variable definition
        F = utils.complex(systems["F"])                                                         # [B... x S_D x S_D]
        B = utils.complex(systems["B"])                                                         # [B... x S_D x I_D]
        H = utils.complex(systems["H"])                                                         # [B... x O_D x S_D]
        sqrt_S_W = utils.complex(systems["sqrt_S_W"])                                           # [B... x S_D x S_D]
        sqrt_S_V = utils.complex(systems["sqrt_S_V"])                                           # [B... x O_D x O_D]

        S_D, I_D, O_D = F.shape[-1], B.shape[-1], H.shape[-2]

        M = F                                                                                   # [B... x S_D x S_D]
        L, V = torch.linalg.eig(M)                                                              # [B... x S_D], [B... x S_D x S_D]
        Vinv = torch.inverse(V)                                                                 # [B... x S_D x S_D]

        Hs = H @ V                                                                              # [B... x O_D x S_D]
        Bs = Vinv @ B                                                                           # [B... x S_D x I_D]
        sqrt_S_Ws = Vinv @ sqrt_S_W                                                             # [B... x S_D x S_D]

        # State evolution noise error
        # Highlight
        ws_current_err = (Hs @ sqrt_S_Ws).norm(dim=(-2, -1)) ** 2                               # [B...]

        cll = L.unsqueeze(-1) * L.unsqueeze(-2)                                                 # [B... x S_D x S_D]
        # DONE: t1 x t1
        # Highlight
        t1t1_w = (Hs.mT @ Hs) * (cll / (1 - cll))                                               # [B... x S_D x S_D]

        # DONE: t1 x t2, t2 x t1
        t1_w_M = L.unsqueeze(-2)                                                                # [B... x 1 x S_D]
        t2_w_N2 = L.unsqueeze(-2)                                                               # [B... x 1 x S_D]

        _t1_w_M = t1_w_M.unsqueeze(-1)                                                          # [... x 1 x S_D x 1]
        _t2_w_N2 = t2_w_N2.unsqueeze(-2)                                                        # [... x 1 x 1 x S_D]

        def k(m: torch.Tensor, n: torch.Tensor):
            mn = m * n
            return mn / (1 - mn)

        _t1t2_w_K2 = k(_t1_w_M, _t2_w_N2)                                                       # [... x 1 x S_D x S_D]
        _t1t2_w_K = -_t1t2_w_K2                                                                 # [... x 1 x S_D x S_D]

        # Highlight
        w = t1t1_w                                                                              # [B... x S_D x S_D]
        ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ w @ sqrt_S_Ws)                      # [B...]

        # Observation noise error
        # Highlight
        v_current_err = sqrt_S_V.norm(dim=(-2, -1)) ** 2                                        # [B...]

        # Highlight
        err = ws_current_err + ws_geometric_err + v_current_err                                 # [B...]
        return err.real

    @classmethod
    def train_func_list(cls, default_train_func: Any) -> Sequence[Any]:
        return ()

    def forward(self, trace: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        return {
            "input_estimation": torch.zeros_like(trace["input"]),
            "observation_estimation": torch.zeros_like(trace["observation"])
        }




