from argparse import Namespace
from typing import *

import torch
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.experiment.training import TrainFunc
from model.base import Predictor


class ZeroPredictor(Predictor):
    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict[str, torch.Tensor],         # [B... x ...]
                                    systems: TensorDict[str, torch.Tensor],     # [B... x ...]
    ) -> Tuple[TensorDict[str, torch.Tensor], Namespace]:                       # [B...]
        # Variable definition
        controller_keys = systems.get(("environment", "B"), {}).keys()
        shape = systems.shape
        default_td = TensorDict({}, batch_size=shape)

        K = utils.complex(systems["environment", "K"])                                                  # [B... x S_D x O_D]
        L = utils.complex(systems["controller", "L"]) if len(controller_keys) > 0 else default_td       # [B... x I_D? x S_D]
        sqrt_S_W = utils.complex(systems["environment", "sqrt_S_W"])                                    # [B... x S_D x S_D]
        sqrt_S_V = utils.complex(systems["environment", "sqrt_S_V"])                                    # [B... x O_D x O_D]

        Fa = utils.complex(systems["F_augmented"])                                                      # [B... x 2S_D x 2S_D]
        Ha = utils.complex(systems["H_augmented"])                                                      # [B... x O_D x 2S_D]
        La = utils.complex(systems["L_augmented"]) if len(controller_keys) > 0 else default_td          # [B... x I_D? x 2S_D]

        S_D, O_D = K.shape[-2:]

        M = Fa                                                                                          # [B... x 2S_D x 2S_D]
        D, V = torch.linalg.eig(M)                                                                      # [B... x 2S_D], [B... x 2S_D x 2S_D]
        Vinv = torch.inverse(V)                                                                         # [B... x 2S_D x 2S_D]

        Has = Ha @ V                                                                                    # [B... x O_D x 2S_D], [B... x O_D x S_Dh]
        Las = La.apply(lambda t: t @ V)                                                                 # [B... x I_D? x 2S_D]
        sqrt_S_Ws = Vinv @ torch.cat([sqrt_S_W, torch.zeros_like(sqrt_S_W)], dim=-2)                    # [B... x 2S_D x S_D]

        # Precomputation
        F = utils.complex(systems["environment", "F"])                                                  # [B... x S_D x S_D]
        BL = utils.complex(torch.zeros((*systems.shape, S_D, S_D)) + sum(
            systems["environment", "B", k] @ systems["controller", "L", k]
            for k in controller_keys
        ))                                                                                              # [B... x S_D x S_D]
        Vinv_BL_F_BLK = Vinv @ torch.cat([-BL, F - BL], dim=-2) @ K                                     # [B... x 2S_D x O_D]

        Dj = D.unsqueeze(-2)                                                                            # [B... x 1 x 2S_D]

        inf_geometric = utils.hadamard_conjugation(Has, Has, Dj, Dj, torch.eye(O_D))

        # State evolution noise error
        # Highlight
        ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ inf_geometric @ sqrt_S_Ws)                  # [B...]

        # Observation noise error
        # Highlight
        v_current_err = torch.norm(sqrt_S_V, dim=[-2, -1]) ** 2                                         # [B...]

        # Highlight
        v_geometric_err = utils.batch_trace(sqrt_S_V.mT @ Vinv_BL_F_BLK.mT @ (
            inf_geometric
        ) @ Vinv_BL_F_BLK @ sqrt_S_V)

        err = torch.real(ws_geometric_err + v_current_err + v_geometric_err)                            # [B...]
        cache = Namespace(
            controller_keys=controller_keys,
            shape=shape, default_td=default_td,
            K=K, L=L, sqrt_S_V=sqrt_S_V,
            Has=Has, Las=Las, sqrt_S_Ws=sqrt_S_Ws,
            Vinv_BL_F_BLK=Vinv_BL_F_BLK, Dj=Dj
        )
        return TensorDict.from_dict({"environment": {"observation": err}}, batch_size=shape), cache


    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return ()

    def forward(self, trace: Dict[str, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        return TensorDict.from_dict(trace, batch_size=()).apply(torch.zeros_like).to_dict()


class ZeroController(ZeroPredictor):
    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict[str, torch.Tensor],         # [B... x ...]
                                    systems: TensorDict[str, torch.Tensor],     # [B... x ...]
    ) -> Tuple[TensorDict[str, torch.Tensor], Namespace]:                       # [B...]
        result, cache = ZeroPredictor._analytical_error_and_cache(kfs, systems)

        # Variable definition
        controller_keys = cache.controller_keys
        shape = cache.shape

        K, L_dict = cache.K, cache.L                                                                    # [B... x S_D x O_D], [B... x I_D? x S_D]
        sqrt_S_V = cache.sqrt_S_V                                                                       # [B... x O_D x O_D]

        Las_dict = cache.Las                                                                            # [B... x I_D? x 2S_D]
        sqrt_S_Ws = cache.sqrt_S_Ws                                                                     # [B... x 2S_D x 2S_D]

        Dj = cache.Dj                                                                                   # [B... x 1 x 2S_D]
        Vinv_BL_F_BLK = cache.Vinv_BL_F_BLK                                                             # [B... x 2S_D x O_D]

        r = dict()
        for k in controller_keys:
            # Precomputation
            L, Las = L_dict[k], Las_dict[k]                                                             # [B... x I_D x S_D], [B... x I_D x 2S_D]
            I_D = L.shape[-2]

            inf_geometric = utils.hadamard_conjugation(Las, Las, Dj, Dj, torch.eye(I_D))

            # State evolution noise error
            # Highlight
            ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ inf_geometric @ sqrt_S_Ws)              # [B...]

            # Observation noise error
            # Highlight
            v_current_err = torch.norm((L @ K) @ sqrt_S_V, dim=[-1, -2]) ** 2                           # [B...]

            # Highlight
            v_geometric_err = utils.batch_trace(sqrt_S_V.mT @ (
                Vinv_BL_F_BLK.mT @ inf_geometric @ Vinv_BL_F_BLK
            ) @ sqrt_S_V)                                                                               # [B...]

            r[k] = torch.real(ws_geometric_err + v_current_err + v_geometric_err)                       # [B...]

        result["controller"] = TensorDict.from_dict(r, batch_size=shape)
        return result, cache




