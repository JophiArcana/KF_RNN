from types import SimpleNamespace
from typing import Sequence

import torch
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.model.base import Predictor


class ZeroPredictor(Predictor):
    def __init__(self, modelArgs: "ZeroPredictor.Config" = None):
        # Used as an analytical baseline (instantiated with ``None`` in engine.error),
        # where no problem shape is available; only initialize the shape-bearing base
        # when real model args are supplied.
        if modelArgs is None:
            torch.nn.Module.__init__(self)
        else:
            Predictor.__init__(self, modelArgs)

    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict,         # [B... x ...]
                                    systems: TensorDict,     # [B... x ...]
    ) -> tuple[TensorDict, SimpleNamespace]:                       # [B...]
        b = Predictor._augmented_plant_modal_decomposition(kfs, systems)
        shape, O_D = b.shape, b.O_D

        inf_geometric = eu.hadamard_conjugation(b.Has, b.Has, b.Dj, b.Dj, torch.eye(O_D))

        # State evolution noise error
        # Highlight
        ws_geometric_err = eu.batch_trace(b.sqrt_S_Ws.mT @ inf_geometric @ b.sqrt_S_Ws)              # [B...]

        # Observation noise error
        # Highlight
        v_current_err = torch.norm(b.sqrt_S_V, dim=[-2, -1]) ** 2                                       # [B...]

        # Highlight
        v_geometric_err = eu.batch_trace(b.sqrt_S_V.mT @ b.Vinv_BL_F_BLK.mT @ (
            inf_geometric
        ) @ b.Vinv_BL_F_BLK @ b.sqrt_S_V)

        err = torch.real(ws_geometric_err + v_current_err + v_geometric_err)                            # [B...]
        cache = SimpleNamespace(
            controller_keys=b.controller_keys,
            shape=shape, default_td=b.default_td,
            K=b.K, L=b.L, sqrt_S_V=b.sqrt_S_V,
            Has=b.Has, Las=b.Las, sqrt_S_Ws=b.sqrt_S_Ws,
            Vinv_BL_F_BLK=b.Vinv_BL_F_BLK, Dj=b.Dj
        )
        return TensorDict.from_dict({"environment": {"observation": err.expand(shape)}}, batch_size=shape), cache

    def training_recipe(self) -> Sequence[str]:
        return []

    def forward(self, trace: dict[str, dict[str, torch.Tensor]], **kwargs) -> dict[str, torch.Tensor]:
        trace = TensorDict(trace, batch_size=trace["environment"]["observation"].shape[:-1])
        valid_keys = [("environment", "observation",),] + [("controller", ac_name) for ac_name in trace["controller"].keys()]
        return TensorDict({
            k: torch.zeros_like(v)
            for k, v in trace.items(include_nested=True, leaves_only=True) if k in valid_keys
        }, batch_size=trace.shape)


class ZeroController(ZeroPredictor):
    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict,         # [B... x ...]
                                    systems: TensorDict,     # [B... x ...]
    ) -> tuple[TensorDict, SimpleNamespace]:                       # [B...]
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

            inf_geometric = eu.hadamard_conjugation(Las, Las, Dj, Dj, torch.eye(I_D))

            # State evolution noise error
            # Highlight
            ws_geometric_err = eu.batch_trace(sqrt_S_Ws.mT @ inf_geometric @ sqrt_S_Ws)              # [B...]

            # Observation noise error
            # Highlight
            v_current_err = torch.norm((L @ K) @ sqrt_S_V, dim=[-1, -2]) ** 2                           # [B...]

            # Highlight
            v_geometric_err = eu.batch_trace(sqrt_S_V.mT @ (
                Vinv_BL_F_BLK.mT @ inf_geometric @ Vinv_BL_F_BLK
            ) @ sqrt_S_V)                                                                               # [B...]

            r[k] = torch.real(ws_geometric_err + v_current_err + v_geometric_err)                       # [B...]

        result["controller"] = TensorDict.from_dict(r, batch_size=shape)
        return result, cache




