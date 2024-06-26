import math
from argparse import Namespace
from typing import *

import numpy as np
import torch
from tensordict import TensorDict

from infrastructure import utils
from system.simple.linear_time_invariant import LTISystem
from model.base import Predictor, Controller


class SequentialPredictor(Predictor):
    @classmethod
    def _evaluate_form(cls,
                       state: torch.Tensor,         # [B x S_D]
                       form: Sequence[torch.Tensor] # [T x D x S_D], [B x T x D]
    ) -> torch.Tensor:                              # [B x T x D]
        weights, biases = form
        return (weights.flatten(0, 1)[None] @ state[:, :, None]).reshape(-1, *biases.shape[1:]) + biases

    @classmethod
    def analytical_error(cls,
                         kfs: TensorDict[str, torch.Tensor],    # [B... x ...]
                         systems: TensorDict[str, torch.Tensor] # [B... x ...]
    ) -> torch.Tensor:                                          # [B...]
        # Variable definition
        Fh_effective = utils.complex(LTISystem.F_effective(kfs))                                # [B... x S_Dh x S_Dh]
        Hh = utils.complex(kfs["H"])                                                            # [B... x O_D x S_Dh]
        Kh = utils.complex(kfs["K"])                                                            # [B... x S_Dh x O_D]

        F_effective = utils.complex(LTISystem.F_effective(systems))                             # [B... x S_D x S_D]
        H = utils.complex(systems["H"])                                                         # [B... x O_D x S_D]
        sqrt_S_W = utils.complex(systems["sqrt_S_W"])                                           # [B... x S_D x S_D]
        sqrt_S_V = utils.complex(systems["sqrt_S_V"])                                           # [B... x O_D x O_D]

        S_D, O_D = F_effective.shape[-1], H.shape[-2]
        S_Dh = Fh_effective.shape[-1]

        M, Mh = F_effective, Fh_effective @ (torch.eye(S_Dh) - Kh @ Hh)                         # [B... x S_D x S_D], [B... x S_Dh x S_Dh]
        L, V = torch.linalg.eig(M)                                                              # [B... x S_D], [B... x S_D x S_D]
        Lh, Vh = torch.linalg.eig(Mh)                                                           # [B... x S_Dh], [B... x S_Dh x S_Dh]
        Vinv, Vhinv = torch.inverse(V), torch.inverse(Vh)                                       # [B... x S_D x S_D], [B... x S_Dh x S_Dh]
        
        Hs, Hhs = H @ V, Hh @ Vh                                                                # [B... x O_D x S_D], [B... x O_D x S_Dh]
        sqrt_S_Ws = Vinv @ sqrt_S_W                                                             # [B... x S_D x S_D]

        # Precomputation
        VhinvFhKh = Vhinv @ Fh_effective @ Kh                                                   # [B... x S_Dh x O_D]
        VhinvFhKhHs = VhinvFhKh @ Hs                                                            # [B... x S_Dh x S_D]

        # State evolution noise error
        # Highlight
        ws_current_err = (Hs @ sqrt_S_Ws).norm(dim=(-2, -1)) ** 2                               # [B...]

        cll = L.unsqueeze(-1) * L.unsqueeze(-2)                                                 # [B... x S_D x S_D]
        # DONE: t1 x t1
        # Highlight
        t1t1_w = (Hs.mT @ Hs) * (cll / (1 - cll))                                               # [B... x S_D x S_D]

        # DONE: t1 x t2, t2 x t1
        t1_w_M = L.unsqueeze(-2)                                                                # [B... x 1 x S_D]
        t1_w_A = Hhs.mT @ Hs                                                                    # [B... x S_Dh x S_D]
        t2_w_N1, t2_w_N2 = Lh.unsqueeze(-1), L.unsqueeze(-2)                                    # [B... x S_Dh x 1], [B... x 1 x S_D]
        t2_w_B = VhinvFhKhHs / (t2_w_N1 - t2_w_N2)                                              # [B... x S_Dh x S_D]

        _t1_w_A, _t1_w_M = t1_w_A.unsqueeze(-1), t1_w_M.unsqueeze(-1)                           # [... x S_Dh x S_D x 1], [... x 1 x S_D x 1]
        _t2_w_B = t2_w_B.unsqueeze(-2)                                                          # [... x S_Dh x 1 x S_D]
        _t2_w_N1, _t2_w_N2 = t2_w_N1.unsqueeze(-2), t2_w_N2.unsqueeze(-2)                       # [... x S_Dh x 1 x 1], [... x 1 x 1 x S_D]

        def k(m: torch.Tensor, n: torch.Tensor):
            mn = m * n
            return mn / (1 - mn)

        _t1t2_w_K1 = k(_t1_w_M, _t2_w_N1)                                                       # [... x S_Dh x S_D x 1]
        _t1t2_w_K2 = k(_t1_w_M, _t2_w_N2)                                                       # [... x 1 x S_D x S_D]
        _t1t2_w_K = _t1t2_w_K1 - _t1t2_w_K2                                                     # [... x S_Dh x S_D x S_D]

        # Highlight
        t1t2_w = (_t1_w_A * _t2_w_B * _t1t2_w_K).sum(dim=-3)                                    # [B... x S_D x S_D]

        # DONE: t2 x t2
        t2t2_w_C = Hhs.mT @ Hhs                                                                 # [B... x S_Dh x S_Dh]

        __t2_w_B10 = _t2_w_B.unsqueeze(-1)                                                      # [... x S_Dh x 1 x S_D x 1]
        __t2_w_B01 = _t2_w_B.unsqueeze(-4)                                                      # [... x 1 x S_Dh x 1 x S_D]
        __t2_w_N10, __t2_w_N20 = _t2_w_N1.unsqueeze(-1), _t2_w_N2.unsqueeze(-1)                 # [... x S_Dh x 1 x 1 x 1], [... x 1 x 1 x S_D x 1]
        __t2_w_N01, __t2_w_N02 = _t2_w_N1.unsqueeze(-4), _t2_w_N2.unsqueeze(-4)                 # [... x 1 x S_Dh x 1 x 1], [... x 1 x 1 x 1 x S_D]
        __t2t2_w_C = t2t2_w_C.unsqueeze(-1).unsqueeze(-1)                                       # [... x S_Dh x S_Dh x 1 x 1]

        __t2t2_w_K11 = k(__t2_w_N10, __t2_w_N01)                                                # [... x S_Dh x S_Dh x 1 x 1]
        __t2t2_w_K21 = k(__t2_w_N20, __t2_w_N01)                                                # [... x 1 x S_Dh x S_D x 1]
        __t2t2_w_K12 = k(__t2_w_N10, __t2_w_N02)                                                # [... x S_Dh x 1 x 1 x S_D]
        __t2t2_w_K22 = k(__t2_w_N20, __t2_w_N02)                                                # [... x 1 x 1 x S_D x S_D]
        __t2t2_w_K = __t2t2_w_K11 - __t2t2_w_K12 - __t2t2_w_K21 + __t2t2_w_K22                  # [... x S_Dh x S_Dh x S_D x S_D]

        # Highlight
        t2t2_w = (__t2_w_B10 * __t2_w_B01 * __t2t2_w_C * __t2t2_w_K).sum(dim=(-4, -3))          # [B... x S_D x S_D]

        # Highlight
        w = t1t1_w - 2 * t1t2_w + t2t2_w                                                        # [B... x S_D x S_D]
        ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ w @ sqrt_S_Ws)                      # [B...]

        # Observation noise error
        # Highlight
        v_current_err = sqrt_S_V.norm(dim=(-2, -1)) ** 2                                        # [B...]

        v_C = t2t2_w_C                                                                          # [B... x S_Dh x S_Dh]
        v_K = 1 / (1 - Lh.unsqueeze(-1) * Lh.unsqueeze(-2))                                     # [B... x S_Dh x S_Dh]

        # Highlight
        _sqrt_S_V = VhinvFhKh @ sqrt_S_V                                                        # [B... x S_Dh x O_D]
        v = v_C * v_K                                                                           # [B... x S_Dh x S_Dh]
        v_geometric_err = utils.batch_trace(_sqrt_S_V.mT @ v @ _sqrt_S_V)                       # [B...]

        err = ws_current_err + ws_geometric_err + v_current_err + v_geometric_err               # [B...]
        return err.real

    def __init__(self, modelArgs: Namespace):
        Predictor.__init__(self, modelArgs)
        self.S_D: int = modelArgs.S_D

    def forward(self, trace: Dict[str, Dict[str, torch.Tensor]], mode: str = None) -> Dict[str, Dict[str, torch.Tensor]]:
        trace = self.trace_to_td(trace)
        actions, observations = trace["controller"], trace["environment"]["observation"]

        state_estimation = torch.randn((*observations.shape[:-2], self.S_D))
        return self.forward_with_initial(state_estimation, actions, observations, mode)

    def forward_with_initial(self,
                             state_estimation: torch.Tensor,
                             actions: TensorDict[str, torch.Tensor],
                             observations: torch.Tensor,
                             mode: str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        L = observations.shape[1]

        if mode is None:
            mode = ["sequential", "form", "form_sqrt"][np.searchsorted([16, 64], L)]

        if mode == "sequential":
            result = []
            for l in range(L):
                result.append(r := self._forward(state_estimation, actions[:, l], observations[:, l]))
                state_estimation = r["environment", "state"]
            return torch.stack(result, dim=-1).to_dict()
        else:
            state_estimations, observation_estimations = [], []
            result_generic = self._forward_generic(actions, observations, mode)

            state_weights, state_biases_list = result_generic["state_form"]                     # [sqrtT x S_D x S_D], sqrtT x [B x ≈sqrtT x S_D]
            observation_weights, observation_biases_list = result_generic["observation_form"]   # [sqrtT x O_D x S_D], sqrtT x [B x ≈sqrtT x O_D]

            for state_biases, observation_biases in zip(state_biases_list, observation_biases_list):
                state_estimations.append(SequentialPredictor._evaluate_form(state_estimation, (state_weights[:state_biases.shape[1]], state_biases)))
                observation_estimations.append(SequentialPredictor._evaluate_form(state_estimation, (observation_weights[:observation_biases.shape[1]], observation_biases)))

                state_estimation = state_estimations[-1][:, -1]

            return {
                "environment": {
                    "state": torch.cat(state_estimations, dim=1),
                    "observation": torch.cat(observation_estimations, dim=1)
                },
                "controller": {}
            }

    def _forward(self,
                 state: torch.Tensor,                   # [B x S_D]
                 action: TensorDict[str, torch.Tensor], # [B x I_D]
                 observation: torch.Tensor,             # [B x O_D]
    ) -> TensorDict[str, torch.Tensor]:                 # [B x S_D], [B x O_D]
        state_estimation = state @ self.F.mT + sum(ac @ self.B[ac_name].mT for ac_name, ac in action.items())
        observation_estimation = state_estimation @ self.H.mT
        state_estimation = state_estimation + (observation - observation_estimation) @ self.K.mT

        return TensorDict.from_dict({
            "environment": {
                "state": state_estimation,
                "observation": observation_estimation
            },
            "controller": {}
        }, batch_size=state.shape[:-1])

    """ forward
        :parameter {
            "input": [B x L x I_D],
            "observation": [B x L x O_D]
        }
        :returns {
            "state_form": ([sqrtT x S_D x S_D], sqrtT x [B x ≈sqrtT x S_D]),
            "observation_form": ([sqrtT x O_D x S_D], sqrtT x [B x ≈sqrtT x O_D])
        }
    """
    def _forward_generic(self,
                         actions: TensorDict[str, torch.Tensor],
                         observations: torch.Tensor,
                         mode: str
    ) -> Dict[str, Tuple[torch.Tensor, Sequence[torch.Tensor]]]:
        # Precomputation
        B, L = actions.shape
        hsqrtL = int(math.ceil(math.sqrt(L)))
        lsqrtL = int(math.ceil(L / hsqrtL))

        E = torch.eye(self.S_D) - self.K @ self.H
        M = E @ self.F

        subL = L if mode == "form" else hsqrtL                                                                                          # Length of vectorized subsequence
        # Compute the weights efficiently by eigenvalue decomposition of (I - KH)F and repeated powers
        """
        L, V = torch.linalg.eig(M)
        eig_powers = torch.diag_embed(torch.pow(L, torch.arange(subL + 1)[:, None]))                                                    # [(subL + 1) x S_D x S_D]
        state_weights = (V @ eig_powers @ torch.inverse(V)).real                                                                        # [(subL + 1) x S_D x S_D]
        state_weights = torch.stack([torch.matrix_power(M, n) for n in range(subL + 1)])
        """
        state_weights = utils.pow_series(M, subL + 1)                                                                                   # [(subL + 1) x S_D x S_D]
        observation_weights = (self.H @ self.F) @ state_weights                                                                         # [(subL + 1) x O_D x S_D]

        # Compute the biases efficiently using the state weights
        buffered_state_weights = torch.cat([
            state_weights,
            torch.zeros((1, self.S_D, self.S_D))
        ], dim=0)                                                                                                                       # [(subL + 2) x S_D x S_D]
        lower_triangular_indices = (torch.arange(subL)[:, None] - torch.arange(subL)).clamp_min(-1)                                     # [subL x subL]

        blocked_lower_triangular_matrix = buffered_state_weights[lower_triangular_indices]                                              # [subL x subL x S_D x S_D]
        lower_triangular_matrix = blocked_lower_triangular_matrix.permute(0, 2, 1, 3).reshape(subL * self.S_D, subL * self.S_D)

        u = torch.zeros((B, L, self.S_D)) + sum(ac @ self.B[ac_name].mT for ac_name, ac in actions.items())                             # [B x L x S_D]
        if mode == "form":
            state_biases = torch.cat([
                torch.zeros((B, 1, self.S_D)),
                ((u @ E.mT + observations @ self.K.mT).view(B, -1) @ lower_triangular_matrix.mT).view(B, L, self.S_D)
            ], dim=1)                                                                                                                   # [B x (L + 1) x S_D]
            observation_biases = (state_biases[:, :-1] @ self.F.mT + u) @ self.H.mT                                                     # [B x L x O_D]

            state_biases = [state_biases[:, 1:]]                                                                                        # 1 x [B x L x S_D]                                                                                               # sqrtT x [B x ≈sqrtT x S_D]
            observation_biases = [observation_biases]                                                                                   # 1 x [B x L x O_D]

        else:
            p = hsqrtL * lsqrtL - L

            reshaped_padded_observations = torch.cat([
                observations, torch.zeros_like(observations[:, :p])
            ], dim=1).reshape(B * lsqrtL, hsqrtL, self.O_D)                                                                             # [BsqrtL x sqrtL x O_D]
            u = torch.cat([u, torch.zeros_like(u[:, :p])], dim=1).reshape(B * lsqrtL, hsqrtL, self.S_D)                                 # [BsqrtL x sqrtL x S_D]

            reshaped_state_biases = torch.cat([
                torch.zeros((B * lsqrtL, 1, self.S_D)),
                ((u @ E.mT + reshaped_padded_observations @ self.K.mT).view(B * lsqrtL, -1) @ lower_triangular_matrix.T).view(B * lsqrtL, hsqrtL, self.S_D)
            ], dim=1)                                                                                                                   # [BsqrtT x (sqrtT + 1) x S_D]
            reshaped_observation_biases = (reshaped_state_biases[:, :-1] @ self.F.mT + u) @ self.H.mT                                   # [BsqrtT x sqrtT x O_D]

            state_biases = list(reshaped_state_biases[:, 1:].view(B, lsqrtL, hsqrtL, self.S_D).transpose(0, 1))                         # sqrtT x [B x sqrtT x S_D]                                                                                               # sqrtT x [B x ≈sqrtT x S_D]
            observation_biases = list(reshaped_observation_biases.view(B, lsqrtL, hsqrtL, self.O_D).transpose(0, 1))                    # sqrtT x [B x sqrtT x O_D]
            if p > 0:
                state_biases[-1] = state_biases[-1][:, :-p]                                                                             # sqrtT x [B x ≈sqrtT x S_D]
                observation_biases[-1] = observation_biases[-1][:, :-p]                                                                 # sqrtT x [B x ≈sqrtT x O_D]

        return {
            "state_form": (state_weights[1:], state_biases),
            "observation_form": (observation_weights[:-1], observation_biases)
        }


class SequentialController(Controller, SequentialPredictor):
    def __init__(self, modelArgs: Namespace):
        SequentialPredictor.__init__(self, modelArgs)

    def forward(self, trace: Dict[str, Dict[str, torch.Tensor]], mode: str = None) -> Dict[str, Dict[str, torch.Tensor]]:
        trace = self.trace_to_td(trace)
        actions, observations = trace["controller"], trace["environment"]["observation"]

        state_estimation = torch.randn((*observations.shape[:-2], self.S_D))
        result = self.forward_with_initial(state_estimation, actions, observations, mode)

        state_estimation_history = torch.cat([
            state_estimation.unsqueeze(-2),
            result["environment"]["state"][..., :-1, :]
        ], dim=1)
        result["controller"] = {
            k: state_estimation_history @ -self.L[k].mT
            for k in vars(self.problem_shape.controller)
        }
        return result




