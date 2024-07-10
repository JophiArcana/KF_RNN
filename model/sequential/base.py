import math
from argparse import Namespace
from typing import *

import numpy as np
import torch
from tensordict import TensorDict

from infrastructure import utils
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
                         kfs: TensorDict[str, torch.Tensor],        # [B... x ...]
                         systems: TensorDict[str, torch.Tensor],    # [B... x ...]
    ) -> torch.Tensor:                                              # [B...]
        # Variable definition
        controller_keys = kfs.get("B", {}).keys()

        Fh = utils.complex(kfs["F"])                                                                    # [B... x S_Dh x S_Dh]
        Hh = utils.complex(kfs["H"])                                                                    # [B... x O_D x S_Dh]
        Kh = utils.complex(kfs["K"])                                                                    # [B... x S_Dh x O_D]
        Bh = utils.complex(kfs["B"])                                                                    # [B... x S_Dh x I_D?]

        K = utils.complex(systems["environment", "K"])                                                  # [B... x S_D x O_D]
        L = utils.complex(systems["controller", "L"])                                                   # [B... x I_D? x S_D]

        Fa = utils.complex(systems["effective", "F"])                                                   # [B... x 2S_D x 2S_D]
        Ha = utils.complex(systems["effective", "H"])                                                   # [B... x O_D x 2S_D]
        sqrt_S_Wa = utils.complex(systems["effective", "sqrt_S_W"])                                     # [B... x 2S_D x 2S_D]
        sqrt_S_Va = utils.complex(systems["effective", "sqrt_S_V"])                                     # [B... x O_D x O_D]
        La = utils.complex(systems["effective", "L"])                                                   # [B... x I_D? x 2S_D]

        S_D, O_D = K.shape[-2:]
        S_Dh = Fh.shape[-1]

        M, Mh = Fa, Fh @ (torch.eye(S_Dh) - Kh @ Hh)                                                    # [B... x 2S_D x 2S_D], [B... x S_Dh x S_Dh]
        D, V = torch.linalg.eig(M)                                                                      # [B... x 2S_D], [B... x 2S_D x 2S_D]
        Dh, Vh = torch.linalg.eig(Mh)                                                                   # [B... x S_Dh], [B... x S_Dh x S_Dh]
        Vinv, Vhinv = torch.inverse(V), torch.inverse(Vh)                                               # [B... x 2S_D x 2S_D], [B... x S_Dh x S_Dh]

        Has, Hhs = Ha @ V, Hh @ Vh                                                                      # [B... x O_D x 2S_D], [B... x O_D x S_Dh]
        Las = La.apply(lambda t: t @ V)                                                                 # [B... x I_D? x 2S_D]
        sqrt_S_Was = Vinv @ sqrt_S_Wa                                                                   # [B... x 2S_D x 2S_D]

        # Precomputation
        HhstHhs = Hhs.mT @ Hhs                                                                          # [B... x S_Dh x S_Dh]
        VhinvFhKh_BhLK = Vhinv @ (Fh @ Kh - sum(Bh[k] @ L[k] for k in controller_keys) @ K)             # [B... x S_Dh x O_D]
        VhinvFhKhHas_BhLas = Vhinv @ (Fh @ Kh @ Has - sum(Bh[k] @ Las[k] for k in controller_keys))     # [B... x S_Dh x 2S_D]

        # State evolution noise error
        # Highlight
        ws_current_err = (Has @ sqrt_S_Was).norm(dim=(-2, -1)) ** 2                                     # [B...]

        def k(A: torch.Tensor,  # [B... x m x n]
              B: torch.Tensor,  # [B... x p x q]
              Ma: torch.Tensor, # [B... x m x n]
              Mb: torch.Tensor, # [B... x p x q]
              C: torch.Tensor   # [B... x m x p]
        ) -> torch.Tensor:      # [B... x n x q]
            P = A.unsqueeze(-1).unsqueeze(-3) * B.unsqueeze(-2).unsqueeze(-4)           # [B... x m x p x n x q]
            _coeff = Ma.unsqueeze(-1).unsqueeze(-3) * Mb.unsqueeze(-2).unsqueeze(-4)    # [B... x m x p x n x q]
            return torch.sum(P * (_coeff / (1 - _coeff)) * C.unsqueeze(-1).unsqueeze(-2), dim=[-3, -4])

        Dj = D.unsqueeze(-2)                                                                            # [B... x 1 x 2S_D]
        Dhi = Dh.unsqueeze(-1)                                                                          # [B... x S_Dh x 1]
        scaled_VhinvFhKhHas_BhLas = VhinvFhKhHas_BhLas / (Dhi - Dj)                                     # [B... x S_Dh x 2S_D]

        # Highlight
        ws_geometric_err = utils.batch_trace(sqrt_S_Was.mT @ (
            k(Has, Has, Dj, Dj, torch.eye(O_D))
            - 2 * k(Hhs.mT @ Has, scaled_VhinvFhKhHas_BhLas, Dj, Dhi, torch.eye(S_Dh))
            + 2 * k(Hhs.mT @ Has, scaled_VhinvFhKhHas_BhLas, Dj, Dj, torch.eye(S_Dh))
            + k(scaled_VhinvFhKhHas_BhLas, scaled_VhinvFhKhHas_BhLas, Dhi, Dhi, HhstHhs)
            - 2 * k(scaled_VhinvFhKhHas_BhLas, scaled_VhinvFhKhHas_BhLas, Dhi, Dj, HhstHhs)
            + k(scaled_VhinvFhKhHas_BhLas, scaled_VhinvFhKhHas_BhLas, Dj, Dj, HhstHhs)
        ) @ sqrt_S_Was)                                                                                 # [B...]

        # Observation noise error
        # Highlight
        v_current_err = sqrt_S_Va.norm(dim=(-2, -1)) ** 2                                               # [B...]

        # Highlight
        v_geometric_err = utils.batch_trace(sqrt_S_Va.mT @ VhinvFhKh_BhLK.mT @ (
            HhstHhs / (1 - Dh.unsqueeze(-1) * Dh.unsqueeze(-2))
        ) @ VhinvFhKh_BhLK @ sqrt_S_Va)                                                                 # [B...]

        err = ws_current_err + ws_geometric_err + v_current_err + v_geometric_err                       # [B...]
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




