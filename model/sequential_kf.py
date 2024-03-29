import math
import numpy as np
import torch
from tensordict import TensorDict
from typing import *

from torch import nn as nn

from model.kf import KF
from infrastructure import utils
from infrastructure.settings import device


class SequentialKF(KF):
    @classmethod
    def evaluate_form(cls,
                      state: torch.Tensor,          # [B x S_D]
                      form: Sequence[torch.Tensor]  # [T x D x S_D], [B x T x D]
    ) -> torch.Tensor:                              # [B x T x D]
        weights, biases = form
        return (weights.flatten(0, 1)[None] @ state[:, :, None]).reshape(-1, *biases.shape[1:]) + biases

    @classmethod
    def analytical_error(cls,
                         kfs: TensorDict[str, torch.Tensor],    # [B... x ...]
                         systems: TensorDict[str, torch.Tensor] # [B... x ...]
    ) -> torch.Tensor:                                          # [B...]
        # Variable definition
        def extract_var(d: TensorDict[str, torch.Tensor], k: str):
            return torch.Tensor(d[k]).to(torch.complex64)
        Fh = extract_var(kfs, 'F')                                                              # [B... x S_Dh x S_Dh]
        Bh = extract_var(kfs, 'B')                                                              # [B... x S_Dh x I_D]
        Hh = extract_var(kfs, 'H')                                                              # [B... x O_D x S_Dh]
        Kh = extract_var(kfs, 'K')                                                              # [B... x S_Dh x O_D]

        F = extract_var(systems, 'F')                                                           # [B... x S_D x S_D]
        B = extract_var(systems, 'B')                                                           # [B... x S_D x I_D]
        H = extract_var(systems, 'H')                                                           # [B... x O_D x S_D]
        sqrt_S_W = extract_var(systems, 'sqrt_S_W')                                             # [B... x S_D x S_D]
        sqrt_S_V = extract_var(systems, 'sqrt_S_V')                                             # [B... x O_D x O_D]

        S_D, I_D, O_D = F.shape[-1], B.shape[-1], H.shape[-2]
        S_Dh = Fh.shape[-1]

        M, Mh = F, Fh @ (torch.eye(S_Dh, dtype=Fh.dtype, device=device) - Kh @ Hh)            # [B... x S_D x S_D], [B... x S_Dh x S_Dh]
        L, V = torch.linalg.eig(M)                                                              # [B... x S_D], [B... x S_D x S_D]
        Lh, Vh = torch.linalg.eig(Mh)                                                           # [B... x S_Dh], [B... x S_Dh x S_Dh]
        Vinv, Vhinv = torch.linalg.inv(V), torch.linalg.inv(Vh)                                 # [B... x S_D x S_D], [B... x S_Dh x S_Dh]
        
        Hs, Hhs = H @ V, Hh @ Vh                                                                # [B... x O_D x S_D], [B... x O_D x S_Dh]
        Bs, Bhs = Vinv @ B, Vhinv @ Bh                                                          # [B... x S_D x I_D], [B... x S_Dh x I_D]
        sqrt_S_Ws = Vinv @ sqrt_S_W                                                             # [B... x S_D x S_D]

        # Precomputation
        VhinvFhKh = Vhinv @ Fh @ Kh                                                             # [B... x S_Dh x O_D]
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
        ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ w @ sqrt_S_Ws) # [Bf x Bs]

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

    @classmethod
    def to_sequential_batch(cls, kfs: TensorDict[str, torch.Tensor], input_enabled: bool) -> TensorDict[str, torch.Tensor]:
        return kfs

    def to_sequential(self) -> nn.Module:
        return self

    def _forward(self,
                 state: torch.Tensor,       # [B x S_D]
                 input: torch.Tensor,       # [B x I_D]
                 observation: torch.Tensor, # [B x O_D]
    ) -> Dict[str, torch.Tensor]:           # [B x S_D], [B x O_D]

        state_estimation = state @ self.F.T + input @ self.B.T
        observation_estimation = state_estimation @ self.H.T
        state_estimation = state_estimation + (observation - observation_estimation) @ self.K.T

        return {
            'state_estimation': state_estimation,
            'observation_estimation': observation_estimation
        }

    def forward(self, trace: Dict[str, torch.Tensor], mode: str = None) -> Dict[str, torch.Tensor]:
        state, inputs, observations = self.extract(trace, self.S_D)
        L = inputs.shape[1]

        if mode is None:
            mode = ['sequential', 'form', 'form_sqrt'][np.searchsorted([16, 64], L)]

        state_estimations, observation_estimations = [], []
        if mode == 'sequential':
            state_estimation = state

            for l in range(L):
                result = self._forward(state_estimation, inputs[:, l], observations[:, l])

                state_estimation, observation_estimation = result['state_estimation'], result['observation_estimation']

                state_estimations.append(state_estimation)
                observation_estimations.append(observation_estimation)

            return {
                'state_estimation': torch.stack(state_estimations, dim=1),
                'observation_estimation': torch.stack(observation_estimations, dim=1)
            }
        else:
            result_generic = self.forward_generic(trace, mode)

            state_weights, state_biases_list = result_generic['state_form']                     # [sqrtT x S_D x S_D], sqrtT x [B x ≈sqrtT x S_D]
            observation_weights, observation_biases_list = result_generic['observation_form']   # [sqrtT x O_D x S_D], sqrtT x [B x ≈sqrtT x O_D]

            for state_biases, observation_biases in zip(state_biases_list, observation_biases_list):
                state_estimations.append(SequentialKF.evaluate_form(state, (state_weights[:state_biases.shape[1]], state_biases)))
                observation_estimations.append(SequentialKF.evaluate_form(state, (observation_weights[:observation_biases.shape[1]], observation_biases)))

                state = state_estimations[-1][:, -1]

            return {
                'state_estimation': torch.cat(state_estimations, dim=1),
                'observation_estimation': torch.cat(observation_estimations, dim=1)
            }

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns {
            'state_form': ([sqrtT x S_D x S_D], sqrtT x [B x ≈sqrtT x S_D]),
            'observation_form': ([sqrtT x O_D x S_D], sqrtT x [B x ≈sqrtT x O_D])
        }
    """
    def forward_generic(self,
                        trace: Dict[str, torch.Tensor],
                        mode: str
    ) -> Dict[str, Tuple[torch.Tensor, Sequence[torch.Tensor]]]:
        inputs, observations = trace['input'], trace['observation']

        # Precomputation
        B, T = inputs.shape[:2]
        hsqrtT = int(math.ceil(math.sqrt(T)))
        lsqrtT = int(math.ceil(T / hsqrtT))

        E = torch.eye(self.S_D, device=device) - self.K @ self.H
        M = E @ self.F

        subT = T if mode == 'form' else hsqrtT                                                                                          # Length of vectorized subsequence
        # Compute the weights efficiently by eigenvalue decomposition of (I - KH)F and repeated powers

        # L, V = torch.linalg.eig(M)
        """
        eig_powers = torch.diag_embed(torch.pow(L[:, None], torch.arange(subT + 1, device=device)).T)                                   # [(subT + 1) x S_D x S_D]
        state_weights = (V @ eig_powers @ torch.linalg.inv(V)).to(torch.double)                                                         # [(subT + 1) x S_D x S_D]
        state_weights = torch.stack([torch.matrix_power(M, n) for n in range(subT + 1)])
        """
        state_weights = utils.pow_series(M, subT + 1)
        observation_weights = self.H @ self.F @ state_weights                                                                           # [(subT + 1) x O_D x S_D]

        # Compute the biases efficiently using the state weights
        buffered_state_weights = torch.cat([state_weights, torch.zeros((1, self.S_D, self.S_D), device=device)])                      # [(subT + 2) x S_D x S_D]
        lower_triangular_indices = (torch.arange(subT, device=device)[:, None] - torch.arange(subT, device=device)).clamp_min(-1)   # [subT x sqrtT]

        blocked_lower_triangular_matrix = buffered_state_weights[lower_triangular_indices]                                              # [subT x subT x S_D x S_D]
        lower_triangular_matrix = blocked_lower_triangular_matrix.permute(0, 2, 1, 3).reshape(subT * self.S_D, subT * self.S_D)

        if mode == 'form':
            state_biases = torch.cat([
                torch.zeros(B, 1, self.S_D, device=device),
                ((inputs @ (E @ self.B).T + observations @ self.K.T).view(B, -1) @ lower_triangular_matrix.T).view(B, T, self.S_D)
            ], dim=1)                                                                                                                   # [B x (T + 1) x S_D]
            observation_biases = (state_biases[:, :-1] @ self.F.T + inputs @ self.B.T) @ self.H.T                                       # [B x T x O_D]

            state_biases = [state_biases[:, 1:]]                                                                                        # 1 x [B x T x S_D]                                                                                               # sqrtT x [B x ≈sqrtT x S_D]
            observation_biases = [observation_biases]                                                                                   # 1 x [B x T x O_D]

        else:
            p = hsqrtT * lsqrtT - T
            reshaped_padded_inputs = torch.constant_pad_nd(inputs, (0, 0, 0, p), 0).reshape(B * lsqrtT, hsqrtT, self.I_D)               # [BsqrtT x sqrtT x I_D]
            reshaped_padded_observations = torch.constant_pad_nd(observations, (0, 0, 0, p), 0).reshape(B * lsqrtT, hsqrtT, self.O_D)   # [BsqrtT x sqrtT x O_D]

            reshaped_state_biases = torch.cat([
                torch.zeros(B * lsqrtT, 1, self.S_D, device=device),
                ((reshaped_padded_inputs @ (E @ self.B).T + reshaped_padded_observations @ self.K.T).view(B * lsqrtT, -1) @ lower_triangular_matrix.T).view(B * lsqrtT, hsqrtT, self.S_D)
            ], dim=1)                                                                                                               # [BsqrtT x (sqrtT + 1) x S_D]
            reshaped_observation_biases = (reshaped_state_biases[:, :-1] @ self.F.T + reshaped_padded_inputs @ self.B.T) @ self.H.T # [BsqrtT x sqrtT x O_D]

            state_biases = list(reshaped_state_biases[:, 1:].view(B, lsqrtT, hsqrtT, self.S_D).transpose(0, 1))                     # sqrtT x [B x sqrtT x S_D]                                                                                               # sqrtT x [B x ≈sqrtT x S_D]
            observation_biases = list(reshaped_observation_biases.view(B, lsqrtT, hsqrtT, self.O_D).transpose(0, 1))                # sqrtT x [B x sqrtT x O_D]
            if p > 0:
                state_biases[-1] = state_biases[-1][:, :-p]                                                                         # sqrtT x [B x ≈sqrtT x S_D]
                observation_biases[-1] = observation_biases[-1][:, :-p]                                                             # sqrtT x [B x ≈sqrtT x O_D]

        return {
            'state_form': (state_weights[1:], state_biases),
            'observation_form': (observation_weights[:-1], observation_biases)
        }




