import math
import numpy as np
import torch
from typing import *

from model.kf import KF
from infrastructure import utils
from infrastructure.settings import dev_type


class SequentialKF(KF):
    @classmethod
    def evaluate_form(cls,
                      state: torch.Tensor,          # [B x S_D]
                      form: Sequence[torch.Tensor]  # [T x D x S_D], [B x T x D]
    ) -> torch.Tensor:                              # [B x T x D]
        weights, biases = form
        return (weights.flatten(0, 1)[None] @ state[:, :, None]).reshape(-1, *biases.shape[1:]) + biases

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

        E = (torch.eye(self.S_D, device=dev_type) - self.K @ self.H)
        M = E @ self.F

        subT = T if mode == 'form' else hsqrtT                                                                                      # Length of vectorized subsequence
        # Compute the weights efficiently by eigenvalue decomposition of (I - KH)F and repeated powers

        # L, V = torch.linalg.eig(M)
        """
        eig_powers = torch.diag_embed(torch.pow(L[:, None], torch.arange(subT + 1, device=device)).T)                               # [(subT + 1) x S_D x S_D]
        state_weights = (V @ eig_powers @ torch.linalg.inv(V)).to(torch.double)                                                     # [(subT + 1) x S_D x S_D]
        state_weights = torch.stack([torch.matrix_power(M, n) for n in range(subT + 1)])
        """
        state_weights = utils.pow_series(M, subT + 1)
        observation_weights = self.H @ self.F @ state_weights                                                                       # [(subT + 1) x O_D x S_D]

        # Compute the biases efficiently using the state weights
        buffered_state_weights = torch.cat([state_weights, torch.zeros_like(state_weights, device=dev_type)])                         # [2(sqrtT + 1) x S_D x S_D]
        lower_triangular_indices = torch.arange(subT, device=dev_type)[:, None] - torch.arange(subT, device=dev_type)                   # [subT x sqrtT]

        blocked_lower_triangular_matrix = buffered_state_weights[lower_triangular_indices]                                          # [subT x subT x S_D x S_D]
        lower_triangular_matrix = blocked_lower_triangular_matrix.permute(0, 2, 1, 3).reshape(subT * self.S_D, subT * self.S_D)

        if mode == 'form':
            state_biases = torch.cat([
                torch.zeros(B, 1, self.S_D, device=dev_type),
                ((inputs @ (E @ self.B).T + observations @ self.K.T).view(B, -1) @ lower_triangular_matrix.T).view(B, T, self.S_D)
            ], dim=1)                                                                                                               # [B x (T + 1) x S_D]
            observation_biases = (state_biases[:, :-1] @ self.F.T + inputs @ self.B.T) @ self.H.T                                   # [B x T x O_D]

            state_biases = [state_biases[:, 1:]]                                                                                    # 1 x [B x T x S_D]                                                                                               # sqrtT x [B x ≈sqrtT x S_D]
            observation_biases = [observation_biases]                                                                               # 1 x [B x T x O_D]

        else:
            p = hsqrtT * lsqrtT - T
            reshaped_padded_inputs = torch.constant_pad_nd(inputs, (0, 0, 0, p), 0).reshape(B * lsqrtT, hsqrtT, self.I_D)               # [BsqrtT x sqrtT x I_D]
            reshaped_padded_observations = torch.constant_pad_nd(observations, (0, 0, 0, p), 0).reshape(B * lsqrtT, hsqrtT, self.O_D)   # [BsqrtT x sqrtT x O_D]

            reshaped_state_biases = torch.cat([
                torch.zeros(B * lsqrtT, 1, self.S_D, device=dev_type),
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




