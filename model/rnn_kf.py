
import math

import torch
import torch.nn as nn
from typing import *

from infrastructure import utils


class RnnKF(nn.Module):
    @classmethod
    def evaluate_form(cls,
                      initial_state: torch.Tensor,  # [B x S_D]
                      form: Sequence[torch.Tensor]  # [T x D x S_D], [B x T x D]
    ) -> torch.Tensor:                              # [B x T x D]
        weights, biases = form
        return (weights.flatten(0, 1)[None] @ initial_state[:, :, None]).reshape(-1, *biases.shape[1:]) + biases


    def __init__(self, state_dim: int, input_dim: int, observation_dim: int):
        super().__init__()

        self.S_D = state_dim
        self.I_D = input_dim
        self.O_D = observation_dim

        self.F: nn.Parameter = nn.Parameter(utils.sample_stable_state_matrix(self.S_D))
        self.B: nn.Parameter = nn.Parameter(torch.zeros(self.S_D, self.I_D))
        torch.nn.init.xavier_normal_(self.B)

        self.H: nn.Parameter = nn.Parameter(torch.zeros(self.O_D, self.S_D))
        torch.nn.init.xavier_normal_(self.H)

        self.K: nn.Parameter = nn.Parameter(torch.zeros(self.S_D, self.O_D))


    def _forward(self,
                state: torch.Tensor,        # [B x S_D]
                input: torch.Tensor,        # [B x I_D]
                observation: torch.Tensor,  # [B x O_D]
    ) -> Sequence[torch.Tensor]:            # [B x S_D], [B x O_D]
        state_estimation = state @ self.F.T + input @ self.B.T
        observation_estimation = state_estimation @ self.H.T
        state_estimation = state_estimation + (observation - observation_estimation) @ self.K.T

        return state_estimation, observation_estimation


    def _forward_generic(self,
                        form: torch.Tensor,         # [S_D x S_D], [B x S_D]
                        input: torch.Tensor,        # [B x I_D]
                        observation: torch.Tensor   # [B x O_D]
    ) -> Sequence[Sequence[torch.Tensor]]:          # ([S_D x S_D], [B x S_D]), ([O_D x S_D], [B x O_D])
        state_weight, state_bias = form

        next_state_weight = self.F @ state_weight
        next_observation_weight = self.H @ next_state_weight
        next_state_weight = next_state_weight - self.K @ next_observation_weight

        next_state_bias = state_bias @ self.F.T + input @ self.B.T
        next_observation_bias = next_state_bias @ self.H.T
        next_state_bias = next_state_bias + (observation - next_observation_bias) @ self.K.T

        return (next_state_weight, next_state_bias), (next_observation_weight, next_observation_bias)


    def _forward_generic_bias(self,
                        state_bias: torch.Tensor,   # [B x S_D]
                        input: torch.Tensor,        # [B x I_D]
                        observation: torch.Tensor   # [B x O_D]
    ) -> Dict[str, torch.Tensor]:                   # [B x S_D], [B x O_D]
        next_state_bias = state_bias @ self.F.T + input @ self.B.T
        next_observation_bias = next_state_bias @ self.H.T
        next_state_bias = next_state_bias + (observation - next_observation_bias) @ self.K.T

        return {
            'state_bias': next_state_bias,
            'observation_bias': next_observation_bias
        }


    def forward(self,
                      initial_state: torch.Tensor,  # [B x S_D]
                      inputs: torch.Tensor,         # [B x T x I_D]
                      observations: torch.Tensor,   # [B x T x O_D]
                      use_form=False
    ) -> Dict[str, torch.Tensor]:                   # [B x T x S_D], [B x T x O_D]
        state_estimations, observation_estimations = [], []
        if use_form:
            result_generic = self.forward_generic(inputs, observations)

            state_weights, state_biases_list = result_generic['state_form']                     # [sqrtT x S_D x S_D], sqrtT x [B x ≈sqrtT x S_D]
            observation_weights, observation_biases_list = result_generic['observation_form']   # [sqrtT x O_D x S_D], sqrtT x [B x ≈sqrtT x O_D]

            for state_biases, observation_biases in zip(state_biases_list, observation_biases_list):
                state_estimations.append(RnnKF.evaluate_form(initial_state, (state_weights, state_biases)))
                observation_estimations.append(RnnKF.evaluate_form(initial_state, (observation_weights, observation_biases)))

                initial_state = state_estimations[-1][:, -1]
        else:
            state_estimation = initial_state

            for i in range(inputs.shape[1]):
                state_estimation, observation_estimation = self._forward(state_estimation, inputs[:, i], observations[:, i])

                state_estimations.append(state_estimation)
                observation_estimations.append(observation_estimation)

        return {
            'state_estimation': torch.stack(state_estimations, dim=1),
            'observation_estimation': torch.stack(observation_estimations, dim=1)
        }


    def forward_generic(self,
                        inputs: torch.Tensor,                                           # [B x T x I_D]
                        observations: torch.Tensor,                                     # [B x T x O_D]
                        ) -> Dict[str, Tuple[torch.Tensor, Sequence[torch.Tensor]]]:    # ([sqrtT x S_D x S_D], sqrtT x [B x ≈sqrtT x S_D]), ([sqrtT x O_D x S_D], sqrtT x [B x ≈sqrtT x O_D])
        device = self.F.device

        B, T = inputs.shape[:2]
        sqrtT = int(math.ceil(math.sqrt(T)))

        E = (torch.eye(self.S_D, device=device) - self.K @ self.H)
        M = E @ self.F

        # Compute the weights efficiently by eigenvalue decomposition of (I - KH)F and repeated powers
        L, V = torch.linalg.eig(M)                                                                                                                  # [S_D], [S_D x S_D]
        eig_powers = torch.diag_embed(torch.pow(L[:, None], torch.arange(sqrtT + 1, device=device)).T)                                              # [(sqrtT + 1) x S_D x S_D]

        state_weights = (V @ eig_powers @ torch.linalg.inv(V)).to(torch.double)                                                                     # [(sqrtT + 1) x S_D x S_D]
        observation_weights = self.H @ self.F @ state_weights                                                                                       # [(sqrtT + 1) x O_D x S_D]

        # Compute the biases efficiently using the state weights
        buffered_state_weights = torch.cat([state_weights, torch.zeros_like(state_weights, device=device)])                                         # [2(sqrtT + 1) x S_D x S_D]
        lower_triangular_indices = torch.arange(T, device=device)[:, None] - torch.arange(T, device=device)                                         # [sqrtT x sqrtT]

        blocked_lower_triangular_matrix = buffered_state_weights[lower_triangular_indices]                                                          # [sqrtT x sqrtT x S_D x S_D]
        lower_triangular_matrix = blocked_lower_triangular_matrix.permute(0, 2, 1, 3).reshape(sqrtT * self.S_D, sqrtT * self.S_D)                   # [(sqrtT S_D) x (sqrtT S_D)]


        p = sqrtT * sqrtT - T
        reshaped_padded_inputs = torch.constant_pad_nd(inputs, (0, 0, 0, p), 0).reshape(B * sqrtT, sqrtT, self.I_D)             # [BsqrtT x sqrtT x I_D]
        reshaped_padded_observations = torch.constant_pad_nd(observations, (0, 0, 0, p), 0).reshape(B * sqrtT, sqrtT, self.O_D) # [BsqrtT x sqrtT x O_D]

        reshaped_state_biases = torch.cat([
            torch.zeros(B * sqrtT, 1, self.S_D, device=device),
            ((reshaped_padded_inputs @ (E @ self.B).T).view(B * sqrtT, -1) @ lower_triangular_matrix.T).view(B * sqrtT, sqrtT, self.S_D) + reshaped_padded_observations @ self.K.T
        ], dim=1)                                                                                                   # [BsqrtT x (sqrtT + 1) x S_D]
        reshaped_observation_biases = (reshaped_state_biases[:, :-1] @ self.F.T + inputs @ self.B.T) @ self.H.T     # [BsqrtT x sqrtT x O_D]

        state_biases = list(reshaped_state_biases[:, 1:].view(B, sqrtT, sqrtT, self.S_D).transpose(0, 1))           # sqrtT x [B x sqrtT x S_D]
        state_biases[-1] = state_biases[-1][:, :-p]                                                                 # sqrtT x [B x ≈sqrtT x S_D]
        observation_biases = list(reshaped_observation_biases.view(B, sqrtT, sqrtT, self.O_D).transpose(0, 1))      # sqrtT x [B x sqrtT x O_D]
        observation_biases[-1] = observation_biases[-1][:, :-p]                                                     # sqrtT x [B x ≈sqrtT x O_D]

        # # Compute the biases by basic forward iteration
        # state_bias = initial_form[1]
        # state_biases, observation_biases = [], []
        #
        # for i in range(T):
        #     state_bias, observation_bias = self._forward_generic_bias(state_bias, inputs[:, i], observations[:, i])
        #
        #     state_biases.append(state_bias)
        #     observation_biases.append(observation_bias)
        #
        # state_biases = torch.stack(state_biases, dim=1)
        # observation_biases = torch.stack(observation_biases, dim=1)

        return {
            'state_form': (state_weights[1:], state_biases),
            'observation_form': (observation_weights[:-1], observation_biases)
        }




