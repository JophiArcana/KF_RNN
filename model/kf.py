
from typing import *

import torch
import torch.nn as nn

from model.linear_system import LinearSystem


class KF(nn.Module):
    def __init__(self, system: LinearSystem):
        super().__init__()
        self.system: LinearSystem = system


    def forward(self,
            initial_state: torch.Tensor,    # [B x S_D]
            inputs: torch.Tensor,           # [B x T x I_D]
            observations: torch.Tensor,     # [B x T x O_D]
    ) -> Dict[str, torch.Tensor]:           # [B x T x S_D], [T x S_D x S_D], [B x T x O_D], [T x O_D x O_D]
        state_estimation = initial_state
        state_covariance = torch.zeros(initial_state.shape[1], initial_state.shape[1], device=self.system.F.device)

        state_estimations, observation_estimations = [], []
        state_covariances, observation_covariances = [], []

        for i in range(inputs.shape[1]):
            # Prediction
            state_estimation = state_estimation @ self.system.F.T + inputs[:, i] @ self.system.B.T
            state_covariance = self.system.F @ state_covariance @ self.system.F.T + self.system.S_W

            observation_estimations.append(
                observation_estimation := state_estimation @ self.system.H.T
            )
            observation_covariances.append(
                observation_covariance := self.system.H @ state_covariance @ self.system.H.T + self.system.S_V
            )

            K = state_covariance @ self.system.H.T @ torch.linalg.inv(observation_covariance)
            state_covariances.append(
                state_covariance := (torch.eye(initial_state.shape[1]) - K @ self.system.H) @ state_covariance
            )

            # Update
            state_estimations.append(
                state_estimation := state_estimation + (observations[:, i] - observation_estimation) @ K.T
            )

        return {
            'state_estimation': torch.stack(state_estimations, dim=1),
            'state_covariance': torch.stack(state_covariances),
            'observation_estimation': torch.stack(observation_estimations, dim=1),
            'observation_covariance': torch.stack(observation_covariances)
        }




