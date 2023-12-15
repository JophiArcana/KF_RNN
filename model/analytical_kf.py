import torch
from typing import *

from model.kf import KF
from model.linear_system import LinearSystem


class AnalyticalKF(KF):
    def __init__(self, system: LinearSystem):
        super().__init__()
        self.system: LinearSystem = system
        self.input_enabled = True

    """ forward
        :parameter {
            'state': [B x S_D],
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns {
            'state_estimation': [B x L x S_D],
            'observation_estimation': [B x L x O_D],
            'state_covariance': [B x L x S_D x S_D],        (Note: first dimension is pseudo-expanded)
            'observation_covariance': [B x L x O_D x O_D]   (Note: first dimension is pseudo-expanded)
        }
    """
    def forward(self, trace: Dict[str, torch.Tensor], steady_state: bool = False) -> Dict[str, torch.Tensor]:
        state, inputs, observations = KF.extract(trace, self.system.S_D)
        B = state.shape[0]
        dev = self.system.F.device

        state_estimation = state
        state_covariance = torch.zeros((state.shape[1], state.shape[1]), device=self.system.F.device)

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

            if steady_state:
                K = self.system.K
            else:
                K = state_covariance @ self.system.H.T @ torch.linalg.inv(observation_covariance)
            state_covariances.append(
                state_covariance := (torch.eye(state.shape[1], device=dev) - K @ self.system.H) @ state_covariance
            )

            # Update
            state_estimations.append(
                state_estimation := state_estimation + (observations[:, i] - observation_estimation) @ K.T
            )

        return {
            'state_estimation': torch.stack(state_estimations, dim=1),
            'state_covariance': torch.stack(state_covariances)[None].expand(B, -1, -1, -1),
            'observation_estimation': torch.stack(observation_estimations, dim=1),
            'observation_covariance': torch.stack(observation_covariances)[None].expand(B, -1, -1, -1)
        }




