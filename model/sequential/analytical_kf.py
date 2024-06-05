from typing import *

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from infrastructure import utils
from model.kf import KF
from model.linear_system import LinearSystem
from model.sequential import SequentialKF


class AnalyticalKF(SequentialKF):
    @classmethod
    def add_targets(cls,
                    kf_arr: np.ndarray[SequentialKF],
                    dataset: TensorDict[str, torch.Tensor] | TensorDictBase[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        with torch.set_grad_enabled(False):
            dataset["target"] = KF.run(
                *utils.stack_module_arr(kf_arr),
                dataset=dataset, kwargs={'steady_state': True}
            )
        return dataset

    def __init__(self, system: LinearSystem):
        super().__init__()
        self.eval()

        # self.system = system
        self.S_D, self.I_D, self.O_D = system.S_D, system.I_D, system.O_D
        self.input_enabled = system.input_enabled

        for attr in ("F", "B", "H", "S_W", "S_V"):
            setattr(self, attr, getattr(system, attr))
        self.register_buffer("K", system.K, persistent=True)

    # @property
    # def K(self):
    #     return self.system.K

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
        if steady_state:
            return super().forward(trace)
        else:
            state, inputs, observations = self.extract(trace, self.S_D)
            B = state.shape[0]

            state_estimation = state
            state_covariance = torch.zeros((state.shape[1], state.shape[1]))

            state_estimations, observation_estimations = [], []
            state_covariances, observation_covariances = [], []

            for i in range(inputs.shape[1]):
                # Prediction
                state_estimation = state_estimation @ self.F.mT + inputs[:, i] @ self.B.mT
                state_covariance = self.F @ state_covariance @ self.F.mT + self.S_W

                observation_estimations.append(
                    observation_estimation := state_estimation @ self.H.T
                )
                observation_covariances.append(
                    observation_covariance := self.H @ state_covariance @ self.H.mT + self.S_V
                )

                K = state_covariance @ self.H.mT @ torch.linalg.inv(observation_covariance)
                state_covariances.append(
                    state_covariance := (torch.eye(state.shape[1]) - K @ self.H) @ state_covariance
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




