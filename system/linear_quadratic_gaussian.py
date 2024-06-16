from typing import *

import torch
import torch.nn as nn

from infrastructure.discrete_are import solve_discrete_are
from system.core import SystemDistribution
from system.linear_time_invariant import LinearSystemGroup


class LinearQuadraticGaussianGroup(LinearSystemGroup):
    class Distribution(SystemDistribution):
        def __init__(self):
            super().__init__(LinearQuadraticGaussianGroup)

    def __init__(self, params: Dict[str, torch.Tensor], input_enabled: bool):
        super().__init__(params, input_enabled)
        assert self.input_enabled, f"Input must be enabled for linear-quadratic Gaussian model but got {self.input_enabled}."

        Q, R = map(params.__getitem__, ("Q", "R"))
        assert Q.shape[-2] == Q.shape[-1] == self.S_D, f"Q matrix must be of shape {(self.S_D, self.S_D)} but got {Q.shape[-2:]}."
        assert R.shape[-2] == R.shape[-1] == self.I_D, f"R matrix must be of shape {(self.I_D, self.I_D)} but got {R.shape[-2:]}."

        self.Q = nn.Parameter(Q.clone())                                                                    # [N... x S_D x S_D]
        self.R = nn.Parameter(R.clone())                                                                    # [N... x I_D x I_D]

        S = solve_discrete_are(self.F, self.B, self.R, self.Q)                                              # [N... x S_D x S_D]
        self.register_buffer("L", torch.inverse(self.B.mT @ S @ self.B + self.R) @ self.B.mT @ S @ self.F)  # [N... x I_D x S_D]

    def supply_input(self,
                     state_estimation: torch.Tensor # [N... x B x S_D]
    ) -> torch.Tensor:
        return state_estimation @ -self.L.mT        # [N... x B x I_D]




