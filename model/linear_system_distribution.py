from argparse import Namespace
from typing import *

import numpy as np
import torch

from infrastructure import utils
from model.linear_system import LinearSystemGroup


class LinearSystemDistribution(object):
    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def sample(self, SHP: Namespace, shape: Tuple[int, ...]) -> LinearSystemGroup:
        return LinearSystemGroup(self.sample_parameters(SHP, shape), SHP.input_enabled)

class MOPDistribution(LinearSystemDistribution):
    def __init__(self, F_mode: str, H_mode: str, W_std: float, V_std: float) -> None:
        assert F_mode in ("gaussian", "uniform"), f"F_mode must be one of (gaussian, uniform) but got {F_mode}."
        self.F_mode = F_mode

        assert H_mode in ("gaussian", "uniform"), f"H_mode must be one of (gaussian, uniform) but got {H_mode}."
        self.H_mode = H_mode

        self.W_std = W_std
        self.V_std = V_std

    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        if self.F_mode == "gaussian":
            F = torch.randn((*shape, SHP.S_D, SHP.S_D))
        else:
            F = torch.zeros((*shape, SHP.S_D, SHP.S_D)).uniform_(-1., 1.)
        F *= (0.95 / torch.linalg.eigvals(F).abs().max(dim=-1).values.unsqueeze(-1).unsqueeze(-2))

        B = torch.randn((*shape, SHP.S_D, SHP.I_D)) / (3 ** 0.5)

        if self.H_mode == "gaussian":
            H = torch.randn((*shape, SHP.O_D, SHP.S_D)) / (3 ** 0.5)
        else:
            H = torch.zeros((*shape, SHP.O_D, SHP.S_D)).uniform_(-1., 1.)

        sqrt_S_W = (torch.eye(SHP.S_D) * self.W_std).expand(*shape, SHP.S_D, SHP.S_D)
        sqrt_S_V = (torch.eye(SHP.O_D) * self.V_std).expand(*shape, SHP.O_D, SHP.O_D)

        return {"F": F, "B": B, "H": H, "sqrt_S_W": sqrt_S_W, "sqrt_S_V": sqrt_S_V}




