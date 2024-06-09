from argparse import Namespace
from typing import *

import numpy as np
import torch

from infrastructure import utils
from model.linear_system import LinearSystemGroup


class LinearSystemDistribution(object):
    def __init__(self, sample_func: Callable[[Namespace, Tuple[int, ...]], Dict[str, torch.Tensor]]):
        self.sample_func = sample_func

    def sample(self, SHP: Namespace, shape: Tuple[int, ...]) -> LinearSystemGroup:
        return LinearSystemGroup(self.sample_func(SHP, shape), SHP.input_enabled)

def get_mop_sample_func(
        F_mode: str,
        H_mode: str,
        W_std: float,
        V_std: float
) -> Callable[[Namespace, Tuple[int, ...]], Dict[str, torch.Tensor]]:
    def sample_func(SHP: Namespace, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        if F_mode == "gaussian":
            F = torch.randn((*shape, SHP.S_D, SHP.S_D))
        elif F_mode == "uniform":
            F = torch.zeros((*shape, SHP.S_D, SHP.S_D)).uniform_(-1., 1.)
        else:
            raise AssertionError(f"F_mode must be one of (gaussian, uniform) but got {F_mode}.")
        F *= (0.95 / torch.linalg.eigvals(F).abs().max(dim=-1).unsqueeze(-1).unsqueeze(-2))

        B = torch.randn((*shape, SHP.S_D, SHP.I_D)) / (3 ** 0.5)

        if H_mode == "gaussian":
            H = torch.randn((*shape, SHP.O_D, SHP.S_D)) / (3 ** 0.5)
        elif H_mode == "uniform":
            H = torch.zeros((*shape, SHP.O_D, SHP.S_D)).uniform_(-1., 1.)
        else:
            raise AssertionError(f"H_mode must be one of (gaussian, uniform) but got {H_mode}.")

        sqrt_S_W = (torch.eye(SHP.S_D) * W_std).expand(*shape, SHP.S_D, SHP.S_D)
        sqrt_S_V = (torch.eye(SHP.O_D) * V_std).expand(*shape, SHP.O_D, SHP.O_D)

        return {"F": F, "B": B, "H": H, "sqrt_S_W": sqrt_S_W, "sqrt_S_V": sqrt_S_V}
    return sample_func



