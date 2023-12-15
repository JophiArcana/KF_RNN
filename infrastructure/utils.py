from argparse import Namespace
import functools
import json
import math
import numpy as np
import torch
import torch.nn as nn
from typing import *


def sample_stable_state_matrix(d: int, lo=0.4, hi=0.8) -> torch.Tensor:
    M = torch.randn(d, d)
    eig_, V = torch.linalg.eig(M)
    eig_abs_ = eig_.abs()
    eig_indices = torch.argmax((eig_abs_[None] == eig_abs_[:, None]).to(torch.int), dim=0)

    eig_abs = torch.empty(d).uniform_(lo, hi)[eig_indices]
    eig = eig_ * (eig_abs / eig_abs_)
    return (V @ torch.diag(eig) @ torch.linalg.inv(V)).real


def pow_series(M: torch.Tensor, n: int) -> torch.Tensor:
    N = M.shape[0]
    k = int(math.ceil(math.log2(n)))

    bits = [M]
    for _ in range(k - 1):
        bits.append(bits[-1] @ bits[-1])

    I = torch.eye(N, device=M.device)
    result = I
    for bit in bits:
        augmented_bit = torch.cat([I, bit], dim=1)
        blocked_result = result @ augmented_bit
        result = torch.cat([blocked_result[:, :N], blocked_result[:, N:]], dim=0)
    return result.reshape(1 << k, N, N)[:n]


def color(z: float, scale: float=120.) -> np.ndarray:
    k = 2 * np.pi * z / scale
    return (1 + np.asarray([np.sin(k), np.sin(k + 2 * np.pi / 3), np.sin(k + 4 * np.pi / 3)], dtype=float)) / 2


def toJSON(n: Namespace):
    d = dict(vars(n))
    for k, v in d.items():
        if type(v) == Namespace:
            d[k] = toJSON(v)
        else:
            try:
                json.dumps(v)
                d[k] = v
            except TypeError as excp:
                d[k] = repr(v)
    return d


def remove_nans_and_infs(x: torch.Tensor) -> torch.Tensor:
    return x[~(x.isnan() + x.isinf())]


def nested_type(o: object) -> object:
    if type(o) in [list, tuple]:
        return type(o)(map(nested_type, o))
    elif type(o) == dict:
        return {k: nested_type(v) for k, v in o.items()}
    else:
        return type(o)


def class_name(cls: type) -> str:
    return str(cls)[8:-2].split('.')[-1]


def run_stacked_modules(
        base_module: nn.Module,
        stacked_modules: Dict[str, nn.Parameter],
        args: Any,
        kwargs: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    def run(kf_dict, ag):
        return nn.utils.stateless.functional_call(base_module, kf_dict, ag, kwargs)
    vmap_run = torch.func.vmap(run, randomness='different')

    return vmap_run(stacked_modules, args)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))




