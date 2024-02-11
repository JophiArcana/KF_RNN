from argparse import Namespace
import functools
import json
import math
import numpy as np
import torch
import torch.nn as nn
from typing import *

"""
System and model functions
"""
# def sample_stable_state_matrix(d: int) -> torch.Tensor:
#     M = torch.DoubleTensor([[2.]])
#     scale = 1
#     while torch.max(torch.abs(torch.linalg.eig(M)[0])) > 1:
#         M = scale * torch.randn(d, d)
#         scale *= 0.99
#     return M

def sample_stable_state_matrix(d: int, lo=0.4, hi=0.8) -> torch.Tensor:
    M = torch.randn(d, d)
    eig_, V = torch.linalg.eig(M)
    eig_abs_ = eig_.abs()
    eig_indices = torch.argmax(torch.Tensor(eig_abs_[None] == eig_abs_[:, None]).to(torch.int), dim=0)

    eig_abs = torch.empty(d).uniform_(lo, hi)[eig_indices]
    eig = eig_ * (eig_abs / eig_abs_)
    return (V @ torch.diag(eig) @ torch.linalg.inv(V)).real

def pow_series(M: torch.Tensor, n: int) -> torch.Tensor:
    N = M.shape[0]
    I = torch.eye(N, device=M.device)
    if n == 1:
        return I[None]
    else:
        k = int(math.ceil(math.log2(n)))
        bits = [M]
        for _ in range(k - 1):
            bits.append(bits[-1] @ bits[-1])

        result = I
        for bit in bits:
            augmented_bit = torch.cat([I, bit], dim=1)
            blocked_result = result @ augmented_bit
            result = torch.cat([blocked_result[:, :N], blocked_result[:, N:]], dim=0)
        return result.reshape(1 << k, N, N)[:n]

def run_stacked_modules(
        base_module: nn.Module,
        stacked_modules: Dict[str, nn.Parameter],
        args: Any,
        kwargs: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    def run(kf_dict, ag):
        return nn.utils.stateless.functional_call(base_module, kf_dict, ag, kwargs)
    return torch.func.vmap(run, randomness='different')(stacked_modules, args)


"""
Miscellaneous
"""
def color(z: float, scale: float = 120.) -> np.ndarray:
    k = 2 * np.pi * z / scale
    return (1 + np.asarray([np.sin(k), np.sin(k + 2 * np.pi / 3), np.sin(k + 4 * np.pi / 3)], dtype=float)) / 2

def toJSON(n: Namespace):
    d = dict(vars(n))
    for k, v in d.items():
        if type(v) is Namespace:
            d[k] = toJSON(v)
        else:
            try:
                json.dumps(v)
                d[k] = v
            except TypeError:
                d[k] = repr(v)
    return d

def batch_trace(x: torch.Tensor) -> torch.Tensor:
    return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

def nested_type(o: object) -> object:
    if type(o) in [list, tuple]:
        return type(o)(map(nested_type, o))
    elif type(o) == dict:
        return {k: nested_type(v) for k, v in o.items()}
    else:
        return type(o)

def class_name(cls: type) -> str:
    return str(cls)[8:-2].split('.')[-1]

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))

def capitalize(s: str):
    return s[0].upper() + s[1:]
