import functools
import hashlib
import json
import math
import os
import sys
from argparse import Namespace
from collections import OrderedDict
from types import MappingProxyType
from typing import *

import numpy as np
import torch
import torch.nn as nn
from dimarray import DimArray, Dataset
from tensordict import TensorDict
from torch.utils._pytree import tree_flatten, tree_unflatten


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

def sample_stable_state_matrix(d: int, batch_size: Tuple[int, ...] = (), lo=0.4, hi=0.9) -> torch.Tensor:
    M = torch.randn((*batch_size, d, d))                        # [B... x D x D]
    eig_, V = torch.linalg.eig(M)                               # [B... x D], [B... x D x D]
    eig_abs_ = eig_.abs()                                       # [B... x D]
    eig_indices = torch.argmax(torch.Tensor(eig_abs_.unsqueeze(-2) == eig_abs_.unsqueeze(-1)).to(torch.int), dim=-2)    # [B... x D]

    eig_abs = torch.take_along_dim(torch.empty((*batch_size, d)).uniform_(lo, hi), eig_indices, dim=-1)                 # [B... x D]
    eig = eig_ * (eig_abs / eig_abs_)                           # [B... x D]
    return (V @ torch.diag_embed(eig) @ torch.inverse(V)).real  # [B... x D x D]

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

def stack_tensor_arr(tensor_arr: np.ndarray[torch.Tensor], dim: int = 0) -> Union[torch.Tensor, TensorDict[str, torch.Tensor]]:
    result = torch.stack((*tensor_arr.ravel(),), dim=dim)
    if tensor_arr.ndim > 1:
        return result.unflatten(dim, tensor_arr.shape)
    elif tensor_arr.ndim == 1:
        return result
    else:
        return result.squeeze(dim)

def stack_module_arr(module_arr: np.ndarray[nn.Module]) -> Tuple[nn.Module, TensorDict[str, torch.Tensor]]:
    params, buffers = torch.func.stack_module_state(module_arr.ravel().tolist())
    td = TensorDict({}, batch_size=module_arr.shape)

    def _unflatten(t: torch.Tensor, dim: int, shape: Tuple[int, ...]):
        if len(shape) == 0:
            return t.squeeze(dim=dim)
        elif len(shape) == 1:
            return t
        else:
            return t.unflatten(dim, shape)

    for k, v in params.items():
        td[k] = nn.Parameter(_unflatten(v, 0, module_arr.shape), requires_grad=v.requires_grad)
    for k, v in buffers.items():
        td[k] = _unflatten(v, 0, module_arr.shape)

    return module_arr.ravel()[0], td

def stack_module_arr_preserve_reference(module_arr: np.ndarray[nn.Module]) -> Tuple[nn.Module, TensorDict[str, torch.Tensor]]:
    flattened_td = torch.stack([
        TensorDict({
            k: v
            for k in dir(module) if isinstance((v := getattr(module, k)), torch.Tensor)
        }, batch_size=())
        for module in module_arr.ravel()
    ])
    td = flattened_td.reshape(module_arr.shape)
    return module_arr.ravel()[0], td

def run_module_arr(
        reference_module: nn.Module,
        module_td: TensorDict[str, torch.Tensor],
        args: Any,  # Note: a TensorDict is only checked for as the immediate argument and will not work inside a nested structure
        kwargs: Dict[str, Any] = MappingProxyType(dict())
) -> Dict[str, torch.Tensor]:
    if "TensorDict" in type(args).__name__:
        args = dict(args)

    if np.prod(module_td.shape) == 1:
        flat_args, args_spec = tree_flatten(args)
        flat_squeezed_args = [
            t.view(*t.shape[module_td.ndim:])
            for t in flat_args
        ]
        squeezed_args = tree_unflatten(flat_squeezed_args, args_spec)

        squeezed_out = nn.utils.stateless.functional_call(reference_module, dict(module_td.view()), squeezed_args, kwargs)
        flat_squeezed_out, args_spec = tree_flatten(squeezed_out)
        flat_out = [
            t.view(*module_td.shape, *t.shape[module_td.ndim:])
            for t in flat_squeezed_out
        ]
        return tree_unflatten(flat_out, args_spec)
    else:
        def vmap_run(module_d, ags):
            return nn.utils.stateless.functional_call(reference_module, module_d, ags, kwargs)
        for _ in range(module_td.ndim):
            vmap_run = torch.func.vmap(vmap_run, randomness="different")
        return vmap_run(dict(module_td), args)

def double_vmap(func: Callable) -> Callable:
    return torch.vmap(torch.vmap(func))

def sqrtm(t: torch.Tensor) -> torch.Tensor:
    L, V = torch.linalg.eig(t)
    return (V @ torch.diag_embed(L ** 0.5) @ torch.inverse(V)).real


"""
NumPy Array Comprehension Operations
"""

def multi_iter(arr: np.ndarray | DimArray) -> Iterable[Any]:
    for x in np.nditer(arr, flags=['refs_ok']):
        yield x[()]

def multi_enumerate(arr: np.ndarray | DimArray) -> Iterable[Tuple[Sequence[int], Any]]:
    it = np.nditer(arr, flags=['multi_index', 'refs_ok'])
    for x in it:
        yield it.multi_index, x[()]

def multi_map(func: Callable[[Any], Any], arr: np.ndarray | DimArray, dtype: type = None):
    if dtype is None:
        dtype = type(func(arr.ravel()[0]))
    result = np.empty_like(arr, dtype=dtype)
    for idx, x in multi_enumerate(arr):
        result[idx] = func(x)
    return DimArray(result, dims=arr.dims) if isinstance(arr, DimArray) else result

def multi_zip(*arrs: np.ndarray) -> np.ndarray:
    result = np.recarray(arrs[0].shape, dtype=[(f"f{i}", arr.dtype) for i, arr in enumerate(arrs)])
    for i, arr in enumerate(arrs):
        setattr(result, f"f{i}", arr)
    return result


"""
DimArray Operations
"""

def dim_array_like(arr: DimArray, dtype: type) -> DimArray:
    empty_arr = np.full_like(arr, None, dtype=dtype)
    return DimArray(empty_arr, dims=arr.dims)

def broadcast_dim_array_shapes(*dim_arrs: Iterable[DimArray]) -> OrderedDict[str, int]:
    dim_dict = OrderedDict()
    for dim_arr in dim_arrs:
        for dim_name, dim_len in zip(dim_arr.dims, dim_arr.shape):
            dim_dict.setdefault(dim_name, []).append(dim_len)
    return OrderedDict((k, np.broadcast_shapes(*v)[0]) for k, v in dim_dict.items())

def broadcast_dim_arrays(*dim_arrs: Iterable[np.ndarray]) -> Iterator[DimArray]:
    _dim_arrs = []
    for dim_arr in dim_arrs:
        if isinstance(dim_arr, DimArray):
            _dim_arrs.append(dim_arr)
        elif isinstance(dim_arr, np.ndarray):
            assert dim_arr.ndim == 0
            _dim_arrs.append(DimArray(dim_arr, dims=[]))
        else:
            _dim_arrs.append(DimArray(array_of(dim_arr), dims=[]))
    dim_arrs = _dim_arrs

    dim_dict = broadcast_dim_array_shapes(*dim_arrs)
    reference_dim_arr = DimArray(np.zeros((*dim_dict.values(),)), dims=(*dim_dict.keys(),))
    return (dim_arr.broadcast(reference_dim_arr) for dim_arr in dim_arrs)

def take_from_dim_array(dim_arr: DimArray | Dataset, idx: Dict[str, Any]):
    dims = set(dim_arr.dims)
    return dim_arr.take(indices={k: v for k, v in idx.items() if k in dims})

def put_in_dim_array(dim_arr: DimArray, idx: Dict[str, Any], value: Any):
    dims = set(dim_arr.dims)
    dim_arr.put(indices={k: v for k, v in idx.items() if k in dims}, values=value)


"""
Recursive attribute functions
"""
def rgetattr(obj: object, attr: str, *args):
    def _getattr(obj: object, attr: str) -> Any:
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rsetattr(obj: object, attr: str, value: Any) -> None:
    def _rsetattr(obj: object, attrs: List[str], value: Any) -> None:
        if len(attrs) == 1:
            setattr(obj, attrs[0], value)
        else:
            _rsetattr(next_obj := getattr(obj, attrs[0], Namespace()), attrs[1:], value)
            setattr(obj, attrs[0], next_obj)
    _rsetattr(obj, attr.split("."), value)

def rhasattr(obj: object, attr: str) -> bool:
    try:
        rgetattr(obj, attr)
        return True
    except AttributeError:
        return False

def rgetattr_default(o: object, format_str: str, try_str: str, default_str: str) -> Any:
    try:
        return rgetattr(o, format_str.format(try_str))
    except AttributeError:
        return rgetattr(o, format_str.format(default_str))


"""
Miscellaneous
"""
class PTR(object):
    def __init__(self, obj: object) -> None:
        self.obj = obj

    def __iter__(self):
        yield self.obj

class print_disabled:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def color(z: float, scale: float = 120.) -> np.ndarray:
    k = 2 * np.pi * z / scale
    return (1 + np.asarray([np.sin(k), np.sin(k + 2 * np.pi / 3), np.sin(k + 4 * np.pi / 3)], dtype=float)) / 2

def toJSON(n: Namespace):
    d = OrderedDict(vars(n))
    for k, v in d.items():
        if type(v) is Namespace:
            d[k] = toJSON(v)
        else:
            try:
                json.dumps(v)
                d[k] = v
            except TypeError:
                d[k] = str(v)
    return d

def str_namespace(n: Namespace) -> str:
    return json.dumps(toJSON(n), indent=4)

def print_namespace(n: Namespace) -> None:
    print(str_namespace(n))

def hash_namespace(n: Namespace) -> str:
    return hashlib.sha256(str_namespace(n).encode("utf-8")).hexdigest()[:8]

def batch_trace(x: torch.Tensor) -> torch.Tensor:
    return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

def array_of(o: object) -> np.ndarray:
    M = np.array(None, dtype=object)
    M[()] = o
    return M

def complex(t: torch.Tensor) -> torch.Tensor:
    return torch.complex(t, torch.zeros_like(t))

def nested_type(o: object) -> object:
    if type(o) in [list, tuple]:
        return type(o)(map(nested_type, o))
    elif type(o) == dict:
        return {k: nested_type(v) for k, v in o.items()}
    else:
        return type(o)

def class_name(cls: type) -> str:
    return str(cls)[8:-2].split('.')[-1]

def capitalize(s: str):
    return s[0].upper() + s[1:]

def npfy_namespace(n: Namespace) -> None:
    for k, v in vars(n).items():
        if isinstance(v, Namespace):
            npfy_namespace(v)
        elif isinstance(n, Iterable):
            setattr(n, k, array_of(v))
        else:
            setattr(n, k, np.array(v))

def broadcast_arrays_preserve_ndims(*arrs: np.ndarray) -> Iterator[np.ndarray]:
    shape = np.broadcast_shapes(*(arr.shape for arr in arrs))
    return (np.broadcast_to(arr, shape[-arr.ndim:]) for arr in arrs)








