"""Minimal in-house labeled-array layer.

Replaces the (unmaintained) ``dimarray`` dependency with the small slice of
functionality the experiment pipeline actually uses: a numpy array tagged with
dim names supporting positional ``take``/``put``/``broadcast`` by name, plus a
heterogeneous ``LabeledDataset`` collection whose members may have different
dims. Indexing is purely positional (no coordinate/label axes), which removes
the label-vs-position and 1-indexing footguns of ``dimarray``.

A ``LabeledArray`` is just ``(values: np.ndarray, dims: tuple[str, ...])`` and so
pickles cleanly via ``torch.save`` without any ``add_safe_globals`` registration.
"""

from collections import OrderedDict
from typing import Any, Iterable

import numpy as np


def array_of(o: Any) -> np.ndarray:
    """Wrap a single object in a 0-d object array (storing the reference).

    Mirrors ``utils.array_of``; duplicated here to avoid an import cycle.
    """
    M = np.array(None, dtype=object)
    M[()] = o
    return M


def put_object(arr: np.ndarray, index: tuple, obj: Any) -> None:
    """Assign ``obj`` into a single cell of ``arr`` at the full integer ``index``.

    Because the destination is a scalar slot, numpy stores the reference rather
    than broadcasting array-likes (``Tensor``/``TensorDict``) across the array.
    """
    arr[index] = obj


class LabeledArray:
    def __init__(self, values: Any, dims: Iterable[str]) -> None:
        if not isinstance(values, np.ndarray):
            values = array_of(values)
        self.values = values
        self.dims = tuple(dims)
        assert self.values.ndim == len(self.dims), (
            f"LabeledArray values have ndim {self.values.ndim} but {len(self.dims)} dims {self.dims}."
        )

    # SECTION: numpy-like attributes
    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def dtype(self):
        return self.values.dtype

    def ravel(self) -> np.ndarray:
        return self.values.ravel()

    def max(self, *args, **kwargs):
        return self.values.max(*args, **kwargs)

    def __getitem__(self, idx):
        return self.values[idx]

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raw_inputs = [x.values if isinstance(x, LabeledArray) else x for x in inputs]
        result = getattr(ufunc, method)(*raw_inputs, **kwargs)
        dims = next((x.dims for x in inputs if isinstance(x, LabeledArray)), None)
        if isinstance(result, np.ndarray) and dims is not None and result.ndim == len(dims):
            return LabeledArray(result, dims)
        return result

    # Operator dunders route through numpy ufuncs (and thus ``__array_ufunc__``),
    # which Python does not invoke for ``-a`` / ``a // b`` unless they are defined.
    def __neg__(self):
        return np.negative(self)

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __truediv__(self, other):
        return np.true_divide(self, other)

    def __rtruediv__(self, other):
        return np.true_divide(other, self)

    def __floordiv__(self, other):
        return np.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return np.floor_divide(other, self)

    def __mod__(self, other):
        return np.mod(self, other)

    def __eq__(self, other):
        return np.equal(self, other)

    def __ne__(self, other):
        return np.not_equal(self, other)

    def __gt__(self, other):
        return np.greater(self, other)

    def __ge__(self, other):
        return np.greater_equal(self, other)

    def __lt__(self, other):
        return np.less(self, other)

    def __le__(self, other):
        return np.less_equal(self, other)

    def __repr__(self) -> str:
        return f"LabeledArray(dims={self.dims}, shape={self.shape}, dtype={self.values.dtype})"

    # SECTION: labeled operations
    def take(self, indices: dict[str, int]) -> "LabeledArray":
        """Positional integer indexing by dim name, dropping indexed dims.

        Indices targeting dims not present on this array are ignored.
        """
        indexer = [slice(None)] * self.values.ndim
        dropped = []
        for dim, i in indices.items():
            if dim in self.dims:
                indexer[self.dims.index(dim)] = i
                dropped.append(dim)
        new_values = self.values[tuple(indexer)]
        new_dims = tuple(d for d in self.dims if d not in dropped)
        if not isinstance(new_values, np.ndarray):
            new_values = array_of(new_values)
        return LabeledArray(new_values, new_dims)

    def put(self, indices: dict[str, int], values: Any) -> None:
        """In-place single-cell write at the full positional ``indices``."""
        put_object(self.values, tuple(indices[d] for d in self.dims), values)

    def broadcast(self, dim_sizes: "OrderedDict[str, int]") -> "LabeledArray":
        """Align/expand to the named dim layout in ``dim_sizes`` (union order)."""
        target_dims = tuple(dim_sizes.keys())
        target_shape = tuple(dim_sizes.values())

        present = [d for d in target_dims if d in self.dims]
        src = np.transpose(self.values, [self.dims.index(d) for d in present])

        reshape_shape, pi = [], 0
        for d in target_dims:
            if d in self.dims:
                reshape_shape.append(src.shape[pi])
                pi += 1
            else:
                reshape_shape.append(1)
        src = src.reshape(reshape_shape)
        return LabeledArray(np.broadcast_to(src, target_shape).copy(), target_dims)


class LabeledDataset:
    """A heterogeneous collection of ``LabeledArray``s keyed by name.

    Members may have different dims; ``take`` slices each member by the subset
    of ``indices`` relevant to its own dims.
    """

    def __init__(self, data: "dict[str, LabeledArray]") -> None:
        self._data = OrderedDict(data)

    def take(self, indices: dict[str, int]) -> "LabeledDataset":
        return LabeledDataset({k: v.take(indices) for k, v in self._data.items()})

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __getitem__(self, key: str) -> LabeledArray:
        return self._data[key]

    def __setitem__(self, key: str, value: LabeledArray) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data
