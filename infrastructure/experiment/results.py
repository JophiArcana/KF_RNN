"""Typed struct-of-arrays containers for experiment results and dataset info.

These replace the ``np.recarray`` (``RESULT_DTYPE``/``INFO_DTYPE``) grids: instead
of one labeled array whose cells are structured records (which forced the ``PTR``
box to hide ``Tensor``/``TensorDict`` ``.dtype`` from numpy's record machinery),
each field is its own ``LabeledArray`` over the shared sweep dims. Object fields
hold ``TensorDict``/``Tensor``/tuple/``SystemGroup`` directly.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from infrastructure import utils
from infrastructure.labeled_array import LabeledArray
from infrastructure.static import INFO_FIELDS, RESULT_FIELDS


@dataclass
class ResultGrid:
    """One ``LabeledArray`` per result field, all sharing the sweep dims."""

    time: LabeledArray
    output: LabeledArray
    learned_kfs: LabeledArray
    systems: LabeledArray
    metrics: LabeledArray

    @classmethod
    def empty(cls, dimensions: "OrderedDict[str, int]") -> "ResultGrid":
        dim_names = tuple(dimensions.keys())
        shape = tuple(dimensions.values())

        def obj_field() -> LabeledArray:
            return LabeledArray(np.full(shape, None, dtype=object), dim_names)

        return cls(
            time=LabeledArray(np.zeros(shape, dtype=float), dim_names),
            output=obj_field(),
            learned_kfs=obj_field(),
            systems=obj_field(),
            metrics=obj_field(),
        )

    @property
    def dims(self) -> tuple[str, ...]:
        return self.time.dims

    @property
    def shape(self) -> tuple[int, ...]:
        return self.time.shape

    @property
    def ndim(self) -> int:
        return self.time.ndim

    def field(self, name: str) -> np.ndarray:
        """The whole-grid ``np.ndarray`` of a field (powers ``get_result_attr``)."""
        return getattr(self, name).values

    def get(self, index: "OrderedDict[str, int]", name: str) -> Any:
        field = getattr(self, name)
        return field.values[tuple(index[d] for d in field.dims)]

    def set(self, index: "OrderedDict[str, int]", **fields: Any) -> None:
        for name, value in fields.items():
            getattr(self, name).put(index, value)


@dataclass
class InfoCell:
    """A single dataset-info record (scalar fields), produced by ``InfoGrid[idx]``."""

    systems: Any
    system_params: Any
    dataset: Any


@dataclass
class InfoGrid:
    """One ``LabeledArray`` per dataset-info field, all sharing dims."""

    systems: LabeledArray
    system_params: LabeledArray
    dataset: LabeledArray

    @classmethod
    def from_fields(cls, fields: "dict[str, LabeledArray]") -> "InfoGrid":
        return cls(**{name: fields[name] for name in INFO_FIELDS})

    def take(self, index: "OrderedDict[str, int]") -> "InfoGrid":
        return InfoGrid(**{
            name: utils.take_from_dim_array(getattr(self, name), index)
            for name in INFO_FIELDS
        })

    def __getitem__(self, idx) -> InfoCell:
        return InfoCell(**{name: getattr(self, name).values[idx] for name in INFO_FIELDS})
