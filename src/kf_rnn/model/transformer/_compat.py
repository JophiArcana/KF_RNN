"""Compatibility shims for transformers APIs newer than the pinned version.

The vendored modeling files (``adasync``, ``observable_mamba``, ``mamba``) were
adapted from a newer ``transformers`` release than the one pinned in
``requirements.txt`` (``~=4.45.2``). Two symbols they rely on do not exist
there yet:

- ``transformers.modeling_layers.GradientCheckpointingLayer`` (added later as a
  base class that transparently applies gradient checkpointing).
- ``transformers.utils.auto_docstring`` (a docstring-generation decorator).

This module re-exports the real implementations when available and otherwise
provides faithful, dependency-light fallbacks so the models import and run on
the pinned version. Both fallbacks are functionally transparent for these
experiments (gradient checkpointing is off by default; ``auto_docstring`` only
affects ``__doc__``).
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional

import torch.nn as nn


# SECTION: GradientCheckpointingLayer
try:  # transformers >= 4.49
    from transformers.modeling_layers import GradientCheckpointingLayer  # type: ignore
except ImportError:  # pragma: no cover - exercised on the pinned 4.45.x
    class GradientCheckpointingLayer(nn.Module):
        """Fallback base class mirroring the newer transformers implementation.

        Subclasses implement ``forward`` as usual; when ``gradient_checkpointing``
        is enabled (and the module is training), the forward pass is routed
        through ``self._gradient_checkpointing_func`` exactly as the upstream
        class does. With the default ``gradient_checkpointing = False`` this is
        a plain ``nn.Module``.
        """

        gradient_checkpointing: bool = False

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if self.gradient_checkpointing and self.training:
                func = getattr(self, "_gradient_checkpointing_func", None)
                if func is not None:
                    return func(partial(super().__call__, **kwargs), *args)
            return super().__call__(*args, **kwargs)


# SECTION: auto_docstring
try:  # transformers >= 4.51
    from transformers.utils import auto_docstring  # type: ignore
except ImportError:  # pragma: no cover - exercised on the pinned 4.45.x
    def auto_docstring(obj: Optional[Callable] = None, **kwargs: Any):
        """No-op fallback supporting both ``@auto_docstring`` and
        ``@auto_docstring(...)`` usage. Returns the decorated object unchanged."""
        if obj is None:
            return lambda decorated: decorated
        return obj


__all__ = ["GradientCheckpointingLayer", "auto_docstring"]
