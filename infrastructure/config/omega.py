"""OmegaConf-backed helpers that replace bespoke namespace serialization and
provide schema-aware validation of sweep targets."""
from __future__ import annotations

from argparse import Namespace
from typing import Any, Iterable

from omegaconf import OmegaConf


KNOWN_BRANCHES = ("problem", "system", "dataset", "model", "training", "experiment", "eval")


def to_yaml(obj: Any) -> str:
    """Serialize a config object to YAML via OmegaConf, tolerating live objects."""
    try:
        container = OmegaConf.create(obj) if not OmegaConf.is_config(obj) else obj
        return OmegaConf.to_yaml(container)
    except Exception:
        from infrastructure import utils
        return utils.str_namespace(obj) if isinstance(obj, Namespace) else str(obj)


def _is_container(o: Any) -> bool:
    return isinstance(o, Namespace) or OmegaConf.is_config(o)


def validate_sweep_targets(
        HP: Namespace,
        iterparams: Iterable[tuple[str, dict[str, Any]]],
) -> None:
    """Validate dotted sweep-parameter targets against the config structure.

    Raises ``ValueError`` for an unknown top-level branch or for a path whose
    already-existing prefix bottoms out at a non-container leaf (a likely typo).
    Legitimately new leaves (e.g. adding a ``.test`` split) are allowed.
    """
    from infrastructure import utils

    for _param_group, params in iterparams:
        for dotted in utils.flatten_nested_dict(params).keys():
            if dotted == "name":
                continue
            parts = dotted.split(".")
            if parts[0] not in KNOWN_BRANCHES:
                raise ValueError(
                    f"Sweep target {dotted!r} does not begin with a known config "
                    f"branch {KNOWN_BRANCHES}."
                )
            node: Any = HP
            for i, seg in enumerate(parts):
                if not _is_container(node):
                    bad = ".".join(parts[:i])
                    raise ValueError(
                        f"Sweep target {dotted!r} is invalid: {bad!r} is a leaf value, "
                        f"not a config sub-tree."
                    )
                node = getattr(node, seg, None)
                if node is None:
                    break  # new leaf below a valid container is allowed
