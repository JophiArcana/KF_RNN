"""Bridge between the typed config tree and the runtime ``argparse.Namespace``.

The experiment pipeline (core/internals/training/metrics) still consumes a
nested ``Namespace`` HP. This module converts a typed ``ExperimentConfig`` into
that Namespace, resolving the shared problem shape once and resolving any
``_target_`` specs into live objects. The reverse direction extracts the
primitive/structured portion for serialization.
"""
from __future__ import annotations

import dataclasses
from argparse import Namespace
from typing import Any

from omegaconf import OmegaConf

from kf_rnn.infrastructure.config import schema
from kf_rnn.infrastructure.config.store import instantiate_target


def _to_namespace(obj: Any) -> Any:
    """Recursively convert a dataclass / OmegaConf config / dict into a nested
    ``Namespace``. Leaf values (and live Python objects) are returned unchanged.

    This replaces the per-branch field-by-field ``Namespace(...)`` construction
    for the regular (object-free) config branches.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return Namespace(**{
            f.name: _to_namespace(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        })
    if OmegaConf.is_config(obj):
        obj = OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return Namespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def split_for(split_cfg: "schema.SplitConfig | Namespace", split: str) -> Any:
    """Resolve a per-split value with train-fallback, accepting either a typed
    ``SplitConfig`` or a Namespace/DefaultingParameter."""
    if isinstance(split_cfg, schema.SplitConfig):
        return split_cfg.for_split(split)
    value = getattr(split_cfg, split, None)
    if value is None:
        value = getattr(split_cfg, "train", None)
    return value


def _problem_shape_namespace(problem: schema.ProblemShape) -> Namespace:
    return Namespace(
        environment=Namespace(observation=problem.environment.observation),
        controller=Namespace(**dict(problem.controller)),
    )


def _split_namespace(split_cfg: schema.SplitConfig) -> Namespace:
    kwargs = {
        name: getattr(split_cfg, name)
        for name in ("train", "valid", "test")
        if getattr(split_cfg, name) is not None
    }
    return Namespace(**kwargs)


def config_to_namespace(cfg: schema.ExperimentConfig) -> Namespace:
    """Convert a typed ``ExperimentConfig`` into the runtime ``Namespace`` HP.

    The regular (object-free) branches (``training``, ``experiment``) are converted
    generically by ``_to_namespace``. The branches that carry live objects or need
    bespoke shaping are handled explicitly:

    - the problem shape is authored once on ``cfg.problem`` and materialized into
      independent ``system`` and ``model`` Namespaces (the runtime applies split
      defaulting to ``system`` only, so the two must not alias the same object);
    - ``SplitConfig`` dataset fields are flattened with train-fallback;
    - ``_target_`` specs (``system.distribution`` / ``model.model``) are resolved;
    - ``model.params`` is flattened onto the ``model`` Namespace;
    - ``eval`` metrics are folded into the ``experiment`` Namespace as sets.
    """
    system = Namespace(
        S_D=cfg.system.S_D,
        problem_shape=_problem_shape_namespace(cfg.problem),
        distribution=instantiate_target(cfg.system.distribution),
        auxiliary=_to_namespace(dict(cfg.system.auxiliary)),
        settings=_to_namespace(dict(cfg.system.settings)),
    )

    dataset = Namespace(
        n_systems=_split_namespace(cfg.dataset.n_systems),
        n_traces=_split_namespace(cfg.dataset.n_traces),
        total_sequence_length=_split_namespace(cfg.dataset.total_sequence_length),
    )

    model = Namespace(
        model=instantiate_target(cfg.model.model),
        problem_shape=_problem_shape_namespace(cfg.problem),
    )
    if cfg.model.S_D is not None:
        model.S_D = cfg.model.S_D
    for k, v in dict(cfg.model.params).items():
        setattr(model, k, v)

    training = _to_namespace(cfg.training)

    experiment = _to_namespace(cfg.experiment)
    if cfg.eval.metrics:
        experiment.metrics = Namespace(**{k: set(v) for k, v in cfg.eval.metrics.items()})
    if cfg.eval.ignore_metrics:
        experiment.ignore_metrics = Namespace(**{k: set(v) for k, v in cfg.eval.ignore_metrics.items()})

    # Let RuntimeConfig.debug drive debug-only global side effects.
    from kf_rnn.infrastructure import settings
    settings.set_debug(cfg.experiment.debug)

    return Namespace(
        system=system,
        dataset=dataset,
        model=model,
        training=training,
        experiment=experiment,
    )
