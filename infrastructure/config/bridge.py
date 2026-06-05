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

from infrastructure.config import schema
from infrastructure.config.store import instantiate_target


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


def _dict_to_namespace(d: dict) -> Namespace:
    return Namespace(**{
        k: _dict_to_namespace(v) if isinstance(v, dict) else v
        for k, v in d.items()
    })


def config_to_namespace(cfg: schema.ExperimentConfig) -> Namespace:
    """Convert a typed ``ExperimentConfig`` into the runtime ``Namespace`` HP.

    The problem shape is authored once on ``cfg.problem`` and materialized into
    independent ``system`` and ``model`` Namespaces (the runtime applies split
    defaulting to ``system`` only, so the two must not alias the same object).
    """
    system = Namespace(
        S_D=cfg.system.S_D,
        problem_shape=_problem_shape_namespace(cfg.problem),
        distribution=instantiate_target(cfg.system.distribution),
        auxiliary=_dict_to_namespace(dict(cfg.system.auxiliary)),
        settings=_dict_to_namespace(dict(cfg.system.settings)),
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

    training = Namespace(
        sampling=Namespace(**dataclasses.asdict(cfg.training.sampling)),
        optimizer=Namespace(**dataclasses.asdict(cfg.training.optimizer)),
        scheduler=Namespace(**dataclasses.asdict(cfg.training.scheduler)),
        loss=cfg.training.loss,
        control_coefficient=cfg.training.control_coefficient,
        ignore_initial=cfg.training.ignore_initial,
    )

    experiment = Namespace(
        exp_name=cfg.experiment.exp_name,
        n_experiments=cfg.experiment.n_experiments,
        ensemble_size=cfg.experiment.ensemble_size,
        backup_frequency=cfg.experiment.backup_frequency,
        checkpoint_frequency=cfg.experiment.checkpoint_frequency,
        print_frequency=cfg.experiment.print_frequency,
        debug=cfg.experiment.debug,
        split_size=cfg.experiment.split_size,
    )
    if cfg.eval.metrics:
        experiment.metrics = Namespace(**{k: set(v) for k, v in cfg.eval.metrics.items()})
    if cfg.eval.ignore_metrics:
        experiment.ignore_metrics = Namespace(**{k: set(v) for k, v in cfg.eval.ignore_metrics.items()})

    # Let RuntimeConfig.debug drive debug-only global side effects.
    from infrastructure import settings
    settings.set_debug(cfg.experiment.debug)

    return Namespace(
        system=system,
        dataset=dataset,
        model=model,
        training=training,
        experiment=experiment,
    )


def namespace_to_config(ns: Namespace) -> schema.ExperimentConfig:
    """Best-effort extraction of the structured/primitive portion of a runtime
    Namespace HP into a typed ``ExperimentConfig`` (object-valued fields such as
    the model class / distribution are left as-is for ``_target_``-free use)."""
    def get(o: Any, name: str, default: Any = None) -> Any:
        return getattr(o, name, default)

    sys_ns = get(ns, "system", Namespace())
    ds_ns = get(ns, "dataset", Namespace())
    model_ns = get(ns, "model", Namespace())
    train_ns = get(ns, "training", Namespace())
    exp_ns = get(ns, "experiment", Namespace())

    ps_ns = get(sys_ns, "problem_shape", get(model_ns, "problem_shape", Namespace()))
    env = get(ps_ns, "environment", Namespace(observation=1))
    controller = get(ps_ns, "controller", Namespace())

    def split(o: Any) -> schema.SplitConfig:
        return schema.SplitConfig(
            train=get(o, "train"), valid=get(o, "valid"), test=get(o, "test"),
        )

    cfg = schema.ExperimentConfig()
    cfg.problem = schema.ProblemShape(
        environment=schema.EnvironmentShape(observation=get(env, "observation", 1)),
        controller={k: v for k, v in vars(controller).items()},
    )
    cfg.system = schema.SystemConfig(
        S_D=get(sys_ns, "S_D"),
        distribution=get(sys_ns, "distribution"),
        auxiliary=dict(vars(get(sys_ns, "auxiliary", Namespace()))),
        settings=dict(vars(get(sys_ns, "settings", Namespace()))),
    )
    cfg.dataset = schema.DataConfig(
        n_systems=split(get(ds_ns, "n_systems", Namespace(train=1))),
        n_traces=split(get(ds_ns, "n_traces", Namespace(train=1))),
        total_sequence_length=split(get(ds_ns, "total_sequence_length", Namespace(train=2000))),
    )
    cfg.model = schema.ModelConfig(model=get(model_ns, "model"), S_D=get(model_ns, "S_D"))
    return cfg
