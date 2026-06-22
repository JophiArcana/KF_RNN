"""Typed dataclass schemas for the KF_RNN experiment hyperparameter tree.

This is the single configuration representation, used for authoring (scripts),
sweeping (the experiment engine copies and overrides these trees), and
serialization (``config_to_jsonable``). Design principles:

- The tree carries *declarative specs* only: primitives, ``Split`` wrappers,
  per-model ``Predictor.Config`` dataclasses, and constructor-like objects
  (HuggingFace ``*Config``s, ``SystemDistribution`` instances). Materialized
  runtime state (sampled systems, datasets, modules) never lives here.
- Per-split (train/valid/test) values are expressed explicitly with ``Split``,
  the only defaulting mechanism: missing splits fall back to ``train``.
- The model branch is polymorphic: each ``Predictor`` subclass owns a nested
  ``Config`` dataclass (see ``kf_rnn.model.base.Observer``), so this module
  never enumerates per-model hyperparameters.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, Generic, Iterable, Optional, Set, TypeVar


_T = TypeVar("_T")

SPLIT_NAMES: tuple[str, ...] = ("train", "valid", "test")


def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


# SECTION: Per-split values with train-fallback defaulting

@dataclass
class Split(Generic[_T]):
    """A value that may differ across the train/valid/test dataset splits.

    Splits left as ``None`` fall back to the ``train`` value via ``for_split``.
    ``update`` / ``reset`` mirror the authoring ergonomics scripts rely on.
    """
    train: Optional[_T] = None
    valid: Optional[_T] = None
    test: Optional[_T] = None

    def for_split(self, split: str) -> _T:
        value = getattr(self, split)
        return self.train if value is None else value

    def default(self) -> _T:
        return self.train

    def update(self, **kwargs: _T) -> None:
        for k, v in kwargs.items():
            assert k in SPLIT_NAMES, f"Unknown split {k!r}; expected one of {SPLIT_NAMES}."
            setattr(self, k, v)

    def reset(self, **kwargs: _T) -> None:
        for k in SPLIT_NAMES:
            setattr(self, k, None)
        self.update(**kwargs)


# SECTION: Problem shape (shared agent/environment contract)

@dataclass
class EnvironmentShape:
    observation: int = 1


@dataclass
class ProblemShape:
    """The shared input/output contract between environment and controller.

    Defined once at the config root (``cfg.problem``) and propagated by
    ``propagate_problem_shape`` into ``system`` and ``model``. ``controller``
    maps each control-channel name to its action dimension; an empty mapping
    means the problem is pure estimation (LQE) with no control (LQR).
    """
    environment: EnvironmentShape = field(default_factory=EnvironmentShape)
    controller: Dict[str, int] = field(default_factory=dict)


def controller_dims(problem_shape: Any) -> Dict[str, int]:
    """Control-channel name -> action dimension.

    Accepts the typed ``ProblemShape`` (``controller`` is a dict) and legacy
    pickled namespaces (``controller`` is an attribute bag), so previously
    saved systems remain loadable.
    """
    controller = problem_shape.controller
    return controller if isinstance(controller, dict) else dict(vars(controller))


def shape_leaves(problem_shape: Any) -> list[str]:
    """Dotted leaf names of the problem shape (loss/metric channel keys)."""
    return [
        "environment.observation",
        *(f"controller.{k}" for k in controller_dims(problem_shape)),
    ]


# SECTION: System configuration

@dataclass
class SystemAuxiliary:
    initial_state_scale: float = 1.0
    control_noise_std: float = 0.0


@dataclass
class SystemSettings:
    include_analytical: bool = True


@dataclass
class SystemConfig:
    S_D: Optional[int] = None
    problem_shape: Optional[ProblemShape] = None     # filled from ExperimentConfig.problem
    distribution: Split = field(default_factory=Split)  # Split[SystemDistribution]
    auxiliary: SystemAuxiliary = field(default_factory=SystemAuxiliary)
    settings: SystemSettings = field(default_factory=SystemSettings)

    def __post_init__(self) -> None:
        # Authoring convenience: a raw distribution becomes the train split.
        if not isinstance(self.distribution, Split):
            self.distribution = Split(train=self.distribution)


# SECTION: Dataset configuration

@dataclass
class DataConfig:
    n_systems: Split = field(default_factory=lambda: Split(train=1))
    n_traces: Split = field(default_factory=lambda: Split(train=1, valid=100, test=500))
    total_sequence_length: Split = field(
        default_factory=lambda: Split(train=2000, valid=20000, test=800000)
    )

    @property
    def sequence_length(self) -> Split:
        """Derived per-trace sequence length (never stored, always consistent
        with the possibly-swept ``total_sequence_length`` / ``n_traces``)."""
        return Split(train=_ceildiv(self.total_sequence_length.train, self.n_traces.train))


# SECTION: Training configuration

@dataclass
class SamplingConfig:
    method: Optional[str] = "subsequence_padded"   # {"full", "subsequence_padded", "subsequence_unpadded"}
    batch_size: Optional[int] = 128
    subsequence_length: Optional[int] = 16


@dataclass
class OptimizerConfig:
    type: str = "AdamW"                  # {"SGD", "AdamW"}
    max_lr: float = 2e-2
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    momentum: float = 0.9                # used by SGD / Adam


@dataclass
class SchedulerConfig:
    type: str = "exponential"            # {"exponential", "cosine", "reduce_on_plateau"}
    warmup_duration: int = 0
    epochs: Optional[int] = 2500
    lr_decay: float = 0.995              # exponential
    T_0: int = 10                        # cosine
    T_mult: int = 2                      # cosine
    num_restarts: int = 8                # cosine
    gradient_cutoff: Optional[float] = None  # used when epochs is None
    factor: Optional[float] = None       # reduce_on_plateau
    patience: Optional[int] = None       # reduce_on_plateau


@dataclass
class TrainConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: str = "mse"
    control_coefficient: float = 1.0
    ignore_initial: bool = False


# SECTION: Runtime / orchestration configuration

@dataclass
class MetricsConfig:
    """Metric-name sets per phase; ``None`` means use the engine's default set."""
    training: Optional[Set[str]] = None
    testing: Optional[Set[str]] = None


@dataclass
class RuntimeConfig:
    exp_name: Optional[str] = None
    n_experiments: int = 1
    ensemble_size: int = 32
    backup_frequency: Optional[int] = None
    checkpoint_frequency: Optional[int] = None
    print_frequency: Optional[int] = None
    debug: bool = False                  # gates global torch anomaly detection
    split_size: int = 1 << 20            # run-chunking threshold (numel)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    ignore_metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @property
    def model_shape(self) -> tuple[int, int]:
        return (self.n_experiments, self.ensemble_size)


# SECTION: The full experiment tree

@dataclass
class ExperimentConfig:
    problem: ProblemShape = field(default_factory=ProblemShape)
    system: SystemConfig = field(default_factory=SystemConfig)
    dataset: DataConfig = field(default_factory=DataConfig)
    model: Any = None                    # an ``Observer.Config`` subclass instance
    training: TrainConfig = field(default_factory=TrainConfig)
    experiment: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        propagate_problem_shape(self)


def propagate_problem_shape(cfg: ExperimentConfig) -> None:
    """Fill ``system.problem_shape`` / ``model.problem_shape`` from ``cfg.problem``.

    Called at construction and again per sweep cell (a swept-in model config
    typically omits its problem shape).
    """
    if cfg.system is not None and getattr(cfg.system, "problem_shape", None) is None:
        cfg.system.problem_shape = cfg.problem
    if is_dataclass(cfg.model) and not isinstance(cfg.model, type):
        if getattr(cfg.model, "problem_shape", None) is None:
            cfg.model.problem_shape = cfg.problem


# SECTION: Config-tree utilities (structural copy, flattening, split resolution)

def copy_config(obj: _T) -> _T:
    """Deep-copy the config *structure* (dataclasses and containers) while
    sharing leaf objects (distributions, HF configs, classes, tensors) by
    reference. Replaces ``deepcopy_namespace`` for the typed tree."""
    if is_dataclass(obj) and not isinstance(obj, type):
        clone = object.__new__(type(obj))
        clone.__dict__.update({k: copy_config(v) for k, v in obj.__dict__.items()})
        return clone
    if isinstance(obj, dict):
        return {k: copy_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(copy_config(v) for v in obj)
    return obj


def resolve_splits(obj: _T, split: str = SPLIT_NAMES[0]) -> _T:
    """Copy of a config subtree with every ``Split`` collapsed to its value for
    ``split`` (train-fallback). Replaces ``index_defaulting_with_attr``."""
    if isinstance(obj, Split):
        return obj.for_split(split)
    if is_dataclass(obj) and not isinstance(obj, type):
        clone = object.__new__(type(obj))
        clone.__dict__.update({k: resolve_splits(v, split) for k, v in obj.__dict__.items()})
        return clone
    if isinstance(obj, dict):
        return {k: resolve_splits(v, split) for k, v in obj.items()}
    return obj


def config_leaves(obj: Any) -> Dict[str, Any]:
    """Flatten a config subtree into ``{dotted leaf name: value}``.

    ``Split`` fields contribute only their explicitly-set (non-``None``)
    splits, so leaf names like ``"distribution.test"`` appear exactly when a
    split-specific value was authored or swept in — the property the sweep
    classification in ``experiment.sweep`` relies on. Replaces ``nested_vars``.
    """
    result: Dict[str, Any] = {}

    def visit(prefix: str, o: Any) -> None:
        def child(name: str) -> str:
            return f"{prefix}.{name}" if prefix else name

        if isinstance(o, Split):
            for s in SPLIT_NAMES:
                v = getattr(o, s)
                if v is not None:
                    result[child(s)] = v
        elif is_dataclass(o) and not isinstance(o, type):
            for k, v in o.__dict__.items():
                visit(child(k), v)
        elif isinstance(o, dict):
            for k, v in o.items():
                visit(child(str(k)), v)
        else:
            result[prefix] = o

    visit("", obj)
    return result


# SECTION: Serialization

def config_to_jsonable(obj: Any) -> Any:
    """Convert a config tree to JSON-serializable data for ``hparams.json``.

    Dataclasses and containers become dicts/lists; classes become their
    qualified names; other live objects fall back to ``str``.
    """
    if isinstance(obj, Split):
        return {s: config_to_jsonable(getattr(obj, s)) for s in SPLIT_NAMES if getattr(obj, s) is not None}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: config_to_jsonable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {str(k): config_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [config_to_jsonable(v) for v in obj]
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


# SECTION: Sweep-target validation

def _flatten_params(params: Dict[str, Any]) -> Iterable[str]:
    for k, v in params.items():
        if isinstance(v, dict):
            for sub in _flatten_params(v):
                yield f"{k}.{sub}"
        else:
            yield k


def _is_container(o: Any) -> bool:
    return (is_dataclass(o) and not isinstance(o, type)) or isinstance(o, (Split, dict))


def validate_sweep_targets(
        cfg: ExperimentConfig,
        iterparams: Iterable[tuple[str, Dict[str, Any]]],
) -> None:
    """Validate dotted sweep-parameter targets against the config structure.

    Raises ``ValueError`` for an unknown top-level branch or for a path whose
    already-existing prefix bottoms out at a non-container leaf (a likely
    typo). Paths through unset branches (e.g. ``model.*`` when the model config
    itself is being swept) are allowed.
    """
    known_branches = tuple(f.name for f in dataclasses.fields(ExperimentConfig))
    for _param_group, params in iterparams:
        for dotted in _flatten_params(params):
            if dotted == "name":
                continue
            parts = dotted.split(".")
            if parts[0] not in known_branches:
                raise ValueError(
                    f"Sweep target {dotted!r} does not begin with a known config "
                    f"branch {known_branches}."
                )
            node: Any = cfg
            for i, seg in enumerate(parts):
                if node is None:
                    break  # below an unset/swept branch is allowed
                if not _is_container(node):
                    bad = ".".join(parts[:i])
                    raise ValueError(
                        f"Sweep target {dotted!r} is invalid: {bad!r} is a leaf value, "
                        f"not a config sub-tree."
                    )
                node = node.get(seg) if isinstance(node, dict) else getattr(node, seg, None)
