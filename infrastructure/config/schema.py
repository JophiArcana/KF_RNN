"""Dataclass schemas describing the KF_RNN experiment hyperparameter tree.

These are plain (OmegaConf-compatible) dataclasses. The well-structured
branches (data / training / eval / runtime / problem shape) are fully typed and
validated. The two branches that legitimately carry live Python objects
(``system.distribution`` and ``model.model``) are kept open and may hold either
a Hydra ``_target_`` instantiation spec or a direct object, resolved by the
bridge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# SECTION: Problem shape (shared agent/environment contract)

@dataclass
class EnvironmentShape:
    observation: int = 1


@dataclass
class ProblemShape:
    """The shared input/output contract between environment and controller.

    Defined once at the config root (``cfg.problem``) and referenced by both
    ``system`` and ``model`` rather than being duplicated. ``controller`` maps
    each control-channel name to its action dimension; an empty mapping means
    the problem is pure estimation (LQE) with no control (LQR).
    """
    environment: EnvironmentShape = field(default_factory=EnvironmentShape)
    controller: Dict[str, int] = field(default_factory=dict)


# SECTION: Per-split values with train-fallback defaulting

@dataclass
class SplitConfig:
    """A value that may differ across the train/valid/test dataset splits.

    Missing splits fall back to the ``train`` value (replacing the implicit
    ``DefaultingParameter.__getattr__`` magic with an explicit method).
    """
    train: Any = None
    valid: Any = None
    test: Any = None

    def for_split(self, split: str) -> Any:
        value = getattr(self, split, None)
        return self.train if value is None else value

    def default(self) -> Any:
        return self.train


# SECTION: Dataset configuration

@dataclass
class DataConfig:
    n_systems: SplitConfig = field(default_factory=lambda: SplitConfig(train=1))
    n_traces: SplitConfig = field(default_factory=lambda: SplitConfig(train=1, valid=100, test=500))
    total_sequence_length: SplitConfig = field(
        default_factory=lambda: SplitConfig(train=2000, valid=20000, test=800000)
    )


# SECTION: Training configuration

@dataclass
class SamplingConfig:
    method: str = "subsequence_padded"   # {"full", "subsequence_padded", "subsequence_unpadded"}
    batch_size: int = 128
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
    warmup_duration: int = 100
    epochs: Optional[int] = 2500
    lr_decay: float = 0.995              # exponential
    T_0: int = 10                        # cosine
    T_mult: int = 2                      # cosine
    num_restarts: int = 8                # cosine
    gradient_cutoff: Optional[float] = None  # used when epochs is None


@dataclass
class TrainConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: str = "mse"
    control_coefficient: float = 1.0
    ignore_initial: bool = False


# SECTION: Evaluation / metrics configuration

@dataclass
class EvalConfig:
    metrics: Dict[str, List[str]] = field(default_factory=dict)
    ignore_metrics: Dict[str, List[str]] = field(default_factory=dict)


# SECTION: Runtime / orchestration configuration

@dataclass
class RuntimeConfig:
    exp_name: str = "Experiment"
    n_experiments: int = 1
    ensemble_size: int = 32
    backup_frequency: Optional[int] = None
    checkpoint_frequency: Optional[int] = None
    print_frequency: Optional[int] = None
    debug: bool = False                  # gates global torch anomaly detection
    split_size: int = 1 << 20            # run-chunking threshold (numel)


# SECTION: System and model branches (carry live objects / _target_ specs)

@dataclass
class SystemConfig:
    S_D: Any = None
    problem_shape: Any = None            # interpolated from cfg.problem by the bridge
    distribution: Any = None             # SystemDistribution object or _target_ spec
    auxiliary: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=lambda: {"include_analytical": True})


@dataclass
class ModelConfig:
    model: Any = None                    # Predictor subclass or _target_ spec
    problem_shape: Any = None            # interpolated from cfg.problem by the bridge
    S_D: Any = None
    params: Dict[str, Any] = field(default_factory=dict)  # extra model hyperparameters


@dataclass
class ExperimentConfig:
    problem: ProblemShape = field(default_factory=ProblemShape)
    system: SystemConfig = field(default_factory=SystemConfig)
    dataset: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    experiment: RuntimeConfig = field(default_factory=RuntimeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
