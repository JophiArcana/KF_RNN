"""Training-stage strategy objects and registry.

This replaces the implicit ``(train_func, terminate_func)`` callable-pair plus
opaque ``cache: Namespace`` contract with explicit ``TrainingStage`` objects and
a single typed ``TrainingContext``. A stage owns its own mutable state and
exposes ``step`` / ``is_done``; the engine drives a sequence of stages.

Models declare a *data* recipe (``training_recipe`` returning a list of stage
names / stage instances) instead of returning callables. Names resolve through
``STAGE_REGISTRY``, which also records a compatibility predicate per stage.

Stage / model compatibility
---------------------------
Most stages are only meaningful for models that expose a particular routine or
parameterization. ``build_stages`` validates each named stage against the model
and raises ``TypeError`` otherwise. The compatibility map:

- ``sgd``                  universal (any ``Predictor`` with trainable params).
- ``analytical_init``      requires ``analytical_initialization``
                           (``CnnAnalyticalPredictor``, ``RnnKalmanPredictor``).
- ``least_squares_init``   requires ``vmap_train_least_squares``
                           (``LeastSquaresPredictor`` subclasses).
- ``online_least_squares`` requires ``train_least_squares_online``
                           (``CnnLeastSquaresPredictor`` family).
- ``newton_init``          requires ``newton_initialization``
                           (``CnnAnalyticalLeastSquaresPredictor``).
- ``random_step``          requires ``vmap_train_least_squares``
                           (``CnnLeastSquaresRandomStepPredictor``).
- ``negation``             requires ``vmap_train_least_squares``
                           (``CnnLeastSquaresNegationPredictor``).
- ``ho_kalman``            requires ``convert_fir``
                           (``RnnHoKalmanBasePredictor`` subclasses).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch

import ecliseutils as eu
from kf_rnn.infrastructure.config.schema import TrainConfig
from ecliseutils.ensemble import EnsembleModule
from kf_rnn.infrastructure.static import ModelPair
from kf_rnn.model.base import Predictor


StepResult = tuple[torch.Tensor, dict[str, torch.Tensor]]


class TrainingContext:
    """Typed bundle passed to every training stage, replacing the positional
    ``(THP, exclusive, model_pair, cache)`` tuple of Namespaces. Per-stage
    mutable state lives on the stage object, not here."""

    def __init__(self, thp: TrainConfig, exclusive: SimpleNamespace, model_pair: ModelPair):
        self.thp = thp
        self.exclusive = exclusive
        self.model_pair = model_pair

    def with_model_pair(self, model_pair: ModelPair) -> "TrainingContext":
        """Return a context targeting a different model pair (e.g. a submodule)."""
        return TrainingContext(self.thp, self.exclusive, model_pair)

    @property
    def model(self) -> EnsembleModule:
        return EnsembleModule.from_pair(self.model_pair)

    @property
    def n_train_systems(self) -> int:
        return getattr(self.exclusive, "n_train_systems", 1)

    @property
    def train_info(self) -> SimpleNamespace:
        return self.exclusive.train_info


class TrainingStage(ABC):
    """One stage of a training recipe (e.g. analytical init, then SGD).

    Mutable state lives in ``self._state`` (a SimpleNamespace owning at least
    ``t``), which the engine serializes for checkpointing.
    """

    name: str = "stage"

    def __init__(self):
        self._state = SimpleNamespace(t=0)

    def reset(self) -> None:
        """Clear per-stage mutable state (called when starting the stage fresh)."""
        self._state = SimpleNamespace(t=0)

    @abstractmethod
    def step(self, ctx: TrainingContext) -> StepResult:
        ...

    @abstractmethod
    def is_done(self, ctx: TrainingContext) -> bool:
        ...

    # Per-stage iteration counter (epoch within the stage).
    @property
    def t(self) -> int:
        return self._state.t

    @t.setter
    def t(self, value: int) -> None:
        self._state.t = value

    # Checkpointable mutable state.
    @property
    def state(self) -> SimpleNamespace:
        return self._state

    def load_state(self, state: SimpleNamespace) -> None:
        self._state = state


class InitializationStage(TrainingStage):
    """One-shot stage that writes a closed-form initialization into the model.

    Subclasses implement ``initialization(ctx) -> (init_dict, error)``. The first
    ``step`` applies the initialization and reports the zero-baseline error; the
    second ``step`` reports the stored initialization error and marks the stage
    done. This mirrors the original two-call ``_train_with_initialization_and_error``
    contract (which coordinated through ``cache.initialization_error`` / ``cache.done``).
    """

    name = "initialization"

    @abstractmethod
    def initialization(self, ctx: TrainingContext) -> tuple[dict[str, Any], torch.Tensor]:
        ...

    def step(self, ctx: TrainingContext) -> StepResult:
        stacked_modules = ctx.model_pair[1]
        if not hasattr(self._state, "initialization_error"):
            initialization, error_ = self.initialization(ctx)
            for k, v in eu.flatten_nested_dict(initialization).items():
                tdk = (*k.split("."),)
                stacked_modules[tdk] = v.expand_as(stacked_modules[tdk])
            self._state.initialization_error = error_.expand(stacked_modules.shape)
            error = Predictor.evaluate_run(
                0, ctx.train_info.dataset, ("environment", "observation")
            ).mean(dim=-1)
        else:
            self._state.done = True
            error = self._state.initialization_error
        return error[None], {}

    def is_done(self, ctx: TrainingContext) -> bool:
        return getattr(self._state, "done", False)


class SubmoduleStage(TrainingStage):
    """Run an inner stage against a named submodule of the model.

    Retargets ``ctx.model_pair`` to ``(reference.get_submodule(attr), stacked[attr])``
    before delegating, so e.g. a Ho-Kalman model can run its FIR submodule's
    stages. Replaces the ``augment_fn`` closure that previously rewrapped the FIR
    class's ``(train, terminate)`` pairs.
    """

    def __init__(self, inner: TrainingStage, attr: str):
        super().__init__()
        self.inner = inner
        self.attr = attr
        self.name = f"{attr}.{inner.name}"

    def reset(self) -> None:
        self.inner.reset()

    def _sub_ctx(self, ctx: TrainingContext) -> TrainingContext:
        reference, stacked = ctx.model_pair
        return ctx.with_model_pair((reference.get_submodule(self.attr), stacked[self.attr]))

    def step(self, ctx: TrainingContext) -> StepResult:
        return self.inner.step(self._sub_ctx(ctx))

    def is_done(self, ctx: TrainingContext) -> bool:
        return self.inner.is_done(self._sub_ctx(ctx))

    # Delegate iteration counter and checkpoint state to the inner stage.
    @property
    def t(self) -> int:
        return self.inner.t

    @t.setter
    def t(self, value: int) -> None:
        self.inner.t = value

    @property
    def state(self) -> SimpleNamespace:
        return self.inner.state

    def load_state(self, state: SimpleNamespace) -> None:
        self.inner.load_state(state)


# SECTION: Concrete model-specific stages (delegate numeric work to model methods)

class AnalyticalInitStage(InitializationStage):
    name = "analytical_init"

    def initialization(self, ctx: TrainingContext) -> tuple[dict[str, Any], torch.Tensor]:
        return ctx.model_pair[0].analytical_initialization(ctx.exclusive)


class LeastSquaresInitStage(InitializationStage):
    name = "least_squares_init"

    def initialization(self, ctx: TrainingContext) -> tuple[dict[str, Any], torch.Tensor]:
        return ctx.model_pair[0].vmap_train_least_squares(ctx.exclusive)


class NewtonStage(InitializationStage):
    name = "newton_init"

    def initialization(self, ctx: TrainingContext) -> tuple[dict[str, Any], torch.Tensor]:
        reference_module, stacked_modules = ctx.model_pair
        return reference_module.newton_initialization(stacked_modules, ctx.exclusive)


class RandomStepStage(InitializationStage):
    name = "random_step"

    def initialization(self, ctx: TrainingContext) -> tuple[dict[str, Any], torch.Tensor]:
        reference_module, stacked_modules = ctx.model_pair
        least_squares = reference_module.vmap_train_least_squares(ctx.exclusive)[0]
        return {
            k: v + torch.normal(0., torch.abs(stacked_modules[k].data - v))
            for k, v in least_squares.items()
        }, torch.full((), torch.nan)


class NegationStage(InitializationStage):
    name = "negation"

    def initialization(self, ctx: TrainingContext) -> tuple[dict[str, Any], torch.Tensor]:
        reference_module, stacked_modules = ctx.model_pair
        least_squares = reference_module.vmap_train_least_squares(ctx.exclusive)[0]
        return {
            k: 2 * v - stacked_modules[k].data
            for k, v in least_squares.items()
        }, torch.full((), torch.nan)


class HoKalmanStage(InitializationStage):
    name = "ho_kalman"

    def initialization(self, ctx: TrainingContext) -> tuple[dict[str, Any], torch.Tensor]:
        reference_module, stacked_modules = ctx.model_pair
        return reference_module.convert_fir(stacked_modules, ctx.exclusive)


class OnlineLeastSquaresStage(TrainingStage):
    """Iterative online least-squares fit (CNN FIR models).

    Delegates to the model's ``train_least_squares_online`` /
    ``terminate_least_squares_online``, which coordinate through ``self._state``.
    """

    name = "online_least_squares"

    def step(self, ctx: TrainingContext) -> StepResult:
        reference_module = ctx.model_pair[0]
        return reference_module.train_least_squares_online(
            ctx.thp, ctx.exclusive, ctx.model_pair, self._state
        )

    def is_done(self, ctx: TrainingContext) -> bool:
        reference_module = ctx.model_pair[0]
        return reference_module.terminate_least_squares_online(
            ctx.thp, ctx.exclusive, ctx.model_pair, self._state
        )


# SECTION: Stage registry (name -> factory + model-compatibility predicate)

@dataclass
class StageEntry:
    factory: Callable[[], TrainingStage]
    requires: Callable[[Any], bool]
    description: str = ""


STAGE_REGISTRY: dict[str, StageEntry] = {}


def register_stage(
    name: str,
    factory: Callable[[], TrainingStage],
    requires: Callable[[Any], bool] = lambda model: True,
    description: str = "",
) -> None:
    STAGE_REGISTRY[name] = StageEntry(factory, requires, description)


def resolve_stage(spec: Any, reference_module: Any) -> TrainingStage:
    """Resolve a recipe entry into a ``TrainingStage``.

    Accepts a ``TrainingStage`` instance (returned as-is) or a registry name.
    Named stages are validated against ``reference_module`` via their ``requires``
    predicate and raise ``TypeError`` when incompatible.
    """
    if isinstance(spec, TrainingStage):
        return spec
    if isinstance(spec, str):
        entry = STAGE_REGISTRY.get(spec)
        if entry is None:
            raise KeyError(f"Unknown training stage {spec!r}; registered: {sorted(STAGE_REGISTRY)}")
        if not entry.requires(reference_module):
            raise TypeError(
                f"Training stage {spec!r} is not compatible with model "
                f"{type(reference_module).__name__}"
                + (f" ({entry.description})" if entry.description else "")
            )
        return entry.factory()
    raise TypeError(f"Cannot interpret training-stage spec {spec!r}.")


def build_stages(recipe: Any, reference_module: Any) -> list[TrainingStage]:
    return [resolve_stage(spec, reference_module) for spec in recipe]


# Register the model-specific stages defined here. The universal ``sgd`` stage is
# registered in infrastructure.experiment.training (where SGDStage lives, to avoid
# an import cycle with the SGD helpers).
register_stage(
    "analytical_init", AnalyticalInitStage,
    requires=lambda m: hasattr(m, "analytical_initialization"),
    description="model must define analytical_initialization()",
)
register_stage(
    "least_squares_init", LeastSquaresInitStage,
    requires=lambda m: hasattr(m, "vmap_train_least_squares"),
    description="model must define vmap_train_least_squares()",
)
register_stage(
    "online_least_squares", OnlineLeastSquaresStage,
    requires=lambda m: hasattr(m, "train_least_squares_online"),
    description="model must define train_least_squares_online()",
)
register_stage(
    "newton_init", NewtonStage,
    requires=lambda m: hasattr(m, "newton_initialization"),
    description="model must define newton_initialization()",
)
register_stage(
    "random_step", RandomStepStage,
    requires=lambda m: hasattr(m, "vmap_train_least_squares"),
    description="model must define vmap_train_least_squares()",
)
register_stage(
    "negation", NegationStage,
    requires=lambda m: hasattr(m, "vmap_train_least_squares"),
    description="model must define vmap_train_least_squares()",
)
register_stage(
    "ho_kalman", HoKalmanStage,
    requires=lambda m: hasattr(m, "convert_fir"),
    description="model must define convert_fir()",
)
