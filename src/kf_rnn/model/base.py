import dataclasses
from types import MappingProxyType, SimpleNamespace
from typing import Any, ClassVar, Sequence

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.infrastructure.config.schema import ProblemShape, TrainConfig
from ecliseutils.ensemble import EnsembleModule


class Observer(nn.Module):
    """Base of the model hierarchy.

    Every subclass owns a nested ``Config`` dataclass holding its
    hyperparameters. Subclasses that introduce hyperparameters declare their
    own ``Config`` (inheriting the base ones); subclasses that don't get one
    auto-generated, so ``SomePredictor.Config`` always identifies exactly
    ``SomePredictor`` via its ``cls`` attribute. The framework instantiates
    models as ``model_config.cls(model_config)``.
    """

    @dataclasses.dataclass
    class Config:
        cls: ClassVar["type[Observer]"]
        problem_shape: ProblemShape = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "Config" not in cls.__dict__:
            # Merge the distinct Config classes of the direct bases so multiple
            # inheritance (e.g. CnnPredictor + LeastSquaresPredictor) combines
            # their hyperparameter fields.
            config_bases: list[type] = []
            for base in cls.__bases__:
                base_config = getattr(base, "Config", None)
                if base_config is not None and base_config not in config_bases:
                    config_bases.append(base_config)
            cls.Config = dataclasses.dataclass(type("Config", tuple(config_bases), {
                "__qualname__": f"{cls.__qualname__}.Config",
                "__module__": cls.__module__,
            }))
        cls.Config.cls = cls

    def __init__(self, modelArgs: "Observer.Config"):
        super().__init__()
        self.problem_shape = modelArgs.problem_shape
        self.O_D: int = self.problem_shape.environment.observation


Observer.Config.cls = Observer


class Predictor(Observer):
    @classmethod
    def impulse(cls,
                kf_arr: np.ndarray[nn.Module],
                length: int
    ) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    def run(cls,
            model_pair: "eu.ModelPair",
            dataset: TensorDict,
            kwargs: dict[str, Any] = MappingProxyType(dict()),
            split_size: int = EnsembleModule.DEFAULT_SPLIT_SIZE,
            method: str = "forward",
    ) -> TensorDict:
        # Thin shim: the ensemble reshape/chunk/vmap plumbing lives in EnsembleModule.
        return EnsembleModule.from_pair(model_pair).run(dataset, kwargs, split_size, method=method)

    @classmethod
    def gradient(cls,
                 model_pair: "eu.ModelPair",
                 dataset: TensorDict,
                 kwargs: dict[str, Any] = MappingProxyType(dict()),
                 split_size: int = EnsembleModule.DEFAULT_SPLIT_SIZE,
                 method: str = "forward",
    ) -> TensorDict:
        return EnsembleModule.from_pair(model_pair).gradient(dataset, kwargs, split_size, method=method)

    @classmethod
    def evaluate_run(cls,
                     result: torch.Tensor | float,                          # [B... x N x B x L x ...]
                     target_dict: TensorDict,            # [B... x N x B x L x ...]
                     target_key: tuple[str, ...],
                     batch_mean: bool = True,
    ) -> torch.Tensor:
        losses = torch.norm(result - target_dict[target_key], dim=-1) ** 2  # [B... x N x B x L]
        mask = target_dict.get("mask", torch.full(target_dict.shape[-1:], True))
        result_ = torch.sum(losses * mask, dim=-1) / torch.sum(mask, dim=-1)
        return result_.mean(dim=-1) if batch_mean else result_

    @classmethod
    def clone_parameter_state(cls, model_pair: "eu.ModelPair") -> "eu.ModelPair":
        return EnsembleModule.from_pair(model_pair).clone().pair

    """ forward
        :parameter {
            'state': [B x S_D],
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns {
            'state_estimation': [B x L x S_D],              (Optional)
            'observation_estimation': [B x L x O_D],
            'state_covariance': [B x L x S_D x S_D],        (Optional)
            'observation_covariance': [B x L x O_D x O_D]   (Optional)
        }
    """
    def forward(self, trace: dict[str, dict[str, torch.Tensor]], **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        raise NotImplementedError()

    @classmethod
    def trace_to_td(cls, trace: dict[str, dict[str, torch.Tensor]]) -> TensorDict:
        return TensorDict.from_dict(trace, batch_size=trace["environment"]["observation"].shape[:-1])

    def training_recipe(self) -> Sequence[Any]:
        """Data-driven training recipe: an ordered sequence of stage specs.

        A spec is a stage-registry name (e.g. ``"sgd"``) or a ``TrainingStage``
        instance. The engine resolves names via ``STAGE_REGISTRY`` and validates
        them against this model. Default: a single SGD stage.
        """
        return ["sgd"]

    @classmethod
    def analytical_error(cls,
                         kfs: TensorDict,                # [B... x ...]
                         sg_td: TensorDict               # [B... x ...]
    ) -> TensorDict:                                     # [B... x ...]
        return cls._analytical_error_and_cache(kfs, sg_td)[0]

    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict,     # [B... x ...]
                                    sg_td: TensorDict    # [B... x ...]
    ) -> tuple[TensorDict, SimpleNamespace]:                   # [B...]
        raise NotImplementedError(f"Analytical error does not exist for model {cls}")

    @classmethod
    def _augmented_plant_modal_decomposition(cls,
                                             kfs: "TensorDict | None",     # [B... x ...]
                                             systems: TensorDict,          # [B... x ...]
    ) -> SimpleNamespace:
        """Shared scaffolding for the closed-form analytical-error computations.

        Casts the relevant LQG system matrices to complex, eigendecomposes the
        augmented closed-loop plant ``F_augmented``, and projects the observation /
        control / process-noise matrices into the plant's modal basis. Every
        formulation-specific ``_analytical_error_and_cache`` (zero, sequential,
        convolutional) consumes this and only adds its own terms, so the common
        algebra lives in exactly one place.
        """
        controller_keys = systems.get(("environment", "B"), {}).keys()
        shape = systems.shape if kfs is None else eu.broadcast_shapes(kfs.shape, systems.shape)
        default_td = TensorDict({}, batch_size=shape)

        F = eu.complex(systems["environment", "F"])                                                  # [B... x S_D x S_D]
        K = eu.complex(systems["environment", "K"])                                                  # [B... x S_D x O_D]
        L = eu.complex(systems["controller", "L"]) if len(controller_keys) > 0 else default_td       # [B... x I_D? x S_D]
        sqrt_S_W = eu.complex(systems["environment", "sqrt_S_W"])                                    # [B... x S_D x S_D]
        sqrt_S_V = eu.complex(systems["environment", "sqrt_S_V"])                                    # [B... x O_D x O_D]

        Fa = eu.complex(systems["F_augmented"])                                                      # [B... x 2S_D x 2S_D]
        Ha = eu.complex(systems["H_augmented"])                                                      # [B... x O_D x 2S_D]
        La = eu.complex(systems["L_augmented"]) if len(controller_keys) > 0 else default_td          # [B... x I_D? x 2S_D]

        S_D, O_D = K.shape[-2:]

        D, V = torch.linalg.eig(Fa)                                                                     # [B... x 2S_D], [B... x 2S_D x 2S_D]
        Vinv = torch.inverse(V)                                                                         # [B... x 2S_D x 2S_D]

        Has = Ha @ V                                                                                    # [B... x O_D x 2S_D]
        Las = La.apply(lambda t: t @ V)                                                                 # [B... x I_D? x 2S_D]
        sqrt_S_Ws = Vinv @ torch.cat([sqrt_S_W, torch.zeros_like(sqrt_S_W)], dim=-2)                    # [B... x 2S_D x S_D]

        Dj = D[..., None, :]                                                                            # [B... x 1 x 2S_D]

        BL = eu.complex(torch.zeros((*shape, S_D, S_D)) + sum(
            systems["environment", "B", k] @ systems["controller", "L", k]
            for k in controller_keys
        ))                                                                                              # [B... x S_D x S_D]
        Vinv_BL_F_BLK = Vinv @ torch.cat([-BL, F - BL], dim=-2) @ K                                     # [B... x 2S_D x O_D]

        return SimpleNamespace(
            controller_keys=controller_keys, shape=shape, default_td=default_td,
            F=F, K=K, L=L, sqrt_S_W=sqrt_S_W, sqrt_S_V=sqrt_S_V,
            Fa=Fa, Ha=Ha, La=La, S_D=S_D, O_D=O_D,
            D=D, V=V, Vinv=Vinv,
            Has=Has, Las=Las, sqrt_S_Ws=sqrt_S_Ws, Dj=Dj,
            BL=BL, Vinv_BL_F_BLK=Vinv_BL_F_BLK,
        )


# A Controller is an Observer (it emits observations/actions); concrete controllers gain
# Predictor machinery by also inheriting a Predictor subclass (e.g. SequentialController,
# TransformerController), so this base is intentionally the lighter-weight Observer.
class Controller(Observer):
    @classmethod
    def compute_losses(
        cls,
        result: TensorDict,
        dataset: TensorDict,
        THP: "TrainConfig",
    ) -> torch.Tensor:
        observation_losses = Predictor.evaluate_run(
            result["environment", "observation"],
            dataset, ("environment", "observation")
        )
        action_losses = sum([
            Predictor.evaluate_run(v, dataset, ("controller", k))
            for k, v in result["controller"].items()
        ])
        return observation_losses + THP.control_coefficient * action_losses




