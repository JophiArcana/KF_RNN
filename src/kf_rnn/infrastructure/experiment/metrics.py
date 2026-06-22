import collections
from types import SimpleNamespace
from typing import Any, Callable, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.infrastructure.static import ModelPair
from kf_rnn.infrastructure.experiment.losses import LossFn
from kf_rnn.model.base import Predictor
from kf_rnn.system.base import SystemGroup


MetricVars = tuple[SimpleNamespace, ModelPair]

class Metric(object):
    @classmethod
    def compute(cls,
                mv: MetricVars,
                dependency: str,
                cache: dict[str, np.ndarray[TensorDict]]
    ) -> np.ndarray[TensorDict]:
        if dependency not in cache:
            exclusive, model_pair = mv
            with torch.set_grad_enabled(False):
                run_arr = eu.multi_map(
                    lambda dataset: Predictor.run(model_pair, dataset),
                    eu.rgetattr(exclusive, f"info.{dependency}.dataset"), dtype=TensorDict,
                )
            cache[dependency] = run_arr
        return cache[dependency]

    def __init__(self, evaluate_func: Callable[[MetricVars, dict[str, np.ndarray[TensorDict]], bool], np.ndarray[torch.Tensor]]) -> None:
        self._evaluate_func = evaluate_func

    def evaluate(self,
                 mv: MetricVars,
                 cache: dict[str, np.ndarray[TensorDict]],
                 sweep_position: str,
                 with_batch_dim: bool
    ) -> torch.Tensor:
        assert sweep_position in ("inside", "outside"), f"Position of hyperparameter sweep must be either before model_shape (outside) or after model_shape (inside), but got {sweep_position}."
        result_arr = self._evaluate_func(mv, cache, with_batch_dim)
        return eu.stack_tensor_arr(result_arr, dim=(0 if sweep_position == "outside" else 2))


METRIC_DICT: OrderedDict[str, Metric] = collections.OrderedDict()
def add_to_metrics(M: Metric, names: tuple[str, ...]):
    for n in names:
        METRIC_DICT[n] = M

def _unsqueeze_if(t: torch.Tensor, b: bool) -> torch.Tensor:
    return t[..., None] if b else t

def _broadcast_group_shaped(group_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reshape a system-``group_shape``-shaped tensor (e.g. the per-system
    irreducible error) so it broadcasts against ``target`` (the per-element
    reducible error).

    The standard datasets are laid out ``[..., N, B(, L)]`` (systems
    second-to-last) while the permuted CNN/RNN baseline datasets are laid out
    ``[N, B, ...]`` (systems first), so a plain right-aligned broadcast is not
    always correct. Locate the alignment offset whose non-singleton group dims
    exactly match ``target`` (rather than merely broadcasting against size-1
    axes), and pad the remaining axes with singletons.
    """
    g_shape = tuple(group_tensor.shape)
    best_offset, best_exact = None, -1
    for offset in range(target.ndim - group_tensor.ndim + 1):
        window = target.shape[offset:offset + group_tensor.ndim]
        if all((g == w) or g == 1 or w == 1 for g, w in zip(g_shape, window)):
            exact = sum(1 for g, w in zip(g_shape, window) if g != 1 and g == w)
            if exact > best_exact:
                best_offset, best_exact = offset, exact
    if best_offset is None:
        raise RuntimeError(
            f"Cannot align group-shaped tensor {g_shape} with target shape {tuple(target.shape)}."
        )
    return group_tensor.reshape(
        (1,) * best_offset + g_shape + (1,) * (target.ndim - group_tensor.ndim - best_offset)
    )








def _get_metric_with_loss_fn_and_dataset_type(ds_type: str, loss_fn: LossFn, kwargs: dict[str, Any]) -> Metric:
    noiseless: bool = kwargs.get("noiseless", False)
    def eval_func(
            mv: MetricVars,
            cache: dict[str, np.ndarray[TensorDict]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:

        def compute_loss_with_result_dataset_pair(args: tuple[TensorDict, TensorDict, SystemGroup,]) -> torch.Tensor:
            result, dataset, sg = args

            if noiseless:
                dataset = dataset.clone()
                prefix = "noiseless_"
                for _k in [*dataset.keys(include_nested=True, leaves_only=True)]:
                    k = (_k,) if isinstance(_k, str) else tuple(_k)
                    new_k = tuple(
                        seg[len(prefix):] if seg.startswith(prefix) else seg
                        for seg in k
                    )
                    if new_k != k:
                        dataset.rename_key_(k, new_k)

            losses = loss_fn(result, dataset, {})
            mask = dataset.get("mask", torch.full(dataset.shape[-2:], True))
            
            dims = [-1,] if with_batch_dim else [-2, -1,]
            reducible_error = torch.sum(losses * mask, dim=dims) / torch.sum(mask, dim=dims)
            
            if noiseless:
                env = sg.environment
                irreducible_error = eu.batch_trace(env.H @ env.S_W @ env.H.mT + env.S_V)
                return reducible_error + _broadcast_group_shaped(irreducible_error, reducible_error)
            else:
                return reducible_error

        return eu.multi_map(compute_loss_with_result_dataset_pair, eu.multi_zip(
            Metric.compute(mv, ds_type, cache),
            eu.rgetattr(mv[0], f"info.{ds_type}.dataset"),
            eu.rgetattr(mv[0], f"info.{ds_type}.systems"),
        ), dtype=torch.Tensor,)
    return Metric(eval_func)








def _get_noiseless_error_with_dataset_type_and_target(ds_type: str, target: tuple[str, ...]) -> Metric:
    def eval_func(
            mv: MetricVars,
            cache: dict[str, np.ndarray[TensorDict]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:
        exclusive, _ = mv

        def noiseless_error(args: tuple[TensorDict, SystemGroup]) -> torch.Tensor:
            dataset, sg = args
            reducible_error = Predictor.evaluate_run(
                dataset[target], dataset, ("environment", "noiseless_observation"),
                batch_mean=not with_batch_dim
            )
            env = sg.environment
            irreducible_error = eu.batch_trace(env.H @ env.S_W @ env.H.mT + env.S_V)[:, None]
            return eu.T(eu.T(reducible_error) + eu.T(irreducible_error))
            
        return eu.multi_map(noiseless_error, eu.multi_zip(
            eu.rgetattr(exclusive, f"info.{ds_type}.dataset"),
            eu.rgetattr(exclusive, f"info.{ds_type}.systems"),
        ), dtype=torch.Tensor)

    return Metric(eval_func)

def _get_comparator_metric_with_dataset_type_and_targets(ds_type: str, target1: tuple[str, ...], target2: tuple[str, ...]) -> Metric:
    def eval_func(
            mv: MetricVars,
            cache: dict[str, np.ndarray[TensorDict]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:
        exclusive, _ = mv
        return eu.multi_map(
            lambda dataset: Predictor.evaluate_run(dataset[target1], dataset, target2, batch_mean=not with_batch_dim),
            eu.rgetattr(exclusive, f"info.{ds_type}.dataset"), dtype=torch.Tensor
        )
    return Metric(eval_func)

def _get_analytical_error_with_dataset_type_and_key(ds_type: str, key: tuple[str, ...]) -> Metric:
    def eval_func(
            mv: MetricVars,
            cache: dict[str, np.ndarray[TensorDict]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:
        exclusive, (reference_module, stacked_modules) = mv

        def analytical_error(sg: SystemGroup) -> torch.Tensor:
            return _unsqueeze_if(reference_module.analytical_error(
                stacked_modules[:, :, None],
                sg.td()[:, None, :]
            )[key], with_batch_dim)

        with torch.set_grad_enabled(False):
            return eu.multi_map(analytical_error, eu.rgetattr(exclusive, f"info.{ds_type}.systems"), dtype=torch.Tensor)
    return Metric(eval_func)

def _get_gradient_norm_with_dataset_type(ds_type: str) -> Metric:
    def eval_func(
            mv: MetricVars,
            cache: dict[str, np.ndarray[TensorDict]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:
        # raise NotImplementedError("This metric is outdated, and we need to derive a new way of determining convergence.")
        exclusive, model_pair = mv

        reset_model_pair = Predictor.clone_parameter_state(model_pair)
        params = [p for p in reset_model_pair[1].values() if isinstance(p, nn.Parameter)]

        dataset_arr = eu.rgetattr(exclusive, f"info.{ds_type}.dataset")
        with torch.set_grad_enabled(True):
            run_arr = eu.multi_map(
                lambda dataset: Predictor.run(reset_model_pair, dataset)["environment", "observation"],
                dataset_arr, dtype=torch.Tensor,
            )
            loss_arr = eu.multi_map(
                lambda pair: Predictor.evaluate_run(pair[0], pair[1], ("environment", "observation")).mean(dim=-1),
                eu.multi_zip(run_arr, dataset_arr), dtype=torch.Tensor,
            )

        def gradient_norm(loss: torch.Tensor) -> torch.Tensor:
            grads = torch.autograd.grad(loss.sum(), params, allow_unused=True)
            return _unsqueeze_if(torch.stack([
                torch.norm(torch.flatten(grad, start_dim=2, end_dim=-1), dim=2) ** 2
                for grad in grads if grad is not None
            ]).sum(dim=0), with_batch_dim)

        return eu.multi_map(
            gradient_norm,
            loss_arr, dtype=torch.Tensor
        )
    return Metric(eval_func)

def _get_irreducible_loss_with_dataset_type_and_key(ds_type: str, key: tuple[str, ...]) -> Metric:
    def eval_func(
            mv: MetricVars,
            cache: dict[str, np.ndarray[TensorDict]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:
        exclusive, _ = mv
        return eu.multi_map(
            lambda sg: _unsqueeze_if(sg.td()[("irreducible_loss", *key)][:, None], with_batch_dim),
            eu.rgetattr(exclusive, f"info.{ds_type}.systems"), dtype=torch.Tensor
        )
    return Metric(eval_func)



from kf_rnn.infrastructure.experiment.losses import (
    _get_mse_loss_fn_with_output_and_target_key,
    _get_finite_difference_mse_loss_fn_with_output_and_target_key,
)

def _is_abbreviation(n: str) -> bool:
    return len(n) <= 4

for ds_type, output_key, target_key, names in [
    ("train", ("environment", "observation"), ("environment", "observation"), ("overfit",)),
    ("valid", ("environment", "observation"), ("environment", "observation"), ("validation",)),
    ("valid", ("environment", "observation"), ("environment", "target_observation_estimation"), ("validation_target",)),
    ("valid", ("controller", "input"), ("controller", "input"), ("validation_controller",)),
    ("test", ("environment", "observation"), ("environment", "observation"), ("testing", "l",)),
]:
    loss_fn = _get_mse_loss_fn_with_output_and_target_key(output_key, target_key)
    nnames = [f"n{n}" if _is_abbreviation(n) else f"noiseless_{n}" for n in names]
    add_to_metrics(_get_metric_with_loss_fn_and_dataset_type(ds_type, loss_fn, {},), names=names,)
    add_to_metrics(_get_metric_with_loss_fn_and_dataset_type(ds_type, loss_fn, dict(noiseless=True),), names=nnames,)

    fd_loss_fn = _get_finite_difference_mse_loss_fn_with_output_and_target_key(output_key, target_key)
    fd_names = [f"fd_{n}" for n in names]
    fd_nnames = [f"fd_{n}" for n in nnames]
    add_to_metrics(_get_metric_with_loss_fn_and_dataset_type(ds_type, fd_loss_fn, {},), names=fd_names,)
    add_to_metrics(_get_metric_with_loss_fn_and_dataset_type(ds_type, fd_loss_fn, dict(noiseless=True),), names=fd_nnames,)







add_to_metrics(_get_comparator_metric_with_dataset_type_and_targets(
    "test", ("environment", "target_observation_estimation"), ("environment", "observation")
), names=("testing_empirical_irreducible", "eil",))

add_to_metrics(_get_noiseless_error_with_dataset_type_and_target(
    "test", ("environment", "target_observation_estimation")
), names=("noiseless_testing_empirical_irreducible", "neil",))

add_to_metrics(_get_analytical_error_with_dataset_type_and_key("valid", ("environment", "observation")), names=("validation_analytical",))
add_to_metrics(_get_analytical_error_with_dataset_type_and_key("valid", ("controller", "input")), names=("validation_controller_analytical",))
add_to_metrics(_get_analytical_error_with_dataset_type_and_key("test", ("environment", "observation")), names=("testing_analytical", "al",))

add_to_metrics(_get_gradient_norm_with_dataset_type("train"), names=("overfit_gradient_norm",))

add_to_metrics(_get_irreducible_loss_with_dataset_type_and_key("test", ("environment", "observation")), names=("testing_irreducible", "il",))




