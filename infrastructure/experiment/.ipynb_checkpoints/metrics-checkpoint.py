import collections
from argparse import Namespace
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tensordict import TensorDict

from infrastructure import utils
from model.base import Predictor
from system.base import SystemGroup


MetricVars = Tuple[Namespace, TensorDict[str, torch.Tensor]]

class Metric(object):
    @classmethod
    def compute(cls,
                mv: MetricVars,
                dependency: str,
                cache: Dict[str, np.ndarray[TensorDict[str, torch.Tensor]]]
    ) -> np.ndarray[TensorDict[str, torch.Tensor]]:
        if dependency not in cache:
            exclusive, ensembled_learned_kfs = mv
            with torch.set_grad_enabled(False):
                run = utils.multi_map(
                    lambda dataset: Predictor.run(exclusive.reference_module, ensembled_learned_kfs, *dataset),
                    utils.rgetattr(exclusive, f"info.{dependency}.dataset"), dtype=TensorDict
                )
            cache[dependency] = run
        return cache[dependency]

    def __init__(self, evaluate_func: Callable[[MetricVars, Dict[str, np.ndarray[TensorDict[str, torch.Tensor]]], bool], np.ndarray[torch.Tensor]]) -> None:
        self._evaluate_func = evaluate_func

    def evaluate(self,
                 mv: MetricVars,
                 cache: Dict[str, np.ndarray[TensorDict[str, torch.Tensor]]],
                 sweep_position: str,
                 with_batch_dim: bool
    ) -> torch.Tensor:
        assert sweep_position in ("inside", "outside"), f"Position of hyperparameter sweep must be either before model_shape (outside) or after model_shape (inside), but got {sweep_position}."
        result_arr = self._evaluate_func(mv, cache, with_batch_dim)
        return utils.stack_tensor_arr(result_arr, dim=(0 if sweep_position == "outside" else 2))


Metrics: OrderedDict[str, Metric] = collections.OrderedDict()
def add_to_metrics(M: Metric, names: str | Iterable[str]):
    if isinstance(names, str):
        names = (names,)
    for n in names:
        Metrics[n] = M

def _unsqueeze_if(t: torch.Tensor, b: bool) -> torch.Tensor:
    return t.unsqueeze(-1) if b else t

def _get_evaluation_metric_with_dataset_type_and_targets(ds_type: str, key: Tuple[str, ...], target: Tuple[str, ...]) -> Metric:
    def eval_func(
            mv: MetricVars,
            cache: Dict[str, np.ndarray[TensorDict[str, torch.Tensor]]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:
        exclusive, ensembled_learned_kfs = mv
        run = Metric.compute(mv, ds_type, cache)

        return utils.multi_map(
            lambda pair: Predictor.evaluate_run(
                pair[0][key], pair[1].obj, target,
                batch_mean=not with_batch_dim
            ), utils.multi_zip(run, utils.rgetattr(exclusive, f"info.{ds_type}.dataset")), dtype=torch.Tensor
        )
    return Metric(eval_func)

def _get_comparator_metric_with_dataset_type_and_targets(ds_type: str, target1: Tuple[str, ...], target2: Tuple[str, ...]) -> Metric:
    def eval_func(
            mv: MetricVars,
            cache: Dict[str, np.ndarray[TensorDict[str, torch.Tensor]]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:
        exclusive, ensembled_learned_kfs = mv
        return utils.multi_map(
            lambda dataset: Predictor.evaluate_run(dataset.obj[target1], dataset.obj, target2, batch_mean=not with_batch_dim),
            utils.rgetattr(exclusive, f"info.{ds_type}.dataset"), dtype=torch.Tensor
        )
    return Metric(eval_func)

def _get_analytical_error_with_dataset_type_and_target(ds_type: str, target: Tuple[str, ...]) -> Metric:
    def eval_func(
            mv: MetricVars,
            cache: Dict[str, np.ndarray[TensorDict[str, torch.Tensor]]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:
        exclusive, ensembled_learned_kfs = mv
        def AE(sg: SystemGroup) -> torch.Tensor:
            return _unsqueeze_if(exclusive.reference_module.analytical_error(
                ensembled_learned_kfs[:, :, None],
                sg.td()[:, None, :]
            )[target], with_batch_dim)

        with torch.set_grad_enabled(False):
            return utils.multi_map(AE, utils.rgetattr(exclusive, f"info.{ds_type}.systems"), dtype=torch.Tensor)
    return Metric(eval_func)

def _get_gradient_norm_with_dataset_type(ds_type: str) -> Metric:
    def eval_func(
            mv: MetricVars,
            cache: Dict[str, np.ndarray[TensorDict[str, torch.Tensor]]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:
        raise NotImplementedError("This metric is outdated, and we need to derive a new way of determining convergence.")
        exclusive, ensembled_learned_kfs = mv

        reset_ensembled_learned_kfs = Predictor.clone_parameter_state(exclusive.reference_module, ensembled_learned_kfs)
        params = [p for p in reset_ensembled_learned_kfs.values() if isinstance(p, nn.Parameter)]

        dataset_arr = utils.rgetattr(exclusive, f"info.{ds_type}.dataset")
        with torch.set_grad_enabled(True):
            run_arr = utils.multi_map(
                lambda dataset: Predictor.run(exclusive.reference_module, reset_ensembled_learned_kfs, *dataset)["environment", "observation"],
                dataset_arr, dtype=torch.Tensor
            )
            loss_arr = utils.multi_map(
                lambda pair: Predictor.evaluate_run(pair[0], pair[1].obj, ("environment", "observation")).mean(dim=-1),
                utils.multi_zip(run_arr, dataset_arr), dtype=torch.Tensor
            )

        def gradient_norm(loss: torch.Tensor) -> torch.Tensor:
            grads = torch.autograd.grad(loss.sum(), params, allow_unused=True)
            return _unsqueeze_if(torch.Tensor(sum(
                torch.norm(torch.flatten(grad, start_dim=2, end_dim=-1), dim=2)
                for grad in grads if grad is not None
            )), with_batch_dim)

        return utils.multi_map(
            gradient_norm,
            loss_arr, dtype=torch.Tensor
        )
    return Metric(eval_func)

def _get_irreducible_loss_with_dataset_type_and_target(ds_type: str, target: Tuple[str, ...]) -> Metric:
    def eval_func(
            mv: MetricVars,
            cache: Dict[str, np.ndarray[TensorDict[str, torch.Tensor]]],
            with_batch_dim: bool
    ) -> np.ndarray[torch.Tensor]:
        exclusive, ensembled_learned_kfs = mv
        return utils.multi_map(
            lambda sg: _unsqueeze_if(sg.td()[("irreducible_loss", *target)][:, None], with_batch_dim),
            utils.rgetattr(exclusive, f"info.{ds_type}.systems"), dtype=torch.Tensor
        )
    return Metric(eval_func)




add_to_metrics(_get_evaluation_metric_with_dataset_type_and_targets(
    "train", ("environment", "observation"), ("environment", "observation")
), names="overfit")
add_to_metrics(_get_evaluation_metric_with_dataset_type_and_targets(
    "valid", ("environment", "observation"), ("environment", "observation")
), names="validation")
add_to_metrics(_get_evaluation_metric_with_dataset_type_and_targets(
    "valid", ("environment", "observation"), ("environment", "target_observation_estimation")
), names="validation_target")
add_to_metrics(_get_evaluation_metric_with_dataset_type_and_targets(
    "test", ("environment", "observation"), ("environment", "observation")
), names=["testing", "l"])
add_to_metrics(_get_evaluation_metric_with_dataset_type_and_targets(
    "valid", ("controller", "input"), ("controller", "input")
), names="validation_controller")

add_to_metrics(_get_comparator_metric_with_dataset_type_and_targets(
    "test", ("environment", "target_observation_estimation"), ("environment", "observation")
), names=["testing_empirical_irreducible", "eil"])

add_to_metrics(_get_analytical_error_with_dataset_type_and_target("valid", ("environment", "observation")), names="validation_analytical")
add_to_metrics(_get_analytical_error_with_dataset_type_and_target("valid", ("controller", "input")), names="validation_controller_analytical")
add_to_metrics(_get_analytical_error_with_dataset_type_and_target("test", ("environment", "observation")), names=["testing_analytical", "al"])

add_to_metrics(_get_gradient_norm_with_dataset_type("train"), names="overfit_gradient_norm")

add_to_metrics(_get_irreducible_loss_with_dataset_type_and_target("test", ("environment", "observation")), names=["testing_irreducible", "il"])









