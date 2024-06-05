import collections
from argparse import Namespace
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.utils import PTR
from model.kf import KF

MetricVars = Tuple[Namespace, Namespace, TensorDict[str, torch.Tensor]]


def _truncation_mask(config: Namespace) -> torch.Tensor:
    return torch.Tensor(torch.arange(config.sequence_length) >= config.sequence_buffer)

class Metric(object):
    @classmethod
    def compute(cls,
                vars: MetricVars,
                dependency: str,
                cache: Dict[str, np.ndarray[torch.Tensor]]
    ) -> np.ndarray[torch.Tensor]:
        if dependency not in cache:
            HP, exclusive, ensembled_learned_kfs = vars
            with torch.set_grad_enabled(True):
                run = utils.multi_map(
                    lambda dataset: KF.run(exclusive.reference_module, ensembled_learned_kfs, *dataset),
                    utils.rgetattr(exclusive, f"info.{dependency}.dataset"), dtype=torch.Tensor
                )
            cache[dependency] = run
        return cache[dependency]

    def __init__(self, evaluate_func: Callable[[MetricVars, Dict[str, np.ndarray[torch.Tensor]], bool], np.ndarray[torch.Tensor]]) -> None:
        self._evaluate_func = evaluate_func

    def evaluate(self, vars: MetricVars, cache: Dict[str, np.ndarray[torch.Tensor]], sweep_position: str, with_batch_dim: bool) -> torch.Tensor:
        assert sweep_position in ("inside", "outside"), f"Position of hyperparameter sweep must be either before model_shape (outside) or after model_shape (inside), but got {sweep_position}."
        result_arr = self._evaluate_func(vars, cache, with_batch_dim)
        return utils.stack_tensor_arr(result_arr, dim=(0 if sweep_position == "outside" else 2))


Metrics: OrderedDict[str, Metric] = collections.OrderedDict()
def add_to_metrics(M: Metric, names: str | Iterable[str]):
    if isinstance(names, str):
        names = (names,)
    for n in names:
        Metrics[n] = M

def _unsqueeze_if(t: torch.Tensor, b: bool) -> torch.Tensor:
    return t.unsqueeze(-1) if b else t

def _get_evaluation_metric_with_dataset_type_and_target(ds_type: str, target) -> Metric:
    def eval_func(vars: MetricVars, cache: Dict[str, np.ndarray[torch.Tensor]], with_batch_dim: bool) -> np.ndarray[torch.Tensor]:
        HP, exclusive, ensembled_learned_kfs = vars
        run = Metric.compute(vars, ds_type, cache)
        if ds_type == "train":
            mask = _truncation_mask(getattr(HP.dataset, ds_type)) * exclusive.train_mask
        else:
            mask = _truncation_mask(getattr(HP.dataset, ds_type))
        return utils.multi_map(
            lambda pair: KF.evaluate_run(pair[0], pair[1].obj[target], mask, batch_mean=not with_batch_dim),
            utils.multi_zip(run, utils.rgetattr(exclusive, f"info.{ds_type}.dataset")), dtype=torch.Tensor
        )
    return Metric(eval_func)

def _get_comparator_metric_with_dataset_type_and_targets(ds_type: str, target1: str, target2: str) -> Metric:
    def eval_func(vars: MetricVars, cache: Dict[str, np.ndarray[torch.Tensor]], with_batch_dim: bool) -> np.ndarray[torch.Tensor]:
        HP, exclusive, ensembled_learned_kfs = vars
        if ds_type == "train":
            mask = _truncation_mask(getattr(HP.dataset, ds_type)) * exclusive.train_mask
        else:
            mask = _truncation_mask(getattr(HP.dataset, ds_type))
        return utils.multi_map(
            lambda dataset: KF.evaluate_run(dataset.obj[target1], dataset.obj[target2], mask, batch_mean=not with_batch_dim),
            utils.rgetattr(exclusive, f"info.{ds_type}.dataset"), dtype=torch.Tensor
        )
    return Metric(eval_func)

def _get_analytical_error_with_dataset_type(ds_type: str) -> Metric:
    def eval_func(vars: MetricVars, cache: Dict[str, np.ndarray[torch.Tensor]], with_batch_dim: bool) -> np.ndarray[torch.Tensor]:
        HP, exclusive, ensembled_learned_kfs = vars
        def AE(stacked_systems_arr_: PTR) -> torch.Tensor:
            return _unsqueeze_if(exclusive.reference_module.analytical_error(
                ensembled_learned_kfs[:, :, None],
                stacked_systems_arr_.obj[:, None, :]
            ), with_batch_dim)

        with torch.set_grad_enabled(False):
            return utils.multi_map(AE, utils.rgetattr(exclusive, f"info.{ds_type}.stacked_systems"), dtype=torch.Tensor)
    return Metric(eval_func)

def _get_gradient_norm_with_dataset_type(ds_type: str) -> Metric:
    def eval_func(vars: MetricVars, cache: Dict[str, np.ndarray[torch.Tensor]], with_batch_dim: bool) -> np.ndarray[torch.Tensor]:
        HP, exclusive, ensembled_learned_kfs = vars

        reset_ensembled_learned_kfs = KF.clone_parameter_state(exclusive.reference_module, ensembled_learned_kfs)
        params = [p for p in reset_ensembled_learned_kfs.values() if isinstance(p, nn.Parameter)]

        dataset_arr = utils.rgetattr(exclusive, f"info.{ds_type}.dataset")
        mask = exclusive.train_mask if ds_type == "train" else None
        with torch.set_grad_enabled(True):
            run_arr = utils.multi_map(
                lambda dataset: KF.run(exclusive.reference_module, reset_ensembled_learned_kfs, *dataset),
                dataset_arr, dtype=torch.Tensor
            )
            loss_arr = utils.multi_map(
                lambda pair: KF.evaluate_run(pair[0], pair[1].obj["observation"], mask).mean(dim=-1),
                utils.multi_zip(run_arr, dataset_arr), dtype=torch.Tensor
            )

        def gradient_norm(loss: torch.Tensor) -> torch.Tensor:
            grads = torch.autograd.grad(loss.sum(), params)
            return _unsqueeze_if(torch.Tensor(sum(
                torch.norm(torch.flatten(grad, start_dim=2, end_dim=-1), dim=2)
                for grad in grads
            )), with_batch_dim)

        return utils.multi_map(
            gradient_norm,
            loss_arr, dtype=torch.Tensor
        )
    return Metric(eval_func)

def _get_irreducible_loss_with_dataset_type(ds_type: str) -> Metric:
    def eval_func(vars: MetricVars, cache: Dict[str, np.ndarray[torch.Tensor]], with_batch_dim: bool) -> np.ndarray[torch.Tensor]:
        HP, exclusive, ensembled_learned_kfs = vars
        return utils.multi_map(
            lambda il: _unsqueeze_if(il.obj[:, None], with_batch_dim),
            utils.rgetattr(exclusive, f"info.{ds_type}.irreducible_loss"), dtype=torch.Tensor
        )
    return Metric(eval_func)




add_to_metrics(_get_evaluation_metric_with_dataset_type_and_target("train", "observation"), names="overfit")
add_to_metrics(_get_evaluation_metric_with_dataset_type_and_target("valid", "observation"), names="validation")
add_to_metrics(_get_evaluation_metric_with_dataset_type_and_target("valid", "target"), names="validation_target")
add_to_metrics(_get_evaluation_metric_with_dataset_type_and_target("test", "observation"), names=["testing", "l"])

add_to_metrics(_get_comparator_metric_with_dataset_type_and_targets("test", "target", "observation"), names=["testing_empirical_irreducible", "eil"])

add_to_metrics(_get_analytical_error_with_dataset_type("valid"), names="validation_analytical")
add_to_metrics(_get_analytical_error_with_dataset_type("test"), names=["testing_analytical", "al"])

add_to_metrics(_get_gradient_norm_with_dataset_type("train"), names="overfit_gradient_norm")

add_to_metrics(_get_irreducible_loss_with_dataset_type("test"), names=["testing_irreducible", "il"])









