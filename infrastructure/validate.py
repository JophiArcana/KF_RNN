from argparse import Namespace
from typing import *

import numpy as np
import torch
import torch.utils.data
from tensordict import TensorDict

from infrastructure.settings import dev_type
from model.analytical_kf import AnalyticalKF
from model.kf import KF
from model.linear_system import LinearSystem


def _compute_metrics(
        vhp: Namespace,
        base_kf: KF,
        flattened_ensembled_learned_kfs: TensorDict[str, torch.Tensor],
        flattened_dataset: TensorDict[str, torch.Tensor],
        metrics: Collection[str] = None
) -> Namespace:
    base_kf.to(dev_type).eval()
    flattened_ensembled_learned_kfs = flattened_ensembled_learned_kfs.to(dev_type)

    mask = torch.Tensor(torch.arange(flattened_dataset.shape[-1], device=dev_type) >= vhp.sequence_buffer)
    with torch.set_grad_enabled(False):
        run_result = base_kf.run(flattened_dataset, dict(flattened_ensembled_learned_kfs))

        pairs: Dict[str, Sequence[torch.Tensor | float]] = {
            'l': (run_result, torch.Tensor(flattened_dataset['observation'])),
            'eil': (torch.Tensor(flattened_dataset['target']), torch.Tensor(flattened_dataset['observation'])),
            'rl': (run_result, torch.Tensor(flattened_dataset['target'])),
            'rn': (0, torch.Tensor(flattened_dataset['target']))
        }

        return Namespace(**{
            metric: KF.evaluate_run(*pairs[metric], mask=mask, batch_mean=False)
            for metric in (metrics if metrics is not None else pairs.keys())
        })


def compute_experiment_metrics(
        vhp: Namespace,
        systems: List[LinearSystem],
        learned_kf_arr: np.ndarray[Tuple[KF, TensorDict]]
) -> Namespace:
    systems = [sys.to(dev_type) for sys in systems]
    analytical_kfs = [AnalyticalKF(sys).eval() for sys in systems]

    n_systems, ensemble_size = learned_kf_arr.flatten()[0][1].shape
    flattened_valid_dataset = AnalyticalKF.add_targets(analytical_kfs, LinearSystem.generate_dataset(
        systems=systems,
        batch_size=vhp.valid_dataset_size,
        seq_length=vhp.valid_sequence_length
    ))[:, None].expand(
        n_systems,
        ensemble_size,
        vhp.valid_dataset_size,
        vhp.valid_sequence_length
    ).flatten(0, 1)
    print('Validation dataset generated')

    metric_list = [_compute_metrics(
        vhp,
        base_kf,
        learned_kf.flatten(),
        flattened_valid_dataset,
        metrics=(None if i == 0 else {'l', 'rl'})
    ) for i, (base_kf, learned_kf) in enumerate(learned_kf_arr.flatten())]

    flattened_result = {
        'eil': metric_list[0].eil,
        'rn': metric_list[0].rn,
        'l': torch.stack([M.l for M in metric_list]),
        'rl': torch.stack([M.rl for M in metric_list])
    }
    if len(learned_kf_arr.shape) > 0:
        flattened_result['l'] = flattened_result['l'].unflatten(0, learned_kf_arr.shape)
        flattened_result['rl'] = flattened_result['rl'].unflatten(0, learned_kf_arr.shape)
    else:
        flattened_result['l'] = flattened_result['l'].squeeze(0)
        flattened_result['rl'] = flattened_result['rl'].squeeze(0)

    return Namespace(**{
        k: v.unflatten(-2, (n_systems, ensemble_size))
        for k, v in flattened_result.items()
    })


def subtraction_normalized_validation_loss(
        vhp: Namespace,
        systems: List[LinearSystem],
        learned_kf_arr: np.ndarray[Tuple[KF, TensorDict]]
) -> torch.Tensor:
    metrics = compute_experiment_metrics(vhp, systems, learned_kf_arr)
    return metrics.l - metrics.eil


def compute_impulse_response(
        base_kf: KF,
        ensembled_kfs: TensorDict[str, torch.Tensor],
        ir_length: int,
        **kwargs
) -> torch.Tensor:
    base_kf.to(dev_type).eval()
    if len(ensembled_kfs.shape) > 1:
        flattened_ensembled_kfs = ensembled_kfs.flatten().to(dev_type)
    else:
        flattened_ensembled_kfs = ensembled_kfs.to(dev_type)
    ir_dataset = TensorDict({
        'input': torch.zeros(base_kf.O_D, ir_length, base_kf.I_D),
        'observation': torch.cat([
            torch.eye(base_kf.O_D)[:, None, :],
            torch.zeros(base_kf.O_D, ir_length - 1, base_kf.O_D)
        ], dim=1),
    }, batch_size=(base_kf.O_D, ir_length), device=dev_type).expand(flattened_ensembled_kfs.shape[0], base_kf.O_D, ir_length)
    return base_kf.run(ir_dataset, dict(flattened_ensembled_kfs), **kwargs).unflatten(0, ensembled_kfs.shape)




