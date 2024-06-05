import copy
import json
from argparse import Namespace
from inspect import signature
from typing import *

import ignite.handlers.param_scheduler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.experiment.metrics import Metrics
from model.kf import KF


# Optimizer configuration
def _get_optimizer(
        params: Iterable[torch.Tensor],
        hp: Namespace
) -> Tuple[optim.Optimizer, ignite.handlers.LRScheduler | ignite.handlers.ConcatScheduler]:
    optim_type = hp.optim_type
    if optim_type == "GD" or optim_type == "SGD":
        optimizer = optim.SGD(params, lr=hp.max_lr, momentum=0.0, weight_decay=hp.weight_decay)
    elif optim_type == "SGDMomentum":
        optimizer = optim.SGD(params, lr=hp.max_lr, momentum=hp.momentum, weight_decay=hp.weight_decay)
    elif optim_type == "Adam":
        optimizer = optim.AdamW(params, lr=hp.max_lr, betas=(hp.momentum, 0.98), weight_decay=hp.weight_decay)
    else:
        raise ValueError(optim_type)

    if hp.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, hp.T_0, T_mult=hp.T_mult, eta_min=hp.min_lr)
        hp.epochs = hp.T_0 * ((hp.T_mult ** hp.num_restarts - 1) // (hp.T_mult - 1))
    elif hp.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, hp.lr_decay)
    else:
        raise ValueError(hp.scheduler)
    
    if (warmup_duration := getattr(hp, 'warmup_duration', None)) is None:
        return optimizer, ignite.handlers.param_scheduler.LRScheduler(scheduler)
    else:
        hp.epochs += (warmup_duration - 1)
        return optimizer, ignite.handlers.param_scheduler.create_lr_scheduler_with_warmup(
            scheduler,
            hp.min_lr,
            hp.warmup_duration
        )

# Training
TrainFunc = Callable[[
    Namespace,
    TensorDict[str, torch.Tensor],
    Namespace
], Tuple[torch.Tensor, bool]]

def _extract_dataset_and_mask_from_indices(
        indices: TensorDict[str, torch.Tensor],
        exclusive: Namespace
) -> Tuple[TensorDict[str, torch.Tensor], torch.Tensor]:
    sequence_indices, start_indices, stop_indices, system_indices = map(indices.__getitem__, (
        "sequence",
        "start",
        "stop",
        "system"
    ))  # [N x E x B]

    """ Smart indexing
        subsequence_length: L_SS
        n_experiment_idx: [N x 1 x 1 x 1], Selects the time experiment is repeated
        ensemble_size_idx: [1 x E x 1 x 1], Selects the time data is resampled
        system_idx: [N x E x B x 1], Selects the system that the data comes from
        sequence_index: [N x E x B x 1], Selects which trace within the dataset
        subsequence_index: [N x E x B x L_SS], Selects the timestep within the trace
    """
    subsequence_length = torch.max(stop_indices - start_indices).item()
    subsequence_offset = torch.arange(subsequence_length)

    n_experiment_idx = torch.arange(indices.shape[0])[:, None, None, None]
    ensemble_size_idx = torch.arange(indices.shape[1])[None, :, None, None]
    system_idx = system_indices.unsqueeze(-1)
    sequence_idx = sequence_indices.unsqueeze(-1)
    subsequence_idx = start_indices.unsqueeze(-1) + subsequence_offset

    """
        input: [N x E x B x L_SS x I_D]
        observation: [N x E x B x L_SS x O_D]
        target: [N x E x B x L_SS x O_D]
    """
    dataset_ss = exclusive.train_info.dataset.obj[
        n_experiment_idx,
        ensemble_size_idx,
        system_idx,
        sequence_idx,
        subsequence_idx
    ]
    mask_ss = torch.Tensor(subsequence_idx < stop_indices.unsqueeze(-1))
    return dataset_ss, mask_ss

def _train_default(
        THP: Namespace,
        exclusive: Namespace,
        ensembled_learned_kfs: TensorDict[str, torch.Tensor],
        cache: Namespace
) -> Tuple[torch.Tensor, bool]:
    def terminate_condition() -> bool:
        return cache.t == THP.epochs
    assert not terminate_condition()

    if not hasattr(cache, "optimizer"):
        # DONE: Need this line because some training functions replace the parameters with untrainable tensors (to preserve gradients)
        for k, v in KF.clone_parameter_state(exclusive.reference_module, ensembled_learned_kfs).items():
            ensembled_learned_kfs[k] = v
        cache.optimizer, cache.scheduler = _get_optimizer((v for v in ensembled_learned_kfs.values() if isinstance(v, nn.Parameter)), THP)
    cache.scheduler(None)

    result = []
    for batch, indices in enumerate(exclusive.supply_train_index_dataloader()):
        # DONE: Use indices to compute the mask for truncation and padding
        dataset_ss, mask_ss = _extract_dataset_and_mask_from_indices(indices, exclusive)
        mask_ss *= (torch.arange(THP.subsequence_length) >= getattr(THP, "sequence_buffer", 0))

        # DONE: Run test on the resulting subsequence block, calculate training loss, and return gradient step
        reference_module = exclusive.reference_module.train()
        with torch.set_grad_enabled(True):
            train_result = KF.run(reference_module, ensembled_learned_kfs, dataset_ss)
        result.append(losses := KF.evaluate_run(train_result, dataset_ss["observation"], mask_ss))

        cache.optimizer.zero_grad()
        torch.sum(losses).backward()

        cache.optimizer.step()
    cache.t += 1

    return torch.stack(result, dim=0), terminate_condition()

# Full training scheme
def _run_training(
        HP: Namespace,
        exclusive: Namespace,
        ensembled_learned_kfs: TensorDict[str, torch.Tensor],   # [N x E x ...]
        print_frequency: int = 10
) -> TensorDict:

    SHP, MHP, _THP, DHP, EHP = map(vars(HP).__getitem__, ("system", "model", "train", "dataset", "experiment"))

    # DONE: Use list of training functions specified by the model
    def DEFAULT_TRAINING_FUNC(
            exclusive_: Namespace,
            ensembled_learned_kfs_: TensorDict[str, torch.Tensor],
            cache_: Namespace
    ):
        return _train_default(THP, exclusive_, ensembled_learned_kfs_, cache_)
    training_funcs: List[TrainFunc] = MHP.model.train_func_list(DEFAULT_TRAINING_FUNC)

    results = []
    for training_func in training_funcs:
        print(f'Training function {training_func.__name__}{signature(training_func)}')
        print("-" * 160)

        # Create optimizer
        THP = copy.deepcopy(_THP)
        done, cache = False, Namespace(t=0)

        while not done:
            # DONE: Set up caching for metrics
            metric_cache = {}
            metrics: set = getattr(EHP, "metrics", {
                "overfit",
                "validation",
                "validation_target",
                "validation_analytical",
                "impulse_target",
                "overfit_gradient_norm"
            } - getattr(EHP, "ignore_metrics", set()))
            if THP.epochs is None:
                metrics.add("overfit_gradient_norm")

            # DONE: Compute necessary metrics (overfit, validation, gradient norm, impulse response difference)
            r = TensorDict({
                m: Metrics[m].evaluate(
                    (HP, exclusive, ensembled_learned_kfs),
                    metric_cache, sweep_position="inside", with_batch_dim=False
                ).detach()
                for m in metrics
            }, batch_size=EHP.model_shape)

            # DONE: Train on train dataset, passing in only train dataset and dataloader, and save the learning rate
            train_result, done = training_func(exclusive, ensembled_learned_kfs, cache)
            r["training"] = train_result.detach()[0]

            # DONE: Check if training uses an LR scheduler, otherwise log the LR as NaN
            if hasattr(cache, "optimizer"):
                lr = cache.optimizer.param_groups[0]["lr"]
            else:
                lr = torch.nan
            r["learning_rate"] = torch.full(r.shape, lr)

            # DONE: Reshape the result and append to results
            results.append(r)

            # DONE: Print losses
            if ((cache.t - 1) % print_frequency == 0) or (training_func is not DEFAULT_TRAINING_FUNC):
                mean_losses = [
                    (loss_type, r[loss_type].reshape(*EHP.model_shape, -1).mean(-1).median(-1).values.mean())
                    for loss_type in ("training", *(lt for lt in Metrics.keys() if lt in metrics))
                ]
                print(f"\tEpoch {cache.t - 1} --- {', '.join([f'{k}: {v:>8f}' for k, v in mean_losses])}, LR: {lr}")

            # DONE: Check for divergence
            if "overfit" in metrics:
                ol = torch.Tensor(r["overfit"])
                divergences = torch.isnan(ol) + torch.isinf(ol)
            else:
                divergences = torch.full(EHP.model_shape, False)
            if torch.any(divergences):
                raise RuntimeError("Model diverged")

            torch.cuda.empty_cache()

    return torch.stack(results, dim=2)

def _run_unit_training_experiment(
        HP: Namespace,
        info: Namespace
) -> Dict[str, TensorDict]:

    SHP, MHP, THP, DHP, EHP = map(vars(HP).__getitem__, ("system", "model", "train", "dataset", "experiment"))

    print("-" * 160)
    print("Hyperparameters:", json.dumps(utils.toJSON(HP), indent=4))
    print("=" * 160)

    # Ensemble learned Kalman Filters to be trained
    learned_kfs = utils.multi_map(
        lambda _: MHP.model(MHP),
        np.empty(EHP.model_shape), dtype=nn.Module
    )
    reference_module, ensembled_learned_kfs = utils.stack_module_arr(learned_kfs)

    # DONE: Create train dataset
    """ train_dataset = {
            'input': [n_experiments x ensemble_size x n_systems x dataset_size x sequence_length x I_D],
            'observation': [n_experiments x ensemble_size x n_systems x dataset_size x sequence_length x O_D]
        }
    """

    # DONE: Create train dataloader
    train_rem = DHP.train.dataset_size * DHP.train.sequence_length - DHP.train.total_sequence_length
    train_mask = torch.ones((DHP.train.dataset_size, DHP.train.sequence_length))
    if train_rem > 0:
        train_mask[-train_rem:, -1] = 0
    train_sequence_lengths = torch.sum(train_mask, dim=1)

    if THP.optim_type == "GD":
        THP.subsequence_length = DHP.train.sequence_length
        THP.batch_size = DHP.train.system.n_systems * DHP.train.dataset_size
    else:
        THP.subsequence_length = min(THP.subsequence_length, DHP.train.sequence_length)


    if THP.optim_type == "GD":
        train_index_dataset = TensorDict({
            "sequence": torch.arange(DHP.train.dataset_size),
            "start": torch.zeros((DHP.train.dataset_size,), dtype=torch.int),
            "stop": train_sequence_lengths
        }, batch_size=(DHP.train.dataset_size,))
    else:
        sequence_indices, start_indices = torch.where(train_mask[:, THP.subsequence_length - 1:])
        train_index_dataset = TensorDict({
            "sequence": sequence_indices,
            "start": start_indices,
            "stop": start_indices + THP.subsequence_length
        }, batch_size=(len(sequence_indices),))
    train_index_dataset = train_index_dataset.expand(DHP.train.system.n_systems, *train_index_dataset.shape)
    train_index_dataset["system"] = torch.arange(DHP.train.system.n_systems)[:, None].expand(*train_index_dataset.shape)
    train_index_dataset = train_index_dataset.flatten()

    index_batch_shape = (THP.iterations_per_epoch, EHP.n_experiments, EHP.ensemble_size, THP.batch_size)
    def supply_train_index_dataloader():
        if THP.optim_type == "GD":
            return train_index_dataset.expand(*index_batch_shape)
        else:
            return train_index_dataset[torch.randint(0, train_index_dataset.shape[0], index_batch_shape)]

    exclusive = Namespace(
        info=info,
        train_info=info.train[()],
        reference_module=reference_module,
        supply_train_index_dataloader=supply_train_index_dataloader,
        train_mask=train_mask,
        n_train_systems=DHP.train.system.n_systems
    )

    # Setup result and run training
    print(f"Mean theoretical irreducible loss " + "-" * 80)
    for ds_type, ds_info in vars(info).items():
        print(f"\t{ds_type}: {utils.multi_map(lambda il: il.obj.mean(), ds_info.irreducible_loss, dtype=float)}")

    return {
        "output": _run_training(HP, exclusive, ensembled_learned_kfs).detach(),
        "learned_kfs": (reference_module, ensembled_learned_kfs),
    }




