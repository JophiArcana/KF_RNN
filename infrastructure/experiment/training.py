import copy
import json
import time
from argparse import Namespace
from inspect import signature
from typing import *

import ignite.handlers.param_scheduler
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.experiment.metrics import Metrics
from infrastructure.settings import *
from infrastructure.utils import PTR
from model.base import Predictor


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
        padded_train_dataset: TensorDict[str, torch.Tensor],
        indices: TensorDict[str, torch.Tensor],
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
    subsequence_offset = torch.arange(subsequence_length, dtype=torch.int)

    n_experiment_idx = torch.arange(indices.shape[0])[:, None, None, None]
    ensemble_size_idx = torch.arange(indices.shape[1])[None, :, None, None]
    system_idx = system_indices.unsqueeze(-1)
    sequence_idx = sequence_indices.unsqueeze(-1)
    subsequence_idx = (start_indices.unsqueeze(-1) + subsequence_offset).clamp_min(-1)

    """
        input: [N x E x B x L_SS x I_D]
        observation: [N x E x B x L_SS x O_D]
        target: [N x E x B x L_SS x O_D]
    """
    dataset_ss = padded_train_dataset[
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
        return cache.t >= THP.epochs

    # SECTION: Setup the index dataloader, optimizer, and scheduler before running iterative training
    if not hasattr(cache, "optimizer"):
        # TODO: Set up the dataset index sampler
        dataset = exclusive.train_info.dataset.obj
        dataset_size, sequence_length = dataset.shape[-2:]

        train_sequence_lengths = torch.sum(exclusive.train_mask, dim=1)
        if THP.optim_type == "GD":
            cache.subsequence_length = sequence_length
            cache.batch_size = exclusive.n_train_systems * dataset_size
            
            train_index_dataset = TensorDict({
                "sequence": torch.arange(dataset_size, dtype=torch.int),
                "start": torch.zeros((dataset_size,), dtype=torch.int),
                "stop": train_sequence_lengths
            }, batch_size=(dataset_size,))
        else:
            cache.subsequence_length = THP.subsequence_length
            cache.batch_size = THP.batch_size
            
            sequence_indices, stop_indices_n1 = torch.where(exclusive.train_mask)
            train_index_dataset = TensorDict({
                "sequence": sequence_indices,
                "start": stop_indices_n1 - cache.subsequence_length + 1,
                "stop": stop_indices_n1 + 1
            }, batch_size=(len(sequence_indices),))
        train_index_dataset = train_index_dataset.expand(exclusive.n_train_systems, *train_index_dataset.shape)
        train_index_dataset["system"] = torch.arange(exclusive.n_train_systems)[:, None].expand(*train_index_dataset.shape)

        cache.train_index_dataset = train_index_dataset.flatten()
        cache.index_batch_shape = (THP.iterations_per_epoch, *ensembled_learned_kfs.shape, cache.batch_size)
        cache.padded_train_dataset = TensorDict({
            k: torch.cat([v, torch.zeros((*dataset.shape[:-1], 1, v.shape[-1]))], dim=-2)
            for k, v in exclusive.train_info.dataset.obj.items()
        }, batch_size=(*dataset.shape[:-1], dataset.shape[-1] + 1))

        # DONE: Need this line because some training functions replace the parameters with untrainable tensors (to preserve gradients)
        for k, v in Predictor.clone_parameter_state(exclusive.reference_module, ensembled_learned_kfs).items():
            ensembled_learned_kfs[k] = v
        cache.optimizer, cache.scheduler = _get_optimizer((v for v in ensembled_learned_kfs.values() if isinstance(v, nn.Parameter)), THP)

    cache.scheduler(None)


    # SECTION: At start of each epoch, generate the training indices for that epoch
    if THP.optim_type == "GD":
        train_index_dataloader = cache.train_index_dataset.expand(*cache.index_batch_shape)
    else:
        train_index_dataloader = cache.train_index_dataset[torch.randint(0, cache.train_index_dataset.shape[0], cache.index_batch_shape)]


    # SECTION: Iterate through train indices and run gradient descent
    result = []
    for batch, indices in enumerate(train_index_dataloader):
        start_t = time.perf_counter()

        # DONE: Use indices to compute the mask for truncation and padding
        dataset_ss, mask_ss = _extract_dataset_and_mask_from_indices(cache.padded_train_dataset, indices)
        mask_ss *= (torch.arange(cache.subsequence_length) >= getattr(THP, "sequence_buffer", 0))
        
        # DONE: Run test on the resulting subsequence block, calculate training loss, and return gradient step
        reference_module = exclusive.reference_module.train()
        with torch.set_grad_enabled(True):
            train_result = Predictor.run(reference_module, ensembled_learned_kfs, dataset_ss)

        losses = Predictor.evaluate_run(
            train_result["observation_estimation"],
            dataset_ss["observation"], mask_ss
        )
        if "input_estimation" in train_result.keys():
            losses = losses + Predictor.evaluate_run(
                train_result["input_estimation"],
                dataset_ss["input"], mask_ss
            )
        result.append(losses)

        cache.optimizer.zero_grad()
        torch.sum(losses).backward()
        for p in cache.optimizer.param_groups[0]["params"]:
            if p.grad is not None:
                p.grad.nan_to_num_()
        cache.optimizer.step()

        end_t = time.perf_counter()
        # print(f"Time for forward and backward pass: {end_t - start_t}s")
        torch.cuda.empty_cache()

    cache.t += 1

    return torch.stack(result, dim=0), terminate_condition()

# Full training scheme
def _run_training(
        HP: Namespace,
        exclusive: Namespace,
        ensembled_learned_kfs: TensorDict[str, torch.Tensor],   # [N x E x ...]
        checkpoint_paths: List[str],
        checkpoint_frequency: int = 100,
        print_frequency: int = 1
) -> TensorDict:

    SHP, MHP, _THP, DHP, EHP = map(vars(HP).__getitem__, ("system", "model", "train", "dataset", "experiment"))

    # TODO: Check if checkpoint exists and if so, load the stored information
    checkpoint = None
    if checkpoint_paths is not None:
        for checkpoint_path in filter(os.path.exists, checkpoint_paths):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                break
            except RuntimeError:
                pass

    if checkpoint is not None:
        training_func_idx, cache, exclusive.reference_module, results = map(checkpoint.__getitem__, (
            "training_func_idx",
            "cache",
            "reference_module",
            "results"
        ))
        # TODO: If an optimizer is used for training, then ensembled_learned_kfs needs to reference the parameters optimized by the optimizer
        if hasattr(cache, "optimizer"):
            checkpoint_params = cache.optimizer.param_groups[0]["params"]
        # TODO: Otherwise, copying the values stored by the checkpointed ensembled_learned_kfs is sufficient
        else:
            checkpoint_params = checkpoint["ensembled_learned_kfs"].values()
        for checkpoint_v, (k, v) in zip(checkpoint_params, ensembled_learned_kfs.items()):
            ensembled_learned_kfs[k] = checkpoint_v
    else:
        training_func_idx = 0
        cache = Namespace(t=0)
        results = []

    # DONE: Use list of training functions specified by the model
    def DEFAULT_TRAINING_FUNC(
            exclusive_: Namespace,
            ensembled_learned_kfs_: TensorDict[str, torch.Tensor],
            cache_: Namespace
    ):
        return _train_default(THP, exclusive_, ensembled_learned_kfs_, cache_)
    training_funcs: List[TrainFunc] = MHP.model.train_func_list(DEFAULT_TRAINING_FUNC)[training_func_idx:]

    # TODO: Run training functions starting from checkpoint if it exists
    counter = 1
    for idx, training_func in enumerate(training_funcs, start=training_func_idx):
        print(f'Training function {training_func.__name__}{signature(training_func)}')
        print("-" * 160)

        # Create optimizer
        THP = copy.deepcopy(_THP)
        done = False
        if idx != training_func_idx:
            cache = Namespace(t=0)

        while not done:
            start_t = time.perf_counter()

            # TODO: Save checkpoint before running so that reloading and saving occur at the same stage
            checkpoint = {
                "training_func_idx": idx,
                "cache": cache,
                "reference_module": exclusive.reference_module,
                "ensembled_learned_kfs": ensembled_learned_kfs,
                "results": results
            }
            if checkpoint_paths is not None and counter % checkpoint_frequency == 0:
                for checkpoint_path in checkpoint_paths:
                    torch.save(checkpoint, checkpoint_path)

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
            exclusive.reference_module.eval()
            r = TensorDict({
                m: Metrics[m].evaluate(
                    (HP, exclusive, ensembled_learned_kfs),
                    metric_cache, sweep_position="inside", with_batch_dim=False
                ).detach()
                for m in metrics
            }, batch_size=EHP.model_shape)

            # DONE: Train on train dataset, passing in only train dataset and dataloader, and save the learning rate
            exclusive.reference_module.train()
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

            counter += 1
            torch.cuda.empty_cache()

            end_t = time.perf_counter()
            # print(f"Time to run one epoch: {end_t - start_t}s")

    # TODO: Delete the checkpoint so that it does not interfere with the next experiment
    if checkpoint_paths is not None:
        for checkpoint_path in filter(os.path.exists, checkpoint_paths):
            os.remove(checkpoint_path)

    if len(results) > 0:
        return torch.stack(results, dim=2)
    else:
        return torch.empty(())

def _run_unit_training_experiment(
        HP: Namespace,
        info: Namespace,
        checkpoint_paths: List[str]
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

    # TODO: Slice the train dataset
    info.train.dataset = utils.multi_map(
        lambda dataset: PTR(dataset.obj[
                            :, :, :,
                            :DHP.train.dataset_size,
                            :DHP.train.sequence_length
        ]), info.train.dataset, dtype=PTR
    )

    # DONE: Create train dataloader
    train_rem = DHP.train.dataset_size * DHP.train.sequence_length - DHP.train.total_sequence_length
    train_mask = torch.ones((DHP.train.dataset_size, DHP.train.sequence_length))
    if train_rem > 0:
        train_mask[-train_rem:, -1] = 0

    exclusive = Namespace(
        info=info,
        reference_module=reference_module,
        train_info=info.train[()],
        train_mask=train_mask,
        n_train_systems=DHP.train.system.n_systems
    )

    # Setup result and run training
    print(f"Mean theoretical irreducible loss " + "-" * 80)
    for ds_type, ds_info in vars(info).items():
        print(f"\t{ds_type}: {utils.multi_map(lambda il: il.obj.mean(), ds_info.irreducible_loss, dtype=float)}")

    return {
        "output": _run_training(HP, exclusive, ensembled_learned_kfs, checkpoint_paths).detach(),
        "learned_kfs": (reference_module, ensembled_learned_kfs),
    }




