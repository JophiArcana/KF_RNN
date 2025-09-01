import copy
import json
from argparse import Namespace
from dataclasses import dataclass
from inspect import signature
from tqdm import tqdm
from typing import Any, Callable, Iterable

import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.experiment.metrics import Metrics
from infrastructure.settings import *
from infrastructure.utils import PTR
from model.base import Predictor


# Optimizer configuration
def _get_optimizer_and_scheduler(
        params: Iterable[tuple[str, torch.Tensor]],
        THP: Namespace
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:

    optimizer_params = THP.optimizer
    optimizer = utils.call_func_with_kwargs(
        getattr(optim, optimizer_params.type),
        (params, optimizer_params.max_lr), vars(optimizer_params)
    )

    scheduler_params = THP.scheduler
    scheduler_type = scheduler_params.type
    match scheduler_type:
        case "cosine":
            scheduler = utils.call_func_with_kwargs(
                optim.lr_scheduler.CosineAnnealingWarmRestarts,
                (optimizer, scheduler_params.T_0), vars(scheduler_params)
            )
            scheduler_params.epochs = scheduler_params.T_0 * ((scheduler_params.T_mult ** scheduler_params.num_restarts - 1) // (scheduler_params.T_mult - 1))
        case "exponential":
            scheduler = utils.call_func_with_kwargs(
                optim.lr_scheduler.ExponentialLR,
                (optimizer, scheduler_params.lr_decay), vars(scheduler_params)
            )
        case "reduce_on_plateau":
            scheduler = utils.call_func_with_kwargs(
                optim.lr_scheduler.ReduceLROnPlateau,
                (optimizer,), vars(scheduler_params)
            )
        case _:
            raise ValueError(scheduler_type)

    if (warmup_duration := getattr(scheduler_params, "warmup_duration", 0)) == 0:
        return optimizer, scheduler
    else:
        scheduler_params.epochs += (warmup_duration - 1)
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=optimizer_params.min_lr / optimizer_params.max_lr,
            total_iters=warmup_duration,
        )
        return optimizer, optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, scheduler],
            milestones=[warmup_duration]
        )

# Training
TrainFunc = tuple[
    Callable[[
        Namespace,
        Namespace,
        TensorDict[str, torch.Tensor],
        Namespace
    ], tuple[torch.Tensor, dict[str, torch.Tensor],]],
    Callable[[
        Namespace,
        Namespace,
        TensorDict[str, torch.Tensor],
        Namespace
    ], bool],
]

def _sample_dataset_indices(
        dataset: TensorDict[str, torch.Tensor],                         # [N x E x S x B x L x ...]
        kwargs: dict[str, Any],
) -> Iterable[TensorDict[str, torch.Tensor]]:
    n_systems, n_traces, max_sequence_length = dataset.shape[-3:]
    model_shape = dataset.shape[:2]

    mask = dataset["mask"][0, 0]                                        # bool: [S x B x L]
    sequence_lengths = torch.sum(mask, dim=-1)                          # int: [S x B]

    if kwargs["subsequence_length"] is None:
        subsequence_length = sequence_lengths                           # int: [S x B]
    else:
        subsequence_length = torch.full_like(sequence_lengths, kwargs["subsequence_length"])

    stop_index_mask = mask                                              # bool: [S x B x L]
    method: str = kwargs["method"]
    if method is None:
        method = "subsequence_unpadded"
    assert method in ("subsequence_unpadded", "subsequence_padded",), f"method must be one of (`subsequence_unpadded`, `subsequence_padded`,) but got {method}"
    if method == "subsequence_unpadded":
        stop_index_mask = stop_index_mask * (torch.arange(max_sequence_length) >= (subsequence_length - 1)[..., None])

    weights = stop_index_mask.to(torch.float) + torch.rand(model_shape + stop_index_mask.shape) # float: [N x E x S x B x L]
    stop_indices = torch.argmax(weights, dim=-1) + 1                                            # int: [N x E x S x B]
    start_indices = stop_indices - subsequence_length                                           # int; [N x E x S x B]
    system_indices, sequence_indices = torch.meshgrid((torch.arange(n_systems), torch.arange(n_traces),), indexing="ij",)

    shape = (*model_shape, n_systems, n_traces,)
    index_td = TensorDict({
        "system": system_indices.expand(shape),
        "sequence": sequence_indices.expand(shape),
        "start": start_indices,
        "stop": stop_indices,
    }, batch_size=shape)                                                # int: [N x E x S x B]
    index_td = index_td.reshape((*model_shape, n_systems * n_traces,))  # int: [N x E x SB]

    permutation = torch.argsort(torch.rand(index_td.shape), dim=-1)     # int: [N x E x SB]
    index_td = TensorDict.gather(index_td, -1, permutation,)            # int: [N x E x SB]

    batch_size: int = kwargs["batch_size"]
    if batch_size is None:
        batch_size = n_systems * n_traces
    lo, hi = 0, n_systems * n_traces
    while lo < hi:
        yield index_td[..., lo:min(hi, lo + batch_size)]
        lo += batch_size


def _extract_dataset_from_indices(
        padded_train_dataset: TensorDict[str, torch.Tensor],
        indices: TensorDict[str, torch.Tensor],
) -> TensorDict[str, torch.Tensor]:
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
    system_idx = system_indices[..., None]
    sequence_idx = sequence_indices[..., None]
    subsequence_idx = (start_indices[..., None] + subsequence_offset).clamp_min(-1)

    """
        input: [N x E x B x L_SS x I_D]
        observation: [N x E x B x L_SS x O_D]
        target: [N x E x B x L_SS x O_D]
    """
    return padded_train_dataset[
        n_experiment_idx,
        ensemble_size_idx,
        system_idx,
        sequence_idx,
        subsequence_idx,
    ]

def TERMINATE_DEFAULT(
        THP: Namespace,
        exclusive: Namespace,
        ensembled_learned_kfs: TensorDict[str, torch.Tensor],
        cache: Namespace,
) -> bool:
    if THP.scheduler.epochs is None:
        reference_module = exclusive.reference_module.eval()
        with torch.set_grad_enabled(True):
            run = Predictor.run(reference_module, ensembled_learned_kfs, cache.padded_train_dataset)
            loss = Predictor.evaluate_run(
                run["environment", "observation"],
                cache.padded_train_dataset, ("environment", "observation")
            )
        grads = torch.autograd.grad(loss.sum(), cache.optimizer.param_groups[0]["params"])
        grad_norm = torch.Tensor(sum(
            grad.flatten(start_dim=2, end_dim=-1).norm(dim=2) ** 2
            for grad in grads
        )).mean()
        print(grad_norm)
        return grad_norm < THP.scheduler.gradient_cutoff
    else:
        return cache.t >= THP.scheduler.epochs

def TRAIN_DEFAULT(
        THP: Namespace,
        exclusive: Namespace,
        ensembled_learned_kfs: TensorDict[str, torch.Tensor],
        cache: Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # SECTION: Setup the index dataloader, optimizer, and scheduler before running iterative training
    if not hasattr(cache, "optimizer"):
        # TODO: Set up the dataset index sampler
        dataset: TensorDict[str, torch.tensor] = exclusive.train_info.dataset.obj
        cache.padded_train_dataset = TensorDict.cat([
            dataset, dataset[..., -1:].apply(torch.zeros_like)
        ], dim=-1)

        # DONE: Need this line because some training functions replace the parameters with untrainable tensors (to preserve gradients)
        for k, v in Predictor.clone_parameter_state(exclusive.reference_module, ensembled_learned_kfs).items():
            ensembled_learned_kfs[k] = v
        cache.optimizer, cache.scheduler = _get_optimizer_and_scheduler((
            (k, v) for (k, v) in ensembled_learned_kfs.items(include_nested=True, leaves_only=True)
            if isinstance(v, nn.Parameter)
        ), THP)

    # SECTION: Iterate through train indices and run gradient descent
    result = []
    for indices in tqdm(_sample_dataset_indices(exclusive.train_info.dataset.obj, vars(THP.sampling),)):
        # DONE: Use indices to compute the mask for truncation and padding
        dataset_ss = _extract_dataset_from_indices(cache.padded_train_dataset, indices)

        # DONE: Run test on the resulting subsequence block, calculate training loss, and return gradient step
        reference_module: Predictor = exclusive.reference_module.train()

        pre_runs: list[torch.Tensor] = []
        def compute_losses() -> torch.Tensor:
            if len(pre_runs) == 0:
                with torch.set_grad_enabled(True):
                    result_ss = Predictor.run(reference_module, ensembled_learned_kfs, dataset_ss)
                    return reference_module.compute_losses(result_ss, dataset_ss, THP)
            else:
                return pre_runs.pop()

        def closure() -> float:
            cache.optimizer.zero_grad()
            loss = torch.sum(compute_losses())
            loss.backward()
            for p in cache.optimizer.param_groups[0]["params"]:
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            return loss.item()

        pre_runs.append(_losses := compute_losses())
        result.append(_losses)

        cache.optimizer.step(closure)

    result = torch.stack(result, dim=0)
    log = {
        "learning_rate": torch.tensor(cache.optimizer.param_groups[0]["lr"]),
        # "imag": sum(torch.norm(v.imag) ** 2 for v in ensembled_learned_kfs.values(include_nested=True, leaves_only=True)),
    }
    if cache.optimizer.param_groups[0]["lr"] >= THP.optimizer.min_lr:
        utils.call_func_with_kwargs(cache.scheduler.step, (), {"metrics": result.mean(-1).median(-1).values.mean().item()})

    return result, log,


@dataclass
class Checkpoint:
    training_func_idx: int
    cache: Namespace
    ensembled_learned_kfs: TensorDict[str, torch.Tensor]
    results: list[TensorDict[str, torch.Tensor]]


# Full training scheme
def _run_training(
        HP: Namespace,
        exclusive: Namespace,
        ensembled_learned_kfs: TensorDict[str, torch.Tensor],   # [N x E x ...]
        checkpoint_paths: list[str],
) -> TensorDict:
    THP, EHP = HP.training, HP.experiment,

    # TODO: Check if checkpoint exists and if so, load the stored information
    checkpoint: Checkpoint = None
    for checkpoint_path in checkpoint_paths:
        try:
            checkpoint = utils.torch_load(checkpoint_path)
            break
        except Exception:
            pass

    if checkpoint is not None:
        training_func_idx = checkpoint.training_func_idx
        cache = checkpoint.cache
        results = checkpoint.results

        if hasattr(cache, "optimizer"):
            # TODO: If an optimizer is used for training, then ensembled_learned_kfs needs to reference the parameters optimized by the optimizer
            optimizer: optim.Optimizer = cache.optimizer
            checkpoint_params = zip(optimizer.param_groups[0]["param_names"], optimizer.param_groups[0]["params"],)
        else:
            # TODO: Otherwise, copying the values stored by the checkpointed ensembled_learned_kfs is sufficient
            checkpoint_params = checkpoint.ensembled_learned_kfs.items(include_nested=True, leaves_only=True)
       
        for k, v in checkpoint_params:
            ensembled_learned_kfs[k] = v
        
        print(f"Checkpoint loaded starting from epoch {cache.t}")
    else:
        training_func_idx = 0
        cache = Namespace(t=0)
        results = []

    # DONE: Use list of training functions specified by the model
    reference_module: Predictor = exclusive.reference_module
    training_funcs: list[TrainFunc] = reference_module.train_func_list((TRAIN_DEFAULT, TERMINATE_DEFAULT,))[training_func_idx:]

    # DONE: List the metrics that need to be computed
    metrics: set = utils.rgetattr(EHP, "metrics.training", {
        "overfit",
        "validation",
        "validation_target",
        "validation_analytical",
        "impulse_target",
        "overfit_gradient_norm"
    } - utils.rgetattr(EHP, "ignore_metrics.training", set()))

    def evaluate_metrics() -> TensorDict[str, torch.Tensor]:
        reference_module.eval()
        metric_cache = {}
        return TensorDict({
            m: Metrics[m].evaluate(
                (exclusive, ensembled_learned_kfs), metric_cache,
                sweep_position="inside", with_batch_dim=False,
            ).detach()
            for m in metrics
        }, batch_size=EHP.model_shape,)
    
    def print_losses(r: TensorDict[str, torch.Tensor], prefix: str, keys: Iterable[str]) -> None:
        mean_losses = [
            (loss_type, r[loss_type].reshape((*EHP.model_shape, -1,)).mean(dim=-1).median(dim=-1).values.mean())
            for loss_type in keys
        ]
        print(f"\t{prefix} --- {', '.join([f'{k}: {v:>9.6f}' for k, v in mean_losses])}")

    # TODO: Run training functions starting from checkpoint if it exists
    for idx, (training_func, _terminate_func,) in enumerate(training_funcs, start=training_func_idx):
        print(f'Training function {training_func.__name__}{signature(training_func)}')
        print("-" * 160)

        # Create optimizer
        if idx != training_func_idx:
            cache = Namespace(t=0)

        def terminate_func(THP, exclusive, ensembled_learned_kfs, cache):
            try:
                return _terminate_func(THP, exclusive, ensembled_learned_kfs, cache)
            except AttributeError:
                return False

        while not terminate_func(THP, exclusive, ensembled_learned_kfs, cache):
            # DONE: Compute necessary metrics (overfit, validation, gradient norm, impulse response difference)
            r = evaluate_metrics()
            
            # DONE: Train on train dataset, passing in only train dataset and dataloader, and save the learning rate
            reference_module.train()
            train_result, log = training_func(THP, exclusive, ensembled_learned_kfs, cache)

            r["training"] = train_result.detach()[0]

            # DONE: Check if training uses an LR scheduler, otherwise log the LR as NaN
            for k, v in log.items():
                r[k] = v.expand(r.shape)

            # DONE: Reshape the result and append to results
            results.append(r)

            # DONE: Print losses
            if (cache.t % EHP.print_frequency == 0) or (training_func is not TRAIN_DEFAULT):
                print_losses(r, f"Epoch {cache.t}", ("training", *metrics, *log.keys(),))
            cache.t += 1

            # TODO: Save checkpoint before running so that reloading and saving occur at the same stage
            if cache.t % EHP.checkpoint_frequency == 0:
                checkpoint = Checkpoint(
                    training_func_idx=idx,
                    cache=cache,
                    ensembled_learned_kfs=ensembled_learned_kfs,
                    results=results,
                )
                for checkpoint_path in checkpoint_paths:
                    torch.save(checkpoint, checkpoint_path)

            utils.empty_cache()
    
    results.append(r := evaluate_metrics())
    print_losses(r, "Final", metrics)

    # TODO: Delete the checkpoint so that it does not interfere with the next experiment
    for checkpoint_path in filter(os.path.exists, checkpoint_paths):
        os.remove(checkpoint_path)

    if len(results) > 0:
        return TensorDict.maybe_dense_stack(results, dim=2)
    else:
        return TensorDict({}, batch_size=(*EHP.model_shape, 0))

def _run_unit_training_experiment(
        HP: Namespace,
        info: Namespace,
        checkpoint_paths: list[str],
        initialization: TensorDict[str, torch.Tensor],
        print_hyperparameters: bool = False
) -> dict[str, Any]:

    MHP, DHP, EHP = map(vars(HP).__getitem__, ("model", "dataset", "experiment"))
    if print_hyperparameters:
        print("-" * 160)
        print("Hyperparameters:", json.dumps(utils.toJSON(HP), indent=4))
    print("=" * 160)

    # Ensemble learned Kalman Filters to be trained
    learned_kfs = utils.multi_map(
        lambda _: MHP.model(MHP),
        np.empty(EHP.model_shape), dtype=nn.Module,
    )
    reference_module, ensembled_learned_kfs = utils.stack_module_arr(learned_kfs)

    # TODO: Load the initialization
    for k, v in initialization.items(include_nested=True, leaves_only=True):
        if k in ensembled_learned_kfs.keys(include_nested=True, leaves_only=True):
            ensembled_learned_kfs[k].data = v.expand_as(ensembled_learned_kfs[k])

    # TODO: Slice the train dataset
    info.train.dataset = utils.multi_map(
        lambda dataset: PTR(utils.mask_dataset_with_total_sequence_length(
            dataset.obj[..., :DHP.n_systems.train, :DHP.n_traces.train, :DHP.sequence_length.train],
            DHP.total_sequence_length.train
        )), info.train.dataset, dtype=PTR,
    )

    # DONE: Create train dataloader
    exclusive = Namespace(
        info=info,
        train_info=info.train[()],
        reference_module=reference_module,
        n_train_systems=DHP.n_systems.train,
    )

    # Setup result and run training
    avg = lambda t: t.mean().item()
    for loss_type in ("zero_predictor_loss", "copy_predictor_loss", "irreducible_loss",):
        print(f"Mean {loss_type.replace('_', ' ')} {'-' * 80}")
        for ds_type, ds_info in vars(info).items():
            try:
                print(f"\t{ds_type}:", utils.multi_map(
                    lambda sg: utils.map_dict(sg.td()[loss_type], avg),
                    ds_info.systems, dtype=dict,
                ))
            except Exception:
                pass
    return {
        "output": PTR(_run_training(HP, exclusive, ensembled_learned_kfs, checkpoint_paths).detach()),
        "learned_kfs": (reference_module, ensembled_learned_kfs),
    }




