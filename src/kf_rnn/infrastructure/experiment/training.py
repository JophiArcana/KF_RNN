import collections
import itertools
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from tqdm import tqdm
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.infrastructure.config import schema
from kf_rnn.infrastructure.config.schema import ExperimentConfig, RuntimeConfig, TrainConfig
from kf_rnn.infrastructure.experiment.engine import error
from kf_rnn.infrastructure.experiment.losses import LOSS_DICT, LossFn
from kf_rnn.infrastructure.experiment.metrics import METRIC_DICT
import kf_rnn.infrastructure.settings  # noqa: F401  (imported for global device/dtype/precision side effects)
from kf_rnn.infrastructure.static import ModelPair
from kf_rnn.infrastructure.experiment.stages import (
    TrainingContext, TrainingStage, build_stages, register_stage,
)
from kf_rnn.model.base import Predictor


# Optimizer configuration
def _get_optimizer_and_scheduler(
        params: Iterable[tuple[str, torch.Tensor]],
        THP: TrainConfig
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:

    optimizer_params = THP.optimizer
    # torch optimizers (<2.5) do not accept ``(name, tensor)`` pairs, so split the
    # named parameters into plain tensors and attach the names to the param group.
    named_params = list(params)
    param_names = [k for k, _ in named_params]
    param_tensors = [v for _, v in named_params]
    optimizer = eu.call_func_with_kwargs(
        getattr(optim, optimizer_params.type),
        (param_tensors, optimizer_params.max_lr), vars(optimizer_params)
    )
    optimizer.param_groups[0]["param_names"] = param_names

    scheduler_params = THP.scheduler
    scheduler_type = scheduler_params.type
    match scheduler_type:
        case "cosine":
            scheduler = eu.call_func_with_kwargs(
                optim.lr_scheduler.CosineAnnealingWarmRestarts,
                (optimizer, scheduler_params.T_0), vars(scheduler_params)
            )
            scheduler_params.epochs = scheduler_params.T_0 * ((scheduler_params.T_mult ** scheduler_params.num_restarts - 1) // (scheduler_params.T_mult - 1))
        case "exponential":
            scheduler = eu.call_func_with_kwargs(
                optim.lr_scheduler.ExponentialLR,
                (optimizer, scheduler_params.lr_decay), vars(scheduler_params)
            )
        case "reduce_on_plateau":
            scheduler = eu.call_func_with_kwargs(
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
def _sample_dataset_indices(
        dataset: TensorDict,                         # [N x E x S x B x L x ...]
        kwargs: dict[str, Any],
) -> Iterable[TensorDict]:
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
        padded_train_dataset: TensorDict,
        indices: TensorDict,
) -> TensorDict:
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

def get_loss_fn(THP: TrainConfig) -> LossFn:
    loss_fn: LossFn = THP.loss
    if not callable(loss_fn):
        loss_fn = LOSS_DICT[loss_fn]
    return loss_fn

class SGDStage(TrainingStage):
    """Default subsequence-SGD training stage (universal).

    Sets up an optimizer/scheduler and padded train dataset on the first ``step``;
    typed mutable state (``optimizer``, ``scheduler``, ``padded_train_dataset``,
    ``t``) lives on ``self._state`` so the checkpoint-resume path can re-point the
    stacked parameters at the optimizer's tensors.
    """

    name = "sgd"

    def is_done(self, ctx: TrainingContext) -> bool:
        THP, model_pair = ctx.thp, ctx.model_pair
        cache = self._state
        if THP.scheduler.epochs is None:
            # The gradient-cutoff criterion depends on the optimizer and padded dataset
            # that the first step sets up, so never terminate before that happens.
            if not hasattr(cache, "optimizer"):
                return False
            model_pair[0].eval()
            loss_fn = get_loss_fn(THP)

            with torch.set_grad_enabled(True):
                result = Predictor.run(model_pair, cache.padded_train_dataset)
                losses = loss_fn(result, cache.padded_train_dataset, vars(THP))
                mask = cache.padded_train_dataset.get("mask", torch.full(cache.padded_train_dataset.shape[-2:], True))
                loss = torch.sum(losses * mask, dim=[-2, -1,]) / torch.sum(mask, dim=[-2, -1,])

            grads = torch.autograd.grad(loss.sum(), cache.optimizer.param_groups[0]["params"])
            grad_norm = torch.stack([
                grad.flatten(start_dim=2, end_dim=-1).norm(dim=2) ** 2
                for grad in grads
            ]).sum(dim=0).mean()
            print(grad_norm)
            return grad_norm < THP.scheduler.gradient_cutoff
        else:
            return cache.t >= THP.scheduler.epochs

    def step(self, ctx: TrainingContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        THP, exclusive, model_pair = ctx.thp, ctx.exclusive, ctx.model_pair
        cache = self._state

        # SECTION: Setup the index dataloader, optimizer, and scheduler before iterating
        if not hasattr(cache, "optimizer"):
            dataset: TensorDict = exclusive.train_info.dataset
            cache.padded_train_dataset = TensorDict.cat([
                dataset, dataset[..., -1:].apply(torch.zeros_like)
            ], dim=-1)

            # Some stages replace parameters with untrainable tensors; re-clone so the
            # optimizer below owns trainable leaves.
            for k, v in Predictor.clone_parameter_state(model_pair)[1].items():
                model_pair[1][k] = v
            cache.optimizer, cache.scheduler = _get_optimizer_and_scheduler((
                (k, v) for (k, v) in model_pair[1].items(include_nested=True, leaves_only=True)
                if isinstance(v, nn.Parameter)
            ), THP)

        # SECTION: Iterate through train indices and run gradient descent
        result = []
        loss_fn: LossFn = get_loss_fn(THP)

        for indices in tqdm(_sample_dataset_indices(exclusive.train_info.dataset, vars(THP.sampling),)):
            dataset_ss = _extract_dataset_from_indices(cache.padded_train_dataset, indices)

            model_pair[0].train()
            pre_runs: list[torch.Tensor] = []
            def compute_losses() -> torch.Tensor:
                if len(pre_runs) == 0:
                    with torch.set_grad_enabled(True):
                        result_ss = Predictor.run(model_pair, dataset_ss)
                        losses = loss_fn(result_ss, dataset_ss, vars(THP))
                        mask = dataset_ss.get("mask", torch.full(dataset_ss.shape[-2:], True))
                        return torch.sum(losses * mask, dim=[-2, -1,]) / torch.sum(mask, dim=[-2, -1,])
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
        }
        if cache.optimizer.param_groups[0]["lr"] >= THP.optimizer.min_lr:
            eu.call_func_with_kwargs(cache.scheduler.step, (), {"metrics": result.mean(-1).median(-1).values.mean().item()})

        return result, log,


# The universal SGD stage, referenceable by name in any model's training recipe.
register_stage("sgd", SGDStage, requires=lambda model: True, description="universal")


@dataclass
class Checkpoint:
    stage_idx: int
    stage_state: SimpleNamespace
    stacked_modules: TensorDict
    results: list[TensorDict]


# Full training scheme
def _run_training(
        HP: ExperimentConfig,
        exclusive: SimpleNamespace,
        model_pair: ModelPair,  # [N x E x ...]
        checkpoint_paths: list[str],
) -> TensorDict:
    THP, EHP = HP.training, HP.experiment,

    # Build the model's training recipe into a sequence of stages and a single
    # typed context (replacing the (train, terminate) callable pairs + cache).
    reference_module: Predictor = model_pair[0]
    stages = build_stages(reference_module.training_recipe(), reference_module)
    ctx = TrainingContext(THP, exclusive, model_pair)

    # Check if a checkpoint exists and if so, restore stage index, stage state, and params.
    checkpoint: Checkpoint = None
    for checkpoint_path in checkpoint_paths:
        try:
            checkpoint = eu.torch_load(checkpoint_path)
            break
        except FileNotFoundError:
            pass
        except (RuntimeError, EOFError, OSError, AttributeError, ModuleNotFoundError) as e:
            print(f"WARNING: failed to load checkpoint from {checkpoint_path}: {e}")

    if checkpoint is not None:
        start_stage_idx = checkpoint.stage_idx
        cache = checkpoint.stage_state
        results = checkpoint.results
        stages[start_stage_idx].load_state(cache)

        if hasattr(cache, "optimizer"):
            # The optimizer owns the parameter tensors, so re-point stacked_modules at them.
            optimizer: optim.Optimizer = cache.optimizer
            checkpoint_params = zip(optimizer.param_groups[0]["param_names"], optimizer.param_groups[0]["params"],)
        else:
            checkpoint_params = checkpoint.stacked_modules.items(include_nested=True, leaves_only=True)

        for k, v in checkpoint_params:
            model_pair[1][k] = v

        print(f"Checkpoint loaded starting from epoch {cache.t}")
    else:
        start_stage_idx = 0
        results = []

    # Metrics computed each epoch (overfit, validation, gradient norm, impulse response).
    metrics: set = EHP.metrics.training
    if metrics is None:
        metrics = {
            "overfit",
            "validation",
            "validation_target",
            "validation_analytical",
            "impulse_target",
            "overfit_gradient_norm"
        } - (EHP.ignore_metrics.training or set())

    def evaluate_metrics() -> TensorDict:
        reference_module.eval()
        metric_cache = {}
        return TensorDict({
            m: METRIC_DICT[m].evaluate(
                (exclusive, model_pair), metric_cache,
                sweep_position="inside", with_batch_dim=False,
            ).detach()
            for m in metrics
        }, batch_size=EHP.model_shape,)
    
    def print_losses(r: TensorDict, prefix: str, keys: Iterable[str]) -> None:
        mean_losses = [
            (loss_type, r[loss_type].reshape((*EHP.model_shape, -1,)).mean(dim=-1).median(dim=-1).values.mean())
            for loss_type in keys
        ]
        print(f"\t{prefix} --- {', '.join([f'{k}: {v:>9.6f}' for k, v in mean_losses])}")

    # Run the training stages, resuming from the checkpointed stage if present.
    for idx in range(start_stage_idx, len(stages)):
        stage = stages[idx]
        print("-" * 160)
        print(f"Training stage {stage.name}")
        print("-" * 160)

        # Reset per-stage state for stages that start fresh (the resumed stage keeps its state).
        if idx != start_stage_idx:
            stage.reset()
        is_sgd_stage = isinstance(stage, SGDStage)

        while not stage.is_done(ctx):
            r = evaluate_metrics()

            reference_module.train()
            train_result, log = stage.step(ctx)

            r["training"] = train_result.detach()[0]

            # Log scheduler-reported values (e.g. learning rate) alongside the metrics.
            for k, v in log.items():
                r[k] = v.expand(r.shape)

            results.append(r)

            # Print every epoch for non-SGD stages, else at the configured frequency.
            # A ``None`` frequency disables periodic printing for SGD stages.
            if (not is_sgd_stage) or (EHP.print_frequency is not None and stage.t % EHP.print_frequency == 0):
                print_losses(r, f"Epoch {stage.t}", ("training", *metrics, *log.keys(),))
            stage.t += 1

            # Save checkpoint before stepping so reloading and saving align on the same stage.
            # A ``None`` frequency disables checkpointing.
            if EHP.checkpoint_frequency is not None and stage.t % EHP.checkpoint_frequency == 0:
                checkpoint = Checkpoint(
                    stage_idx=idx,
                    stage_state=stage.state,
                    stacked_modules=model_pair[1],
                    results=results,
                )
                for checkpoint_path in checkpoint_paths:
                    torch.save(checkpoint, checkpoint_path)

            eu.empty_cache()
    
    r = evaluate_metrics()
    # Models with an empty training recipe (e.g. ``ZeroPredictor``) run no stages,
    # so ``results`` can be empty; only backfill train-only keys when present.
    if results:
        for k, v in results[0].items(include_nested=True, leaves_only=True):
            if k not in r:
                r[k] = torch.full_like(v, torch.nan)
    results.append(r)
    print_losses(r, "Final", metrics)

    # Delete the checkpoint so it does not interfere with the next experiment.
    for checkpoint_path in filter(os.path.exists, checkpoint_paths):
        os.remove(checkpoint_path)

    if len(results) > 0:
        return TensorDict.maybe_dense_stack(results, dim=2)
    else:
        return TensorDict({}, batch_size=(*EHP.model_shape, 0))


def _build_model_ensemble(
        MHP: "Predictor.Config",
        model_shape: tuple[int, ...],
        initialization: TensorDict,
) -> ModelPair:
    """Construct the ensemble of models and load any provided initialization.

    The model class is identified by the config type itself (``Config.cls``)."""
    learned_kfs = eu.multi_map(
        lambda _: MHP.cls(MHP),
        np.empty(model_shape), dtype=nn.Module,
    )
    model_pair = eu.stack_module_arr(learned_kfs)

    for k, v in initialization.items(include_nested=True, leaves_only=True):
        if k in model_pair[1].keys(include_nested=True, leaves_only=True):
            model_pair[1][k].data = v.expand_as(model_pair[1][k])
    return model_pair


def _print_baseline_errors(info: SimpleNamespace) -> None:
    """Print zero/copy/irreducible baseline errors per dataset type for comparison."""
    loss_types = ("zero_predictor", "copy_predictor", "irreducible",)
    for ds_type, ds_info in vars(info).items():
        print(f"Mean loss for dataset type {ds_type} {'-' * 80}")
        evaluation_targets = collections.OrderedDict([
            ("analytical", ds_info.systems.values,),
            ("empirical", eu.multi_map(eu.identity, ds_info.dataset, dtype=TensorDict,).values,),
        ])
        shape = [*evaluation_targets.values()][0].shape
        assert all(arr.shape == shape for arr in evaluation_targets.values()), "Not sure when this is not true but we'll deal with it later."

        loss_arr_dict = {}
        for args in itertools.product(enumerate(evaluation_targets.items()), enumerate(loss_types),):
            idx, ((target_type, td_arr), loss_type,) = zip(*args)
            try:
                err_dict_arr = eu.multi_map(lambda td: error(loss_type, td), td_arr, dtype=TensorDict,)
                for k in err_dict_arr.ravel()[0].keys(include_nested=True, leaves_only=True):
                    loss_arr_dict.setdefault(k, np.full((len(evaluation_targets), len(loss_types), *shape,), None, dtype=object))[idx] = eu.multi_map(
                        lambda td: td[k].mean().item(), err_dict_arr, dtype=float,
                    )
            except (RuntimeError, ValueError, KeyError, TypeError, AssertionError):
                # Baseline-error printing is best-effort diagnostics; skip targets
                # that a given loss type does not support (e.g. an empty / zero-length
                # train dataset, which the ensemble reshape rejects).
                pass

        df_dict = {}
        for k, v in loss_arr_dict.items():
            eu.rsetitem(df_dict, ".".join(map(str.upper, k)), pd.DataFrame(v, evaluation_targets.keys(), loss_types))
        eu.print_dict(df_dict)


def _run_unit_training_experiment(
        HP: ExperimentConfig,
        info: SimpleNamespace,
        checkpoint_paths: list[str],
        initialization: TensorDict,
        print_hyperparameters: bool = False
) -> dict[str, Any]:

    MHP, DHP, EHP = HP.model, HP.dataset, HP.experiment,
    if print_hyperparameters:
        print("-" * 160)
        print("Hyperparameters:", json.dumps(schema.config_to_jsonable(HP), indent=4))
    print("=" * 160)

    model_pair = _build_model_ensemble(MHP, EHP.model_shape, initialization)

    # Slice the train dataset down to the configured train shape.
    info.train.dataset = eu.multi_map(
        lambda dataset: eu.mask_dataset_with_total_sequence_length(
            dataset[..., :DHP.n_systems.train, :DHP.n_traces.train, :DHP.sequence_length.train],
            DHP.total_sequence_length.train
        ), info.train.dataset, dtype=TensorDict,
    )

    exclusive = SimpleNamespace(
        info=info,
        train_info=info.train[()],
        n_train_systems=DHP.n_systems.train,
    )

    _print_baseline_errors(info)

    return {
        "output": _run_training(HP, exclusive, model_pair, checkpoint_paths).detach(),
        "learned_kfs": model_pair,
    }




