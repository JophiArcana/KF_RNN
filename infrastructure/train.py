import collections
from argparse import Namespace
from inspect import signature
from typing import *

import ignite.handlers.param_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tensordict import TensorDict

from infrastructure.settings import dev_type
from model.kf import KF


# Optimizer configuration
def get_optimizer(params, hp):
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
    Dict[str, Any],
    Dict[str, nn.Parameter],
    optim.Optimizer
], Tuple[torch.Tensor, bool]]


def train(
        hp: Namespace,
        exclusive: Dict[str, Any],
        flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
        optimizer: optim.Optimizer
) -> Tuple[torch.Tensor, bool]:
    mhp, thp, ehp = hp.model, hp.train, hp.experiment
    n_models = ehp.n_systems * ehp.ensemble_size

    result = []
    for batch, indices in enumerate(exclusive['training_dataloader']()):
        # DONE: Use indices to index into datasets and retrieve subsequences
        indices = indices.reshape(n_models, -1)
        sequence_indices, start_indices, stop_indices = map(torch.Tensor, (
            indices['sequence'],
            indices['start'],
            indices['stop']
        ))  # [NE x B]

        """ Smart indexing
            subsequence_length: L_SS
            model_index: [NE x 1 x 1], Selects individual model
            sequence_index: [NE x B x 1], Selects which trace within the dataset
            subsequence_index: [NE x B x L_SS], Selects subsequence within the trace
        """
        subsequence_length = torch.max(stop_indices - start_indices)
        subsequence_offset = torch.arange(subsequence_length.item(), device=dev_type)

        model_idx = torch.arange(n_models)[:, None, None]
        sequence_idx = sequence_indices[:, :, None]
        subsequence_idx = start_indices[:, :, None] + subsequence_offset

        """
            state: [NE x B x S_D]
            input: [NE x B x L_SS x I_D]
            observation: [NE x B x L_SS x O_D]
        """
        dataset_ss = exclusive['training_dataset'][model_idx, sequence_idx, subsequence_idx]

        # DONE: Use indices to compute the mask for truncation and padding
        mask = torch.Tensor(subsequence_idx < stop_indices[:, :, None])

        # DONE: Run test on the resulting subsequence block, calculate training loss, and return gradient step
        base_model = exclusive['base_model'].train()
        with torch.set_grad_enabled(True):
            train_result = base_model.run(dataset_ss, flattened_ensembled_learned_kfs)
        result.append(losses := KF.evaluate_run(train_result, dataset_ss['observation'], mask))

        optimizer.zero_grad()
        torch.sum(losses).backward()
        optimizer.step()

    return torch.stack(result, dim=-1), False


# Full training scheme
def run_training(
        hp: Namespace,
        shared: Dict[str, Any],
        exclusive: Dict[str, Any],
        flattened_ensembled_learned_kfs: Dict[str, torch.nn.Parameter],  # [NE x ...]
        print_frequency: int = 10
) -> TensorDict:
    shp, mhp, thp, ehp = hp.system, hp.model, hp.train, hp.experiment
    n_models = ehp.n_systems * ehp.ensemble_size

    # DONE: Check whether to use default or overridden training function provided by model
    training_funcs = getattr(mhp.model, 'train_override', lambda func: (func,))(train)

    results = []
    for training_func in training_funcs:
        print(f'Training function {training_func.__name__}{signature(training_func)}')
        print('-' * 200)

        # Create optimizer
        optimizer, scheduler = get_optimizer(tuple(flattened_ensembled_learned_kfs.values()), thp)

        done, t = False, 0
        done_matrix = torch.full((ehp.n_systems, ehp.ensemble_size), False, device=dev_type)

        # DONE: Compute masking
        def truncation_mask(n: int) -> torch.Tensor:
            return torch.Tensor(torch.arange(n, device=dev_type) >= thp.sequence_buffer)

        overfit_mask = truncation_mask(thp.train_sequence_length) * exclusive['training_mask']
        valid_mask = truncation_mask(thp.valid_sequence_length)

        base_model = exclusive['base_model'].eval()
        while not done:
            # DONE: Set up caching for metrics
            run_dict = dict()
            def get_or_run(key: str, dataset: TensorDict, grad_enabled: bool) -> torch.Tensor:
                if key not in run_dict:
                    with torch.set_grad_enabled(grad_enabled):
                        run_dict[key] = base_model.run(dataset, flattened_ensembled_learned_kfs)
                return run_dict[key]
            def get_evaluate_func(key: str, target_key: str, dataset: TensorDict, mask: torch.Tensor = None, grad_enabled: bool = False) -> Callable[[], torch.Tensor]:
                return lambda: KF.evaluate_run(get_or_run(key, dataset, grad_enabled), dataset[target_key], mask)

            metrics: set = getattr(ehp, 'metrics', {
                'overfit',
                'validation',
                'validation_target',
                'impulse_target',
                'gradient_norm'
            } - getattr(ehp, 'ignore_metrics', set()))
            if thp.epochs is None:
                metrics.add('gradient_norm')

            def gradient_norm_():
                optimizer.zero_grad()
                torch.sum(get_or_run('overfit', exclusive['training_dataset'], True)).backward()
                return torch.sum(torch.stack([
                    torch.norm(v.grad, dim=[1, 2]) ** 2
                    for v in flattened_ensembled_learned_kfs.values()]), dim=0)

            metric_dict: Dict[str, Callable[[], torch.Tensor]] = {
                'overfit': get_evaluate_func('overfit', 'observation', exclusive['training_dataset'], mask=overfit_mask, grad_enabled=(thp.epochs is None) or ('gradient_norm' in metrics)),
                'validation': get_evaluate_func('validation', 'observation', shared['validation_dataset'], mask=valid_mask),
                'validation_target': get_evaluate_func('validation', 'target', shared['validation_dataset'], mask=valid_mask),
                'impulse_target': get_evaluate_func('impulse', 'target', shared['impulse_dataset']),
                'gradient_norm': gradient_norm_
            }

            # DONE: Compute necessary metrics (overfit, validation, gradient norm, impulse response difference)
            r = TensorDict({m: metric_dict[m]().detach() for m in metrics}, batch_size=(n_models,))

            # DONE: Train on train dataset, passing in only train dataset and dataloader, and save the learning rate
            scheduler(None)
            train_result, done_override = training_func(hp, exclusive, flattened_ensembled_learned_kfs, optimizer)
            r['training'] = train_result.detach()[:, 0]
            r['learning_rate'] = torch.full((n_models,), (lr := optimizer.param_groups[0]['lr']))

            results.append(r := r.reshape(ehp.n_systems, ehp.ensemble_size))

            # DONE: Print losses
            if (training_func is train and t % print_frequency == 0) or (training_func is not train):
                mean_losses = collections.OrderedDict([
                    (loss_type, torch.Tensor(r[loss_type]).median(-1).values.mean())
                    for loss_type in ('training', 'overfit', 'validation') if loss_type in {'training'} | metrics
                ])
                print(f'\tEpoch {t} --- {", ".join([f"{k}: {v:>8f}" for k, v in mean_losses.items()])}, LR: {lr}')
                # for n in range(ehp.n_systems):
                #     print('\t' + ''.join(['\u25A0' if divergences[n, e] else '\u25A1' for e in range(ehp.ensemble_size)]))
            t += 1

            # DONE: Check for divergence
            if 'overfit' in metrics:
                ol = torch.Tensor(r['overfit'])
                divergences = torch.isnan(ol) + torch.isinf(ol)
            else:
                divergences = torch.full((ehp.n_systems, ehp.ensemble_size), False, device=dev_type)
            if torch.any(divergences):
                raise RuntimeError('Model diverged')

            # DONE: Check for convergence
            if thp.epochs is None:
                threshold = 0.01
                il, gn = shared['irreducible_loss'], torch.Tensor(r['gradient_norm'])
                convergence_metric = gn / (il[:, None] ** 2)
                convergences = convergence_metric < threshold
                torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
                print(f'Convergence metric:\n{convergence_metric} < {threshold}')

                done_matrix += convergences + divergences
                done = torch.all(done_matrix)
            else:
                done = t == thp.epochs
            done |= done_override

            torch.cuda.empty_cache()

    return torch.stack(results, dim=-1)




