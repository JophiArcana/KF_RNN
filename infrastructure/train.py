from argparse import Namespace
import collections
import copy
import itertools
import json
import os
from types import MappingProxyType
from typing import *
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import ignite.handlers.param_scheduler
from tensordict import TensorDict, TensorDictBase

from infrastructure import utils
from model.linear_system import LinearSystem
from model.analytical_kf import AnalyticalKF
from model.kf import KF


# Optimizer configuration
def get_optimizer(params, hp):
    optim_type = hp.optim_type
    if optim_type == "GD" or optim_type == "SGD":
        optimizer = optim.SGD(params, lr=hp.max_lr, momentum=0.0, weight_decay=hp.l2_reg)
    elif optim_type == "SGDMomentum":
        optimizer = optim.SGD(params, lr=hp.max_lr, momentum=hp.momentum, weight_decay=hp.l2_reg)
    elif optim_type == "Adam":
        optimizer = optim.AdamW(params, lr=hp.max_lr, betas=(hp.momentum, 0.98), weight_decay=hp.l2_reg)
    else:
        raise ValueError(optim_type)
    scheduler = ignite.handlers.param_scheduler.create_lr_scheduler_with_warmup(
        optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, hp.T_0, T_mult=hp.T_mult, eta_min=hp.min_lr),
        hp.min_lr,
        hp.warmup_duration
    )
    return optimizer, scheduler


def run(
        dataset: TensorDict[str, torch.Tensor],
        base_module: KF,
        flattened_ensembled_learned_kfs: Dict[str, nn.Parameter]
) -> torch.Tensor:  # [NE]
    return utils.run_stacked_modules(
        base_module=base_module,
        stacked_modules=flattened_ensembled_learned_kfs,
        args=dict(dataset),
        kwargs=dict()
    )['observation_estimation']


def evaluate_run(result: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, batch_mean: bool = True) -> torch.Tensor:
    losses = torch.norm(result - target, dim=-1) ** 2
    if mask is not None:
        mask = mask.expand(losses.shape[-3:])
        result_ = torch.sum(losses * mask, dim=-1) / torch.sum(mask, dim=-1)
    else:
        result_ = torch.mean(losses, dim=-1)
    return result_.mean(-1) if batch_mean else result_


# Training
def train(
        hp: Namespace,
        exclusive: Dict[str, Any],
        flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
        optimizer: optim.Optimizer,
        dev_type: str,
) -> torch.Tensor:
    mhp, thp, ehp = hp.model, hp.train, hp.experiment
    n_models = ehp.n_systems * ehp.ensemble_size

    result = []
    for batch, indices in enumerate(exclusive['training_dataloader']()):
        # DONE: Use indices to index into datasets and retrieve subsequences
        indices = indices.reshape(n_models, -1)
        sequence_indices, start_indices, stop_indices = map(torch.tensor, (
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
            train_result = run(dataset_ss, base_model, flattened_ensembled_learned_kfs)
        result.append(losses := evaluate_run(train_result, dataset_ss['observation'], mask))

        optimizer.zero_grad()
        torch.sum(losses).backward()
        optimizer.step()

    return torch.stack(result, dim=-1)


def generate_dataset(
        systems: List[LinearSystem],
        batch_size: int,
        seq_length: int,
        dev_type: str
) -> TensorDict[str, torch.Tensor]:
    state = torch.randn(len(systems), batch_size, systems[0].S_D, device=dev_type)
    inputs = torch.randn(len(systems), batch_size, seq_length, systems[0].I_D, device=dev_type)
    observations = utils.run_stacked_modules(
        base_module=systems[0],
        stacked_modules=torch.func.stack_module_state(systems)[0],
        args=(state, inputs),
        kwargs=dict()
    )['observation']

    return TensorDict({
        'input': inputs,
        'observation': observations
    }, batch_size=(len(systems), batch_size, seq_length), device=dev_type)


def add_targets(
        kfs: List[AnalyticalKF],
        dataset: TensorDict[str, torch.Tensor] | TensorDictBase[str, torch.Tensor]
) -> TensorDict[str, torch.Tensor]:
    kfs[0].eval()
    with torch.set_grad_enabled(False):
        dataset['target'] = utils.run_stacked_modules(
            base_module=kfs[0],
            stacked_modules=torch.func.stack_module_state(kfs)[0],
            args=dict(dataset),
            kwargs={'steady_state': True}
        )['observation_estimation']
    return dataset


# Full training scheme
def run_training(
        hp: Namespace,
        shared: Dict[str, Any],
        exclusive: Dict[str, Any],
        flattened_ensembled_learned_kfs: Dict[str, torch.nn.Parameter],  # [NE x ...]
        dev_type: str
) -> TensorDict:
    shp, mhp, thp, ehp = hp.system, hp.model, hp.train, hp.experiment
    n_models = ehp.n_systems * ehp.ensemble_size

    # Create optimizer
    optimizer, scheduler = get_optimizer(tuple(flattened_ensembled_learned_kfs.values()), thp)

    results = []
    done, t = False, 0
    done_matrix = torch.full((ehp.n_systems, ehp.ensemble_size), False, device=dev_type)
    while not done:
        # DONE: Run learned KFs on the train, validation, IR datasets
        base_model = exclusive['base_model'].eval()
        with torch.set_grad_enabled(thp.epochs is None):
            overfit_result = run(exclusive['training_dataset'], base_model, flattened_ensembled_learned_kfs)
        with torch.set_grad_enabled(False):
            valid_result = run(shared['validation_dataset'], base_model, flattened_ensembled_learned_kfs)
            if shared['impulse_dataset'] is not None:
                ir_result = run(shared['impulse_dataset'], base_model, flattened_ensembled_learned_kfs)

        # DONE: Compute necessary metrics (overfit, validation, gradient norm, impulse response difference)
        def truncation_mask(n: int) -> torch.Tensor:
            return torch.Tensor(torch.arange(n, device=dev_type) >= thp.sequence_buffer)

        overfit_mask = truncation_mask(thp.train_sequence_length) * exclusive['training_mask']
        valid_mask = truncation_mask(thp.valid_sequence_length)

        r = TensorDict({
            'overfit': evaluate_run(overfit_result, exclusive['training_dataset']['observation'], overfit_mask),
            'validation': evaluate_run(valid_result, shared['validation_dataset']['observation'], valid_mask),
            'validation_target': evaluate_run(valid_result, shared['validation_dataset']['target'], valid_mask)
        }, batch_size=(n_models,))
        # DONE: Compute gradient norms and unit impulse responses
        if thp.epochs is None:
            optimizer.zero_grad()
            torch.sum(overfit_result).backward()
            r['gradient_norm'] = torch.sum(torch.stack([
                torch.norm(v.grad, dim=[1, 2]) ** 2
                for v in flattened_ensembled_learned_kfs.values()]), dim=0).detach()
        if shared['impulse_dataset'] is not None:
            r['impulse_target'] = evaluate_run(ir_result, shared['impulse_dataset']['target'])

        # DONE: Check whether to use default or overridden training function provided by model
        training_func = getattr(shared['model'], 'train_override', train)
        
        # DONE: Train on train dataset, passing in only train dataset and dataloader
        r['training'] = training_func(hp, exclusive, flattened_ensembled_learned_kfs, optimizer, dev_type)[:, 0]
        results.append(r := r.reshape(ehp.n_systems, ehp.ensemble_size, -1))
        
        scheduler(None)

        mean_losses = [r[loss_type].median(-1).values.mean() for loss_type in ('training', 'overfit', 'validation')]        
        print(f'Epoch {t + 1} --- Training loss: {mean_losses[0]:>8f}, Overfit loss: {mean_losses[1]:>8f}, Validation loss: {mean_losses[2]:>8f}')
        print(f'\tLR: {optimizer.param_groups[0]["lr"]}')

        ol = torch.Tensor(r['overfit'])
        divergences = torch.isnan(ol) + torch.isinf(ol)
        t += 1
        if thp.epochs is None:
            il = shared['irreducible_loss']
            gn = torch.Tensor(r['gradient_norm'])
            convergence_metric = gn / (il[:, None] ** 2)
            convergences = convergence_metric < 0.01
            torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
            print(f'Convergence metric:\n{convergence_metric} < 0.01')

            done_matrix += convergences + divergences
            done = torch.all(done_matrix)
        else:
            done = t == thp.epochs
        for n in range(ehp.n_systems):
            print('\t' + ''.join(['\u25A0' if divergences[n, e] else '\u25A1' for e in range(ehp.ensemble_size)]))

    return torch.stack(results, dim=-1)


def run_experiment(
        hp: Namespace,
        shared: Dict[str, Any],
        dev_type: str
) -> Dict[str, TensorDict]:
    shp, mhp, thp, ehp = hp.system, hp.model, hp.train, hp.experiment
    print('=' * 200)
    print(ehp.exp_name)
    print('=' * 200)
    print("Hyperparameters:", json.dumps(utils.toJSON(hp), indent=4))
    print('=' * 200)

    # Ensemble learned Kalman Filters to be trained
    n_models = ehp.n_systems * ehp.ensemble_size
    flattened_learned_kfs = [
        shared['model'](mhp).to(dev_type)
        for _ in range(n_models)]
    flattened_ensembled_learned_kfs = torch.func.stack_module_state(flattened_learned_kfs)[0]

    if getattr(thp, 'initialization', None) is not None:
        initialization = thp.initialization.expand(ehp.n_systems, ehp.ensemble_size).flatten()
        for k in flattened_ensembled_learned_kfs:
            if k in initialization.keys():
                flattened_ensembled_learned_kfs[k] = initialization[k].clone().to(dev_type).requires_grad_()

    # DONE: Create train dataset
    """ train_dataset = {
            'input': [NE x S_T x L_T x I_D],
            'observation': [NE x S_T x L_T x O_D]
        }
    """
    thp.train_sequence_length = (thp.total_train_sequence_length + thp.train_dataset_size - 1) // thp.train_dataset_size

    train_rem = thp.train_dataset_size * thp.train_sequence_length - thp.total_train_sequence_length
    train_mask = torch.full((thp.train_dataset_size, thp.train_sequence_length), True, device=dev_type)
    if train_rem > 0:
        train_mask[-train_rem:, -1] = False
    train_sequence_lengths = torch.sum(train_mask, dim=1)

    if thp.optim_type == 'GD':
        thp.subsequence_length = thp.train_sequence_length
        thp.batch_size = thp.train_dataset_size
    else:
        thp.subsequence_length = min(thp.subsequence_length, thp.train_sequence_length)

    train_dataset = generate_dataset(
        systems=shared['system'],
        batch_size=ehp.ensemble_size * thp.train_dataset_size,
        seq_length=thp.train_sequence_length,
        dev_type=dev_type
    ).reshape(n_models, thp.train_dataset_size, thp.train_sequence_length)

    # DONE: Create train dataloader
    total_batch_size = ehp.n_systems * ehp.ensemble_size * thp.batch_size
    if thp.optim_type == 'GD':
        train_index_dataset = TensorDict({
            'sequence': torch.arange(thp.train_dataset_size).repeat(n_models),
            'start': torch.zeros(total_batch_size, dtype=torch.int),
            'stop': train_sequence_lengths.repeat(n_models)
        }, batch_size=(total_batch_size,), device=dev_type).expand(thp.iterations_per_epoch, total_batch_size)
    else:
        sequence_indices, start_indices = torch.where(train_mask[:, thp.subsequence_length - 1:])
        train_index_dataset = TensorDict({
            'sequence': sequence_indices,
            'start': start_indices,
            'stop': start_indices + thp.subsequence_length
        }, batch_size=(len(sequence_indices),), device=dev_type)

    def supply_train_index_dataloader():
        if thp.optim_type == 'GD':
            return train_index_dataset
        else:
            return train_index_dataset[torch.randint(0, len(sequence_indices), (thp.iterations_per_epoch, total_batch_size), device=dev_type)]

    exclusive = {
        'base_model': flattened_learned_kfs[0],
        'training_dataset': train_dataset,
        'training_dataloader': supply_train_index_dataloader,
        'training_mask': train_mask
    }

    # Setup result and run training
    print(f'Mean theoretical irreducible loss: {torch.mean(shared["irreducible_loss"]).item()}')
    return {
        'output': run_training(
            hp,
            shared,
            exclusive,
            flattened_ensembled_learned_kfs,
            dev_type
        ).detach().cpu(),
        'learned_kf': TensorDict(
            flattened_ensembled_learned_kfs, batch_size=(n_models,), device='cpu'
        ).reshape(ehp.n_systems, ehp.ensemble_size).detach()
    }


def run_experiments(
        model: type,
        hp: Namespace,
        iterp: OrderedDict[str, Dict[str, Sequence[Any]]],
        dev_type: str,
        output_kwargs: Dict[str, Any],
        systems: List[LinearSystem] = None,
        system_kwargs: Dict[str, torch.Tensor] = MappingProxyType({}),
) -> TensorDict[str, torch.Tensor | TensorDict]:

    shp, mhp, thp, ehp = hp.system, hp.model, hp.train, hp.experiment

    # Setup file names
    ehp.exp_name = exp_name = f'{utils.class_name(model)}_{ehp.exp_name}'
    output_dir = f'output/{output_kwargs["dir"]}/{exp_name}'
    output_fname = f'{output_dir}/{output_kwargs["fname"]}.pt'
    output_fname_backup = f'{output_dir}/{output_kwargs["fname"]}_backup.pt'

    # Setup hyperparameter tuning shape
    zipped_params, param_shape = collections.OrderedDict(), []
    for param_group, params in iterp.items():
        zpn, zpv = tuple(params.keys()), tuple(params.values())
        zipped_params[param_group] = zip([zpn] * len(zpv[0]), zip(*zpv))

        param_shape.append(len(zpv[0]))
        for n, v in params.items():
            idx = n.rfind('.')
            setattr(utils.rgetattr(hp, n[:idx]), n[idx + 1:], v)

    # Setup result dict
    if os.path.exists(output_fname):
        try:
            with open(output_fname, 'rb') as fp:
                result = torch.load(fp, map_location=torch.device('cpu'))
        except RuntimeError:
            with open(output_fname_backup, 'rb') as fp:
                result = torch.load(fp, map_location=torch.device('cpu'))
    else:
        os.makedirs(output_dir, exist_ok=True)
        result = {
            'time': torch.zeros(param_shape, dtype=torch.float),
            'output': np.empty(param_shape, dtype=TensorDict),
            'learned_kf': np.empty(param_shape, dtype=TensorDict)
        }

    # System setup
    sys_fname = f'output/{output_kwargs["dir"]}/systems.pt'
    if os.path.exists(sys_fname):
        print('Systems found')
        with open(sys_fname, 'rb') as fp:
            systems = torch.load(fp, map_location=torch.device(dev_type))
    elif systems is None:
        print('No systems found, generating new systems')
        systems = [
            LinearSystem.sample_stable_system(hp.system, **system_kwargs).to(dev_type)
            for _ in range(hp.experiment.n_systems)]
    if not os.path.exists(sys_fname):
        with open(sys_fname, 'wb') as fp:
            torch.save(systems, fp)

    # DONE: System preprocessing (Analytical KFs, Irreducible loss)
    analytical_kfs = list(map(AnalyticalKF, systems))
    irreducible_loss = torch.stack([torch.trace(sys.S_observation_inf) for sys in systems])

    # DONE: Create validation and IR datasets
    """ valid_dataset = {
            'input': [NE x S_V x L_V x I_D],
            'observation': [NE x S_V x L_V x O_D]
        }
        ir_dataset = {
            'input': [NE x O_D x L_I x I_D],
            'observation': [NE x O_D x L_I x O_D]
        }
    """
    thp.valid_sequence_length = (thp.total_valid_sequence_length + thp.valid_dataset_size - 1) // thp.valid_dataset_size
    valid_dataset = add_targets(analytical_kfs, generate_dataset(
        systems=systems,
        batch_size=ehp.ensemble_size * thp.valid_dataset_size,
        seq_length=thp.valid_sequence_length,
        dev_type=dev_type
    )).reshape(ehp.n_systems * ehp.ensemble_size, thp.valid_dataset_size, thp.valid_sequence_length)
    if thp.ir_length is not None:
        ir_dataset = add_targets(analytical_kfs, TensorDict({
            'input': torch.zeros(shp.O_D, thp.ir_length, shp.I_D),
            'observation': torch.cat([
                torch.eye(shp.O_D)[:, None, :],
                torch.zeros(shp.O_D, thp.ir_length - 1, shp.O_D)
            ], dim=1),
        }, batch_size=(shp.O_D, thp.ir_length), device=dev_type).expand(ehp.n_systems, shp.O_D, thp.ir_length))
        ir_dataset = ir_dataset[:, None].expand(ehp.n_systems, ehp.ensemble_size, shp.O_D, thp.ir_length).flatten(0, 1)
    else:
        ir_dataset = None

    shared = {
        'model': model,
        'system': systems,
        'analytical_kf': analytical_kfs,
        'irreducible_loss': irreducible_loss,
        'validation_dataset': valid_dataset,
        'impulse_dataset': ir_dataset
    }

    # Run experiments for hyperparameter tuning
    print('=' * 200)
    print(hp.experiment.exp_name)
    print('=' * 200)
    print("Hyperparameters:", json.dumps(utils.toJSON(hp), indent=4))

    counter = 0
    for enumerated_args in itertools.product(*map(enumerate, zipped_params.values())):
        # Setup experiment hyperparameters
        indices, args = zip(*enumerated_args)
        experiment_hp = copy.deepcopy(hp)
        for arg_names, arg_values in args:
            for n, v in zip(arg_names, arg_values):
                idx = n.rfind('.')
                setattr(utils.rgetattr(experiment_hp, n[:idx]), n[idx + 1:], v)

        done = result['done'] if 'done' in result else (result['time'] > 0)
        if not done[indices]:
            print('=' * 200)
            print(f'Experiment {done.sum().item()}/{done.numel()}')

            start_t = time.perf_counter()
            experiment_result = run_experiment(
                experiment_hp,
                shared,
                dev_type
            )
            end_t = time.perf_counter()
            result['output'][indices] = experiment_result['output']
            result['learned_kf'][indices] = experiment_result['learned_kf']
            result['time'][indices] = end_t - start_t

            print('\n' + '#' * 200)
            if counter % ehp.backup_frequency == 0:
                with open(output_fname_backup, 'wb') as fp:
                    torch.save(result, fp)
                print(f'{os.path.getsize(output_fname_backup)} bytes written to {output_fname_backup}')
            with open(output_fname, 'wb') as fp:
                torch.save(result, fp)
                print(f'{os.path.getsize(output_fname)} bytes written to {output_fname}')
            print('#' * 200 + '\n')

            counter += 1

    with open(f'{output_dir}/hparams.json', 'w') as fp:
        json.dump(utils.toJSON(hp), fp, indent=4)

    return result




