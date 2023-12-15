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
import torch.utils as ptu
import torch.utils.data
import ignite.handlers.param_scheduler
import tensordict
import tensordict.utils
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


# Testing
def test(
        dataset: TensorDict[str, torch.Tensor],
        indices: TensorDict,  # [NEB x ...]
        base_module: KF,
        flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
        dev_type: str,
        buffer: int = 0
) -> TensorDict[str, torch.Tensor]:  # [NE]

    n_models = dataset['input'].shape[0]
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
    dataset_ss = dataset[model_idx, sequence_idx, subsequence_idx]
    observation_ss = utils.run_stacked_modules(
        base_module=base_module,
        stacked_modules=flattened_ensembled_learned_kfs,
        args=dict(dataset_ss),
        kwargs=dict()
    )['observation_estimation']

    mask = (buffer <= subsequence_offset) * (subsequence_idx < stop_indices[:, :, None])

    def evaluate(input, target):
        losses = mask * torch.norm(input - target, dim=-1) ** 2
        return torch.sum(losses, dim=-1) / torch.sum(mask, dim=-1)  # [NE x B], L2 averaged over L_SS, sum over O_D

    result = {'prediction_true': evaluate(observation_ss, dataset_ss['observation'])}
    if 'target' in dataset_ss.keys():
        result.update({
            'prediction_target': evaluate(observation_ss, dataset_ss['target']),
            'naive_target': evaluate(0, dataset_ss['target'])
        })
    return TensorDict(result, batch_size=sequence_indices.shape, device=dev_type)


# Training
def train(
        model: type,
        hp: Namespace,
        dataset: Dict[str, TensorDict[str, torch.Tensor]],
        dataloader: Dict[str, ptu.data.DataLoader],
        flattened_ensembled_learned_kfs: Dict[str, nn.Parameter],
        optimizer: optim.Optimizer,
        dev_type: str,
) -> TensorDict:
    mhp, thp, ehp = hp.model, hp.train, hp.experiment
    n_models = ehp.n_systems * ehp.ensemble_size
    base_module = model(mhp).to(dev_type)

    result = []
    for batch, indices in enumerate(dataloader['training']):
        # Pass to obtain the observations to be lossed on
        with torch.set_grad_enabled(True):
            losses = torch.Tensor(test(
                dataset['training'],
                indices,
                base_module,
                flattened_ensembled_learned_kfs,
                dev_type
            )['prediction_true']).mean(-1)
        mask = torch.isnan(losses) + torch.isinf(losses)

        if batch % ehp.log_frequency == 0:
            with torch.set_grad_enabled(thp.epochs is None):
                overfit_losses = torch.Tensor(test(
                    dataset['training'],
                    tensordict.utils.expand_right(
                        list(dataloader['overfit'])[0][None], (n_models, thp.train_dataset_size)
                    ).flatten(),
                    base_module,
                    flattened_ensembled_learned_kfs,
                    dev_type,
                    buffer=getattr(thp, 'sequence_buffer', 0)
                )['prediction_true']).mean(-1)

            with torch.set_grad_enabled(False):
                valid_result = test(
                    dataset['validation'],
                    tensordict.utils.expand_right(
                        list(dataloader['validation'])[0][None], (n_models, thp.valid_dataset_size)
                    ).flatten(),
                    base_module,
                    flattened_ensembled_learned_kfs,
                    dev_type,
                    buffer=getattr(thp, 'sequence_buffer', 0)
                )

            r = TensorDict({
                'training': losses,
                'overfit': overfit_losses,
                'validation': torch.Tensor(valid_result['prediction_true']).mean(-1),
                'validation_target': torch.Tensor(valid_result['prediction_target']).mean(-1),
                'validation_target_norm': torch.Tensor(valid_result['naive_target']).mean(-1)
            }, batch_size=(n_models,))

            if thp.epochs is None:
                optimizer.zero_grad()
                torch.sum(overfit_losses[~mask]).backward()
                r['gradient_norm'] = torch.sum(torch.stack([
                    torch.norm(v.grad, dim=[1, 2]) ** 2
                    for v in flattened_ensembled_learned_kfs.values()]), dim=0).detach()
            if thp.ir_length is not None:
                with torch.set_grad_enabled(False):
                    impulse_result = test(
                        dataset['impulse_response'],
                        tensordict.utils.expand_right(
                            list(dataloader['impulse_response'])[0][None], (n_models, mhp.O_D)
                        ).flatten(),
                        base_module,
                        flattened_ensembled_learned_kfs,
                        dev_type
                    )
                r['impulse_target'] = torch.Tensor(impulse_result['prediction_target']).mean(-1)
                r['impulse_target_norm'] = torch.Tensor(impulse_result['naive_target']).mean(-1)

            result.append(r)

            if batch % ehp.print_frequency == 0:
                mean_losses = [
                    utils.remove_nans_and_infs(loss).median().item()
                    for loss in (losses, overfit_losses, valid_result['prediction_true'])]
                print(
                    f'\tTrain loss: {mean_losses[0]:>8f}, Overfit loss: {mean_losses[1]:>8f}, Valid loss: {mean_losses[2]:>8f}  [{batch * thp.batch_size:>5d}/{(thp.iterations_per_epoch * thp.batch_size):>5d}]')

        optimizer.zero_grad()
        torch.sum(losses[~mask]).backward()
        optimizer.step()

    return torch.stack(result, dim=1)


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


def generate_dataloader(seq_lengths: torch.Tensor, dev_type: str) -> ptu.data.DataLoader:
    N = len(seq_lengths)
    return ptu.data.DataLoader(
        TensorDict({
            'sequence': torch.arange(N),
            'start': torch.zeros(N, dtype=torch.int),
            'stop': seq_lengths
        }, batch_size=(N,), device=dev_type),
        batch_size=N,
        collate_fn=lambda x: x
    )


def add_targets(
        kfs: List[AnalyticalKF],
        dataset: TensorDict[str, torch.Tensor] | TensorDictBase[str, torch.Tensor]
) -> TensorDict[str, torch.Tensor]:
    with torch.set_grad_enabled(False):
        dataset['target'] = utils.run_stacked_modules(
            base_module=kfs[0],
            stacked_modules=torch.func.stack_module_state(kfs)[0],
            args=dict(dataset),
            kwargs={'steady_state': True}
        )['observation_estimation']
    return dataset


def validate(
        dataset: TensorDict[str, torch.Tensor],
        base_kf: KF,
        learned_kfs: TensorDictBase[str, torch.Tensor],
        sequence_buffer: int,
        dev_type: str
) -> TensorDict[str, torch.Tensor]:
    dataset_size, sequence_length = dataset.shape[1:]
    dataloader = generate_dataloader(torch.full((dataset_size,), sequence_length), dev_type)

    learned_kfs = learned_kfs.to(dev_type)

    n_systems, ensemble_size = learned_kfs.shape
    n_models = n_systems * ensemble_size

    with torch.set_grad_enabled(False):
        return test(
            dataset[:, None].expand(n_systems, ensemble_size, dataset_size, sequence_length).flatten(0, 1),
            tensordict.utils.expand_right(
                list(dataloader)[0][None], (n_models, dataset_size)
            ).flatten(),
            base_kf,
            dict(learned_kfs.flatten(0, 1)),
            dev_type,
            buffer=sequence_buffer
        ).reshape(n_systems, ensemble_size, -1).cpu()


def subtraction_normalized_validation_loss(
        systems: List[LinearSystem],
        base_kf: KF,
        learned_kf_arr: np.ndarray[TensorDict],
        dev_type: str
) -> torch.Tensor:
    systems = [sys.to(dev_type) for sys in systems]
    ValidArgs = Namespace(
        valid_dataset_size=500,
        valid_sequence_length=800,
        sequence_buffer=50
    )
    valid_dataset = generate_dataset(
        systems=systems,
        batch_size=ValidArgs.valid_dataset_size,
        seq_length=ValidArgs.valid_sequence_length,
        dev_type=dev_type
    )

    # Compute the empirical irreducible loss: [n_systems x 1 x valid_dataset_size]
    analytical_kfs = list(map(AnalyticalKF, systems))
    eil = validate(
        dataset=valid_dataset,
        base_kf=analytical_kfs[0],
        learned_kfs=TensorDict(torch.func.stack_module_state(analytical_kfs)[0], batch_size=(len(systems),))[:, None],
        sequence_buffer=ValidArgs.sequence_buffer,
        dev_type=dev_type
    )['prediction_true']

    vl = torch.stack([validate(
        dataset=valid_dataset,
        base_kf=base_kf,
        learned_kfs=learned_kf,
        sequence_buffer=ValidArgs.sequence_buffer,
        dev_type=dev_type
    ) for learned_kf in learned_kf_arr.flatten()]).unflatten(0, learned_kf_arr.shape)['prediction_true']

    return vl - eil


# Full training scheme
def run_training(
        model: type,
        hp: Namespace,
        systems: List[LinearSystem],  # [N x ...]
        flattened_ensembled_learned_kfs: Dict[str, torch.nn.Parameter],  # [NE x ...]
        irreducible_loss: torch.Tensor,  # [N x E]
        dev_type: str
) -> TensorDict:
    shp, mhp, thp, ehp = hp.system, hp.model, hp.train, hp.experiment

    n_models = ehp.n_systems * ehp.ensemble_size

    thp.train_sequence_length = (thp.total_train_sequence_length + thp.train_dataset_size - 1) // thp.train_dataset_size
    thp.valid_sequence_length = (thp.total_valid_sequence_length + thp.valid_dataset_size - 1) // thp.valid_dataset_size

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

    # Ensembled dataset setup
    """ Datasets
        train_dataset = {
            'input': [NE x S_T x L_T x I_D],
            'observation': [NE x S_T x L_T x O_D]
        }
        valid_dataset = {
            'input': [NE x S_V x L_V x I_D],
            'observation': [NE x S_V x L_V x O_D]
        }
    """
    kfs = list(map(AnalyticalKF, systems))
    train_dataset = generate_dataset(
        systems=systems,
        batch_size=ehp.ensemble_size * thp.train_dataset_size,
        seq_length=thp.train_sequence_length,
        dev_type=dev_type
    ).reshape(n_models, thp.train_dataset_size, thp.train_sequence_length)
    valid_dataset = add_targets(kfs, generate_dataset(
        systems=systems,
        batch_size=ehp.ensemble_size * thp.valid_dataset_size,
        seq_length=thp.valid_sequence_length,
        dev_type=dev_type
    )).reshape(n_models, thp.valid_dataset_size, thp.valid_sequence_length)

    # Dataloaders indexing into datasets to save computation
    total_batch_size = ehp.n_systems * ehp.ensemble_size * thp.batch_size

    if thp.optim_type == 'GD':
        train_index_dataset = TensorDict({
            'sequence': torch.arange(thp.train_dataset_size).repeat(thp.iterations_per_epoch * n_models),
            'start': torch.zeros(total_batch_size, dtype=torch.int).repeat(thp.iterations_per_epoch),
            'stop': train_sequence_lengths.repeat(thp.iterations_per_epoch * n_models)
        }, batch_size=(thp.iterations_per_epoch * total_batch_size,), device=dev_type)
        train_index_sampler = None
    else:
        sequence_indices, start_indices = torch.where(train_mask[:, thp.subsequence_length - 1:])
        train_index_dataset = TensorDict({
            'sequence': sequence_indices,
            'start': start_indices,
            'stop': start_indices + thp.subsequence_length
        }, batch_size=(len(sequence_indices),), device=dev_type)
        train_index_sampler = ptu.data.RandomSampler(
            train_index_dataset,
            replacement=True,
            num_samples=thp.iterations_per_epoch * total_batch_size
        )

    def identity(x): return x
    overfit_index_dataloader = generate_dataloader(train_sequence_lengths, dev_type)
    valid_index_dataloader = generate_dataloader(torch.full((thp.valid_dataset_size,), thp.valid_sequence_length), dev_type)

    """ Ensembled impulse response
        ir_inputs: [N x EO_D x L_I x I_D]
        ir_observations: [N x EO_D x L_I x O_D]
    """
    if thp.ir_length is not None:
        ir_dataset = add_targets(kfs, TensorDict({
            'input': torch.zeros(shp.O_D, thp.ir_length, shp.I_D),
            'observation': torch.cat([
                torch.eye(shp.O_D)[:, None, :],
                torch.zeros(shp.O_D, thp.ir_length - 1, shp.O_D)
            ], dim=1),
        }, batch_size=(shp.O_D, thp.ir_length), device=dev_type).expand(ehp.n_systems, shp.O_D, thp.ir_length))
        ir_dataset = ir_dataset[:, None].expand(ehp.n_systems, ehp.ensemble_size, shp.O_D, thp.ir_length).flatten(0, 1)
        ir_index_dataloader = generate_dataloader(torch.full((shp.O_D,), thp.ir_length), dev_type)
    else:
        ir_dataset, ir_index_dataloader = None, None

    # Create optimizer
    optimizer, scheduler = get_optimizer(tuple(flattened_ensembled_learned_kfs.values()), thp)

    results = []
    done = False
    t = 0
    done_matrix = torch.full((ehp.n_systems, ehp.ensemble_size), False, device=dev_type)
    while not done:
        print(f'Epoch {t + 1} ' + '-' * 100)

        train_index_dataloader = ptu.data.DataLoader(
            train_index_dataset,
            sampler=train_index_sampler,
            batch_size=total_batch_size,
            collate_fn=identity
        )

        results.append(train(model, hp, {
            'training': train_dataset,
            'validation': valid_dataset,
            'impulse_response': ir_dataset
        }, {
            'training': train_index_dataloader,
            'overfit': overfit_index_dataloader,
            'validation': valid_index_dataloader,
            'impulse_response': ir_index_dataloader
        }, flattened_ensembled_learned_kfs, optimizer, dev_type).reshape(ehp.n_systems, ehp.ensemble_size, -1))

        scheduler(None)
        print(f'LR: {optimizer.param_groups[0]["lr"]}')

        ol = torch.Tensor(results[-1]['overfit'][:, :, -1])
        divergences = torch.isnan(ol) + torch.isinf(ol)
        t += 1
        if thp.epochs is None:
            il = irreducible_loss
            gn = torch.Tensor(results[-1]['gradient_norm'][:, :, -1])
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

    return torch.cat(results, dim=-1)


# Experimentation
def load_experiment(output_fname: str, dev_type: str = 'cuda') -> TensorDict[str, Union[torch.Tensor, TensorDict]]:
    with open(output_fname, 'rb') as fp:
        return torch.load(fp, map_location=torch.device(dev_type))


def run_experiment(
        model: type,
        hp: Namespace,
        systems: List[LinearSystem],
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
        model(mhp).to(dev_type)
        for _ in range(n_models)]
    flattened_ensembled_learned_kfs = torch.func.stack_module_state(flattened_learned_kfs)[0]

    if getattr(thp, 'initialization', None) is not None:
        initialization = thp.initialization.expand(ehp.n_systems, ehp.ensemble_size).flatten()
        for k in flattened_ensembled_learned_kfs:
            if k in initialization.keys():
                flattened_ensembled_learned_kfs[k] = initialization[k].clone().to(dev_type).requires_grad_()

    irreducible_loss = torch.stack([torch.trace(sys.S_observation_inf) for sys in systems])
    print(f'Mean theoretical irreducible loss: {torch.mean(irreducible_loss).item()}')

    # Setup result and run training
    return {
        'output': run_training(
            model,
            hp,
            systems,
            flattened_ensembled_learned_kfs,
            irreducible_loss,
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
                model,
                experiment_hp,
                systems,
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




