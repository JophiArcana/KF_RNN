import copy
import itertools
import json
import os
import shutil
import time
from argparse import Namespace
from typing import *

import numpy as np
import torch
import torch.utils.data
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.settings import device
from infrastructure.train import run_training
from infrastructure.validate import _compute_metrics
from model.sequential_kf import SequentialKF
from model.analytical_kf import AnalyticalKF
from model.linear_system import LinearSystem


def run_experiment(
        hp: Namespace,
        shared: Dict[str, Any]
) -> Dict[str, TensorDict]:
    shp, mhp, thp, vhp, ehp = hp.system, hp.model, hp.train, hp.valid, hp.experiment
    print('=' * 200)
    print(ehp.exp_name)
    print('=' * 200)
    print("Hyperparameters:", json.dumps(utils.toJSON(hp), indent=4))
    print('=' * 200)

    # Ensemble learned Kalman Filters to be trained
    flattened_learned_kfs = [mhp.model(mhp).to(device) for _ in range(shared['n_models'])]
    flattened_ensembled_learned_kfs = torch.func.stack_module_state(flattened_learned_kfs)[0]

    if getattr(thp, 'initialization', None) is not None:
        initialization = thp.initialization.expand(shared['model_shape']).flatten()
        for k in flattened_ensembled_learned_kfs:
            if k in initialization.keys():
                flattened_ensembled_learned_kfs[k].data = initialization[k].to(device)

    # DONE: Create train dataset
    """ train_dataset = {
            'input': [NE x S_T x L_T x I_D],
            'observation': [NE x S_T x L_T x O_D]
        }
    """
    thp.train_sequence_length = (thp.total_train_sequence_length + thp.train_dataset_size - 1) // thp.train_dataset_size
    train_dataset = shared['full_training_dataset'][:, :thp.train_dataset_size, :thp.train_sequence_length]

    # DONE: Create train dataloader
    train_rem = thp.train_dataset_size * thp.train_sequence_length - thp.total_train_sequence_length
    train_mask = torch.full((thp.train_dataset_size, thp.train_sequence_length), True, device=device)
    if train_rem > 0:
        train_mask[-train_rem:, -1] = False
    train_sequence_lengths = torch.sum(train_mask, dim=1)

    if thp.optim_type == 'GD':
        thp.subsequence_length = thp.train_sequence_length
        thp.batch_size = thp.train_dataset_size
    else:
        thp.subsequence_length = min(thp.subsequence_length, thp.train_sequence_length)

    total_batch_size = ehp.n_systems * ehp.ensemble_size * thp.batch_size
    if thp.optim_type == 'GD':
        train_index_dataset = TensorDict({
            'sequence': torch.arange(thp.train_dataset_size).repeat(shared['n_models']),
            'start': torch.zeros(total_batch_size, dtype=torch.int),
            'stop': train_sequence_lengths.repeat(shared['n_models'])
        }, batch_size=(total_batch_size,), device=device).expand(thp.iterations_per_epoch, total_batch_size)
    else:
        sequence_indices, start_indices = torch.where(train_mask[:, thp.subsequence_length - 1:])
        train_index_dataset = TensorDict({
            'sequence': sequence_indices,
            'start': start_indices,
            'stop': start_indices + thp.subsequence_length
        }, batch_size=(len(sequence_indices),), device=device)

    def supply_train_index_dataloader():
        if thp.optim_type == 'GD':
            return train_index_dataset
        else:
            return train_index_dataset[torch.randint(0, len(sequence_indices), (thp.iterations_per_epoch, total_batch_size), device=device)]

    exclusive = {
        'base_model': flattened_learned_kfs[0],
        'training_dataset': train_dataset,
        'training_dataloader': supply_train_index_dataloader,
        'training_mask': train_mask
    }

    # Setup result and run training
    print(f'Mean theoretical irreducible loss: {torch.mean(shared["irreducible_loss"]).item()}')
    result = run_training(hp, shared, exclusive, flattened_ensembled_learned_kfs).detach()

    learned_kf = (flattened_learned_kfs[0], TensorDict(
        flattened_ensembled_learned_kfs, batch_size=(shared['n_models'],), device=device
    ).reshape(shared['model_shape']).detach())

    with torch.set_grad_enabled(False):
        metric = _compute_metrics(
            vhp,
            learned_kf[0],
            learned_kf[1].flatten(),
            shared['full_testing_dataset'][:, :vhp.valid_dataset_size, :vhp.valid_sequence_length]
        ).reshape(shared['model_shape'])
    
        stacked_systems = TensorDict(torch.func.stack_module_state(shared['system'])[0], batch_size=(ehp.n_systems,))[:, None]
        metric['al'] = mhp.model.analytical_error(learned_kf[1], stacked_systems)[:, :, None]
        metric['il'] = shared['irreducible_loss'][:, None, None].expand(-1, ehp.ensemble_size, -1)

    return {
        'output': result,
        'learned_kf': learned_kf,
        'metric': metric
    }


"""
    iterp = [
        ('optimizer', {
            'name': ['Adam', 'SGD', 'LeastSquares'],
            'model.model': [CnnKF, CnnKF, CnnKFLeastSquares],
            'train.optim_type': ['Adam', 'SGDMomentum', None],
            'train.max_lr': [2e-2, 2e-3, None],
            'train.min_lr': [1e-6, 1e-7, None],
            'train.weight_decay': [1e-1, 1e-2, None]
        }),
        ('model', {
            'model.model': [CnnKF, CnnKFLeastSquares, CnnKFPretrainLeastSquares]
        }),
        ('ir_length', {
            'model.ir_length': [1, 2, 4, 8, 16, 32]
        }),
        ('total_trace_length', {
            'train.total_train_sequence_length': [100, 200, 500, 1000, 2000, 5000, 10000]
        })
    ]
"""
def run_experiments(
        hp: Namespace,
        iterp: List[Tuple[str, Dict[str, List[Any]]]],
        output_kwargs: Dict[str, Any],
        systems: List[LinearSystem] = None
) -> Dict[str, np.ndarray | torch.Tensor | TensorDict]:

    # Set up hyperparameter tuning shape
    zipped_params, param_shape = [], []
    for param_group, params in iterp:
        zpn, zpv = tuple(params.keys()), tuple(params.values())
        zipped_params.append(zip([zpn] * len(zpv[0]), zip(*zpv)))

        param_shape.append(len(zpv[0]))
        for n, v in params.items():
            if n != 'name':
                idx = n.rfind('.')
                setattr(utils.rgetattr(hp, n[:idx]), n[idx + 1:], v)
    param_shape = tuple(param_shape)

    shp, mhp, thp, vhp, ehp = hp.system, hp.model, hp.train, hp.valid, hp.experiment

    # Set up file names
    root_dir = f'output/{output_kwargs["dir"]}'
    if hasattr(mhp.model, '__getitem__'):
        class_name = utils.class_name(mhp.model[0])
        # class_name = f"[{', '.join(map(utils.class_name, mhp.model))}]"
    else:
        class_name = utils.class_name(mhp.model)
    ehp.exp_name = exp_name = f'{class_name}_{ehp.exp_name}'
    os.makedirs((output_dir := f'{root_dir}/{exp_name}'), exist_ok=True)
    output_fname = f'{output_dir}/{output_kwargs["fname"]}.pt'
    output_fname_backup = f'{output_dir}/{output_kwargs["fname"]}_backup.pt'

    # Save code to for experiment reproducibility
    code_base_dir = f'{output_dir}/code'
    os.makedirs(code_base_dir, exist_ok=True)
    for dir_name in ('infrastructure', 'model'):
        code_dir = f'{code_base_dir}/{dir_name}'
        shutil.copytree(dir_name, code_dir, dirs_exist_ok=True)

    # System setup
    sys_fname = f'{root_dir}/systems.pt'
    if os.path.exists(sys_fname):
        print('Systems found')
        with open(sys_fname, 'rb') as fp:
            systems = torch.load(fp, map_location=torch.device(device))
    elif systems is None:
        print('No systems found, generating new systems')
        systems = [LinearSystem.sample_stable_system(shp).to(device) for _ in range(ehp.n_systems)]
    else:
        systems = [sys.to(device) for sys in systems]
    if not os.path.exists(sys_fname):
        with open(sys_fname, 'wb') as fp:
            torch.save(systems, fp)

    # DONE: System preprocessing (Analytical KFs, Irreducible loss)
    model_shape = (ehp.n_systems, ehp.ensemble_size)
    n_models = ehp.n_systems * ehp.ensemble_size

    analytical_kfs = list(map(AnalyticalKF, systems))
    irreducible_loss = torch.stack([torch.trace(sys.S_observation_inf) for sys in systems])

    # Dataset setup
    dataset_fname = f'{output_dir}/datasets.pt'
    thp.valid_sequence_length = (thp.total_valid_sequence_length + thp.valid_dataset_size - 1) // thp.valid_dataset_size
    if os.path.exists(dataset_fname):
        with open(dataset_fname, 'rb') as fp:
            dataset = torch.load(fp, map_location=torch.device(device))
    else:
        print('Generating new dataset')

        # DONE: Create full training dataset, and validation and IR datasets
        """ train_dataset = {
                'input': [NE x S_T x L_T x I_D],
                'observation': [NE x S_T x L_T x O_D]
            }
            valid_dataset = {
                'input': [NE x S_V x L_V x I_D],
                'observation': [NE x S_V x L_V x O_D]
            }
            ir_dataset = {
                'input': [NE x O_D x L_I x I_D],
                'observation': [NE x O_D x L_I x O_D]
            }
        """
        # Train dataset
        if hasattr(thp.train_dataset_size, '__getitem__'):
            max_train_dataset_size = max(thp.train_dataset_size)
            min_train_dataset_size = min(thp.train_dataset_size)
        else:
            max_train_dataset_size = min_train_dataset_size = thp.train_dataset_size
        if hasattr(thp.total_train_sequence_length, '__getitem__'):
            max_total_train_sequence_length = max(thp.total_train_sequence_length)
        else:
            max_total_train_sequence_length = thp.total_train_sequence_length
        max_train_sequence_length = (max_total_train_sequence_length + min_train_dataset_size - 1) // min_train_dataset_size
        full_train_dataset = AnalyticalKF.add_targets(analytical_kfs, LinearSystem.generate_dataset(
            systems=systems,
            batch_size=ehp.ensemble_size * max_train_dataset_size,
            seq_length=max_train_sequence_length,
        )).reshape(n_models, max_train_dataset_size, max_train_sequence_length)

        # Validation dataset
        valid_dataset = AnalyticalKF.add_targets(analytical_kfs, LinearSystem.generate_dataset(
            systems=systems,
            batch_size=ehp.ensemble_size * thp.valid_dataset_size,
            seq_length=thp.valid_sequence_length,
        )).reshape(n_models, thp.valid_dataset_size, thp.valid_sequence_length)

        # Impulse dataset
        ir_dataset = AnalyticalKF.add_targets(analytical_kfs, TensorDict({
            'input': torch.zeros(shp.O_D, thp.ir_length, shp.I_D),
            'observation': torch.cat([
                torch.eye(shp.O_D)[:, None, :],
                torch.zeros(shp.O_D, thp.ir_length - 1, shp.O_D)
            ], dim=1),
        }, batch_size=(shp.O_D, thp.ir_length), device=device).expand(ehp.n_systems, shp.O_D, thp.ir_length))
        ir_dataset = ir_dataset[:, None].expand(*model_shape, shp.O_D, thp.ir_length).flatten(0, 1)

        # Testing dataset
        if hasattr(vhp.valid_dataset_size, '__getitem__'):
            max_test_dataset_size = max(vhp.valid_dataset_size)
        else:
            max_test_dataset_size = vhp.valid_dataset_size
        if hasattr(vhp.valid_sequence_length, '__getitem__'):
            max_test_sequence_length = max(vhp.valid_sequence_length)
        else:
            max_test_sequence_length = vhp.valid_sequence_length
        full_test_dataset = AnalyticalKF.add_targets(analytical_kfs, LinearSystem.generate_dataset(
            systems=systems,
            batch_size=max_test_dataset_size,
            seq_length=max_test_sequence_length
        ))[:, None].expand(
            ehp.n_systems,
            ehp.ensemble_size,
            vhp.valid_dataset_size,
            vhp.valid_sequence_length
        ).flatten(0, 1)

        # Set up new dataset dict
        dataset = {
            'full_training': full_train_dataset,
            'validation': valid_dataset,
            'impulse': ir_dataset,
            'full_testing': full_test_dataset
        }
        with open(dataset_fname, 'wb') as fp:
            torch.save(dataset, fp)

    # Result setup
    if os.path.exists(output_fname):
        try:
            with open(output_fname, 'rb') as fp:
                result = torch.load(fp, map_location=torch.device(device))
        except RuntimeError:
            with open(output_fname_backup, 'rb') as fp:
                result = torch.load(fp, map_location=torch.device(device))
    else:
        # Set up new result dict
        shape = param_shape + model_shape
        result = {
            'time': torch.zeros(param_shape, dtype=torch.double),
            'output': np.empty(param_shape, dtype=TensorDict),
            'learned_kf': np.empty(param_shape, dtype=tuple),
            'metric': TensorDict({**{
                k: torch.empty((*shape, vhp.valid_dataset_size), dtype=torch.double)
                for k in ('eil', 'rn', 'l', 'rl')
            },
                'al': torch.empty((*shape, 1), dtype=torch.double),
                'il': torch.empty((*shape, 1), dtype=torch.double),
            }, batch_size=shape, device=device)
        }

    # Set up shared dict
    shared = {
        'system': systems,
        'analytical_kf': analytical_kfs,
        'irreducible_loss': irreducible_loss,
        'model_shape': model_shape,
        'n_models': n_models
    }
    shared.update({k + '_dataset': v for k, v in dataset.items()})

    # Run experiments for hyperparameter tuning
    print('=' * 200)
    print(hp.experiment.exp_name)
    print('=' * 200)
    print("Hyperparameters:", json.dumps(utils.toJSON(hp), indent=4))

    counter = 0
    for enumerated_args in itertools.product(*map(enumerate, zipped_params)):
        # Setup experiment hyperparameters
        experiment_hp = copy.deepcopy(hp)
        if len(enumerated_args) == 0:
            indices = ()
        else:
            indices, args = zip(*enumerated_args)
            for arg_names, arg_values in args:
                for n, v in zip(arg_names, arg_values):
                    if n != 'name':
                        idx = n.rfind('.')
                        setattr(utils.rgetattr(experiment_hp, n[:idx]), n[idx + 1:], v)

        done = result['done'] if 'done' in result else (result['time'] > 0)
        if not done[indices]:
            print('=' * 200)
            print(f'Experiment {done.sum().item()}/{done.numel()}')

            start_t = time.perf_counter()
            experiment_result = run_experiment(experiment_hp, shared)
            end_t = time.perf_counter()

            for k, v in experiment_result.items():
                result[k][indices] = v
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
        torch.cuda.empty_cache()

    # Convert metrics to namespace
    result['metric'] = Namespace(**result['metric'])

    # Write hyperparameters to JSON
    hp_fname = f'{output_dir}/hparams.json'
    if not os.path.exists(hp_fname):
        with open(hp_fname, 'w') as fp:
            json.dump(utils.toJSON(hp), fp, indent=4)

    # Clean up result_backup
    if os.path.exists(output_fname_backup):
        os.remove(output_fname_backup)

    return result




