import copy
import itertools
import json
import os
import time
from argparse import Namespace
from types import MappingProxyType
from typing import *

import numpy as np
import torch
import torch.utils.data
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.settings import dev_type
from infrastructure.train import run_training
from infrastructure.validate import _compute_metrics, compute_experiment_metrics
from model.analytical_kf import AnalyticalKF
from model.linear_system import LinearSystem


def run_experiment(
        hp: Namespace,
        shared: Dict[str, Any]
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
        mhp.model(mhp).to(dev_type)
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
    train_dataset = shared['training_dataset'][:, :thp.train_dataset_size, :thp.train_sequence_length]

    # DONE: Create train dataloader
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
        ).detach().cpu(),
        'learned_kf': (flattened_learned_kfs[0].cpu(), TensorDict(
            flattened_ensembled_learned_kfs, batch_size=(n_models,), device='cpu'
        ).reshape(ehp.n_systems, ehp.ensemble_size).detach())
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
        systems: List[LinearSystem] = None,
        system_kwargs: Dict[str, torch.Tensor] = MappingProxyType({}),
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

    # System setup
    sys_fname = f'{root_dir}/systems.pt'
    if os.path.exists(sys_fname):
        print('Systems found')
        with open(sys_fname, 'rb') as fp:
            systems = torch.load(fp, map_location=torch.device(dev_type))
    elif systems is None:
        print('No systems found, generating new systems')
        systems = [
            LinearSystem.sample_stable_system(hp.system, **system_kwargs).to(dev_type)
            for _ in range(hp.experiment.n_systems)]
    else:
        systems = [sys.to(dev_type) for sys in systems]
    if not os.path.exists(sys_fname):
        with open(sys_fname, 'wb') as fp:
            torch.save(systems, fp)

    # DONE: System preprocessing (Analytical KFs, Irreducible loss)
    analytical_kfs = list(map(AnalyticalKF, systems))
    irreducible_loss = torch.stack([torch.trace(sys.S_observation_inf) for sys in systems])

    # Dataset setup
    dataset_fname = f'{output_dir}/datasets.pt'
    thp.valid_sequence_length = (thp.total_valid_sequence_length + thp.valid_dataset_size - 1) // thp.valid_dataset_size
    if os.path.exists(dataset_fname):
        with open(dataset_fname, 'rb') as fp:
            dataset = torch.load(fp, map_location=torch.device(dev_type))
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
        n_models = ehp.n_systems * ehp.ensemble_size
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
        full_train_dataset = LinearSystem.generate_dataset(
            systems=systems,
            batch_size=ehp.ensemble_size * max_train_dataset_size,
            seq_length=max_train_sequence_length,
        ).reshape(n_models, max_train_dataset_size, max_train_sequence_length)

        valid_dataset = AnalyticalKF.add_targets(analytical_kfs, LinearSystem.generate_dataset(
            systems=systems,
            batch_size=ehp.ensemble_size * thp.valid_dataset_size,
            seq_length=thp.valid_sequence_length,
        )).reshape(n_models, thp.valid_dataset_size, thp.valid_sequence_length)

        ir_dataset = AnalyticalKF.add_targets(analytical_kfs, TensorDict({
            'input': torch.zeros(shp.O_D, thp.ir_length, shp.I_D),
            'observation': torch.cat([
                torch.eye(shp.O_D)[:, None, :],
                torch.zeros(shp.O_D, thp.ir_length - 1, shp.O_D)
            ], dim=1),
        }, batch_size=(shp.O_D, thp.ir_length), device=dev_type).expand(ehp.n_systems, shp.O_D, thp.ir_length))
        ir_dataset = ir_dataset[:, None].expand(ehp.n_systems, ehp.ensemble_size, shp.O_D, thp.ir_length).flatten(0, 1)

        test_dataset = AnalyticalKF.add_targets(analytical_kfs, LinearSystem.generate_dataset(
            systems=systems,
            batch_size=vhp.valid_dataset_size,
            seq_length=vhp.valid_sequence_length
        ))[:, None].expand(
            ehp.n_systems,
            ehp.ensemble_size,
            vhp.valid_dataset_size,
            vhp.valid_sequence_length
        ).flatten(0, 1)

        # Set up new dataset dict
        dataset = {
            'training': full_train_dataset,
            'validation': valid_dataset,
            'impulse': ir_dataset,
            'testing': test_dataset
        }
        with open(dataset_fname, 'wb') as fp:
            torch.save(dataset, fp)

    # Result setup
    metric_shape = (ehp.n_systems, ehp.ensemble_size, vhp.valid_dataset_size)
    if os.path.exists(output_fname):
        try:
            with open(output_fname, 'rb') as fp:
                result = torch.load(fp, map_location=torch.device('cpu'))
        except RuntimeError:
            with open(output_fname_backup, 'rb') as fp:
                result = torch.load(fp, map_location=torch.device('cpu'))
    else:
        # Set up new result dict
        result = {
            'time': torch.zeros(param_shape, dtype=torch.double),
            'output': np.empty(param_shape, dtype=TensorDict),
            'learned_kf': np.empty(param_shape, dtype=tuple),
            'metric': Namespace(
                eil=None,
                rn=None,
                l=torch.empty((*param_shape, *metric_shape), dtype=torch.double),
                rl=torch.empty((*param_shape, *metric_shape), dtype=torch.double)
            )
        }

    # Set up shared dict
    shared = {
        'system': systems,
        'analytical_kf': analytical_kfs,
        'irreducible_loss': irreducible_loss
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
                    if n != 'name' and v is not None:
                        idx = n.rfind('.')
                        setattr(utils.rgetattr(experiment_hp, n[:idx]), n[idx + 1:], v)

        done = result['done'] if 'done' in result else (result['time'] > 0)
        if not done[indices]:
            print('=' * 200)
            print(f'Experiment {done.sum().item()}/{done.numel()}')

            start_t = time.perf_counter()
            experiment_result = run_experiment(experiment_hp, shared)
            end_t = time.perf_counter()

            result['output'][indices] = experiment_result['output']
            result['learned_kf'][indices] = experiment_result['learned_kf']
            result['time'][indices] = end_t - start_t

            M = _compute_metrics(
                vhp,
                experiment_result['learned_kf'][0],
                experiment_result['learned_kf'][1].flatten(),
                dataset['testing'],
                metrics={'l', 'rl'} | ({'eil'} if result['metric'].eil is None else set()) | ({'rn'} if result['metric'].rn is None else set())
            )
            if result['metric'].eil is None:
                result['metric'].eil = M.eil
            if result['metric'].rn is None:
                result['metric'].rn = M.rn
            result['metric'].l[indices] = M.l
            result['metric'].rl[indices] = M.rl

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

    # Write hyperparameters to JSON
    hp_fname = f'{output_dir}/hparams.json'
    if not os.path.exists(hp_fname):
        with open(hp_fname, 'w') as fp:
            json.dump(utils.toJSON(hp), fp, indent=4)

    # Clean up result_backup
    if os.path.exists(output_fname_backup):
        os.remove(output_fname_backup)

    return result




