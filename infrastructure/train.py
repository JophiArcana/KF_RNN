
import copy
import itertools
import json
import os
from typing import *
from argparse import Namespace

from KF_RNN.model.linear_system import LinearSystem
from KF_RNN.model.rnn_kf import RnnKF
from KF_RNN.infrastructure import utils

import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.utils as ptu
import torch.optim as optim
from torchdata.datapipes.map import Zipper, SequenceWrapper
import tensordict
from tensordict import TensorDict


# Optimizer configuration
def get_optimizer(params, hp):
    optim_type = hp.optim_type
    if optim_type == "GD" or optim_type == "SGD":
        optimizer = optim.SGD(params, lr=hp.lr, momentum=0.0, weight_decay=hp.l2_reg)
    elif optim_type == "SGDMomentum":
        optimizer = optim.SGD(params, lr=hp.lr, momentum=hp.momentum, weight_decay=hp.l2_reg)
    elif optim_type == "Adam":
        optimizer = optim.AdamW(params, lr=hp.lr, betas=(hp.momentum, 0.999), weight_decay=hp.l2_reg)
    else:
        raise ValueError(optim_type)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, hp.lr_decay)
    return optimizer, scheduler


# Testing
def test(
    hp: Namespace,
    dataset: Tuple[torch.DoubleTensor],
    flattened_ensembled_learned_kfs: Tuple[Dict[str, nn.Parameter]],
    dev_type: str
) -> torch.DoubleTensor:                                                                                    # [NE]
    mhp, thp, ehp = hp.model, hp.train, hp.experiment

    base_RnnKF = RnnKF(mhp.S_D, mhp.I_D, mhp.O_D).to(dev_type)
    base_RnnKF.eval()
    def run_kf(kf_dicts, state, inputs, observations):
        return torch.func.functional_call(base_RnnKF, kf_dicts, (state, inputs, observations))

    dataset_size, sequence_length = dataset[0].shape[2:4]

    n_models = ehp.n_systems * ehp.ensemble_size
    with torch.no_grad():
        X_initial = torch.randn(n_models, dataset_size, mhp.S_D, device=dev_type)                           # [NE x S x S_D]
        X_input, y = dataset                                                                                # [N x E x S x L x I_D], [N x E x S x L x O_D]
        X_input = X_input.view(n_models, dataset_size, sequence_length, mhp.I_D)                            # [NE x S x L x I_D]
        y = y.view(n_models, dataset_size, sequence_length, mhp.O_D)                                        # [NE x S x L x O_D]

        observation = torch.func.vmap(run_kf)(
            flattened_ensembled_learned_kfs,
            X_initial, X_input, y
        )['observation_estimation']                                                                         # [NE x S x L x O_D]
        losses = torch.sum(torch.mean(Fn.mse_loss(observation, y, reduction='none'), dim=(1, 2)), dim=-1)   # [NE], L2 averaged over B x L_SS, sum over O_D

    return losses


# Training
def train(
    hp: Namespace,
    dataset: Dict[str, Tuple[torch.DoubleTensor]],
    dataloader: Dict[str, ptu.data.DataLoader],
    flattened_ensembled_learned_kfs: Tuple[Dict[str, nn.Parameter]],
    optimizer: optim.Optimizer,
    dev_type: str,
    loss_threshold: float=float('inf')  # 1.e18
) -> Dict[str, torch.DoubleTensor]:
    mhp, thp, ehp = hp.model, hp.train, hp.experiment

    base_RnnKF = RnnKF(mhp.S_D, mhp.I_D, mhp.O_D).to(dev_type)
    base_RnnKF.train()
    def run_kf(kf_dicts, state, inputs, observations):
        return torch.func.functional_call(base_RnnKF, kf_dicts, (state, inputs, observations))

    n_models = ehp.n_systems * ehp.ensemble_size
    train_losses_list, overfit_losses_list, valid_losses_list = [], [], []
    for batch, (sequence_indices, start_indices) in enumerate(dataloader['training']):                                          # [NEB], [NEB]
        sequence_indices = sequence_indices.view(n_models, thp.batch_size)                                                      # [NE x B]
        start_indices = start_indices.view(n_models, thp.batch_size)                                                            # [NE x B]

        X_input, y = dataset['training']                                                                                        # [N x E x S x L x I_D], [N x E x S x L x O_D]
        X_input = X_input.view(n_models, thp.train_dataset_size, thp.train_sequence_length, mhp.I_D)                            # [NE x S x L x I_D]
        y = y.view(n_models, thp.train_dataset_size, thp.train_sequence_length, mhp.O_D)                                        # [NE x S x L x O_D]

        # Smart indexing
        model_index = torch.arange(n_models)[:, None, None]                                                                     # Selects individual model
        sequence_index = sequence_indices[:, :, None]                                                                           # Selects which trace in the dataset
        subsequence_index = start_indices[:, :, None] + torch.arange(thp.subsequence_length, device=dev_type)                   # Selects subsequence of trace

        X_initial_ss = torch.randn(n_models, thp.batch_size, mhp.S_D, device=dev_type)                                          # [NE x B x S_D]
        X_input_ss = X_input[model_index, sequence_index, subsequence_index]                                                    # [NE x B x L_SS x I_D]
        y_ss = y[model_index, sequence_index, subsequence_index]                                                                # [NE x B x L_SS x O_D]

        # Pass to obtain the observations to be lossed on
        observation_ss = torch.func.vmap(run_kf)(
            flattened_ensembled_learned_kfs,
            X_initial_ss, X_input_ss, y_ss
        )['observation_estimation']                                                                                             # [NE x B x L_SS x O_D]
        losses = torch.sum(torch.mean(Fn.mse_loss(observation_ss, y_ss, reduction='none'), dim=(1, 2)), dim=-1)                 # [NE], L2 averaged over B x L_SS, sum over O_D
        
        divergences = torch.isnan(losses) + torch.isinf(losses) + (losses > loss_threshold)
        losses[divergences] = 0

        optimizer.zero_grad()
        torch.sum(losses).backward()
        optimizer.step()

        if batch % ehp.log_frequency == 0:
            train_losses_list.append(losses)
            overfit_losses_list.append(overfit_losses := test(hp, dataset['training'], flattened_ensembled_learned_kfs, dev_type))
            valid_losses_list.append(valid_losses := test(hp, dataset['validation'], flattened_ensembled_learned_kfs, dev_type))
            if batch % ehp.print_frequency == 0:
                print(f"\tTrain loss: {losses.mean().item():>8f}, Overfit loss: {overfit_losses.mean().item():>8f}, Valid loss: {valid_losses.mean().item():>8f}, Divergences: {divergences.sum().item()}  [{batch * thp.batch_size:>5d}/{(thp.iterations_per_epoch * thp.batch_size):>5d}]")

    return TensorDict({
        'training_loss': torch.stack(train_losses_list, dim=1),
        'overfit_loss': torch.stack(overfit_losses_list, dim=1),
        'validation_loss': torch.stack(valid_losses_list, dim=1)
    }, batch_size=(n_models, len(train_losses_list)))


# Full training scheme
def run_training(
    hp: Namespace,
    ensembled_systems: Tuple[Dict[str, torch.DoubleTensor]],                # [N x ...]
    flattened_ensembled_learned_kfs: Tuple[Dict[str, torch.DoubleTensor]],  # [NE x ...]
    dev_type: str
) -> Tuple[torch.DoubleTensor]:
    mhp, thp, ehp = hp.model, hp.train, hp.experiment

    thp.train_sequence_length = (thp.total_train_sequence_length + thp.train_dataset_size - 1) // thp.train_dataset_size
    thp.valid_sequence_length = (thp.total_valid_sequence_length + thp.valid_dataset_size - 1) // thp.valid_dataset_size

    if thp.optim_type == 'GD':
        thp.subsequence_length = thp.train_sequence_length  # f'For full GD, subsequence length {thp.subsequence_length} must be equal to train sequence length {thp.train_sequence_length}'
        thp.batch_size = thp.train_dataset_size             # f'For full GD, batch size {thp.batch_size} must equal train dataset size {thp.train_dataset_size}'
    else:
        thp.subsequence_length = min(thp.subsequence_length, thp.train_sequence_length)

    # Ensembled dataset setup
    total_dataset_size = thp.train_dataset_size + thp.valid_dataset_size
    initial_state = torch.randn(ehp.n_systems, ehp.ensemble_size * total_dataset_size, mhp.S_D, device=dev_type)                                # [N x EB x S_D]

    base_LS = LinearSystem.sample_stable_system(mhp)
    def propagate_system(system_dicts, state, inputs):
        return torch.func.functional_call(base_LS, system_dicts, (state, inputs))
    vmap_propagate_system = torch.func.vmap(propagate_system, randomness='different')

    # Ensembled training data
    train_initial_state = initial_state[:, :ehp.ensemble_size * thp.train_dataset_size]                                                         # [N x ES_T x S_D]
    train_inputs = torch.randn(ehp.n_systems, ehp.ensemble_size * thp.train_dataset_size, thp.train_sequence_length, mhp.I_D, device=dev_type)  # [N x ES_T x L_T x I_D]
    train_observations = vmap_propagate_system(ensembled_systems, train_initial_state, train_inputs)['observation']                             # [N x ES_T x L_T x O_D]

    # Ensembled validation data
    valid_initial_state = initial_state[:, ehp.ensemble_size * thp.train_dataset_size:]                                                         # [N x ES_V x S_D]
    valid_inputs = torch.randn(ehp.n_systems, ehp.ensemble_size * thp.valid_dataset_size, thp.valid_sequence_length, mhp.I_D, device=dev_type)  # [N x ES_V x L_V x I_D]
    valid_observations = vmap_propagate_system(ensembled_systems, valid_initial_state, valid_inputs)['observation']                             # [N x ES_V x L_V x O_D]

    # Datasets
    train_dataset = (
        train_inputs.view(ehp.n_systems, ehp.ensemble_size, thp.train_dataset_size, thp.train_sequence_length, mhp.I_D),                # [N x E x S_T x L_T x I_D]
        train_observations.view(ehp.n_systems, ehp.ensemble_size, thp.train_dataset_size, thp.train_sequence_length, mhp.O_D)           # [N x E x S_T x L_T x O_D]
    )
    valid_dataset = (
        valid_inputs.view(ehp.n_systems, ehp.ensemble_size, thp.valid_dataset_size, thp.valid_sequence_length, mhp.I_D),                # [N x E x S_V x L_V x I_D]
        valid_observations.view(ehp.n_systems, ehp.ensemble_size, thp.valid_dataset_size, thp.valid_sequence_length, mhp.O_D)           # [N x E x S_V x L_V x O_D]
    )

    # Dataloaders indexing into datasets to save computation
    start_min = thp.replay_buffer if thp.subsequence_initial_mode == 'replay_buffer' else 0
    start_max = thp.train_sequence_length - thp.subsequence_length + 1

    total_batch_size = ehp.n_systems * ehp.ensemble_size * thp.batch_size
    
    if thp.optim_type == 'GD':
        train_index_dataset = Zipper(
            SequenceWrapper(torch.arange(thp.train_dataset_size, device=dev_type).repeat(ehp.n_systems * ehp.ensemble_size * thp.iterations_per_epoch)),
            SequenceWrapper(torch.zeros(total_batch_size * thp.iterations_per_epoch, dtype=int, device=dev_type))
        )
        train_index_sampler = None
    else:
        train_index_dataset = Zipper(
            SequenceWrapper(torch.arange(thp.train_dataset_size, device=dev_type).repeat_interleave(start_max - start_min, dim=0)),
            SequenceWrapper(torch.arange(start_min, start_max, device=dev_type).repeat(thp.train_dataset_size))
        )
        train_index_sampler = ptu.data.RandomSampler(
            train_index_dataset,
            replacement=True,
            num_samples=thp.iterations_per_epoch * total_batch_size
        )

    overfit_index_dataloader = ptu.data.DataLoader(Zipper(
        SequenceWrapper(torch.arange(thp.train_dataset_size, device=dev_type)),
        SequenceWrapper(torch.zeros(thp.train_dataset_size, device=dev_type))
    ), batch_size=thp.train_dataset_size)

    valid_index_dataloader = ptu.data.DataLoader(Zipper(
        SequenceWrapper(torch.arange(thp.valid_dataset_size, device=dev_type)),
        SequenceWrapper(torch.zeros(thp.valid_dataset_size, device=dev_type))
    ), batch_size=thp.valid_dataset_size)

    # Create optimizer
    optimizer, scheduler = get_optimizer(tuple(flattened_ensembled_learned_kfs[0].values()), thp)

    results = []
    for t in range(thp.epochs):
        print(f'Epoch {t + 1} ' + '-' * 100)
            
        train_index_dataloader = ptu.data.DataLoader(
            train_index_dataset,
            sampler=train_index_sampler,
            batch_size=total_batch_size
        )

        results.append(train(hp, {
            'training': train_dataset,
            'validation': valid_dataset
        }, {
            'training': train_index_dataloader,
            'overfit': overfit_index_dataloader,
            'validation': valid_index_dataloader
        }, flattened_ensembled_learned_kfs, optimizer, dev_type).reshape(ehp.n_systems, ehp.ensemble_size, -1))

        scheduler.step()

    return torch.cat(results, dim=-1)


# Experimentation
def load_experiment(output_fname: str, dev_type: str='cuda') -> TensorDict[str, Union[torch.DoubleTensor, TensorDict]]:
    with open(output_fname, 'rb') as fp:
        # return pickle.load(fp)
        return torch.load(fp, map_location=torch.device(dev_type))


def run_experiment(
    hp: Namespace,
    dev_type: str,
    systems: Sequence[LinearSystem]=None,
    system_kwargs: Dict[str, torch.DoubleTensor]=None,
    output_mode: str='load',
    output_kwargs: Dict[str, Any]=dict(),
) -> TensorDict[str, Union[torch.DoubleTensor, TensorDict]]:

    mhp, thp, ehp = hp.model, hp.train, hp.experiment
    print('=' * 200)
    print(getattr(ehp, 'exp_name', 'Ensembled experiment'))
    print('=' * 200)
    print("Hyperparameters:", json.dumps(utils.toJSON(hp), indent=4))

    # If we are performing more experiments on the same system, load system configurations from existing experiments
    output_dir = f'output/{ehp.output_dir}/{ehp.exp_name}'
    output_fname = f'{output_dir}/{output_kwargs.get("fname", "result")}.pt'
    if output_mode == 'load':
        if os.path.exists(output_fname):
            print()
            return load_experiment(output_fname)
        else:
            output_mode = 'concatenate'
    if output_mode == 'concatenate' and output_kwargs['dim'] == 2 and os.path.exists(output_fname):
        with open(output_fname, 'rb') as fp:
            # current_result = pickle.load(fp)
            current_result = torch.load(fp, map_location=torch.device(dev_type))
        systems = [
            LinearSystem.sample_stable_system(mhp).to(dev_type)
        for _ in range(ehp.n_systems)]
        for n, sys in enumerate(systems):
            sys.load_state_dict(current_result['system'][n, 0])
    else:
        if systems is None:
            systems = [None for _ in range(ehp.n_systems)]
        systems = [
            LinearSystem.sample_stable_system(mhp, base_system=sys, **system_kwargs).to(dev_type)
        for sys in systems]
    print('=' * 200)

    # Ensemble linear systems to generate data
    ensembled_systems = torch.func.stack_module_state(systems)

    # Ensemble learned Kalman Filters to be trained
    flattened_learned_kfs = [
        RnnKF(mhp.S_D, mhp.I_D, mhp.O_D).to(dev_type)
    for _ in range(ehp.n_systems * ehp.ensemble_size)]
    flattened_ensembled_learned_kfs = torch.func.stack_module_state(flattened_learned_kfs)
    [p.to(dev_type).requires_grad_() for p in flattened_ensembled_learned_kfs[0].values()]

    irreducible_loss = torch.DoubleTensor([torch.trace(sys.S_observation_inf) for sys in systems])
    print(f'Mean theoretical irreducible loss: {torch.mean(irreducible_loss).item()}')

    # Setup result and run training
    result = {
        'system': tensordict.utils.expand_right(torch.stack([
            TensorDict(sys.state_dict(), batch_size=())
        for sys in systems]), (ehp.n_systems, ehp.ensemble_size)),
        'learned_kf': torch.stack([
            TensorDict(learned_kf.state_dict(), batch_size=())
        for learned_kf in flattened_learned_kfs]).reshape(ehp.n_systems, ehp.ensemble_size),
        'irreducible_loss': tensordict.utils.expand_right(irreducible_loss, (ehp.n_systems, ehp.ensemble_size))
    }
    result.update(run_training(hp, ensembled_systems, flattened_ensembled_learned_kfs, dev_type))
    result = TensorDict(result, batch_size=(ehp.n_systems, ehp.ensemble_size))

    # Setup output directory
    if output_mode == 'none':
        pass
    elif output_mode == 'reset' or output_mode == 'concatenate':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/hparams.json', 'w') as fp:
                json.dump(utils.toJSON(hp), fp, indent=4)
        if output_mode == 'concatenate' and os.path.exists(output_fname):
            with open(output_fname, 'rb') as fp:
                # current_result = pickle.load(fp)
                current_result = torch.load(fp, map_location=torch.device(dev_type))
            try:
                result = torch.cat([current_result, result], dim=output_kwargs['dim'])
            except Exception:
                pass
        with open(output_fname, 'wb') as fp:
            # pickle.dump(result, fp)
            torch.save(result, fp)
    print()

    return result


def run_experiments(
    hp: Namespace,
    iterp: List[str],
    dev_type: str,
    systems: Sequence[LinearSystem]=None,
    system_kwargs: Dict[str, torch.DoubleTensor]=None,
    output_kwargs: Dict[str, Any]=dict(),
) -> TensorDict[str, Union[torch.DoubleTensor, TensorDict]]:
    
    mhp, thp, ehp = hp.model, hp.train, hp.experiment
    print('=' * 200)
    print(getattr(ehp, 'exp_name', 'Ensembled experiment'))
    print('=' * 200)
    print("Hyperparameters:", json.dumps(utils.toJSON(hp), indent=4))

    output_dir = f'output/{ehp.output_dir}/{ehp.exp_name}'
    output_fname = f'{output_dir}/{output_kwargs.get("fname", "result")}.pt'
    
    if os.path.exists(output_fname):
        with open(output_fname, 'rb') as fp:
            result = torch.load(fp, map_location=torch.device(dev_type))
    else:
        result = TensorDict(dict(), batch_size=(ehp.n_systems, ehp.ensemble_size))
        os.makedirs(output_dir, exist_ok=True)
        with open(output_fname, 'wb') as fp:
            torch.save(result, fp)

    for args in itertools.product(*(thp.__dict__[p] for p in iterp)):
        str_args = tuple(map(str, args))
        if str_args not in result.keys(include_nested=True):
            exp_hp = copy.copy(hp)
            exp_hp.train = copy.copy(hp.train)
            exp_hp.train.__dict__.update({k: v for k, v in zip(iterp, args)})

            result[str_args] = run_experiment(
                exp_hp,
                dev_type,
                systems=systems,
                system_kwargs=system_kwargs,
                output_mode='none',
                output_kwargs=output_kwargs
            )

            with open(output_fname, 'wb') as fp:
                torch.save(result, fp)
    
    with open(f'{output_dir}/hparams.json', 'w') as fp:
        json.dump(utils.toJSON(hp), fp, indent=4)
    
    return result