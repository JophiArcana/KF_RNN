import itertools
import os
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from dimarray import DimArray, Dataset

from infrastructure import utils
from infrastructure.experiment.sweep import (
    is_dataset_param_of_type,
    is_system_param_of_type,
    is_system_nonauxiliary_param_of_type,
    support_param_targets_type,
    support_param_is_nonauxiliary_of_type,
)
from infrastructure.settings import DEVICE
from infrastructure.static import (
    DATASET_SUPPORT_PARAMS,
    INFO_DTYPE,
    PARAM_GROUP_FORMATTER,
    TRAINING_DATASET_TYPES,
)
from infrastructure.utils import PTR
from system.base import SystemGroup


def _supports_dataset_condition(HP: Namespace, ds_type: str) -> Callable[[str], bool]:
    train_support_names = [f"dataset.{TRAINING_DATASET_TYPES[0]}.{p}" for p in DATASET_SUPPORT_PARAMS]
    ds_support_names = [f"dataset.{ds_type}.{p}" for p in DATASET_SUPPORT_PARAMS]

    def condition(n: str) -> bool:
        if n in ds_support_names:
            return False
        else:
            for tsn, dsn in zip(train_support_names, ds_support_names):
                if n == tsn:
                    return utils.rhasattr(HP, dsn)
            return True
    return condition

def _construct_dependency_dict_and_params_dataset(
        HP: Namespace,
        iterparams: list[tuple[str, dict[str, Any]]],
        assertion_conditions: Iterable[tuple[Callable[[str], bool], str]] = ()
) -> tuple[OrderedDict[str, tuple[int, list[str]]], Dataset]:

    HP.experiment.model_shape = (HP.experiment.n_experiments, HP.experiment.ensemble_size)

    # SECTION: Set up the hyperparameters in data structures that make if convenient to perform the iteration
    # DONE: Set up 0d NumPy arrays to preserve objects before adding iterated hyperparameters
    dependency_dict, dataset = OrderedDict(), OrderedDict()
    for param_group, params in iterparams:
        params = utils.flatten_nested_dict(params)
        for n, v in params.items():
            for condition, message in assertion_conditions:
                assert condition(n), message

            if n != "name":
                utils.rsetattr(HP, n, v)

        # DONE: Need to pre-broadcast at some point before constructing Dataset structures or else they will do something stupid
        vs = (*map(np.array, params.values()),)
        param_group_shape = np.broadcast_shapes(*(v.shape for v in vs))
        _vs = (np.broadcast_to(v, param_group_shape[-v.ndim:]) for v in vs)

        dim_names = [
            PARAM_GROUP_FORMATTER.format(param_group, d)
            for d in range(-len(param_group_shape), 0)
        ]
        for k, _v in zip(params.keys(), _vs):
            dataset[k] = DimArray(_v, dims=dim_names[-_v.ndim:])
            for dn, dim in zip(dataset[k].dims, param_group_shape[-_v.ndim:]):
                dependency_dict.setdefault(dn, (dim, []))[1].append(k)

    return dependency_dict, Dataset(dataset)

def _filter_dimensions_if_any_satisfy_condition(
        dependency_dict: dict[str, tuple[int, list[str]]],
        condition: Callable[[str], Any]
) -> OrderedDict[str, int]:
    return OrderedDict([
        (k, d) for k, (d, dependencies) in dependency_dict.items()
        if any(condition(param) for param in dependencies)
    ])

def _iterate_HP_with_params(
        HP: Namespace,
        dimensions: OrderedDict[str, int],
        params_dataset: Dataset,
) -> Iterable[tuple[OrderedDict[str, int], Namespace]]:
    for idx in itertools.product(*map(range, dimensions.values())):
        dict_idx = OrderedDict([*zip(dimensions.keys(), idx)])
        sub_HP = utils.deepcopy_namespace(HP)
        for n, v in params_dataset.take(indices=dict_idx).items():
            utils.rsetattr(sub_HP, n, v.values[()])
        yield dict_idx, sub_HP

def _map_HP_with_params(
        HP: Namespace,
        dimensions: OrderedDict[str, int],
        params_dataset: Dataset,
        func: Callable[[OrderedDict[str, int], Namespace], Any],
        dtype: type
) -> DimArray:
    result_arr = DimArray(np.empty([*dimensions.values()], dtype=dtype), dims=[*dimensions.keys()])
    for dict_idx, sub_HP in _iterate_HP_with_params(HP, dimensions, params_dataset):
        result_arr.put(indices=dict_idx, values=func(dict_idx, sub_HP))
    return result_arr

def _get_param_dimarr(
        HP: Namespace,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: Dataset,
        param: str, dtype: type
):
    filter_dimensions = _filter_dimensions_if_any_satisfy_condition(dimensions, param.__eq__)
    return _map_HP_with_params(
        HP, filter_dimensions, params_dataset,
        lambda _, sub_HP: utils.rgetattr(sub_HP, param), dtype=dtype
    )

def _resolve_system_params(
        HP: Namespace,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: Dataset,
        info_dict: OrderedDict[str, OrderedDict[str, DimArray]],
        ds_type: str,
        save_dict: dict[str, dict[str, Any]],
        system_param_dimensions: OrderedDict[str, int],
) -> DimArray:
    """Resolve the array of system *parameters* (matrices) for a dataset type.

    Precondition: at least one system support hyperparameter targets ``ds_type``.

    - if previously-saved params exist, reuse them;
    - else if a non-auxiliary system parameter is swept for this type, sample fresh ones;
    - else default to the training split's system parameters.
    """
    if "system_params" in save_dict and ds_type in save_dict["system_params"]:
        print(f"System matrices found for dataset type {ds_type}")
        return save_dict["system_params"][ds_type]

    system_support_hyperparameters = utils.nested_vars(HP.system).keys()
    if any(support_param_is_nonauxiliary_of_type(param, ds_type) for param in system_support_hyperparameters):
        n_systems_arr = _get_param_dimarr(HP, dimensions, params_dataset, f"dataset.n_systems.{ds_type}", dtype=int)
        max_n_systems = n_systems_arr.max()

        def sample_system_parameters_with_sub_hyperparameters(_, sub_HP: Namespace) -> PTR:
            return PTR(utils.rgetattr(sub_HP, f"system.distribution.{ds_type}").sample_parameters(
                utils.index_defaulting_with_attr(sub_HP.system), (HP.experiment.n_experiments, max_n_systems)
            ))

        print(f"Sampling new system matrices for dataset type {ds_type}")
        return _map_HP_with_params(
            HP, system_param_dimensions, params_dataset,
            sample_system_parameters_with_sub_hyperparameters, dtype=PTR
        )
    else:
        print(f"Defaulting to train system matrices for dataset type {ds_type}")
        return info_dict[TRAINING_DATASET_TYPES[0]]["system_params"]

def _construct_systems(
        HP: Namespace,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: Dataset,
        ds_type: str,
        system_param_dimensions: OrderedDict[str, int],
        system_params_arr: DimArray,
) -> DimArray:
    """Build the array of SystemGroups for a dataset type from resolved system params."""
    system_dimensions = OrderedDict(system_param_dimensions)
    system_dimensions.update(_filter_dimensions_if_any_satisfy_condition(
        dimensions, lambda param: is_system_param_of_type(param, ds_type)
    ))

    def construct_system_with_sub_hyperparameters(dict_idx: OrderedDict[str, int], sub_HP: Namespace) -> SystemGroup:
        dist = utils.rgetattr(sub_HP, f"system.distribution.{ds_type}")
        system_params = utils.take_from_dim_array(system_params_arr, dict_idx).values[()].obj

        sub_HP = utils.index_defaulting_with_attr(sub_HP, ds_type)
        return dist.system_type(sub_HP.system, system_params)

    return _map_HP_with_params(
        HP, system_dimensions, params_dataset,
        construct_system_with_sub_hyperparameters, dtype=SystemGroup,
    )

def _resolve_dataset(
        HP: Namespace,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: Dataset,
        info_dict: OrderedDict[str, OrderedDict[str, DimArray]],
        ds_type: str,
        save_dict: dict[str, dict[str, Any]],
        systems_arr: DimArray,
        system_support_hyperparameters: Iterable[str],
) -> DimArray:
    """Resolve the rollout dataset for a dataset type.

    - if a previously-saved dataset exists, reuse it;
    - else if any system or dataset support parameter is swept for this type, sample fresh;
    - else default to the training dataset.
    """
    if "dataset" in save_dict and ds_type in save_dict["dataset"]:
        print(f"Dataset found for dataset type {ds_type}")
        return save_dict["dataset"][ds_type]

    dataset_support_hyperparameters = utils.nested_vars(HP.dataset).keys()
    if not any(support_param_targets_type(param, ds_type) for param in (
        *system_support_hyperparameters,
        *dataset_support_hyperparameters,
    )):
        print(f"Defaulting to train dataset for dataset type {ds_type}")
        return info_dict[TRAINING_DATASET_TYPES[0]]["dataset"]

    print(f"Generating new dataset for dataset type {ds_type}")
    dataset_dimensions = OrderedDict([*zip(systems_arr.dims, systems_arr.shape)])

    n_traces_arr, total_sequence_length_arr = utils.broadcast_dim_arrays(
        _get_param_dimarr(HP, dimensions, params_dataset, f"dataset.n_traces.{ds_type}", dtype=int),
        _get_param_dimarr(HP, dimensions, params_dataset, f"dataset.total_sequence_length.{ds_type}", dtype=int)
    )
    sequence_length_arr = utils.ceildiv(total_sequence_length_arr, n_traces_arr)

    max_n_traces = n_traces_arr.max()
    max_sequence_length = sequence_length_arr.max()
    max_batch_size = (HP.experiment.ensemble_size if ds_type == TRAINING_DATASET_TYPES[0] else 1) * max_n_traces

    def sample_dataset_with_sub_hyperparameters(dict_idx: OrderedDict[str, int], _) -> PTR:
        sg = utils.take_from_dim_array(systems_arr, dict_idx).values[()]
        dataset = sg.generate_dataset(max_batch_size, max_sequence_length).detach()

        if ds_type == TRAINING_DATASET_TYPES[0]:
            return PTR(dataset.unflatten(2, (HP.experiment.ensemble_size, max_n_traces)).permute(0, 2, 1, 3, 4))
        else:
            return PTR(dataset.unsqueeze(1).expand(
                HP.experiment.n_experiments,
                HP.experiment.ensemble_size,
                sg.group_shape[1],
                max_n_traces,
                max_sequence_length
            ))

    return _map_HP_with_params(
        HP, dataset_dimensions, params_dataset,
        sample_dataset_with_sub_hyperparameters, dtype=PTR
    )

def _construct_info_dict(
        HP: Namespace,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: Dataset,
        info_dict: OrderedDict[str, OrderedDict[str, DimArray]],
        ds_type: str,
        save_dict: dict[str, dict[str, Any]],
        systems: dict[str, DimArray] = None,
) -> OrderedDict[str, DimArray]:
    """Resolve the systems, system parameters, and rollout dataset for one dataset type.

    Resolution order for systems/parameters:
    - if explicit (non-auxiliary) system matrix parameters are swept -> sample new matrices;
    - else if only auxiliary parameters change -> reuse train matrices but rebuild systems;
    - else -> default to the training systems entirely.
    Saved systems (from disk) short-circuit this and are used directly.
    """
    result = OrderedDict()
    # DONE: Check for saved systems, if not then construct distribution and save systems
    if "systems" in save_dict:
        systems = save_dict["systems"]

    system_support_hyperparameters = utils.nested_vars(HP.system).keys()
    system_param_dimensions = _filter_dimensions_if_any_satisfy_condition(
        dimensions, lambda param: is_system_nonauxiliary_param_of_type(param, ds_type)
    )

    if systems is None or ds_type not in systems:
        if any(support_param_targets_type(param, ds_type) for param in system_support_hyperparameters):
            system_params_arr = _resolve_system_params(
                HP, dimensions, params_dataset, info_dict, ds_type, save_dict, system_param_dimensions,
            )
            systems_arr = _construct_systems(
                HP, dimensions, params_dataset, ds_type, system_param_dimensions, system_params_arr,
            )
        else:
            print(f"Defaulting to train systems for dataset type {ds_type}")
            systems_arr = info_dict[TRAINING_DATASET_TYPES[0]]["systems"]
            system_params_arr = info_dict[TRAINING_DATASET_TYPES[0]]["system_params"]
    else:
        print(f"Systems found for dataset type {ds_type}")
        systems_arr = systems[ds_type]

        def retrieve_system_params_from_system(dict_idx: OrderedDict[str, int], _) -> PTR:
            return PTR(systems_arr.take(indices=dict_idx).values.ravel()[0].td())

        system_params_arr = _map_HP_with_params(
            HP, system_param_dimensions, params_dataset,
            retrieve_system_params_from_system, dtype=PTR,
        )

    result["system_params"] = system_params_arr
    result["systems"] = systems_arr
    result["dataset"] = _resolve_dataset(
        HP, dimensions, params_dataset, info_dict, ds_type, save_dict,
        systems_arr, system_support_hyperparameters,
    )
    return result

def _construct_info_dict_from_dataset_types(
        HP: Namespace,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: Dataset,
        info_dict: OrderedDict[str, OrderedDict[str, DimArray]],
        dataset_types: Sequence[str],
        output_dir: str,
        default_systems: dict[str, DimArray] = None
) -> OrderedDict[str, OrderedDict[str, DimArray]]:
    saved_fname_dict, unsaved_fname_dict = {}, {}
    for attr in INFO_DTYPE.names:
        fname = f"{output_dir}/{attr}.pt"
        (saved_fname_dict if os.path.exists(fname) else unsaved_fname_dict)[attr] = fname

    save_dict = {}
    if output_dir is not None:
        for attr, fname in saved_fname_dict.items():
            save_dict[attr] = utils.torch_load(fname)
            print(f"Loaded {fname} from disk.")

    for ds_type in dataset_types:
        info_dict[ds_type] = _construct_info_dict(HP, dimensions, params_dataset, info_dict, ds_type, save_dict, systems=default_systems)

    if output_dir is not None:
        for attr, fname in unsaved_fname_dict.items():
            torch.save(OrderedDict([(k, v[attr]) for k, v in info_dict.items()]), fname)

    return info_dict

def _process_info_dict(ds_info: OrderedDict[str, DimArray]) -> DimArray:
    ds_info = OrderedDict(filter(
        lambda p: p[0] in INFO_DTYPE.names,
        zip(ds_info.keys(), utils.broadcast_dim_arrays(*ds_info.values()))
    ))
    ref = (*ds_info.values(),)[0]

    info_recarr = np.recarray(ref.shape, dtype=INFO_DTYPE)
    for k, v in ds_info.items():
        setattr(info_recarr, k, v.values)
    return DimArray(info_recarr, dims=ref.dims)

def _populate_values(HP: Namespace) -> None:
    total_sequence_length = HP.dataset.total_sequence_length.train
    n_traces = HP.dataset.n_traces.train
    HP.dataset.sequence_length = utils.DefaultingParameter(train=utils.ceildiv(total_sequence_length, n_traces))





