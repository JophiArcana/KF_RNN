import itertools
import os
from collections import OrderedDict
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.infrastructure.config import schema
from kf_rnn.infrastructure.config.schema import ExperimentConfig
from kf_rnn.infrastructure.experiment.results import InfoGrid
from kf_rnn.infrastructure.experiment.sweep import (
    is_system_param_of_type,
    is_system_nonauxiliary_param_of_type,
    support_param_targets_type,
    support_param_is_nonauxiliary_of_type,
)
from ecliseutils.labeled_array import LabeledArray, LabeledDataset
from kf_rnn.infrastructure.settings import DEVICE
from kf_rnn.infrastructure.static import (
    DATASET_SUPPORT_PARAMS,
    INFO_FIELDS,
    PARAM_GROUP_FORMATTER,
    TRAINING_DATASET_TYPES,
)
from kf_rnn.system.base import SystemGroup


def _supports_dataset_condition(HP: ExperimentConfig, ds_type: str) -> Callable[[str], bool]:
    """Forbid sweeping over parameters that determine the shape of the ``ds_type``
    dataset. Sweeping the *train* counterpart is allowed only when the ``ds_type``
    split is explicitly pinned (so the swept train value never leaks into it)."""
    def condition(n: str) -> bool:
        segs = n.split(".")
        if segs[0] != "dataset" or ".".join(segs[1:-1]) not in DATASET_SUPPORT_PARAMS:
            return True
        if segs[-1] == ds_type:
            return False
        if segs[-1] == TRAINING_DATASET_TYPES[0]:
            split_cfg: schema.Split = eu.rgetattr(HP, ".".join(segs[:-1]))
            return getattr(split_cfg, ds_type) is not None
        return True
    return condition

def _construct_dependency_dict_and_params_dataset(
        HP: ExperimentConfig,
        iterparams: list[tuple[str, dict[str, Any]]],
        assertion_conditions: Iterable[tuple[Callable[[str], bool], str]] = ()
) -> tuple[OrderedDict[str, tuple[int, list[str]]], LabeledDataset]:

    # SECTION: Set up the hyperparameters in data structures that make if convenient to perform the iteration
    # DONE: Set up 0d NumPy arrays to preserve objects before adding iterated hyperparameters
    dependency_dict, dataset = OrderedDict(), OrderedDict()
    for param_group, params in iterparams:
        params = eu.flatten_nested_dict(params)
        for n, v in params.items():
            for condition, message in assertion_conditions:
                assert condition(n), message

            if n != "name":
                # Register the swept path on the base config so split leaves like
                # ``distribution.test`` become visible to ``config_leaves``-based
                # support classification (which keys off path presence, not value).
                # Graft a single representative value rather than the whole value
                # list: whole-branch overrides (e.g. the polymorphic ``model``
                # config) must keep their structural type so deeper dotted paths
                # like ``model.ir_length`` can still be registered on top of them.
                representative = v[0] if isinstance(v, (list, tuple)) and len(v) else v
                eu.rsetattr(HP, n, schema.copy_config(representative))

        # DONE: Need to pre-broadcast at some point before constructing Dataset structures or else they will do something stupid
        vs = (*map(np.array, params.values()),)
        param_group_shape = np.broadcast_shapes(*(v.shape for v in vs))
        _vs = (np.broadcast_to(v, param_group_shape[-v.ndim:]) for v in vs)

        dim_names = [
            PARAM_GROUP_FORMATTER.format(param_group, d)
            for d in range(-len(param_group_shape), 0)
        ]
        for k, _v in zip(params.keys(), _vs):
            dataset[k] = LabeledArray(_v, dim_names[-_v.ndim:])
            for dn, dim in zip(dataset[k].dims, param_group_shape[-_v.ndim:]):
                dependency_dict.setdefault(dn, (dim, []))[1].append(k)

    return dependency_dict, LabeledDataset(dataset)

def _filter_dimensions_if_any_satisfy_condition(
        dependency_dict: dict[str, tuple[int, list[str]]],
        condition: Callable[[str], Any]
) -> OrderedDict[str, int]:
    return OrderedDict([
        (k, d) for k, (d, dependencies) in dependency_dict.items()
        if any(condition(param) for param in dependencies)
    ])

def _iterate_HP_with_params(
        HP: ExperimentConfig,
        dimensions: OrderedDict[str, int],
        params_dataset: LabeledDataset,
) -> Iterable[tuple[OrderedDict[str, int], ExperimentConfig]]:
    """Yield a fresh per-cell ``ExperimentConfig`` for each sweep index.

    Each cell config is a structural copy of the base config with the swept
    values applied functionally — no shared mutable tree. Whole-branch
    overrides (e.g. swapping the entire ``model`` config) are applied before
    deeper dotted overrides so the latter land on the swapped-in branch.
    """
    for idx in itertools.product(*map(range, dimensions.values())):
        dict_idx = OrderedDict([*zip(dimensions.keys(), idx)])
        sub_HP = schema.copy_config(HP)
        overrides = sorted(
            params_dataset.take(indices=dict_idx).items(),
            key=lambda kv: kv[0].count("."),
        )
        for n, v in overrides:
            # Only apply overrides that ``take`` reduced to a scalar for this
            # cell: when iterating a subset of the sweep dimensions, params over
            # un-iterated dims stay as arrays and must keep their base value
            # (grafting an array would clobber whole-branch configs like
            # ``model`` and break deeper paths like ``model.ir_length``).
            if v.values.ndim != 0:
                continue
            # Copy applied values too: sweep-axis values (e.g. model configs)
            # are shared across cells and must never be mutated in place.
            eu.rsetattr(sub_HP, n, schema.copy_config(v.values[()]))
        schema.propagate_problem_shape(sub_HP)
        yield dict_idx, sub_HP

def _map_HP_with_params(
        HP: ExperimentConfig,
        dimensions: OrderedDict[str, int],
        params_dataset: LabeledDataset,
        func: Callable[[OrderedDict[str, int], ExperimentConfig], Any],
        dtype: type
) -> LabeledArray:
    result_arr = LabeledArray(np.empty([*dimensions.values()], dtype=dtype), [*dimensions.keys()])
    for dict_idx, sub_HP in _iterate_HP_with_params(HP, dimensions, params_dataset):
        result_arr.put(dict_idx, func(dict_idx, sub_HP))
    return result_arr

def _get_split_param_dimarr(
        HP: ExperimentConfig,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: LabeledDataset,
        param: str, ds_type: str, dtype: type
):
    """Per-sweep-cell values of ``<param>.for_split(ds_type)`` (train-fallback)."""
    filter_dimensions = _filter_dimensions_if_any_satisfy_condition(dimensions, f"{param}.{ds_type}".__eq__)
    return _map_HP_with_params(
        HP, filter_dimensions, params_dataset,
        lambda _, sub_HP: eu.rgetattr(sub_HP, param).for_split(ds_type), dtype=dtype
    )

def _resolve_system_params(
        HP: ExperimentConfig,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: LabeledDataset,
        info_dict: OrderedDict[str, OrderedDict[str, LabeledArray]],
        ds_type: str,
        save_dict: dict[str, dict[str, Any]],
        system_param_dimensions: OrderedDict[str, int],
) -> LabeledArray:
    """Resolve the array of system *parameters* (matrices) for a dataset type.

    Precondition: at least one system support hyperparameter targets ``ds_type``.

    - if previously-saved params exist, reuse them;
    - else if a non-auxiliary system parameter is swept for this type, sample fresh ones;
    - else default to the training split's system parameters.
    """
    if "system_params" in save_dict and ds_type in save_dict["system_params"]:
        print(f"System matrices found for dataset type {ds_type}")
        return save_dict["system_params"][ds_type]

    system_support_hyperparameters = schema.config_leaves(HP.system).keys()
    if any(support_param_is_nonauxiliary_of_type(param, ds_type) for param in system_support_hyperparameters):
        n_systems_arr = _get_split_param_dimarr(HP, dimensions, params_dataset, "dataset.n_systems", ds_type, dtype=int)
        max_n_systems = n_systems_arr.max()

        def sample_system_parameters_with_sub_hyperparameters(_, sub_HP: ExperimentConfig) -> TensorDict:
            return sub_HP.system.distribution.for_split(ds_type).sample_parameters(
                schema.resolve_splits(sub_HP.system), (HP.experiment.n_experiments, max_n_systems)
            )

        print(f"Sampling new system matrices for dataset type {ds_type}")
        return _map_HP_with_params(
            HP, system_param_dimensions, params_dataset,
            sample_system_parameters_with_sub_hyperparameters, dtype=object
        )
    else:
        print(f"Defaulting to train system matrices for dataset type {ds_type}")
        return info_dict[TRAINING_DATASET_TYPES[0]]["system_params"]

def _construct_systems(
        HP: ExperimentConfig,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: LabeledDataset,
        ds_type: str,
        system_param_dimensions: OrderedDict[str, int],
        system_params_arr: LabeledArray,
) -> LabeledArray:
    """Build the array of SystemGroups for a dataset type from resolved system params."""
    system_dimensions = OrderedDict(system_param_dimensions)
    system_dimensions.update(_filter_dimensions_if_any_satisfy_condition(
        dimensions, lambda param: is_system_param_of_type(param, ds_type)
    ))

    def construct_system_with_sub_hyperparameters(dict_idx: OrderedDict[str, int], sub_HP: ExperimentConfig) -> SystemGroup:
        dist = sub_HP.system.distribution.for_split(ds_type)
        system_params = eu.take_from_dim_array(system_params_arr, dict_idx).values[()]

        system_cfg = schema.resolve_splits(sub_HP.system, ds_type)
        return dist.system_type(system_cfg, system_params)

    return _map_HP_with_params(
        HP, system_dimensions, params_dataset,
        construct_system_with_sub_hyperparameters, dtype=SystemGroup,
    )

def _resolve_dataset(
        HP: ExperimentConfig,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: LabeledDataset,
        info_dict: OrderedDict[str, OrderedDict[str, LabeledArray]],
        ds_type: str,
        save_dict: dict[str, dict[str, Any]],
        systems_arr: LabeledArray,
        system_support_hyperparameters: Iterable[str],
) -> LabeledArray:
    """Resolve the rollout dataset for a dataset type.

    - if a previously-saved dataset exists, reuse it;
    - else if any system or dataset support parameter is swept for this type, sample fresh;
    - else default to the training dataset.
    """
    if "dataset" in save_dict and ds_type in save_dict["dataset"]:
        print(f"Dataset found for dataset type {ds_type}")
        return save_dict["dataset"][ds_type]

    dataset_support_hyperparameters = schema.config_leaves(HP.dataset).keys()
    if not any(support_param_targets_type(param, ds_type) for param in (
        *system_support_hyperparameters,
        *dataset_support_hyperparameters,
    )):
        print(f"Defaulting to train dataset for dataset type {ds_type}")
        return info_dict[TRAINING_DATASET_TYPES[0]]["dataset"]

    print(f"Generating new dataset for dataset type {ds_type}")
    dataset_dimensions = OrderedDict([*zip(systems_arr.dims, systems_arr.shape)])

    n_traces_arr, total_sequence_length_arr = eu.broadcast_dim_arrays(
        _get_split_param_dimarr(HP, dimensions, params_dataset, "dataset.n_traces", ds_type, dtype=int),
        _get_split_param_dimarr(HP, dimensions, params_dataset, "dataset.total_sequence_length", ds_type, dtype=int)
    )
    sequence_length_arr = eu.ceildiv(total_sequence_length_arr, n_traces_arr)

    max_n_traces = n_traces_arr.max()
    max_sequence_length = sequence_length_arr.max()
    max_batch_size = (HP.experiment.ensemble_size if ds_type == TRAINING_DATASET_TYPES[0] else 1) * max_n_traces

    def sample_dataset_with_sub_hyperparameters(dict_idx: OrderedDict[str, int], _) -> TensorDict:
        sg = eu.take_from_dim_array(systems_arr, dict_idx).values[()]
        dataset = sg.generate_dataset(max_batch_size, max_sequence_length).detach()

        if ds_type == TRAINING_DATASET_TYPES[0]:
            return dataset.unflatten(2, (HP.experiment.ensemble_size, max_n_traces)).permute(0, 2, 1, 3, 4)
        else:
            return dataset.unsqueeze(1).expand(
                HP.experiment.n_experiments,
                HP.experiment.ensemble_size,
                sg.group_shape[1],
                max_n_traces,
                max_sequence_length
            )

    return _map_HP_with_params(
        HP, dataset_dimensions, params_dataset,
        sample_dataset_with_sub_hyperparameters, dtype=object
    )

def _construct_info_dict(
        HP: ExperimentConfig,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: LabeledDataset,
        info_dict: OrderedDict[str, OrderedDict[str, LabeledArray]],
        ds_type: str,
        save_dict: dict[str, dict[str, Any]],
        systems: dict[str, LabeledArray] = None,
) -> OrderedDict[str, LabeledArray]:
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

    system_support_hyperparameters = schema.config_leaves(HP.system).keys()
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

        def retrieve_system_params_from_system(dict_idx: OrderedDict[str, int], _) -> TensorDict:
            return systems_arr.take(indices=dict_idx).values.ravel()[0].td()

        system_params_arr = _map_HP_with_params(
            HP, system_param_dimensions, params_dataset,
            retrieve_system_params_from_system, dtype=object,
        )

    result["system_params"] = system_params_arr
    result["systems"] = systems_arr
    result["dataset"] = _resolve_dataset(
        HP, dimensions, params_dataset, info_dict, ds_type, save_dict,
        systems_arr, system_support_hyperparameters,
    )
    return result

def _construct_info_dict_from_dataset_types(
        HP: ExperimentConfig,
        dimensions: OrderedDict[str, tuple[int, list[str]]],
        params_dataset: LabeledDataset,
        info_dict: OrderedDict[str, OrderedDict[str, LabeledArray]],
        dataset_types: Sequence[str],
        output_dir: str,
        default_systems: dict[str, LabeledArray] = None
) -> OrderedDict[str, OrderedDict[str, LabeledArray]]:
    saved_fname_dict, unsaved_fname_dict = {}, {}
    for attr in INFO_FIELDS:
        fname = f"{output_dir}/{attr}.pt"
        (saved_fname_dict if os.path.exists(fname) else unsaved_fname_dict)[attr] = fname

    save_dict = {}
    if output_dir is not None:
        for attr, fname in saved_fname_dict.items():
            save_dict[attr] = eu.torch_load(fname)
            print(f"Loaded {fname} from disk.")

    for ds_type in dataset_types:
        info_dict[ds_type] = _construct_info_dict(HP, dimensions, params_dataset, info_dict, ds_type, save_dict, systems=default_systems)

    if output_dir is not None:
        for attr, fname in unsaved_fname_dict.items():
            torch.save(OrderedDict([(k, v[attr]) for k, v in info_dict.items()]), fname)

    return info_dict

def _process_info_dict(ds_info: OrderedDict[str, LabeledArray]) -> InfoGrid:
    fields = OrderedDict(filter(
        lambda p: p[0] in INFO_FIELDS,
        zip(ds_info.keys(), eu.broadcast_dim_arrays(*ds_info.values()))
    ))
    return InfoGrid.from_fields(fields)
