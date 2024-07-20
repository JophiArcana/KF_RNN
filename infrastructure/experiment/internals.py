import copy
import itertools
import os
from argparse import Namespace
from collections import OrderedDict

import torch
from dimarray import DimArray, Dataset

from infrastructure import utils
from infrastructure.experiment.static import *
from infrastructure.settings import DEVICE
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

def _prologue(
        HP: Namespace,
        iterparams: List[Tuple[str, Dict[str, Any]]],
        assertion_conditions: Iterable[Tuple[Callable[[str], bool], str]] = ()
) -> Tuple[OrderedDict[str, Tuple[int, List[str]]], Dataset]:

    HP.experiment.model_shape = (HP.experiment.n_experiments, HP.experiment.ensemble_size)

    # SECTION: Set up the hyperparameters in data structures that make if convenient to perform the iteration
    # DONE: Set up 0d NumPy arrays to preserve objects before adding iterated hyperparameters
    dependency_dict, dataset = OrderedDict(), OrderedDict()
    for param_group, params in iterparams:
        for n, v in params.items():
            for condition, message in assertion_conditions:
                assert condition(n), message

            if n != "name":
                utils.rsetattr(HP, n, v)

        # DONE: Need to pre-broadcast at some point before constructing Dataset structures or else they will do something stupid
        _vs, param_group_shape = utils.broadcast_arrays_preserve_ndims(*map(np.array, params.values()))

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
        dependency_dict: Dict[str, Tuple[int, List[str]]],
        condition: Callable[[str], bool]
) -> OrderedDict[str, int]:
    return OrderedDict([
        (k, d) for k, (d, dependencies) in dependency_dict.items()
        if any(condition(param) for param in dependencies)
    ])

def _iterate_HP_with_params(
        HP: Namespace,
        dimensions: OrderedDict[str, int],
        params_dataset: Dataset,
) -> Iterable[Tuple[OrderedDict[str, int], Namespace]]:
    for idx in itertools.product(*map(range, dimensions.values())):
        sub_HP = copy.deepcopy(HP)
        for n, v in params_dataset.take(indices=idx).items():
            utils.rsetattr(sub_HP, n, v.values[()])
        yield OrderedDict([*zip(dimensions.keys(), idx)]), sub_HP

def _map_HP_with_params(
        HP: Namespace,
        dimensions: OrderedDict[str, int],
        params_dataset: Dataset,
        func: Callable[[OrderedDict[str, int], Namespace], Any],
        dtype: type
) -> DimArray:
    result_arr = DimArray(np.empty([*dimensions.values()], dtype=dtype), dims=[*dimensions.keys()])
    for dict_idx, sub_HP in _iterate_HP_with_params(HP, dimensions, params_dataset):
        result_arr[dict_idx] = func(dict_idx, sub_HP)
    return result_arr

def _construct_info_dict(
        HP: Namespace,
        dimensions: OrderedDict[str, Tuple[int, List[str]]],
        params_dataset: Dataset,
        info_dict: OrderedDict[str, OrderedDict[str, DimArray]],
        ds_type: str,
        save_dict: Dict[str, Dict[str, Any]],
        systems: Dict[str, DimArray] = None,
) -> OrderedDict[str, DimArray]:

    def _rgetattr_default(format_str: str) -> Any:
        attr = format_str.format(ds_type if utils.rhasattr(HP, format_str.format(ds_type)) else TRAINING_DATASET_TYPES[0])
        return params_dataset.get(attr, DimArray(utils.array_of(utils.rgetattr(HP, attr))))

    result = OrderedDict()
    # Dataset setup
    # DONE: Check for saved systems, if not then construct distribution and save systems
    if "systems" in save_dict:
        if utils.rhasattr(HP, f"dataet.{ds_type}.system.distribution"):
            print(f"Restoring distributions found for dataset type {ds_type}")
            result["distributions"] = DimArray(utils.rgetattr(HP, f"dataset.{ds_type}.system.distribution"))

        systems = save_dict["systems"]

    if systems is None or ds_type not in systems:
        if utils.rhasattr(HP, f"dataset.{ds_type}.system"):
            # DONE: If no saved systems, then construct distributions based on the provided sample functions
            if utils.rhasattr(HP, f"dataset.{ds_type}.system.distribution"):
                distribution_dimensions = _filter_dimensions_if_any_satisfy_condition(
                    dimensions, lambda param: param == f"dataset.{ds_type}.system.distribution"
                )
                distributions_arr = _map_HP_with_params(
                    HP, distribution_dimensions, params_dataset,
                    lambda _, sub_HP: utils.rgetattr(sub_HP, f"dataset.{ds_type}.system.distribution"), dtype=object
                )
            else:
                print(f"Defaulting to train distributions for dataset type {ds_type}")
                distributions_arr = info_dict[TRAINING_DATASET_TYPES[0]]["distributions"]
                distribution_dimensions = OrderedDict([*zip(distributions_arr.dims, distributions_arr.shape)])
            result["distributions"] = distributions_arr


            # TODO: Sample minimal system parameters from array of distributions in the shape of (n_experiments, n_systems)
            system_param_dimensions = distribution_dimensions
            system_param_dimensions.update(_filter_dimensions_if_any_satisfy_condition(
                dimensions, lambda param: param == "system.S_D" or param.startswith("system.problem_shape.") or param.startswith(f"dataset.{ds_type}.system.")
            ))

            def sample_system_parameters_with_sub_hyperparameters(_, sub_HP: Namespace) -> PTR:
                return PTR(utils.rgetattr(sub_HP, f"dataset.{ds_type}.system.distribution").sample_parameters(
                    sub_HP.system, (sub_HP.experiment.n_experiments, utils.rgetattr(sub_HP, f"dataset.{ds_type}.system.n_systems"))
                ))

            print(f"Sampling new systems for dataset type {ds_type}")
            system_params_arr = _map_HP_with_params(
                HP, system_param_dimensions, params_dataset,
                sample_system_parameters_with_sub_hyperparameters, dtype=PTR
            )

            # TODO: Sample systems from array of distributions in the shape of (n_experiments, n_systems)
            system_dimensions = system_param_dimensions
            system_dimensions.update(_filter_dimensions_if_any_satisfy_condition(
                dimensions, lambda param: param.startswith("system.")
            ))

            def construct_system_with_sub_hyperparameters(dict_idx: OrderedDict[str, int], sub_HP: Namespace) -> SystemGroup:
                dist = utils.take_from_dim_array(distributions_arr, dict_idx).values[()]
                system_params = utils.take_from_dim_array(system_params_arr, dict_idx).values[()].obj
                return dist.system_type(sub_HP.system, system_params)

            systems_arr = _map_HP_with_params(
                HP, system_dimensions, params_dataset,
                construct_system_with_sub_hyperparameters, dtype=SystemGroup
            )
        else:
            print(f"Defaulting to train systems for dataset type {ds_type}")
            systems_arr = info_dict[TRAINING_DATASET_TYPES[0]]["systems"]
    else:
        print(f"Systems found for dataset type {ds_type}")
        systems_arr = systems[ds_type]

    # DONE: Refresh the systems with the same parameters so that gradients will pass through properly in post-experiment analysis
    # systems_arr = utils.multi_map(
    #     lambda sg: type(sg)(SHP, sg.td()),
    #     systems_arr, dtype=SystemGroup
    # )
    result["systems"] = systems_arr

    # DONE: Check for saved dataset, otherwise sample and save datasets
    if "dataset" not in save_dict or ds_type not in save_dict["dataset"]:
        if utils.rhasattr(HP, f"dataset.{ds_type}"):
            print(f"Generating new dataset for dataset type {ds_type}")
            dataset_dimensions = OrderedDict([*zip(systems_arr.dims, systems_arr.shape)])

            dataset_size_arr, total_sequence_length_arr = utils.broadcast_dim_arrays(
                _rgetattr_default("dataset.{0}.dataset_size"),
                _rgetattr_default("dataset.{0}.total_sequence_length")
            )
            sequence_length_arr = (total_sequence_length_arr - 1) // dataset_size_arr + 1

            max_dataset_size = dataset_size_arr.max()
            max_sequence_length = sequence_length_arr.max()
            max_batch_size = (HP.experiment.ensemble_size if ds_type == TRAINING_DATASET_TYPES[0] else 1) * max_dataset_size

            def sample_dataset_with_sub_hyperparameters(dict_idx: OrderedDict[str, int], sub_HP: Namespace) -> PTR:
                sg = utils.take_from_dim_array(systems_arr, dict_idx).values[()]
                dataset = sg.generate_dataset(max_batch_size, max_sequence_length)

                if ds_type == TRAINING_DATASET_TYPES[0]:
                    return PTR(dataset.unflatten(2, (HP.experiment.ensemble_size, max_dataset_size)).permute(0, 2, 1, 3, 4))
                else:
                    return PTR(dataset.unsqueeze(1).expand(
                        HP.experiment.n_experiments,
                        HP.experiment.ensemble_size,
                        sg.group_shape[1],
                        max_dataset_size,
                        max_sequence_length
                    ))

            dataset_arr = _map_HP_with_params(
                HP, dataset_dimensions, params_dataset,
                sample_dataset_with_sub_hyperparameters, dtype=PTR
            )
        else:
            print(f"Defaulting to train dataset for dataset type {ds_type}")
            dataset_arr = info_dict[TRAINING_DATASET_TYPES[0]]["dataset"]
    else:
        print(f"Dataset found for dataset type {ds_type}")
        dataset_arr = save_dict["dataset"][ds_type]
    result["dataset"] = dataset_arr
    return result

def _construct_info_dict_from_dataset_types(
        HP: Namespace,
        dimensions: OrderedDict[str, Tuple[int, List[str]]],
        params_dataset: Dataset,
        info_dict: OrderedDict[str, OrderedDict[str, DimArray]],
        dataset_types: Sequence[str],
        output_dir: str,
        default_systems: Dict[str, DimArray] = None
) -> OrderedDict[str, OrderedDict[str, DimArray]]:
    saved_fname_dict, unsaved_fname_dict = {}, {}
    for attr in INFO_DTYPE.names:
        fname = f"{output_dir}/{attr}.pt"
        (saved_fname_dict if os.path.exists(fname) else unsaved_fname_dict)[attr] = fname

    save_dict = {}
    if output_dir is not None:
        for attr, fname in saved_fname_dict.items():
            save_dict[attr] = torch.load(fname, map_location=DEVICE)
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
    def _rgetattr_default(format_str: str, ds_type: str) -> Any:
        return utils.rgetattr_default(HP.dataset, format_str, ds_type, TRAINING_DATASET_TYPES[0])

    for ds_type, ds_config in vars(HP.dataset).items():
        if isinstance(ds_config, Namespace):
            dataset_size = _rgetattr_default("{0}.dataset_size", ds_type)
            total_sequence_length = _rgetattr_default("{0}.total_sequence_length", ds_type)
            ds_config.sequence_length = (total_sequence_length - 1) // dataset_size + 1




