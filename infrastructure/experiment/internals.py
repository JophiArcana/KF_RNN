import copy
from argparse import Namespace
from collections import OrderedDict
from typing import *

import torch
from dimarray import DimArray, Dataset

from infrastructure import utils
from infrastructure.experiment.static import *
from model.linear_system import LinearSystemGroup, AnalyticalKFGroup
from model.linear_system_distribution import LinearSystemDistribution


def _supports_dataset_condition(HP: Namespace, ds_type: str) -> Callable[[str, Any], bool]:
    train_support_names = [f"dataset.{TRAINING_DATASET_TYPES[0]}.{p}" for p in DATASET_SUPPORT_PARAMS]
    ds_support_names = [f"dataset.{ds_type}.{p}" for p in DATASET_SUPPORT_PARAMS]

    def condition(n: str, v: Any) -> bool:
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
        assertion_conditions: Iterable[Tuple[Callable[[str, Any], bool], str]] = ()
) -> Tuple[Namespace, List[Tuple[str, Dict[str, DimArray]]]]:

    # SECTION: Set up the hyperparameters in data structures that make if convenient to perform the iteration
    # DONE: Set up 0d NumPy arrays to preserve objects before adding iterated hyperparameters
    numpy_HP = copy.deepcopy(HP)
    utils.npfy_namespace(numpy_HP)

    _iterparams = []
    for param_group, params in iterparams:
        for n, v in params.items():
            for condition, message in assertion_conditions:
                assert condition(n, v), message

            if n != "name":
                utils.rsetattr(HP, n, v)

        # DONE: Need to pre-broadcast at some point before constructing Dataset structures or else they will do something stupid
        _vs = utils.broadcast_arrays_preserve_ndims(*map(np.array, params.values()))
        _iterparams.append((param_group, {
            k: DimArray(_v, dims=[
                PARAM_GROUP_FORMATTER.format(param_group, k)
                for k in range(-_v.ndim, 0)
            ])
            for k, _v in zip(params.keys(), _vs)
        }))
    iterparams = _iterparams

    # DONE: Update the hyperparameter copy so non-swept parameters are stored in standard ndarrays, while swept parameters are stored in DimArrays
    for param_group, params in iterparams:
        for n, v in params.items():
            utils.rsetattr(numpy_HP, n, v)

    return numpy_HP, iterparams

def _construct_dataset_from_iterparam(iterparam: Tuple[str, Dict[str, DimArray]]) -> Tuple[Dataset, Sequence[str], Sequence[int]]:
    param_group, params = iterparam

    param_group_dataset = Dataset(params)
    param_group_dim_names = (*(
        PARAM_GROUP_FORMATTER.format(param_group, k)
        for k in range(-param_group_dataset.ndim, 0)
    ),)
    param_group_shape = np.broadcast_shapes(*(v.shape for v in params.values()))

    return param_group_dataset, param_group_dim_names, param_group_shape

def _construct_info_dict(
        HP: Namespace,
        info_dict: Dict[str, OrderedDict[str, DimArray]],
        ds_type: str,
        save_dict: Dict[str, Dict[str, Any]],
        systems: Dict[str, DimArray] = None,
) -> OrderedDict[str, DimArray]:
    SHP, MHP, THP, DHP, EHP = map(vars(HP).__getitem__, ("system", "model", "train", "dataset", "experiment"))

    def _rgetattr_default(format_str: str) -> Any:
        return utils.rgetattr_default(DHP, format_str, ds_type, TRAINING_DATASET_TYPES[0])

    result = OrderedDict()
    # Dataset setup
    # DONE: Check for saved systems, if not then construct distribution and save systems
    if "systems" in save_dict:
        print(f"Systems found for dataset type {ds_type}")
        systems_arr = save_dict["systems"][ds_type]
    else:
        if systems is None or ds_type not in systems:
            if utils.rhasattr(DHP, f"{ds_type}.system"):
                # DONE: If no saved systems, then construct distributions based on the provided sample functions
                try:
                    sample_func_arr, = utils.broadcast_dim_arrays(
                        utils.rgetattr(DHP, f"{ds_type}.system.distribution.sample_func"),
                    )
                    distributions_arr = utils.multi_map(LinearSystemDistribution, sample_func_arr, dtype=LinearSystemDistribution)
                except AttributeError:
                    print(f"Defaulting to train distributions for dataset type {ds_type}")
                    distributions_arr = info_dict[TRAINING_DATASET_TYPES[0]]["distributions"]
                result["distributions"] = distributions_arr

                # DONE: Sample systems from array of distributions in the shape of (n_experiments, n_systems)
                SHP_arrs = OrderedDict(vars(SHP))
                broadcasted_arrs = utils.broadcast_dim_arrays(
                    distributions_arr,
                    _rgetattr_default("{0}.system.n_systems"),
                    *SHP_arrs.values()
                )
                distributions_arr, n_systems_arr = next(broadcasted_arrs), next(broadcasted_arrs)
                SHP_arrs = OrderedDict(zip(SHP_arrs.keys(), broadcasted_arrs))

                print(f"Sampling new systems for dataset type {ds_type}")
                systems_arr = utils.dim_array_like(distributions_arr, dtype=LinearSystemGroup)
                for idx, dist in utils.multi_enumerate(distributions_arr):

                    SHP_copy = copy.deepcopy(SHP)
                    for k, v in SHP_arrs.items():
                        setattr(SHP_copy, k, utils.take_from_dim_array(v, dict(zip(distributions_arr.dims, idx))))

                    systems_arr[idx] = dist.sample(SHP_copy, (EHP.n_experiments, n_systems_arr[idx]))
            else:
                print(f"Defaulting to train systems for dataset type {ds_type}")
                systems_arr = info_dict[TRAINING_DATASET_TYPES[0]]["systems"]
        else:
            print(f"Systems found for dataset type {ds_type}")
            systems_arr = systems[ds_type]

    # DONE: Refresh the systems with the same parameters so that gradients will pass through properly in post-experiment analysis
    systems_arr = utils.multi_map(
        lambda lsg: LinearSystemGroup(lsg.state_dict(), SHP.input_enabled),
        systems_arr, dtype=LinearSystemGroup
    )
    result["systems"] = systems_arr

    # DONE: Construct dictionary with the analytical KFs that correspond to the generated systems
    analytical_kfs_arr = utils.multi_map(
        lambda lsg: AnalyticalKFGroup(lsg),
        systems_arr, dtype=AnalyticalKFGroup
    )

    # DONE: Check for saved dataset, otherwise sample and save datasets
    if "dataset" in save_dict:
        print(f"Dataset found for dataset type {ds_type}")
        dataset_arr = save_dict["dataset"][ds_type]
    else:
        if hasattr(DHP, ds_type):
            print(f"Generating new dataset for dataset type {ds_type}")
            systems_arr, analytical_kfs_arr = utils.broadcast_dim_arrays(
                systems_arr,
                analytical_kfs_arr
            )
            dataset_size_arr, total_sequence_length_arr = utils.broadcast_dim_arrays(
                _rgetattr_default("{0}.dataset_size"),
                _rgetattr_default("{0}.total_sequence_length")
            )
            sequence_length_arr = (total_sequence_length_arr + dataset_size_arr - 1) // dataset_size_arr

            max_dataset_size = dataset_size_arr.max()
            max_sequence_length = sequence_length_arr.max()
            max_batch_size = (EHP.ensemble_size if ds_type == TRAINING_DATASET_TYPES[0] else 1) * max_dataset_size

            dataset_arr = utils.dim_array_like(systems_arr, dtype=TensorDict)
            for idx, system_group in utils.multi_enumerate(systems_arr):
                dataset_subarr = analytical_kfs_arr.values[idx].add_targets(
                    system_group.generate_dataset(
                        batch_size=max_batch_size,
                        seq_length=max_sequence_length
                    )
                )
                # DONE: For valid and test, don't generate over the ensemble
                if ds_type == TRAINING_DATASET_TYPES[0]:
                    dataset_subarr = dataset_subarr.unflatten(2, (EHP.ensemble_size, max_dataset_size)).permute(0, 2, 1, 3, 4)
                else:
                    dataset_subarr = dataset_subarr.unsqueeze(1).expand(
                        EHP.n_experiments,
                        EHP.ensemble_size,
                        system_group.group_shape[1],
                        max_dataset_size,
                        max_sequence_length
                    )
                dataset_arr[idx] = PTR(dataset_subarr)
        else:
            print(f"Defaulting to train dataset for dataset type {ds_type}")
            dataset_arr = info_dict[TRAINING_DATASET_TYPES[0]]["dataset"]
    result["dataset"] = dataset_arr

    # DONE: Compute the irreducible losses that correspond to the generated systems
    irreducible_loss_arr = utils.dim_array_like(systems_arr, dtype=tuple)
    for idx, systems_subarr in utils.multi_enumerate(systems_arr):
        S_observation_inf = utils.stack_tensor_arr(utils.multi_map(
            lambda sys: sys.S_observation_inf, systems_subarr, dtype=torch.Tensor
        ))

        # DONE: Need to store as a tuple because directly storing a tensor breaks NumPy internals
        irreducible_loss_arr[idx] = PTR(utils.batch_trace(S_observation_inf))
    result["irreducible_loss"] = irreducible_loss_arr

    return result

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

def _populate_default_values(HP: Namespace) -> None:
    HP.experiment.model_shape = (HP.experiment.n_experiments, HP.experiment.ensemble_size)

    def _rgetattr_default(format_str: str, ds_type: str) -> Any:
        return utils.rgetattr_default(HP.dataset, format_str, ds_type, TRAINING_DATASET_TYPES[0])

    for ds_type, ds_config in vars(HP.dataset).items():
        if isinstance(ds_config, Namespace):
            dataset_size = _rgetattr_default("{0}.dataset_size", ds_type)
            total_sequence_length = _rgetattr_default("{0}.total_sequence_length", ds_type)
            ds_config.sequence_length = (total_sequence_length + dataset_size - 1) // dataset_size

            if not hasattr(ds_config, "sequence_buffer"):
                ds_config.sequence_buffer = 0




