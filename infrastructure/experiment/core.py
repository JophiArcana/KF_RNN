import json
import os
import re
import shutil
import time
from argparse import Namespace

import numpy.core.records as np_records
import torch
from dimarray import DimArray

from infrastructure import utils
from infrastructure.utils import PTR
from infrastructure.experiment.internals import (
    _filter_dimensions_if_any_satisfy_condition,
    _supports_dataset_condition,
    _construct_dependency_dict_and_params_dataset,
    _construct_info_dict_from_dataset_types,
    _iterate_HP_with_params,
    _process_info_dict,
    _populate_values,
)
from infrastructure.experiment.metrics import Metrics
from infrastructure.static import *
from infrastructure.experiment.training import _run_unit_training_experiment
from infrastructure.settings import DEVICE


def run_experiments(
        HP: Namespace,
        iterparams: List[Tuple[str, Dict[str, List[Any] | np.ndarray[Any]]]],
        output_kwargs: Dict[str, Any],
        systems: Dict[str, DimArray] = None,
        initialization: DimArray = None,
        save_experiment: bool = True
) -> Tuple[DimArray, OrderedDict[str, OrderedDict[str, DimArray]]]:
    HP = utils.deepcopy_namespace(HP)

    training_iterparams = []
    for param_group, params in iterparams:
        _training_params, _testing_params = {}, {}
        for n, v in params.items():
            (_testing_params if re.match("dataset(\\..*\\.|\\.)test$", n) else _training_params)[n] = v
        if len(_training_params) > 0:
            training_iterparams.append((param_group, _training_params))

    training_result, info_dict = run_training_experiments(
        HP, training_iterparams, output_kwargs,
        systems=systems, initialization=initialization, save_experiment=save_experiment
    )

    return run_testing_experiments(
        HP, iterparams, output_kwargs,
        systems=systems, result=training_result, info_dict=info_dict, save_experiment=save_experiment,
    )


"""
    iterparams = [
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


def run_training_experiments(
        HP: Namespace,
        iterparams: List[Tuple[str, Dict[str, List[Any] | np.ndarray[Any]]]],
        output_kwargs: Dict[str, Any],
        systems: Dict[str, DimArray] = None,
        initialization: DimArray = None,
        save_experiment: bool = True,
        print_hyperparameters: bool = False,
) -> Tuple[DimArray, OrderedDict[str, OrderedDict[str, DimArray]]]:
    HP = utils.deepcopy_namespace(HP)

    # Set up file names
    output_kwargs.setdefault("fname", "result")
    output_kwargs.setdefault("training_dir", "training")

    if save_experiment:
        root_dir = f"output/{output_kwargs['dir']}"
        output_dir = f"{root_dir}/{HP.experiment.exp_name}"
        os.makedirs((train_output_dir := f"{output_dir}/{output_kwargs['training_dir']}"), exist_ok=True)

        tiers = 2
        output_fnames = [f"{train_output_dir}/{output_kwargs['fname']}{'_' * tier}.pt" for tier in range(tiers)]
        checkpoint_paths = [f"{train_output_dir}/checkpoint-{output_kwargs['fname']}{'_' * tier}.pt" for tier in range(tiers)]
    else:
        output_dir = train_output_dir = None
        output_fnames = []
        checkpoint_paths = []


    # SECTION: Run prologue to construct basic data structures
    conditions = (
        (lambda n: not re.match(r"experiment\.", n), "Cannot sweep over experiment parameters."),
        (lambda n: not re.match(r"dataset(\..*\.|\.)test$", n), "Cannot sweep over test dataset hyperparameters during training."),
        # (_supports_dataset_condition(HP, "valid"), "Cannot sweep over hyperparameters that determine shape of the validation dataset."),
    )
    dimensions, params_dataset = _construct_dependency_dict_and_params_dataset(HP, iterparams, conditions)

    # SECTION: Construct dataset-relevant metadata provided to the experiment
    INFO_DICT = _construct_info_dict_from_dataset_types(
        HP, dimensions, params_dataset, OrderedDict(), TRAINING_DATASET_TYPES, train_output_dir,
        default_systems=systems,
    )


    # DONE: Result setup
    output_fname: str = None
    result: DimArray = None
    for output_fname in output_fnames:
        try:
            result = utils.torch_load(output_fname)
            break
        except Exception:
            pass
    
    def check_done() -> np.ndarray:
        return np_records.fromrecords(result.values, dtype=RESULT_DTYPE).time > 0

    train_dimensions = _filter_dimensions_if_any_satisfy_condition(
        dimensions, lambda param: not re.match("dataset\\.", param) or re.match("dataset(\\..*\\.|\\.)train$", param)
    )
    if result is not None:
        done = check_done()
        if done.sum().item() == done.size:
            print(f"Complete result recovered from file {output_fname}.")
            return result, INFO_DICT,
    else:
        # DONE: Filter out the hyperparameter sweeps that do not influence training
        result = DimArray(
            np.recarray([*train_dimensions.values()], dtype=RESULT_DTYPE),
            dims=[*train_dimensions.keys()], dtype=RESULT_DTYPE,
        )

    # DONE: Restructure INFO_DICT for easier access during training
    PROCESSED_INFO_DICT: Dict[str, DimArray] = {k: _process_info_dict(v) for k, v in INFO_DICT.items()}


    # SECTION: Run the training experiments
    print("=" * 160)
    print(HP.experiment.exp_name)
    if print_hyperparameters:
        print("=" * 160)
        print("Hyperparameters:", json.dumps(utils.toJSON(HP), indent=4))

    counter = 1
    for experiment_dict_index, EXPERIMENT_HP in _iterate_HP_with_params(HP, train_dimensions, params_dataset):
        experiment_record = result.take(experiment_dict_index)
        if experiment_record.time == 0:
            done = check_done()
            print("=" * 160)
            print(f'Experiment {done.sum().item()}/{done.size}')

            # DONE: Set up experiment hyperparameters
            _populate_values(EXPERIMENT_HP)

            INFO = Namespace(**{
                ds_type: np_records.fromrecords(utils.take_from_dim_array(ds_info, experiment_dict_index), dtype=ds_info.dtype)
                for ds_type, ds_info in PROCESSED_INFO_DICT.items()
            })

            # TODO: Set up experiment initialization if it exists
            if initialization is not None:
                _initialization = utils.take_from_dim_array(initialization, experiment_dict_index).values[()].obj
            else:
                _initialization = TensorDict({}, batch_size=())

            start_t = time.perf_counter()
            experiment_result = _run_unit_training_experiment(EXPERIMENT_HP, INFO, checkpoint_paths, _initialization)
            end_t = time.perf_counter()

            for k, v in experiment_result.items():
                setattr(experiment_record, k, v)

            experiment_record.time = end_t - start_t
            experiment_record.systems = INFO.train.systems[()]

            print("\n" + "#" * 160)
            if save_experiment:
                backup_frequency = utils.rgetattr(HP, "experiment.backup_frequency", None)
                if backup_frequency is None:
                    backup_frequency = 10000
                for tier, output_fname in enumerate(output_fnames):
                    if counter % (backup_frequency ** tier) == 0:
                        torch.save(result, output_fname)
                        print(f'{os.path.getsize(output_fname)} bytes written to {output_fname}')
            print("#" * 160 + "\n")

            utils.empty_cache()
            counter += 1

    # SECTION: Save relevant information
    if save_experiment:
        # Save full results to final output file
        for output_fname in output_fnames[1:]:
            if os.path.exists(output_fname):
                os.remove(output_fname)

        # Save code to for experiment reproducibility
        code_base_dir = f"{output_dir}/code"
        os.makedirs(code_base_dir, exist_ok=True)
        for dir_name in ("infrastructure", "model", "system"):
            code_dir = f"{code_base_dir}/{dir_name}"
            if not os.path.exists(code_dir):
                shutil.copytree(dir_name, code_dir, dirs_exist_ok=True)

        # Write hyperparameters to JSON
        hp_fname = f"{train_output_dir}/hparams.json"
        if not os.path.exists(hp_fname):
            with open(hp_fname, "w") as fp:
                json.dump(utils.toJSON(HP), fp, indent=4)

    return result, INFO_DICT,


def run_testing_experiments(
        HP: Namespace,
        iterparams: List[Tuple[str, Dict[str, List[Any] | np.ndarray[Any]]]],
        output_kwargs: Dict[str, Any],
        systems: Dict[str, DimArray] = None,
        result: DimArray = None,
        info_dict: OrderedDict[str, OrderedDict[str, DimArray]] = None,
        save_experiment: bool = True
) -> Tuple[DimArray, OrderedDict[str, OrderedDict[str, DimArray]]]:
    HP = utils.deepcopy_namespace(HP)

    # Set up file names
    output_kwargs.setdefault("fname", "result")
    output_kwargs.setdefault("training_dir", "training")
    output_kwargs.setdefault("testing_dir", "testing")

    if save_experiment:
        root_dir = f"output/{output_kwargs['dir']}"
        output_dir = f"{root_dir}/{HP.experiment.exp_name}"
        os.makedirs((test_output_dir := f"{output_dir}/{output_kwargs['testing_dir']}"), exist_ok=True)

        tiers = 1
        output_fnames = [f"{test_output_dir}/{output_kwargs['fname']}{'_' * tier}.pt" for tier in range(tiers)]
    else:
        output_dir = test_output_dir = None
        output_fnames = []


    # SECTION: Run prologue to construct basic data structures
    conditions = (
        (lambda n: not re.match(r"experiment\.", n), "Cannot sweep over experiment parameters."),
        (_supports_dataset_condition(HP, "test"), "Cannot sweep over hyperparameters that determine shape of the testing dataset."),
    )
    dimensions, params_dataset = _construct_dependency_dict_and_params_dataset(HP, iterparams, conditions)

    # SECTION: Construct dataset-relevant metadata provided to the experiment
    if info_dict is None:
        info_dict = OrderedDict()
    INFO_DICT = _construct_info_dict_from_dataset_types(
        HP, dimensions, params_dataset, info_dict, (TESTING_DATASET_TYPE,), test_output_dir,
        default_systems=systems
    )


    # TODO: Result setup
    output_fname: str = None
    result: DimArray = None
    for output_fname in output_fnames:
        try:
            result = utils.torch_load(output_fname)
            break
        except Exception:
            pass
    
    if result is None:
        train_output_fname = f"{output_dir}/{output_kwargs['training_dir']}/{output_kwargs['fname']}.pt"
        assert os.path.exists(train_output_fname), f"Training result was not provided, and could not be found at {train_output_fname}."
        result = utils.torch_load(train_output_fname)

    def check_done() -> np.ndarray:
        return ~np.array(get_result_attr(result, "metrics") == None)

    done = check_done()
    if done.sum().item() == done.size:
        print(f"Complete result recovered from file {output_fname}.")
        return result, INFO_DICT,


    # SECTION: Construct DimRecarray from accumulated information
    for ds_info in INFO_DICT.values():
        shapes = utils.stack_tensor_arr(utils.multi_map(
            lambda dataset: torch.IntTensor([*dataset.obj.shape, ]),
            ds_info["dataset"].values, dtype=tuple
        ))
        flattened_shapes = shapes.reshape(-1, shapes.shape[-1])
        assert torch.all(flattened_shapes == flattened_shapes[0]), f"Cannot sweep over hyperparameters that determine shape of the testing dataset. Got dataset shapes {shapes}."

    # DONE: Restructure INFO_DICT for easier access during training
    TEST_INFO = _process_info_dict(INFO_DICT[TESTING_DATASET_TYPE])


    # SECTION: Run the testing metrics
    counter = 1
    train_dimensions = OrderedDict(zip(result.dims, result.shape,))
    for experiment_dict_index, EXPERIMENT_HP in _iterate_HP_with_params(HP, train_dimensions, params_dataset):
        experiment_record = result.take(experiment_dict_index)
        if experiment_record.metrics is None:
            done = ~np.array(get_result_attr(result, "metrics") == None)
            print(f"Computing metrics for experiment {done.sum().item()}/{done.size}")

            # DONE: Set up experiment hyperparameters
            _populate_values(EXPERIMENT_HP)

            # DONE: Set up metric information
            reference_module, ensembled_learned_kfs = experiment_record.learned_kfs
            reference_module.eval()

            INFO = np_records.fromrecords(utils.take_from_dim_array(TEST_INFO, experiment_dict_index), dtype=TEST_INFO.dtype)
            exclusive = Namespace(info=Namespace(test=INFO), reference_module=reference_module)

            # DONE: Compute testing metrics
            metric_cache = {}
            metrics: set = utils.rgetattr(HP, "experiment.metrics.testing", {
                "nl", "al", "il", "neil"
            } - utils.rgetattr(HP, "experiment.ignore_metrics.testing", set()))

            metric_result, metric_shape = {}, (
                *INFO.dataset.shape,
                *EXPERIMENT_HP.experiment.model_shape,
                utils.rgetattr(EXPERIMENT_HP, f"dataset.n_systems.{TESTING_DATASET_TYPE}")
            )
            for m in metrics:
                try:
                    r = Metrics[m].evaluate(
                        (exclusive, ensembled_learned_kfs),
                        metric_cache, sweep_position="outside", with_batch_dim=True,
                    ).detach()
                    metric_result[m] = r.expand(*metric_shape, *r.shape[len(metric_shape):])
                except Exception:
                    pass
            try:
                metric_result["output"] = utils.stack_tensor_arr(metric_cache["test"])
            except KeyError:
                pass
            metric_result = TensorDict(metric_result, batch_size=metric_shape)
            experiment_record.metrics = PTR(metric_result)

            if save_experiment:
                # TODO: Save if backup frequency or if it was the last experiment
                backup_frequency = utils.rgetattr(HP, "experiment.backup_frequency", None)
                if backup_frequency is None:
                    backup_frequency = 10000

                for tier, output_fname in enumerate(output_fnames):
                    if (counter % (backup_frequency ** tier) == 0) or (done.sum().item() + 1 == done.size):
                        torch.save(result, output_fname)
                        print("\n" + "#" * 160)
                        print(f'{os.path.getsize(output_fname)} bytes written to {output_fname}')
                        print("\n" + "#" * 160)

            utils.empty_cache()
            counter += 1

    # SECTION: Save relevant information
    if save_experiment:
        # Save full results to final output file
        for output_fname in output_fnames[1:]:
            if os.path.exists(output_fname):
                os.remove(output_fname)

        # Write hyperparameters to JSON
        hp_fname = f"{test_output_dir}/hparams.json"
        if not os.path.exists(hp_fname):
            with open(hp_fname, "w") as fp:
                json.dump(utils.toJSON(HP), fp, indent=4)

    return result, INFO_DICT,


def get_result_attr(r: DimArray, attr: str) -> np.ndarray[Any]:
    return getattr(np_records.fromrecords(r.values, dtype=RESULT_DTYPE), attr)


def get_metric_namespace_from_result(r: DimArray) -> Namespace:
    result = Namespace()
    for k, v in utils.stack_tensor_arr(utils.multi_map(
            lambda metrics: metrics.obj,
            get_result_attr(r, "metrics"), dtype=TensorDict
    )).items(include_nested=True, leaves_only=True):
        if isinstance(k, str):
            setattr(result, k, v)
        else:
            utils.rsetattr(result, ".".join(k), v)
    return result




