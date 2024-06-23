import copy
import itertools
import json
import os
import shutil
import time
from argparse import Namespace

import numpy.core.records as np_records
import torch
from dimarray import DimArray

from infrastructure import utils
from infrastructure.experiment.internals import _supports_dataset_condition, _prologue, \
    _construct_info_dict_from_dataset_types, \
    _process_info_dict, _construct_dataset_from_iterparam, _populate_values
from infrastructure.experiment.metrics import Metrics
from infrastructure.experiment.static import *
from infrastructure.experiment.training import _run_unit_training_experiment
from infrastructure.settings import DEVICE


def run_experiments(
        HP: Namespace,
        iterparams: List[Tuple[str, Dict[str, List[Any] | np.ndarray[Any]]]],
        output_kwargs: Dict[str, Any],
        systems: Dict[str, DimArray] = None,
        save_experiment: bool = True
) -> Tuple[DimArray, DimArray]:
    HP = copy.deepcopy(HP)

    training_iterparams = []
    for param_group, params in iterparams:
        _training_params, _testing_params = {}, {}
        for n, v in params.items():
            (_testing_params if n.startswith("dataset.test.") else _training_params)[n] = v
        if len(_training_params) > 0:
            training_iterparams.append((param_group, _training_params))

    training_result, training_cache = run_training_experiments(
        HP, training_iterparams, output_kwargs,
        systems=systems, save_experiment=save_experiment
    )

    return run_testing_experiments(
        HP, iterparams, output_kwargs,
        systems=systems, result=training_result, cache=training_cache, save_experiment=save_experiment
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
        save_experiment: bool = True
) -> Tuple[DimArray, Namespace]:
    HP = copy.deepcopy(HP)

    # Set up file names
    output_kwargs.setdefault("fname", "result")
    output_kwargs.setdefault("training_dir", "training")

    if save_experiment:
        root_dir = f"output/{output_kwargs['dir']}"
        output_dir = f"{root_dir}/{HP.experiment.exp_name}"
        os.makedirs((train_output_dir := f"{output_dir}/{output_kwargs['training_dir']}"), exist_ok=True)

        output_fname = f"{train_output_dir}/{output_kwargs['fname']}.pt"
        output_fname_backup = f"{train_output_dir}/{output_kwargs['fname']}_backup.pt"
        checkpoint_paths = [
            f"{train_output_dir}/checkpoint.pt",
            f"{train_output_dir}/checkpoint_backup.pt"
        ]
    else:
        output_dir = train_output_dir = output_fname = output_fname_backup = checkpoint_paths = None

    # SECTION: Run prologue to construct basic data structures
    cache = Namespace()
    conditions = (
        (lambda n: not n.startswith("experiment."), "Cannot sweep over experiment parameters."),
        (lambda n: not n.startswith("dataset.test."),
         "Cannot sweep over test dataset hyperparameters during training."),
        (_supports_dataset_condition(HP, "valid"),
         "Cannot sweep over hyperparameters that determine shape of the validation dataset."),
    )
    numpy_HP, iterparams = _prologue(HP, iterparams, conditions)

    # DONE: Filter out the hyperparameter sweeps that do not influence training
    train_iterparams = []
    for param_group, params in iterparams:
        _params = {}
        for n, v in params.items():
            if not (n.startswith("dataset.") and not n.startswith("dataset.train.")):
                _params[n] = v
        if len(_params) > 0:
            train_iterparams.append((param_group, _params))

    # DONE: Construct Dataset structures
    iterparam_datasets = (*map(_construct_dataset_from_iterparam, iterparams),)
    train_iterparam_datasets = (*map(_construct_dataset_from_iterparam, train_iterparams),)

    if len(train_iterparam_datasets) > 0:
        train_datasets, train_dim_names, train_shapes = zip(*train_iterparam_datasets)
    else:
        train_datasets, train_dim_names, train_shapes = (), (), ()
    cache.train_dim_names = train_dim_names = (*itertools.chain(*train_dim_names),)
    cache.train_param_shape = train_param_shape = (*itertools.chain(*train_shapes),)

    # SECTION: Construct dataset-relevant metadata provided to the experiment
    _INFO_DICT = _construct_info_dict_from_dataset_types(
        numpy_HP, OrderedDict(), TRAINING_DATASET_TYPES, train_output_dir,
        default_systems=systems
    )

    # SECTION: Construct DimRecarrays from accumulated information
    # DONE: Cache unprocessed INFO_DICT to use during testing
    cache.info_dict = _INFO_DICT
    INFO_DICT: Dict[str, DimArray] = {k: _process_info_dict(v) for k, v in _INFO_DICT.items()}

    # SECTION: Run the experiments for hyperparameter sweeps
    # Result setup
    if save_experiment and os.path.exists(output_fname):
        try:
            result = torch.load(output_fname, map_location=DEVICE)
        except RuntimeError:
            result = torch.load(output_fname_backup, map_location=DEVICE)
    else:
        # Set up new result DimRecarray
        result = DimArray(np.recarray(train_param_shape, dtype=RESULT_DTYPE), dims=train_dim_names, dtype=RESULT_DTYPE)

    print("=" * 160)
    print(HP.experiment.exp_name)
    print("=" * 160)
    print("Hyperparameters:", json.dumps(utils.toJSON(HP), indent=4))

    counter = 0
    for experiment_indices in itertools.product(*map(range, train_param_shape)):
        experiment_dict_index = dict(zip(train_dim_names, experiment_indices))
        experiment_record = result.take(experiment_dict_index)

        if experiment_record.time == 0:
            done = np_records.fromrecords(result.values, dtype=RESULT_DTYPE).time > 0
            print("=" * 160)
            print(f'Experiment {done.sum().item()}/{done.size}')

            # DONE: Set up experiment hyperparameters
            EXPERIMENT_HP = copy.deepcopy(HP)
            _populate_values(EXPERIMENT_HP, iterparam_datasets, experiment_dict_index)

            INFO = Namespace(**{
                ds_type: np_records.fromrecords(utils.take_from_dim_array(ds_info, experiment_dict_index),
                                                dtype=ds_info.dtype)
                for ds_type, ds_info in INFO_DICT.items()
            })

            start_t = time.perf_counter()
            experiment_result = _run_unit_training_experiment(EXPERIMENT_HP, INFO, checkpoint_paths)
            end_t = time.perf_counter()

            for k, v in experiment_result.items():
                setattr(experiment_record, k, v)

            experiment_record.time = end_t - start_t
            experiment_record.systems = INFO.train.systems[()]

            print("\n" + "#" * 160)
            if save_experiment:
                if counter % HP.experiment.backup_frequency == 0:
                    torch.save(result, output_fname_backup)
                    print(f'{os.path.getsize(output_fname_backup)} bytes written to {output_fname_backup}')
                torch.save(result, output_fname)
                print(f'{os.path.getsize(output_fname)} bytes written to {output_fname}')
            print("#" * 160 + "\n")

            counter += 1
        torch.cuda.empty_cache()

    # SECTION: Save relevant information
    if save_experiment:
        # Save code to for experiment reproducibility
        code_base_dir = f"{output_dir}/code"
        os.makedirs(code_base_dir, exist_ok=True)
        for dir_name in ("infrastructure", "model", "system"):
            code_dir = f"{code_base_dir}/{dir_name}"
            if not os.path.exists(code_dir):
                shutil.copytree(dir_name, code_dir, dirs_exist_ok=True)

        # Save cached training information to be used at test time
        cache_fname = f"{train_output_dir}/cache.pt"
        torch.save(cache, cache_fname)

        # Write hyperparameters to JSON
        hp_fname = f"{train_output_dir}/hparams.json"
        if not os.path.exists(hp_fname):
            with open(hp_fname, "w") as fp:
                json.dump(utils.toJSON(HP), fp, indent=4)

        # Clean up result_backup
        if os.path.exists(output_fname_backup):
            os.remove(output_fname_backup)

    return result, cache


def run_testing_experiments(
        HP: Namespace,
        iterparams: List[Tuple[str, Dict[str, List[Any] | np.ndarray[Any]]]],
        output_kwargs: Dict[str, Any],
        systems: Dict[str, DimArray] = None,
        result: DimArray = None,
        cache: Namespace = None,
        save_experiment: bool = True
) -> Tuple[DimArray, DimArray]:
    HP = copy.deepcopy(HP)

    # Set up file names
    output_kwargs.setdefault("fname", "result")
    output_kwargs.setdefault("training_dir", "training")
    output_kwargs.setdefault("testing_dir", "testing")

    if save_experiment or cache is None:
        root_dir = f"output/{output_kwargs['dir']}"
        output_dir = f"{root_dir}/{HP.experiment.exp_name}"
        if save_experiment:
            os.makedirs((test_output_dir := f"{output_dir}/{output_kwargs['testing_dir']}"), exist_ok=True)
            output_fname = f"{test_output_dir}/{output_kwargs['fname']}.pt"
        else:
            test_output_dir = output_fname = None
    else:
        output_dir = test_output_dir = output_fname = None

    if save_experiment and os.path.exists(output_fname):
        result = torch.load(output_fname, map_location=DEVICE)
    elif result is None:
        train_output_fname = f"{output_dir}/{output_kwargs['training_dir']}/{output_kwargs['fname']}.pt"
        assert os.path.exists(
            train_output_fname), f"Training result was not provided, and could not be found at {train_output_fname}."
        result = torch.load(train_output_fname, map_location=DEVICE)

    if cache is None:
        training_cache_fname = f"{output_dir}/{output_kwargs['training_dir']}/cache.pt"
        assert os.path.exists(
            training_cache_fname), f"Training cache was not provided, and could not be found at {training_cache_fname}."
        cache = torch.load(training_cache_fname, map_location=DEVICE)

    # SECTION: Run prologue to construct basic data structures
    conditions = (
        (lambda n: not n.startswith("experiment."), "Cannot sweep over experiment parameters."),
        (_supports_dataset_condition(HP, "test"),
         "Cannot sweep over hyperparameters that determine shape of the testing dataset."),
    )
    numpy_HP, iterparams = _prologue(HP, iterparams, conditions)

    # DONE: Construct Dataset structures
    iterparam_datasets = (*map(_construct_dataset_from_iterparam, iterparams),)

    # SECTION: Construct dataset-relevant metadata provided to the experiment
    INFO_DICT = _construct_info_dict_from_dataset_types(
        numpy_HP, cache.info_dict, (TESTING_DATASET_TYPE,), test_output_dir,
        default_systems=systems
    )

    for ds_info in INFO_DICT.values():
        shapes = utils.stack_tensor_arr(utils.multi_map(
            lambda dataset: torch.IntTensor([*dataset.obj.shape, ]),
            ds_info["dataset"].values, dtype=tuple
        ))
        flattened_shapes = shapes.reshape(-1, shapes.shape[-1])
        assert torch.all(flattened_shapes == flattened_shapes[
            0]), f"Cannot sweep over hyperparameters that determine shape of the testing dataset. Got dataset shapes {shapes}."

    # SECTION: Construct DimRecarray from accumulated information
    TEST_INFO = _process_info_dict(INFO_DICT[TESTING_DATASET_TYPE])

    # SECTION: Run the testing metrics
    counter = 0
    for experiment_indices in itertools.product(*map(range, cache.train_param_shape)):
        experiment_dict_index = dict(zip(cache.train_dim_names, experiment_indices))
        experiment_record = result.take(experiment_dict_index)

        if experiment_record.metrics is None:
            done = ~np.array(get_result_attr(result, "metrics") == None)
            print(f"Computing metrics for experiment {done.sum().item()}/{done.size}")

            # DONE: Set up experiment hyperparameters
            EXPERIMENT_HP = copy.deepcopy(HP)
            _populate_values(EXPERIMENT_HP, iterparam_datasets, experiment_dict_index)

            # DONE: Set up metric information
            reference_module, ensembled_learned_kfs = experiment_record.learned_kfs
            reference_module.eval()

            INFO = np_records.fromrecords(utils.take_from_dim_array(TEST_INFO, experiment_dict_index),
                                          dtype=TEST_INFO.dtype)
            exclusive = Namespace(info=Namespace(test=INFO), reference_module=reference_module)

            # DONE: Compute testing metrics
            metric_cache = {}
            metrics = ("l", "al", "il", "eil")

            metric_result, metric_shape = {}, (
                *INFO.dataset.shape,
                *EXPERIMENT_HP.experiment.model_shape,
                utils.rgetattr_default(EXPERIMENT_HP, "dataset.{0}.system.n_systems", TESTING_DATASET_TYPE, TRAINING_DATASET_TYPES[0])
            )
            for m in metrics:
                try:
                    r = Metrics[m].evaluate(
                        (exclusive, ensembled_learned_kfs),
                        metric_cache, sweep_position="outside", with_batch_dim=True
                    ).detach()
                    metric_result[m] = r.expand(*metric_shape, *r.shape[len(metric_shape):])
                except NotImplementedError:
                    pass
            metric_result = TensorDict(metric_result, batch_size=metric_shape)
            metric_result["output"] = utils.stack_tensor_arr(metric_cache["test"])
            experiment_record.metrics = PTR(metric_result)

            if save_experiment:
                # TODO: Save if backup frequency or if it was the last experiment
                if (counter % HP.experiment.backup_frequency == 0) or (done.sum().item() + 1 == done.size):
                    print("\n" + "#" * 160)
                    torch.save(result, output_fname)
                    print(f'{os.path.getsize(output_fname)} bytes written to {output_fname}')
                    print("#" * 160 + "\n")

            counter += 1
        torch.cuda.empty_cache()

    # SECTION: Save relevant information
    if save_experiment:
        # Write hyperparameters to JSON
        hp_fname = f"{test_output_dir}/hparams.json"
        if not os.path.exists(hp_fname):
            with open(hp_fname, "w") as fp:
                json.dump(utils.toJSON(HP), fp, indent=4)

    return result, INFO_DICT[TESTING_DATASET_TYPE]["dataset"]


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




