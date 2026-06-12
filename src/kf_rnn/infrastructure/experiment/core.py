import collections
import json
import os
import subprocess
import time
import traceback
from types import SimpleNamespace
from typing import Any, OrderedDict

import numpy as np
import torch
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.infrastructure.settings import OUTPUT_PATH
from kf_rnn.infrastructure.config import schema
from kf_rnn.infrastructure.config.schema import ExperimentConfig, validate_sweep_targets
from ecliseutils.labeled_array import LabeledArray
from kf_rnn.infrastructure.experiment.internals import (
    _filter_dimensions_if_any_satisfy_condition,
    _supports_dataset_condition,
    _construct_dependency_dict_and_params_dataset,
    _construct_info_dict_from_dataset_types,
    _iterate_HP_with_params,
    _process_info_dict,
)
from kf_rnn.infrastructure.experiment.metrics import METRIC_DICT, Metric
from kf_rnn.infrastructure.experiment.results import ResultGrid, InfoGrid
from kf_rnn.infrastructure.experiment.sweep import (
    is_experiment_param,
    is_dataset_param,
    is_dataset_param_of_type,
)
from kf_rnn.infrastructure.static import (
    TRAINING_DATASET_TYPES,
    TESTING_DATASET_TYPE,
)
from kf_rnn.infrastructure.experiment.training import _run_unit_training_experiment
from kf_rnn.infrastructure.settings import DEVICE


# SECTION: Shared output / result-persistence helpers used by training and testing

def _setup_output_paths(
        HP: ExperimentConfig,
        output_kwargs: dict[str, Any],
        save_experiment: bool,
        subdir_key: str,
        tiers: int,
        with_checkpoints: bool = False,
) -> tuple[str, str, list[str], list[str]]:
    """Resolve the experiment output directory and tiered result/checkpoint file paths.

    Returns ``(output_dir, subdir, output_fnames, checkpoint_paths)``. When
    ``save_experiment`` is False, directories are None and the path lists are empty.
    """
    if not save_experiment:
        return None, None, [], []

    output_dir = os.path.join(OUTPUT_PATH, output_kwargs['dir'], HP.experiment.exp_name)
    subdir = f"{output_dir}/{output_kwargs[subdir_key]}"
    os.makedirs(subdir, exist_ok=True)

    fname = output_kwargs["fname"]
    output_fnames = [f"{subdir}/{fname}{'_' * tier}.pt" for tier in range(tiers)]
    checkpoint_paths = [
        f"{subdir}/checkpoint-{fname}{'_' * tier}.pt" for tier in range(tiers)
    ] if with_checkpoints else []
    return output_dir, subdir, output_fnames, checkpoint_paths

def _recover_result(output_fnames: list[str], kind: str) -> tuple[ResultGrid, str]:
    """Load the most recent existing result file, returning ``(result_or_None, last_fname)``."""
    last_fname: str = None
    for last_fname in output_fnames:
        try:
            return eu.torch_load(last_fname), last_fname
        except FileNotFoundError:
            pass
        except (RuntimeError, EOFError, OSError, AttributeError, ModuleNotFoundError, ValueError):
            print(f"WARNING: failed to load existing {kind} result from {last_fname}:\n{traceback.format_exc()}")
    return None, last_fname

def _backup_frequency(HP: ExperimentConfig) -> int:
    backup_frequency = HP.experiment.backup_frequency
    return 10000 if backup_frequency is None else backup_frequency

def _save_result_tiers(
        result: ResultGrid,
        output_fnames: list[str],
        counter: int,
        backup_frequency: int,
        force: bool = False,
) -> None:
    for tier, output_fname in enumerate(output_fnames):
        if force or (counter % (backup_frequency ** tier) == 0):
            torch.save(result, output_fname)
            print(f'{os.path.getsize(output_fname)} bytes written to {output_fname}')

def _cleanup_backup_tiers(output_fnames: list[str]) -> None:
    """Remove the higher-frequency backup tiers, leaving only the final result file."""
    for output_fname in output_fnames[1:]:
        if os.path.exists(output_fname):
            os.remove(output_fname)

def _write_hparams(subdir: str, HP: ExperimentConfig) -> None:
    hp_fname = f"{subdir}/hparams.json"
    if not os.path.exists(hp_fname):
        with open(hp_fname, "w") as fp:
            json.dump(schema.config_to_jsonable(HP), fp, indent=4)


def _record_code_version(subdir: str) -> None:
    """Record the git commit, dirty flag, and working-tree diff for reproducibility.

    Replaces copying entire source trees into the output directory: the code state
    is reproducible by checking out ``commit`` and applying ``diff``. Degrades
    gracefully (records ``commit: None``) outside a git repo or if git is missing.
    """
    def _git(*args: str) -> str:
        return subprocess.run(
            ("git", *args),
            capture_output=True, text=True, check=True,
        ).stdout

    version: dict[str, Any] = {"commit": None, "dirty": None, "diff": None}
    try:
        version["commit"] = _git("rev-parse", "HEAD").strip()
        version["dirty"] = bool(_git("status", "--porcelain").strip())
        version["diff"] = _git("diff", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(f"WARNING: failed to record git code version: {e}")

    with open(f"{subdir}/code_version.json", "w") as fp:
        json.dump(version, fp, indent=4)


def run_experiments(
        HP: ExperimentConfig,
        iterparams: list[tuple[str, dict[str, list[Any] | np.ndarray[Any]]]],
        output_kwargs: dict[str, Any],
        systems: dict[str, LabeledArray] = None,
        initialization: LabeledArray = None,
        save_experiment: bool = True
) -> tuple[ResultGrid, OrderedDict[str, OrderedDict[str, LabeledArray]]]:
    HP = schema.copy_config(HP)
    validate_sweep_targets(HP, iterparams)

    training_iterparams = []
    for param_group, params in iterparams:
        _training_params, _testing_params = {}, {}
        for n, v in params.items():
            (_testing_params if is_dataset_param_of_type(n, TESTING_DATASET_TYPE) else _training_params)[n] = v
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
        HP: ExperimentConfig,
        iterparams: list[tuple[str, dict[str, list[Any] | np.ndarray[Any]]]],
        output_kwargs: dict[str, Any],
        systems: dict[str, LabeledArray] = None,
        initialization: LabeledArray = None,
        save_experiment: bool = True,
        print_hyperparameters: bool = False,
) -> tuple[ResultGrid, OrderedDict[str, OrderedDict[str, LabeledArray]]]:
    HP = schema.copy_config(HP)
    validate_sweep_targets(HP, iterparams)

    # Set up file names
    output_kwargs.setdefault("fname", "result")
    output_kwargs.setdefault("training_dir", "training")

    output_dir, train_output_dir, output_fnames, checkpoint_paths = _setup_output_paths(
        HP, output_kwargs, save_experiment, "training_dir", tiers=2, with_checkpoints=True,
    )


    # SECTION: Run prologue to construct basic data structures
    conditions = (
        (lambda n: not is_experiment_param(n), "Cannot sweep over experiment parameters."),
        (lambda n: not is_dataset_param_of_type(n, TESTING_DATASET_TYPE), "Cannot sweep over test dataset hyperparameters during training."),
        # (_supports_dataset_condition(HP, "valid"), "Cannot sweep over hyperparameters that determine shape of the validation dataset."),
    )
    dimensions, params_dataset = _construct_dependency_dict_and_params_dataset(HP, iterparams, conditions)

    # SECTION: Construct dataset-relevant metadata provided to the experiment
    INFO_DICT = _construct_info_dict_from_dataset_types(
        HP, dimensions, params_dataset, collections.OrderedDict(), TRAINING_DATASET_TYPES, train_output_dir,
        default_systems=systems,
    )


    # DONE: Result setup
    result, output_fname = _recover_result(output_fnames, "training")

    def check_done() -> np.ndarray:
        return get_result_attr(result, "time") > 0

    train_dimensions = _filter_dimensions_if_any_satisfy_condition(
        dimensions, lambda param: not is_dataset_param(param) or is_dataset_param_of_type(param, TRAINING_DATASET_TYPES[0])
    )
    if result is not None:
        done = check_done()
        if done.sum().item() == done.size:
            print(f"Complete result recovered from file {output_fname}.")
            return result, INFO_DICT,
    else:
        # DONE: Filter out the hyperparameter sweeps that do not influence training
        result = ResultGrid.empty(train_dimensions)

    # DONE: Restructure INFO_DICT for easier access during training
    PROCESSED_INFO_DICT: dict[str, InfoGrid] = {k: _process_info_dict(v) for k, v in INFO_DICT.items()}


    # SECTION: Run the training experiments
    print("=" * 160)
    print(HP.experiment.exp_name)
    if print_hyperparameters:
        print("=" * 160)
        print("Hyperparameters:", json.dumps(schema.config_to_jsonable(HP), indent=4))

    counter = 1
    for experiment_dict_index, EXPERIMENT_HP in _iterate_HP_with_params(HP, train_dimensions, params_dataset):
        if result.get(experiment_dict_index, "time") == 0:
            done = check_done()
            print("=" * 160)
            print(f'Experiment {done.sum().item()}/{done.size}')

            INFO = SimpleNamespace(**{
                ds_type: ds_info.take(experiment_dict_index)
                for ds_type, ds_info in PROCESSED_INFO_DICT.items()
            })

            # TODO: Set up experiment initialization if it exists
            if initialization is not None:
                _initialization = eu.take_from_dim_array(initialization, experiment_dict_index).values[()]
            else:
                _initialization = TensorDict({}, batch_size=())

            start_t = time.perf_counter()
            experiment_result = _run_unit_training_experiment(EXPERIMENT_HP, INFO, checkpoint_paths, _initialization)
            end_t = time.perf_counter()

            result.set(
                experiment_dict_index,
                time=end_t - start_t,
                systems=INFO.train.systems[()],
                **experiment_result,
            )

            print("\n" + "#" * 160)
            if save_experiment:
                _save_result_tiers(result, output_fnames, counter, _backup_frequency(HP))
            print("#" * 160 + "\n")

            eu.empty_cache()
            counter += 1

    # SECTION: Save relevant information
    if save_experiment:
        # Save full results to final output file
        _cleanup_backup_tiers(output_fnames)

        # Record code version (git commit + diff) for experiment reproducibility
        _record_code_version(output_dir)

        # Write hyperparameters to JSON
        _write_hparams(train_output_dir, HP)

    return result, INFO_DICT,


def run_testing_experiments(
        HP: ExperimentConfig,
        iterparams: list[tuple[str, dict[str, list[Any] | np.ndarray[Any]]]],
        output_kwargs: dict[str, Any],
        systems: dict[str, LabeledArray] = None,
        result: ResultGrid = None,
        info_dict: OrderedDict[str, OrderedDict[str, LabeledArray]] = None,
        save_experiment: bool = True
) -> tuple[ResultGrid, OrderedDict[str, OrderedDict[str, LabeledArray]]]:
    HP = schema.copy_config(HP)
    validate_sweep_targets(HP, iterparams)

    # Set up file names
    output_kwargs.setdefault("fname", "result")
    output_kwargs.setdefault("training_dir", "training")
    output_kwargs.setdefault("testing_dir", "testing")

    output_dir, test_output_dir, output_fnames, _ = _setup_output_paths(
        HP, output_kwargs, save_experiment, "testing_dir", tiers=1,
    )


    # SECTION: Run prologue to construct basic data structures
    conditions = (
        (lambda n: not is_experiment_param(n), "Cannot sweep over experiment parameters."),
        (_supports_dataset_condition(HP, TESTING_DATASET_TYPE), "Cannot sweep over hyperparameters that determine shape of the testing dataset."),
    )
    dimensions, params_dataset = _construct_dependency_dict_and_params_dataset(HP, iterparams, conditions)

    # SECTION: Construct dataset-relevant metadata provided to the experiment
    if info_dict is None:
        info_dict = collections.OrderedDict()
    INFO_DICT = _construct_info_dict_from_dataset_types(
        HP, dimensions, params_dataset, info_dict, (TESTING_DATASET_TYPE,), test_output_dir,
        default_systems=systems
    )


    # SECTION: Recover any previously-saved testing result, else fall back to the training result
    found_result, output_fname = _recover_result(output_fnames, "testing")
    if found_result is not None:
        result = found_result
    
    if result is None:
        train_output_fname = f"{output_dir}/{output_kwargs['training_dir']}/{output_kwargs['fname']}.pt"
        assert os.path.exists(train_output_fname), f"Training result was not provided, and could not be found at {train_output_fname}."
        result = eu.torch_load(train_output_fname)

    def check_done() -> np.ndarray:
        return ~np.array(get_result_attr(result, "metrics") == None)

    done = check_done()
    if done.sum().item() == done.size:
        print(f"Complete result recovered from file {output_fname}.")
        return result, INFO_DICT,


    # SECTION: Validate that the testing dataset shape is constant across the sweep
    for ds_info in INFO_DICT.values():
        shapes = eu.stack_tensor_arr(eu.multi_map(
            lambda dataset: torch.IntTensor([*dataset.shape, ]),
            ds_info["dataset"].values, dtype=tuple
        ))
        flattened_shapes = shapes.reshape(-1, shapes.shape[-1])
        assert torch.all(flattened_shapes == flattened_shapes[0]), f"Cannot sweep over hyperparameters that determine shape of the testing dataset. Got dataset shapes {shapes}."

    # DONE: Restructure INFO_DICT for easier access during training
    TEST_INFO = _process_info_dict(INFO_DICT[TESTING_DATASET_TYPE])


    # SECTION: Run the testing metrics
    counter = 1
    train_dimensions = collections.OrderedDict(zip(result.dims, result.shape,))
    for experiment_dict_index, EXPERIMENT_HP in _iterate_HP_with_params(HP, train_dimensions, params_dataset):
        if result.get(experiment_dict_index, "metrics") is None:
            done = ~np.array(get_result_attr(result, "metrics") == None)
            print(f"Computing metrics for experiment {done.sum().item()}/{done.size}")

            # DONE: Set up metric information
            reference_module, stacked_modules = result.get(experiment_dict_index, "learned_kfs")
            reference_module.eval()

            INFO = TEST_INFO.take(experiment_dict_index)
            exclusive = SimpleNamespace(info=SimpleNamespace(test=INFO), reference_module=reference_module)

            # DONE: Compute testing metrics
            metric_cache = {}
            metrics: set = HP.experiment.metrics.testing
            if metrics is None:
                metrics = {
                    "nl", "al", "il", "neil"
                } - (HP.experiment.ignore_metrics.testing or set())

            metric_result, metric_shape = {}, (
                *INFO.dataset.shape,
                *EXPERIMENT_HP.experiment.model_shape,
                EXPERIMENT_HP.dataset.n_systems.for_split(TESTING_DATASET_TYPE)
            )
            metric_vars = exclusive, (reference_module, stacked_modules,),
            for m in metrics:
                try:
                    r = METRIC_DICT[m].evaluate(
                        metric_vars, metric_cache,
                        sweep_position="outside", with_batch_dim=True,
                    ).detach()
                    metric_result[m] = r.expand(*metric_shape, *r.shape[len(metric_shape):])
                except (RuntimeError, ValueError, KeyError, IndexError, TypeError):
                    # Per-metric resilience boundary: a single failing metric should
                    # not abort an otherwise-successful sweep.
                    print(f"WARNING: testing metric {m!r} failed and was skipped:\n{traceback.format_exc()}")
            metric_result["output"] = eu.stack_tensor_arr(Metric.compute(metric_vars, "test", metric_cache))
            metric_result = TensorDict(metric_result, batch_size=metric_shape)
            result.set(experiment_dict_index, metrics=metric_result)

            if save_experiment:
                # Save if backup frequency or if it was the last experiment
                _save_result_tiers(
                    result, output_fnames, counter, _backup_frequency(HP),
                    force=(done.sum().item() + 1 == done.size),
                )

            eu.empty_cache()
            counter += 1

    # SECTION: Save relevant information
    if save_experiment:
        # Save full results to final output file
        _cleanup_backup_tiers(output_fnames)

        # Write hyperparameters to JSON
        _write_hparams(test_output_dir, HP)

    return result, INFO_DICT,


def get_result_attr(r: ResultGrid, attr: str) -> np.ndarray[Any]:
    return r.field(attr)


def get_metric_namespace_from_result(r: ResultGrid) -> SimpleNamespace:
    result = SimpleNamespace()
    for k, v in eu.stack_tensor_arr(
            get_result_attr(r, "metrics")
    ).items(include_nested=True, leaves_only=True):
        keys = (k,) if isinstance(k, str) else tuple(k)
        node = result
        for seg in keys[:-1]:
            if not hasattr(node, seg):
                setattr(node, seg, SimpleNamespace())
            node = getattr(node, seg)
        setattr(node, keys[-1], v)
    return result




