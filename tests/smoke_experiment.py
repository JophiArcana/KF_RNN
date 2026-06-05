"""Minimal end-to-end smoke test used to guard the refactor.

Runs a tiny single-system training + testing experiment and prints a couple of
final metric values. Intended to be run with the project's conda interpreter:

    python tests/smoke_experiment.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

from argparse import Namespace

import numpy as np
import torch

from infrastructure import loader
from infrastructure.experiment import run_experiments, get_result_attr
from model.convolutional import CnnPredictor
from system.linear_time_invariant import MOPDistribution


def _build_typed_args():
    """Build the same experiment via the typed ExperimentConfig path."""
    from infrastructure.config import (
        ExperimentConfig, ProblemShape, EnvironmentShape, SystemConfig,
        DataConfig, SplitConfig, ModelConfig, RuntimeConfig, EvalConfig,
    )
    cfg = ExperimentConfig(
        problem=ProblemShape(environment=EnvironmentShape(observation=2), controller={}),
        system=SystemConfig(
            S_D=4,
            distribution={
                "_target_": "system.linear_time_invariant.MOPDistribution",
                "F_mode": "gaussian", "H_mode": "gaussian", "W_std": 0.1, "V_std": 0.1,
            },
            settings={"include_analytical": True},
        ),
        dataset=DataConfig(
            n_systems=SplitConfig(train=1, valid=1, test=1),
            n_traces=SplitConfig(train=4, valid=4, test=4),
            total_sequence_length=SplitConfig(train=200, valid=200, test=200),
        ),
        model=ModelConfig(
            model={"_target_": "model.convolutional.cnn_predictor.CnnPredictor", "_partial_": True},
            S_D=4, params={"ir_length": 4},
        ),
        eval=EvalConfig(metrics={"training": ["overfit", "validation"], "testing": ["l", "il"]}),
    )
    cfg.experiment = RuntimeConfig(
        exp_name="Smoke", n_experiments=1, ensemble_size=2,
        print_frequency=1, checkpoint_frequency=10000,
    )
    args = loader.args_from_config(cfg)
    import os
    args.training.scheduler.epochs = int(os.environ.get("SMOKE_EPOCHS", "0"))
    args.training.sampling.batch_size = 16
    args.training.scheduler.warmup_duration = 2
    return args


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    import os
    if os.environ.get("SMOKE_CONFIG") == "typed":
        args = _build_typed_args()
        result, _ = run_experiments(
            args, [("trivial", {"model.ir_length": [2, 4]})],
            {"dir": "_smoke", "fname": "result"}, None, save_experiment=False,
        )
        times = get_result_attr(result, "time")
        print("SMOKE_OK times>0:", bool(np.all(times > 0)), "shape:", result.shape)
        return

    dist = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
    SHP = Namespace(
        distribution=dist, S_D=4,
        problem_shape=Namespace(
            environment=Namespace(observation=2),
            controller=Namespace(),
        ),
        auxiliary=Namespace(),
        settings=Namespace(include_analytical=True),
    )
    args = loader.generate_args(SHP)

    model_kind = os.environ.get("SMOKE_MODEL", "cnn")
    if model_kind == "analytical":
        from model.convolutional import CnnAnalyticalPredictor
        args.model.model = CnnAnalyticalPredictor          # recipe: analytical init only
    else:
        args.model.model = CnnPredictor                    # recipe: SGD only
    args.model.ir_length = 4
    args.model.S_D = 4

    args.dataset.n_systems.reset(train=1, valid=1, test=1)
    args.dataset.n_traces.reset(train=4, valid=4, test=4)
    args.dataset.total_sequence_length.reset(train=200, valid=200, test=200)

    args.experiment.n_experiments = 1
    args.experiment.ensemble_size = 2
    args.experiment.exp_name = "Smoke"
    args.experiment.print_frequency = 1
    args.experiment.checkpoint_frequency = 10000
    args.experiment.metrics = Namespace(
        training={"overfit", "validation"},
        testing={"l", "il"},
    )
    import os
    args.training.scheduler.epochs = int(os.environ.get("SMOKE_EPOCHS", "0"))
    args.training.sampling.batch_size = 16
    args.training.ignore_initial = False
    args.training.scheduler.warmup_duration = 2

    result, _ = run_experiments(
        args, [("trivial", {"model.ir_length": [2, 4]})],
        {"dir": "_smoke", "fname": "result"}, None, save_experiment=False,
    )

    times = get_result_attr(result, "time")
    print("SMOKE_OK times>0:", bool(np.all(times > 0)), "shape:", result.shape)


if __name__ == "__main__":
    main()
