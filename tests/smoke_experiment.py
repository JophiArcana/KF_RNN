"""Minimal end-to-end smoke test used to guard the refactor.

Runs a tiny single-system training + testing experiment and prints a couple of
final metric values. Intended to be run with the project's conda interpreter:

    python tests/smoke_experiment.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from kf_rnn.infrastructure.config import (
    DataConfig,
    EnvironmentShape,
    ExperimentConfig,
    MetricsConfig,
    ProblemShape,
    Split,
    SystemConfig,
)
from kf_rnn.infrastructure.experiment import run_experiments, get_result_attr
from kf_rnn.model.convolutional import CnnPredictor
from kf_rnn.system.linear_time_invariant import MOPDistribution


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    if os.environ.get("SMOKE_MODEL", "cnn") == "analytical":
        from kf_rnn.model.convolutional import CnnAnalyticalPredictor
        model_cls = CnnAnalyticalPredictor                  # recipe: analytical init only
    else:
        model_cls = CnnPredictor                            # recipe: SGD only

    cfg = ExperimentConfig(
        problem=ProblemShape(environment=EnvironmentShape(observation=2), controller={}),
        system=SystemConfig(
            S_D=4,
            distribution=MOPDistribution("gaussian", "gaussian", 0.1, 0.1),
        ),
        dataset=DataConfig(
            n_systems=Split(train=1, valid=1, test=1),
            n_traces=Split(train=4, valid=4, test=4),
            total_sequence_length=Split(train=200, valid=200, test=200),
        ),
        model=model_cls.Config(ir_length=4),
    )

    cfg.experiment.n_experiments = 1
    cfg.experiment.ensemble_size = 2
    cfg.experiment.exp_name = "Smoke"
    cfg.experiment.print_frequency = 1
    cfg.experiment.checkpoint_frequency = 10000
    cfg.experiment.metrics = MetricsConfig(
        training={"overfit", "validation"},
        testing={"l", "il"},
    )
    cfg.training.scheduler.epochs = int(os.environ.get("SMOKE_EPOCHS", "0"))
    cfg.training.sampling.batch_size = 16
    cfg.training.ignore_initial = False
    cfg.training.scheduler.warmup_duration = 2

    result, _ = run_experiments(
        cfg, [("trivial", {"model.ir_length": [2, 4]})],
        {"dir": "_smoke", "fname": "result"}, None, save_experiment=False,
    )

    times = get_result_attr(result, "time")
    print("SMOKE_OK times>0:", bool(np.all(times > 0)), "shape:", result.shape)


if __name__ == "__main__":
    main()
