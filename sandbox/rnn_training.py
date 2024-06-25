import os
import sys
from argparse import Namespace

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader
from infrastructure.experiment import *
from system.linear_time_invariant import MOPDistribution
from model.sequential import RnnPredictorPretrainAnalytical


if __name__ == "__main__":
    base_exp_name = "OptimizerComparison_LBFGS"
    output_dir = "system2_RNN"
    output_fname = "result"


    SHP = Namespace(S_D=2, I_D=1, O_D=1, input_enabled=False)
    args = loader.generate_args(SHP)

    args.dataset.train.system.distribution = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
    args.model.model = RnnPredictorPretrainAnalytical
    args.model.S_D = args.system.S_D
    args.train.sampling.method = "full"

    args.experiment.exp_name = base_exp_name
    args.experiment.metrics = {"validation_analytical"}

    configurations = [
        ("optimizer", {
            "train.optimizer": [
                Namespace(
                    type="SGD",
                    max_lr=1e-3, min_lr=1e-6, momentum=0.0, weight_decay=0.0
                ), Namespace(
                    type="Adam",
                    max_lr=2e-2, min_lr=1e-6, momentum=0.9, weight_decay=0.9
                ), Namespace(
                    type="LBFGS",
                    max_lr=1.0, history_size=10
                )
            ],
            "train.scheduler.warmup_duration": [100, 100, 0],
            "train.scheduler.epochs": [2000, 2000, 200],
        }),
        ("total_trace_length", {
            "dataset.train.total_sequence_length": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        })
    ]

    result, dataset = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True
    )

    M = get_metric_namespace_from_result(result)
    # plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical")




