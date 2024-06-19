import os
import sys
from argparse import Namespace

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader
from infrastructure.experiment import *
from model.convolutional import *
from system.linear_time_invariant import MOPDistribution


if __name__ == "__main__":
    base_exp_name = "OptimizerComparison"
    output_dir = "system2_RNN"
    output_fname = "result"


    SHP = Namespace(S_D=2, I_D=1, O_D=1, input_enabled=False)
    args = loader.generate_args(SHP)
    # system2, args = loader.load_system_and_args("data/2dim_scalar_system_matrices")
    args.dataset.train.system.distribution = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
    args.model.S_D = args.system.S_D
    # args.model.ir_length = 16
    args.train.epochs = 2000
    args.train.subsequence_length = 32
    args.experiment.exp_name = base_exp_name
    args.experiment.metrics = {"validation_analytical"}

    from model.sequential.rnn_predictor import RnnPredictor, RnnPredictorPretrainAnalytical
    configurations = [
        ("optimizer", {
            "name": ["adam", "gd", "adam_analytical_initialization", "gd_analytical_initialization"],
            "model.model": [RnnPredictor, RnnPredictor, RnnPredictorPretrainAnalytical, RnnPredictorPretrainAnalytical],
            "train.optim_type": ["Adam", "GD", "Adam", "GD"],
            "train.max_lr": [2e-2, 1e-3, 2e-2, 1e-3]
        }),
        ("total_trace_length", {
            "dataset.train.total_sequence_length": [1, 2, 5, 10, 20, 50, 100, 200, 500],
            # "dataset.train.total_sequence_length": [100, 200, 500, 1000, 2000, 5000, 10000]
        }),
        # ("validation_distribution", {
        #     "dataset.valid.system.distribution.sample_func": [get_mop_sample_func("gaussian", "gaussian", 0.1, 0.1)] * 2
        # })
    ]

    result, dataset = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True
    )

    M = get_metric_namespace_from_result(result)
    # print(M.output.shape)

    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical")




