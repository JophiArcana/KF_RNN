import os

from infrastructure import loader
from infrastructure.experiment import *
from model.sequential import *
from model.convolutional import *
from model.linear_system_distribution import get_mop_sample_func


if __name__ == "__main__":
    os.chdir("..")
    base_exp_name = "SingleTrace_with_checkpointing"
    output_dir = "system2_CNN"
    output_fname = "result"

    system2, args = loader.load_system_and_args("data/2dim_scalar_system_matrices")
    args.model.S_D = args.system.S_D
    args.model.ir_length = 16
    args.train.epochs = 500
    args.train.subsequence_length = 32
    args.experiment.exp_name = base_exp_name
    args.experiment.metrics = {"validation_analytical"}

    configurations = [
        ("model", {
            "model.model": [
                # RnnKF,
                # RnnKFAnalytical,
                # RnnKFPretrainAnalytical,
                CnnKF,
                CnnKFLeastSquares,
                # CnnKFPretrainLeastSquares,
                # CnnKFAnalytical,
                # CnnKFAnalyticalLeastSquares,
                # CnnKFLeastSquaresRandomStep,
                # CnnKFLeastSquaresNegation
            ]
        }),
        ("total_trace_length", {
            # "dataset.train.total_sequence_length": [100, 200]
            "dataset.train.total_sequence_length": [100, 200, 500, 1000, 2000, 5000, 10000]
        }),
        # ("validation_distribution", {
        #     "dataset.valid.system.distribution.sample_func": [get_mop_sample_func("gaussian", "gaussian", 0.1, 0.1)] * 2
        # })
    ]

    result = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=False
    )

    M = get_metric_namespace_from_result(result)
    print(M.output.shape)

    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical")




