from infrastructure import loader
from infrastructure.experiment import *
from model.convolutional import *


if __name__ == "__main__":
    base_exp_name = "SingleTrace"
    output_dir = "system2_CNN"
    output_fname = "result"

    system2, args = loader.load_system_and_args("data/2dim_scalar_system_matrices")
    args.model.ir_length = 16
    args.train.epochs = 500
    args.train.subsequence_length = 32
    args.experiment.exp_name = base_exp_name
    args.experiment.metrics = {"validation_analytical"}

    configurations = [
        ("model", {
            "model.model": [CnnKF, CnnKFLeastSquares]
        }),
        ("total_trace_length", {
            "dataset.train.total_sequence_length": [100, 200, 500, 1000, 2000, 5000, 10000]
        })
    ]

    result = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=True
    )

    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical")

