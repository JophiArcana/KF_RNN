import os
import sys

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader
from infrastructure.experiment import *
from model.convolutional import *


if __name__ == "__main__":
    base_exp_name = "MultipleTraceLSTSQ"
    output_dir = "system2_CNN"
    output_fname = "result"


    system2, args = loader.load_system_and_args("data/2dim_scalar_system_matrices")
    args.model.ir_length = 16
    args.dataset.train.dataset_size = 5
    args.train.epochs = 2000
    args.train.subsequence_length = 32
    args.train.batch_size = 64
    args.experiment.exp_name = base_exp_name
    args.experiment.metrics = {"validation_analytical"}

    configurations = [
        ("model", {
            "model.model": [CnnPredictorLeastSquares]
        }),
        ("total_trace_length", {
            "dataset.train.total_sequence_length": [100, 128, 200, 256, 500, 768, 1000, 1536, 2000, 3072, 5000, 7680, 10000]
        })
    ]

    result, dataset = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=True
    )

    M = get_metric_namespace_from_result(result)
    # print(M.output.shape)

    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical")




