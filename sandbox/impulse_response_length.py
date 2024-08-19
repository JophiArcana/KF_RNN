from argparse import Namespace

import torch
from matplotlib import pyplot as plt


from infrastructure import loader
from infrastructure.experiment import run_experiments, plot_experiment, get_result_attr
from model.convolutional import CnnPredictorLeastSquares, CnnPredictorAnalytical, CnnPredictorAnalyticalLeastSquares


if __name__ == "__main__":
    base_exp_name = "ImpulseResponseLengthAnalytical"
    output_dir = "system6_CNN"
    output_fname = "result"

    system2, args = loader.load_system_and_args("data/6dim_scalar_system_matrices")

    args.dataset.total_sequence_length.reset(train=10000)
    args.experiment.metrics = Namespace(
        training={"validation_analytical"},
        testing={"al", "il"}
    )
    args.experiment.exp_name = base_exp_name

    max_ir_length = 64
    configurations = [
        ("model", {
            "model.model": [
                CnnPredictorLeastSquares,
                CnnPredictorAnalytical,
                CnnPredictorAnalyticalLeastSquares
            ]
        }),
        ("ir_length", {
            "model.ir_length": list(range(1, max_ir_length + 1))
        })
    ]

    result, _, _ = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=False
    )
    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical", lstsq=False, xscale="linear")




