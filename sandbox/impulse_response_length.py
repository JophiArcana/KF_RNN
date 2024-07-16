from argparse import Namespace

from infrastructure import loader
from infrastructure.experiment import run_experiments, plot_experiment
from model.convolutional import CnnPredictorLeastSquares, CnnPredictorAnalytical, CnnPredictorAnalyticalLeastSquares


if __name__ == "__main__":
    base_exp_name = "ImpulseResponseLengthAnalytical"
    output_dir = "system2_CNN"
    output_fname = "result"

    system2, args = loader.load_system_and_args("data/2dim_scalar_system_matrices")

    args.dataset.train.total_sequence_length = 10000
    args.dataset.valid = args.dataset.test = Namespace(
        dataset_size=1,
        total_sequence_length=10000
    )
    args.experiment.metrics = {"validation_analytical"}
    args.experiment.test_metrics = {"al", "il"}
    args.experiment.exp_name = base_exp_name

    configurations = [
        ("model", {
            "model.model": [
                CnnPredictorLeastSquares,
                CnnPredictorAnalytical,
                CnnPredictorAnalyticalLeastSquares
            ]
        }),
        ("ir_length", {
            "model.ir_length": list(range(1, 65))
        })
    ]

    result, _, _ = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=True
    )
    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical", lstsq=False, xscale="linear")




