from argparse import Namespace

from infrastructure import loader, utils
from infrastructure.experiment import run_experiments, plot_experiment, get_result_attr
from system.linear_time_invariant import MOPDistribution
from model.convolutional import CnnPredictorLeastSquares, CnnPredictorAnalytical, CnnPredictorAnalyticalLeastSquares


if __name__ == "__main__":
    base_exp_name = "Debug"
    output_dir = "debug"
    output_fname = "result"

    SHP = Namespace(
        distribution=MOPDistribution("gaussian", "gaussian", 0.1, 0.1),
        S_D=2, problem_shape=Namespace(
            environment=Namespace(observation=2),
            controller=Namespace(),
        ),
        auxiliary=Namespace(control_noise_std=0.0)
    )
    args = loader.generate_args(SHP)

    args.system.auxiliary.control_noise_std.update(valid=0.0, test=0.0)
    args.dataset.dataset_size.update(valid=1, test=1)
    args.dataset.total_sequence_length.update(train=20000, valid=10000, test=10000)

    args.experiment.metrics = Namespace(
        training={"validation_analytical"},
        testing={"al", "il"}
    )
    args.experiment.exp_name = base_exp_name

    configurations = [
        ("training_distribution", {
            "system.distribution.train": [MOPDistribution("gaussian", "gaussian", 0.1, 0.1), MOPDistribution("gaussian", "gaussian", 1.0, 1.0)]
        }),
        # ("n_train_systems", {
        #     "dataset.n_systems.train": [2, 12, 20],
        #     "dataset.dataset_size.train": [2, 12, 20],
        # }),
        ("system_dimension", {
            "system.S_D.train": [5, 7],
        }),
        ("control_noise_std", {
            "system.auxiliary.control_noise_std.train": [0.0, 0.5, 1.0, 1.5, 2.0],
        }),
        ("model", {
            "model.model": [
                CnnPredictorLeastSquares,
                CnnPredictorAnalytical,
                CnnPredictorAnalyticalLeastSquares
            ]
        }),
        ("ir_length", {
            "model.ir_length": list(range(1, 65))
        }),
        ("total_trace_length", {
            "dataset.total_sequence_length.train": [100, 200, 500]
        })
    ]

    result, _, _ = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=False
    )
    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical", lstsq=False, xscale="linear")




