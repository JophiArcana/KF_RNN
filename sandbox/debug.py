from argparse import Namespace

from infrastructure import loader
from infrastructure.experiment import run_experiments, plot_experiment, get_result_attr
from system.linear_time_invariant import MOPDistribution
from model.convolutional import CnnPredictorLeastSquares, CnnPredictorAnalytical, CnnPredictorAnalyticalLeastSquares


if __name__ == "__main__":
    base_exp_name = "Debug"
    output_dir = "debug"
    output_fname = "result"

    SHP = Namespace(S_D=2, problem_shape=Namespace(
        environment=Namespace(observation=2),
        controller=Namespace(),
    ))
    args = loader.generate_args(SHP)

    # args.dataset.train.system.distribution = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
    args.dataset.train.total_sequence_length = 10000
    args.dataset.valid = args.dataset.test = Namespace(
        dataset_size=1,
        total_sequence_length=10000
    )
    args.experiment.metrics = Namespace(
        training={"validation_analytical"},
        testing={"al", "il"}
    )
    args.experiment.exp_name = base_exp_name

    configurations = [
        ("training_distribution", {
            "dataset.train.system.distribution": [MOPDistribution("gaussian", "gaussian", 0.1, 0.1), MOPDistribution("gaussian", "gaussian", 1.0, 1.0)]
        }),
        # ("n_train_systems", {
        #     "dataset.train.system.n_systems": [2, 12, 20]
        # }),
        # ("system_dimension", {
        #     "system.S_D": [5, 7],
        # }),
        ("control_noise_std", {
            "system.control_noise_std": [0.0, 0.5, 1.0, 1.5, 2.0]
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
            "dataset.train.total_sequence_length": [100, 200, 500]
        })
    ]

    result, _, _ = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=False
    )
    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical", lstsq=False, xscale="linear")




