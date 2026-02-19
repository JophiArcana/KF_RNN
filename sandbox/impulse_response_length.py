from argparse import Namespace

from infrastructure import loader
from infrastructure.experiment import run_experiments, plot_experiment, get_result_attr
from model.convolutional import CnnLeastSquaresPredictor, CnnAnalyticalPredictor, CnnAnalyticalLeastSquaresPredictor
from model.sequential import RnnHoKalmanAnalyticalPredictor, RnnHoKalmanAnalyticalLeastSquaresPredictor
from system.linear_time_invariant import OrthonormalDistribution, MOPDistribution


if __name__ == "__main__":
    base_exp_name = "ImpulseResponseLengthAnalytical"
    output_dir = "system6_CNN"
    output_fname = "result"

    system2, args = loader.load_system_and_args("data/6dim_scalar_system_matrices")
    # dist = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
    # SHP = Namespace(
    #     distribution=dist, S_D=6,
    #     problem_shape=Namespace(
    #         environment=Namespace(observation=1),
    #         controller=Namespace()
    #     ),
    #     auxiliary=Namespace(),
    #     settings=Namespace(include_analytical=True),
    # )
    # args = loader.generate_args(SHP)
    # system2 = None

    context_length = 100
    n_train_traces = 1
    n_valid_traces = 100

    args.model.S_D = args.system.S_D.default()
    args.dataset.n_traces.reset(train=n_train_traces, valid=n_valid_traces, test=n_valid_traces)
    args.dataset.total_sequence_length.reset(train=n_train_traces * context_length, valid=n_valid_traces * context_length, test=n_valid_traces * context_length)
    args.training.sampling.batch_size = 4096
    args.experiment.metrics = Namespace(
        training={"validation_analytical",},
        testing={"al", "il", "l", "eil",},
    )
    args.experiment.checkpoint_frequency = 10000
    args.experiment.print_frequency = 1
    args.experiment.exp_name = base_exp_name

    max_ir_length = 40
    configurations = [
        ("model", {
            "model.model": [
                CnnLeastSquaresPredictor,
                CnnAnalyticalPredictor,
                CnnAnalyticalLeastSquaresPredictor,
                # RnnHoKalmanAnalyticalPredictor,
                # RnnHoKalmanAnalyticalLeastSquaresPredictor,
            ]
        }),
        ("ir_length", {
            "model.ir_length": list(range(1, max_ir_length + 1))
        })
    ]

    result, _ = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=False
    )
    plot_experiment(
        f"{output_dir}/{base_exp_name}", configurations, result,
        loss_type="analytical", lstsq=False, xscale="linear",
    )




