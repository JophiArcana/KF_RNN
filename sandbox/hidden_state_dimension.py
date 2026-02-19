#%%
# This line needs to be added since some terminals will not recognize the current directory
import sys
sys.path.append("/workspace/KF_RNN")

from argparse import Namespace

from infrastructure import loader
from infrastructure.experiment import run_experiments, plot_experiment, get_result_attr
from model.sequential import RnnKalmanInitializedPredictor, RnnLeastSquaresPredictor
from model.convolutional import CnnLeastSquaresPredictor


if __name__ == "__main__":
    base_exp_name = "OnlineLeastSquares"
    output_dir = "system6_RNN"
    output_fname = "result"

    # from system.linear_time_invariant import MOPDistribution
    # dist = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
    # SHP = Namespace(
    #     distribution=dist, S_D=3,
    #     problem_shape=Namespace(
    #         environment=Namespace(observation=1),
    #         controller=Namespace()
    #     ), auxiliary=Namespace()
    # )
    # args = loader.generate_args(SHP)

    system2, args = loader.load_system_and_args("data/2dim_scalar_system_matrices")
    total_dataset_length = 2000

    args.model.ridge = 1.0
    args.dataset.n_traces.reset(train=1)
    args.dataset.total_sequence_length.reset(train=total_dataset_length)
    args.training.sampling = Namespace(method="full")
    args.training.optimizer = Namespace(
        type="SGD",
        max_lr=1e-4, min_lr=1e-8,
        weight_decay=0.0, momentum=0.9,
    )
    args.training.scheduler = Namespace(
        type="exponential",
        lr_decay=0.995, warmup_duration=100,
        epochs=2000, gradient_cutoff=1e-6,
    )
    args.training.iterations_per_epoch = 20
    args.experiment.metrics = Namespace(
        training={"validation_analytical"},
        testing={"al", "il"}
    )
    # args.experiment.ensemble_size = 1
    args.experiment.exp_name = base_exp_name

    max_state_dimension = 10
    configurations = [
        ("model", {
            "model.model": [RnnLeastSquaresPredictor, CnnLeastSquaresPredictor],
            "model.S_D": [args.system.S_D.default(), None],
            "model.ir_length": [None, 2],
        }),
        ("total_trace_length", {
            "dataset.total_sequence_length.train": [*range(10, total_dataset_length + 1)]
            # "dataset.total_sequence_length.train": [10, 20, 50, 100, 200, 500, 1000, 2000]
        }),
    ]

    result, _, _ = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=True
    )
    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical", lstsq=False, xscale="log")





# %%
