#%%
# This line needs to be added since some terminals will not recognize the current directory
import os
import sys
os.chdir("/home/wentinn/workspace/KF_RNN")
sys.path.append("/home/wentinn/workspace/KF_RNN")

from argparse import Namespace

from infrastructure import loader
from infrastructure.experiment import run_experiments, plot_experiment, get_result_attr
from model.convolutional import CnnLeastSquaresPredictor, CnnAnalyticalPredictor, CnnAnalyticalLeastSquaresPredictor
from system.linear_time_invariant import OrthonormalDistribution


if __name__ == "__main__":
    base_exp_name = "ImpulseResponseLengthAnalytical"
    output_dir = "system6_CNN"
    output_fname = "result"

    system2, args = loader.load_system_and_args("data/6dim_scalar_system_matrices")
    # dist = OrthonormalDistribution()
    # SHP = Namespace(
    #     distribution=dist, S_D=6,
    #     problem_shape=Namespace(
    #         environment=Namespace(observation=6),
    #         controller=Namespace()
    #     ),
    #     auxiliary=Namespace(),
    #     settings=Namespace(include_analytical=False),
    # )
    # args = loader.generate_args(SHP)
    # system2 = None




    args.dataset.total_sequence_length.reset(train=100)
    args.training.sampling.batch_size = 4096
    args.experiment.metrics = Namespace(
        training={"validation_analytical"},
        testing={"al", "il"},
    )
    args.experiment.exp_name = base_exp_name

    max_ir_length = 40
    configurations = [
        ("model", {
            "model.model": [
                CnnLeastSquaresPredictor,
                CnnAnalyticalPredictor,
                CnnAnalyticalLeastSquaresPredictor,
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





# %%
