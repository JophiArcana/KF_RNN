from argparse import Namespace

import torch
from matplotlib import pyplot as plt

from infrastructure import loader, utils
from infrastructure.experiment import run_experiments, get_result_attr
from model.convolutional import CnnLeastSquaresPredictor
from system.linear_time_invariant import MOPDistribution


if __name__ == "__main__":
    base_exp_name = "RNNLeastSquaresApproximation"
    output_dir = "system6_CNN"
    output_fname = "result"

    SHP = Namespace(S_D=6, problem_shape=Namespace(
        environment=Namespace(observation=3),
        controller=Namespace()
    ))
    args = loader.generate_args(SHP)

    args.model.model = CnnLeastSquaresPredictor
    args.model.ir_length = 64

    args.dataset.train.total_sequence_length = 10000
    args.dataset.train.system.distribution = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
    args.dataset.valid = args.dataset.test = Namespace(
        n_traces=1,
        total_sequence_length=10000
    )
    args.experiment.metrics = Namespace(
        training={"validation_analytical"},
        testing={"al", "il"}
    )
    args.experiment.exp_name = base_exp_name

    configurations = []

    result, _, _ = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True
    )

    learned_kfs = get_result_attr(result, "learned_kfs")
    reference_module, module_arr = learned_kfs[()]
    observation_IR = module_arr["observation_IR"][0, 0].permute(1, 2, 0)

    L = torch.linalg.eigvals(observation_IR)
    x = torch.arange(args.model.ir_length)
    for i in range(L.shape[-1]):
        plt.plot(x, L[:, i].log().real, label=f"$Re(\\log\\lambda{i})$")
        # plt.scatter(x, L[:, i].log().imag, label=f"$Im(\\log\\lambda{i})$")
    plt.xlabel("timestep")
    # plt.yscale("log")
    plt.ylabel("eigenvalue")

    plt.legend()
    plt.show()
    raise Exception()



