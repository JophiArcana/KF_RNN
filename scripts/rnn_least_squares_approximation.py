import torch
from matplotlib import pyplot as plt

from kf_rnn.infrastructure.config import (
    EnvironmentShape,
    ExperimentConfig,
    MetricsConfig,
    ProblemShape,
    SystemConfig,
)
from kf_rnn.infrastructure.experiment import run_experiments, get_result_attr
from kf_rnn.model.convolutional import CnnLeastSquaresPredictor
from kf_rnn.system.linear_time_invariant import MOPDistribution


if __name__ == "__main__":
    base_exp_name = "RNNLeastSquaresApproximation"
    output_dir = "system6_CNN"
    output_fname = "result"

    ir_length = 64
    cfg = ExperimentConfig(
        problem=ProblemShape(environment=EnvironmentShape(observation=3), controller={}),
        system=SystemConfig(
            S_D=6,
            distribution=MOPDistribution("gaussian", "gaussian", 0.1, 0.1),
        ),
        model=CnnLeastSquaresPredictor.Config(ir_length=ir_length),
    )

    cfg.dataset.total_sequence_length.reset(train=10000, valid=10000, test=10000)
    cfg.dataset.n_traces.reset(train=1, valid=1, test=1)
    cfg.experiment.metrics = MetricsConfig(
        training={"validation_analytical"},
        testing={"al", "il"}
    )
    cfg.experiment.exp_name = base_exp_name

    configurations = []

    result, _, _ = run_experiments(
        cfg, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True
    )

    learned_kfs = get_result_attr(result, "learned_kfs")
    reference_module, module_arr = learned_kfs[()]
    observation_IR = module_arr["observation_IR"][0, 0].permute(1, 2, 0)

    L = torch.linalg.eigvals(observation_IR)
    x = torch.arange(ir_length)
    for i in range(L.shape[-1]):
        plt.plot(x, L[:, i].log().real, label=f"$Re(\\log\\lambda{i})$")
        # plt.scatter(x, L[:, i].log().imag, label=f"$Im(\\log\\lambda{i})$")
    plt.xlabel("timestep")
    # plt.yscale("log")
    plt.ylabel("eigenvalue")

    plt.legend()
    plt.show()
    raise Exception()
