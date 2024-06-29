import os
import sys
from argparse import Namespace
from typing import *

import numpy as np
import torch
from matplotlib import pyplot as plt
from tensordict import TensorDict

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader, utils
from infrastructure.experiment import *
from infrastructure.experiment.plotting import COLOR_LIST
from system.simple.base import SystemGroup
from system.actionable.base import ActionableSystemGroup
from system.controller import NNControllerGroup


if __name__ == "__main__":
    from system.actionable.linear_quadratic_gaussian import LQGSystem, LQGDistribution
    from model.sequential.rnn_controller import RnnController

    # SECTION: Run imitation learning experiment across different control noises
    SHP = Namespace(
        S_D=3, problem_shape=Namespace(
            environment=Namespace(observation=2),
            controller=Namespace(input=2),
        )
    )
    hp_name = "control_noise_std"

    dist_ = LQGDistribution("gaussian", "gaussian", 0.1, 0.1, 1.0, 1.0)
    lqg_ = dist_.sample(SHP, ())

    class ConstantLQGDistribution(LQGSystem.Distribution):
        def __init__(self, params: TensorDict[str, torch.Tensor], cns: float):
            LQGSystem.Distribution.__init__(self)
            self.params = params
            self.control_noise_std = cns

        def sample(self, SHP: Namespace, shape: Tuple[int, ...]) -> SystemGroup:
            return LQGSystem(SHP.problem_shape, self.params.expand(*shape), self.control_noise_std)

    # Experiment setup
    exp_name = "ControlNoiseComparison"
    output_dir = "imitation_learning"
    output_fname = "result"

    args = loader.generate_args(SHP)
    args.model.model = RnnController
    args.model.S_D = SHP.S_D
    args.dataset.train = Namespace(
        dataset_size=1,
        total_sequence_length=2000,
        system=Namespace(n_systems=1)
    )
    args.dataset.valid = args.dataset.test = Namespace(
        dataset_size=10,
        total_sequence_length=100000,
        system=Namespace(
            n_systems=1,
            distribution=ConstantLQGDistribution(lqg_.td(), 0.0)
        )
    )
    args.train.sampling = Namespace(method="full")
    args.train.optimizer = Namespace(
        type="Adam",
        max_lr=1e-2, min_lr=1e-9,
        weight_decay=0.0
    )
    args.train.scheduler = Namespace(
        type="exponential",
        warmup_duration=100,
        epochs=2000, lr_decay=0.995,
    )

    args.experiment.n_experiments = 1
    args.experiment.ensemble_size = 32
    args.experiment.exp_name = exp_name
    args.experiment.metrics = {"validation_analytical"}

    control_noise_std = [0.0, 0.1, 0.2, 0.3]
    dist_list = [ConstantLQGDistribution(lqg_.td(), cns) for cns in control_noise_std]
    configurations = [
        (hp_name, {"dataset.train.system.distribution": dist_list})
    ]

    result, systems, dataset = run_experiments(args, configurations, {
        "dir": output_dir,
        "fname": output_fname
    }, save_experiment=True)

    # DONE: After running experiment, refresh LQG because the saved system overrides the one sampled at the start
    lqg = LQGSystem(SHP.problem_shape, systems.values[()].td().squeeze(1).squeeze(0))
    dist_list = [ConstantLQGDistribution(lqg.td(), cns) for cns in control_noise_std]
    lqg_list = [dist_.sample(SHP, ()) for dist_ in dist_list]

    """
    # Plot the training loss curve
    ax = plt.gca()
    ax2 = ax.twinx()

    training_outputs = list(get_result_attr(result, "output"))
    for idx, (cns, training_output) in enumerate(zip(control_noise_std, training_outputs)):
        out = training_output.squeeze(0)
        tl = out["training"]
        al = out["validation_analytical"].squeeze(-1)

        ax.plot(tl.median(dim=0).values, color=0.7 * COLOR_LIST[idx], linestyle="--")
        ax2.plot(al.median(dim=0).values, color=COLOR_LIST[idx], linestyle="-", label=f"{hp_name}{cns}_validation_analytical")

    plt.xlabel("epoch")
    ax.set_ylabel(r"loss: $\frac{1}{L}|| F_\theta(\tau) - \tau ||^2$")
    plt.title("training_curve")

    plt.legend()
    plt.show()
    """

    # SECTION: LQG system visualization
    batch_size, horizon = 1024, 1000
    datasets = [lqg_.generate_dataset(batch_size, horizon) for lqg_ in lqg_list]

    optimal_states = datasets[0]["environment", "state"].flatten(0, -2)
    U, S, Vh = torch.linalg.svd(optimal_states, full_matrices=False)
    s0, s1 = S[:2] * len(optimal_states) ** -0.5
    compression = Vh.H[:, :2]

    """
    # Plot covariances of sampled states
    indices = torch.randint(0, batch_size * horizon, (2000,))

    for idx, (cns, ds_) in enumerate(zip(control_noise_std, datasets)):
        compressed_states = ds_["environment", "state"].flatten(0, -2) @ compression
        color = COLOR_LIST[idx]

        plt.scatter(*compressed_states[indices].mT, s=3, color=color, alpha=0.15)
        utils.confidence_ellipse(
            *compressed_states.mT, plt.gca(),
            n_std=2.0, linewidth=1.5, linestyle='--', edgecolor=0.7 * color, label=f"{hp_name}{cns}_states", zorder=12
        )

    plt.xlabel("$\\sigma_0$")
    plt.xlim(left=-3 * s0, right=3 * s0)
    plt.ylabel("$\\sigma_1$")
    plt.ylim(bottom=-3 * s0, top=3 * s0)
    plt.title("state_covariance")

    plt.legend()
    plt.show()
    """

    # SECTION: Visualize trajectory generated using the learned controllers
    learned_controllers = [
        NNControllerGroup(SHP.problem_shape, reference_module, module_td.squeeze(0))
        for reference_module, module_td in get_result_attr(result, "learned_kfs")
    ]

    small_batch_size, small_horizon = 5, 10
    trace = lqg.generate_dataset_with_controller_arr(np.array([lqg.controller] + learned_controllers), small_batch_size, small_horizon)

    optimal_trajectory = trace[0, 0]["environment", "state"]
    learned_trajectories = trace[1:]["environment", "state"]

    trace_idx, ensemble_idx = 0, 0
    for idx, (cns, learned_trajectory) in enumerate(zip(control_noise_std, learned_trajectories)):
        trajectory = learned_trajectory[ensemble_idx, trace_idx] @ compression
        plt.plot(*trajectory.mT, color=COLOR_LIST[idx], marker=".", markersize="8", label=f"{hp_name}{cns}_trajectory")
    plt.plot(*(optimal_trajectory[trace_idx] @ compression).mT, color="black", linestyle="--", marker="*", markersize="8", label="optimal_trajectory")

    plt.xlabel("$\\sigma_0$")
    plt.xlim(left=-3 * s0, right=3 * s0)
    plt.ylabel("$\\sigma_1$")
    plt.ylim(bottom=-3 * s0, top=3 * s0)
    plt.title("trajectory")

    plt.legend()
    plt.show()

    raise Exception()


    # Plot cumulative loss over horizon
    def loss(ds_: TensorDict[str, torch.Tensor], lqg_: LQGSystem) -> torch.Tensor:
        state_loss = (
                ds_["environment", "state"].unsqueeze(-2) @
                sum(lqg_.controller.Q.values()) @
                ds_["environment", "state"].unsqueeze(-1)
        ).squeeze(-2).squeeze(-1)
        control_loss = sum(
            (ds_["controller", k].unsqueeze(-2) @ v @ ds_["controller", k].unsqueeze(-1)).squeeze(-2).squeeze(-1)
            for k, v in lqg_.controller.R.items()
        )
        return state_loss + control_loss

    for idx, (cns, lqg_, ds_) in enumerate(zip(control_noise_std, lqg_list, datasets)):
        l = loss(ds_, lqg_)
        plt.plot(torch.cumsum(l, dim=1).median(dim=0).values.detach(), color=COLOR_LIST[idx], label=f"{hp_name}{cns}_regret")

    plt.xlabel("horizon")
    plt.ylabel("cumulative_loss")
    plt.title("regret_growth")

    plt.legend()
    plt.show()





