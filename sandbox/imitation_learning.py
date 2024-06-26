import os
import sys
from argparse import Namespace
from typing import *

import torch
from matplotlib import pyplot as plt
from tensordict import TensorDict

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader, utils
from infrastructure.experiment import *
from system.core import SystemGroup

if __name__ == "__main__":
    from system.linear_quadratic_gaussian import LinearQuadraticGaussianGroup
    from system.linear_quadratic_gaussian import LQGDistribution
    from model.sequential.rnn_controller import RnnController

    # SECTION: Run imitation learning experiment across different control noises
    SHP = Namespace(S_D=3, I_D=2, O_D=2, input_enabled=True)
    hp_name = "control_noise_std"

    dist = LQGDistribution("gaussian", "gaussian", 0.1, 0.1, 1.0, 1.0)
    lqg = dist.sample(SHP, ())

    class ConstantLQGDistribution(LinearQuadraticGaussianGroup.Distribution):
        def __init__(self, params: TensorDict[str, torch.Tensor], cns: float):
            LinearQuadraticGaussianGroup.Distribution.__init__(self)
            self.params = params
            self.control_noise_std = cns

        def sample(self,
                   SHP: Namespace,
                   shape: Tuple[int, ...],
                   params: TensorDict[str, torch.Tensor] | Dict[str, torch.Tensor] = None
        ) -> SystemGroup:
            return LinearQuadraticGaussianGroup(self.params.expand(*shape), SHP.input_enabled, control_noise_std=self.control_noise_std)

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
            distribution=ConstantLQGDistribution(lqg.td(), 0.0)
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
    dist_list = [ConstantLQGDistribution(lqg.td(), cns) for cns in control_noise_std]
    configurations = [
        (hp_name, {"dataset.train.system.distribution": dist_list})
    ]

    result, systems, dataset = run_experiments(args, configurations, {
        "dir": output_dir,
        "fname": output_fname
    }, save_experiment=False)


    # SECTION: LQG system visualization
    lqg_list = [dist_.sample(SHP, ()) for dist_ in dist_list]

    batch_size, horizon = 1024, 1000
    datasets = [lqg_.generate_dataset(batch_size, horizon) for lqg_ in lqg_list]

    # Plot covariances of sampled states
    from infrastructure.experiment.plotting import COLOR_LIST
    indices = torch.randint(0, batch_size * horizon, (2000,))

    for idx, (cns, ds_) in enumerate(zip(control_noise_std, datasets)):
        states = ds_["state"].flatten(0, -2)
        color = COLOR_LIST[idx]

        plt.scatter(*states[indices].mT, s=3, color=color, alpha=0.15)
        utils.confidence_ellipse(
            *states.mT, plt.gca(),
            n_std=2.0, linewidth=1.5, linestyle='--', edgecolor=0.7 * color, label=f"{hp_name}{cns}_states", zorder=12
        )

    plt.xlim(left=-0.5, right=0.5)
    plt.ylim(bottom=-0.5, top=0.5)
    plt.title("state_covariance")

    plt.legend()
    plt.show()

    # Plot cumulative loss over horizon
    def loss(ds_: TensorDict[str, torch.Tensor], lqg_: LinearQuadraticGaussianGroup) -> torch.Tensor:
        sl = (ds_["state"].unsqueeze(-2) @ lqg_.Q @ ds_["state"].unsqueeze(-1)).squeeze(-2).squeeze(-1)
        il = (ds_["input"].unsqueeze(-2) @ lqg_.R @ ds_["input"].unsqueeze(-1)).squeeze(-2).squeeze(-1)
        # il = 0
        return sl + il

    for idx, (cns, lqg_, ds_) in enumerate(zip(control_noise_std, lqg_list, datasets)):
        l = loss(ds_, lqg_)
        plt.plot(torch.cumsum(l, dim=1).median(dim=0).values.detach(), color=COLOR_LIST[idx], label=f"{hp_name}{cns}_regret")

    plt.xlabel("horizon")
    plt.ylabel("cumulative_loss")
    plt.legend()
    plt.show()




