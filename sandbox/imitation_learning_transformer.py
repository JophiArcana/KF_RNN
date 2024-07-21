import os
import sys
from argparse import Namespace

import numpy as np
import torch
from dimarray import DimArray
from matplotlib import pyplot as plt
from tensordict import TensorDict
from transformers import TransfoXLConfig

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader, utils
from infrastructure.experiment import *
from infrastructure.experiment.plotting import COLOR_LIST
from infrastructure.static import PARAM_GROUP_FORMATTER
from infrastructure.settings import DEVICE
from system.controller import NNControllerGroup


if __name__ == "__main__":
    from system.linear_time_invariant import LTISystem, MOPDistribution
    from model.transformer.transformerxl_iccontroller import TransformerXLInContextController

    # Experiment setup
    exp_name = "ControlNoiseComparison"
    output_dir = "imitation_learning_transformer"
    output_fname = "result"

    # SECTION: Run imitation learning experiment across different control noises
    SHP = Namespace(
        S_D=3, problem_shape=Namespace(
            environment=Namespace(observation=2),
            controller=Namespace(input=2),
        )
    )
    hp_name = "control_noise_std"


    os.makedirs(f"output/{output_dir}", exist_ok=True)
    lqg_fname = f"output/{output_dir}/systems.pt"
    if not os.path.exists(lqg_fname):
        dist = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
        lqg = dist.sample(SHP, ())
        torch.save(lqg, lqg_fname)
    else:
        lqg = torch.load(lqg_fname, map_location=DEVICE)

    control_noise_std = [0.0, 0.5, 1.5, 2.0]
    lqg_list = [LTISystem(SHP.problem_shape, lqg.td(), cns) for cns in control_noise_std]

    train_systems = DimArray([
        LTISystem(SHP.problem_shape, lqg.td().expand(1, 1), cns)
        for cns in control_noise_std
    ], dims=(PARAM_GROUP_FORMATTER.format(hp_name, -1),))
    test_systems = DimArray(LTISystem(SHP.problem_shape, lqg.td().expand(1, 1)), dims=())
    if not os.path.exists(train_dir := f"output/{output_dir}/{exp_name}/training"):
        os.makedirs(train_dir)
        torch.save({"train": train_systems, "valid": test_systems}, f"{train_dir}/systems.pt")
    if not os.path.exists(test_dir := f"output/{output_dir}/{exp_name}/testing"):
        os.makedirs(test_dir)
        torch.save({"test": test_systems}, f"{test_dir}/systems.pt")

    args = loader.generate_args(SHP)

    d_embed = 8
    n_layer = 3
    n_head = 2
    d_inner = 2 * d_embed

    args.model.model = TransformerXLInContextController
    args.model.transformerxl = TransfoXLConfig(
        d_model=d_embed,
        d_embed=d_embed,
        n_layer=n_layer,
        n_head=n_head,
        d_head=d_embed // n_head,
        d_inner=d_inner,
        dropout=0.0,
    )

    args.dataset.train = Namespace(
        dataset_size=1,
        total_sequence_length=2000,
        system=Namespace(n_systems=1)
    )
    args.dataset.valid = args.dataset.test = Namespace(
        dataset_size=10,
        total_sequence_length=20000,
        system=Namespace(n_systems=1)
    )

    # args.train.sampling = Namespace(
    #     method="subsequence_padded",
    #     subsequence_length=200,
    #     batch_size=32
    # )
    args.train.sampling = Namespace(method="full")
    args.train.optimizer = Namespace(
        type="Adam",
        max_lr=3e-4, min_lr=1e-6,
        weight_decay=1e-2, momentum=0.9
    )
    args.train.scheduler = Namespace(
        type="exponential",
        epochs=10000, lr_decay=0.9998,

    )
    args.train.iterations_per_epoch = 1

    args.experiment.n_experiments = 1
    args.experiment.ensemble_size = 1
    args.experiment.exp_name = exp_name
    args.experiment.metrics = {"validation", "validation_controller"}

    configurations = [
        (hp_name, {"system.control_noise_std": control_noise_std})
    ]

    result, _, _ = run_experiments(args, configurations, {
        "dir": output_dir,
        "fname": output_fname
    }, save_experiment=True)

    # DONE: After running experiment, refresh LQG because the saved system overrides the one sampled at the start
    il_observation = lqg.irreducible_loss.environment.observation
    il_controller = lqg.irreducible_loss.controller.input

    # """
    # SECTION: Plot the training loss curve
    training_outputs = list(get_result_attr(result, "output"))

    fig, ax_observation = plt.subplots()
    ax_controller = ax_observation.twinx()
    clip = 100
    for cns, training_output, color in zip(control_noise_std, training_outputs, COLOR_LIST):
        out = training_output.squeeze(0)
        tl = out["training"]
        vl = out["validation"].squeeze(-1)
        vcl = out["validation_controller"].squeeze(-1)

        def plot_with_clip(ax, y, **kwargs):
            return ax.plot(torch.arange(clip, len(y)).cpu(), y[clip:].cpu(), **kwargs)

        # plot_with_clip(ax_observation, (tl.median(dim=0).values - il_observation).detach(), color=0.7 * color, linestyle="--")
        plot_with_clip(ax_observation, (vl.median(dim=0).values - il_observation).detach(), color=color, linestyle="-", label=f"{hp_name}{cns}_validation")

        plot_with_clip(ax_controller, (vcl.median(dim=0).values - il_controller).detach(), color=0.7 * color, linestyle="--")

    plt.xlabel("epoch")
    ax_observation.set_yscale("log")
    ax_observation.set_ylabel("analytical_observation_loss")
    ax_observation.legend()

    ax_controller.set_yscale("log")
    ax_controller.set_ylabel("analytical_controller_loss")

    plt.title("training_curve")
    plt.show()
    # """

    # LQG system visualization
    batch_size, horizon = 128, 200
    datasets = [lqg_.generate_dataset(batch_size, horizon) for lqg_ in lqg_list]

    optimal_states = datasets[0]["environment", "state"].flatten(0, -2)
    U, S, Vh = torch.linalg.svd(optimal_states, full_matrices=False)
    s0, s1 = S[:2] * len(optimal_states) ** -0.5
    compression = Vh.H[:, :2]

    # """
    # SECTION: Visualize trajectory generated using the learned controllers
    learned_controllers = [
        NNControllerGroup(SHP.problem_shape, reference_module, module_td.squeeze(0))
        for reference_module, module_td in get_result_attr(result, "learned_kfs")
    ]

    small_horizon = 12
    trace_idx, ensemble_idx = 0, 0

    trace = lqg.generate_dataset_with_controller_arr(
        np.array([lqg.controller] + learned_controllers),
        batch_size, horizon
    )

    optimal_trajectory = trace[0]
    learned_trajectories = trace[1:]

    for cns, learned_trajectory, color in zip(control_noise_std, learned_trajectories, COLOR_LIST):
        trajectory = learned_trajectory["environment", "state"][ensemble_idx, trace_idx, :small_horizon] @ compression
        plt.plot(*trajectory.mT.cpu(), color=color, marker=".", markersize="8", label=f"{hp_name}{cns}_learned_trajectory")
    plt.plot(*(optimal_trajectory["environment", "state"][ensemble_idx, trace_idx, :small_horizon] @ compression).mT.cpu(), color="black", linestyle="--", marker="*", markersize="8", label="optimal_trajectory")

    plt.xlabel("$\\sigma_0$")
    # plt.xlim(left=-3 * s0, right=3 * s0)
    plt.ylabel("$\\sigma_1$")
    # plt.ylim(bottom=-3 * s0, top=3 * s0)
    plt.title("trajectory")

    plt.legend(fontsize=6)
    plt.show()
    # """

    # """
    # SECTION: Plot covariances of sampled states
    indices = torch.randint(0, batch_size * horizon, (2000,))

    for cns, ds_, color in zip(control_noise_std, datasets, COLOR_LIST):
        compressed_states = ds_["environment", "state"].flatten(0, -2) @ compression
        plt.scatter(*compressed_states[indices].mT.cpu(), s=3, color=color, alpha=0.15)
        utils.confidence_ellipse(
            *compressed_states.mT.cpu(), plt.gca(),
            n_std=2.0, linewidth=2, linestyle='--', edgecolor=0.7 * color, label=f"{hp_name}{cns}_states", zorder=12
        )

    plt.xlabel("$\\sigma_0$")
    # plt.xlim(left=-3 * s0, right=3 * s0)
    plt.ylabel("$\\sigma_1$")
    # plt.ylim(bottom=-3 * s0, top=3 * s0)
    plt.title("state_covariance")

    plt.legend()
    plt.show()


    for cns, ds_, color in zip(control_noise_std, learned_trajectories, COLOR_LIST):
        compressed_states = ds_["environment", "state"].flatten(0, -2) @ compression
        plt.scatter(*compressed_states[indices].mT.cpu(), s=3, color=color, alpha=0.15)
        utils.confidence_ellipse(
            *compressed_states.mT.cpu(), plt.gca(),
            n_std=2.0, linewidth=2, linestyle='-', edgecolor=0.7 * color, label=f"{hp_name}{cns}_learned_states", zorder=12
        )

    compressed_optimal_states = optimal_trajectory["environment", "state"].flatten(0, -2) @ compression
    plt.scatter(*compressed_optimal_states[indices].mT.cpu(), s=3, color="black", alpha=0.15)
    utils.confidence_ellipse(
        *compressed_optimal_states.mT.cpu(), plt.gca(),
        n_std=2.0, linewidth=2, linestyle='-', edgecolor="black", label=f"optimal_states", zorder=12
    )

    plt.xlabel("$\\sigma_0$")
    # plt.xlim(left=-3 * s0, right=3 * s0)
    plt.ylabel("$\\sigma_1$")
    # plt.ylim(bottom=-3 * s0, top=3 * s0)
    plt.title("learned_state_covariance")

    plt.legend()
    plt.show()
    # """

    # """
    # SECTION: Plot cumulative loss over horizon
    def loss(ds_: TensorDict[str, torch.Tensor], lqg_: LTISystem) -> torch.Tensor:
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

    for cns, lqg_, ds_, color in zip(control_noise_std, lqg_list, datasets, COLOR_LIST):
        l = loss(ds_, lqg_)
        plt.plot(torch.cumsum(l, dim=-1).median(dim=-2).values.detach().cpu(), color=color, label=f"{hp_name}{cns}_regret")

    plt.xlabel("horizon")
    plt.ylabel("loss")
    plt.title("cumulative_regret")

    plt.legend()
    plt.show()
    # """




