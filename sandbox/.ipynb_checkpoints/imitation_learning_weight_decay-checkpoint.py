import os
import sys
from argparse import Namespace

import numpy as np
import torch
from matplotlib import pyplot as plt
from tensordict import TensorDict
from transformers import TransfoXLConfig

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader, utils
from infrastructure.experiment import *
from infrastructure.experiment.plotting import COLOR_LIST
from infrastructure.settings import DEVICE
from system.controller import NNControllerGroup
from model.zero_predictor import ZeroController


if __name__ == "__main__":
    from system.linear_time_invariant import LTISystem, MOPDistribution
    from model.sequential.rnn_controller import RnnController
    from model.transformer.transformerxl_iccontroller import TransformerXLInContextController

    # Experiment setup
    exp_name = "ControlNoiseComparison_ZeroInit_Debug"
    output_dir = "imitation_learning"
    output_fname = "result"

    # SECTION: Run imitation learning experiment across different control noises
    hp_name = "control_noise_std"
    hp_values = torch.linspace(*torch.Tensor([0.0, 2.0]).exp(), 5).log().tolist()

    S_D = 3
    problem_shape = Namespace(
        environment=Namespace(observation=2),
        controller=Namespace(input=2),
    )
    SHP = Namespace(
        distribution=MOPDistribution("gaussian", "gaussian", 0.1, 0.1),
        S_D=S_D, problem_shape=problem_shape, auxiliary=Namespace(**{hp_name: hp_values[0]})
    )

    args = loader.generate_args(SHP)
    getattr(args.system.auxiliary, hp_name).update(valid=hp_values[0], test=hp_values[0])

    d_embed = 8 # 2 * S_D
    n_layer = 3
    n_head = 1
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

    args.dataset.dataset_size.update(train=1, valid=10, test=10)
    args.dataset.total_sequence_length.update(train=2000, valid=20000, test=20000)

    args.training.sampling = Namespace(method="full")
    args.training.optimizer = Namespace(
        type="Adam",
        max_lr=3e-4, min_lr=1e-9,
        weight_decay=0.0
    )
    args.training.scheduler = Namespace(
        type="exponential",
        warmup_duration=100,
        epochs=10000, lr_decay=0.9998,
    )
    args.training.iterations_per_epoch = 1
    # args.training.control_coefficient = 1e-12

    args.experiment.n_experiments = 5
    args.experiment.ensemble_size = 1
    args.experiment.exp_name = exp_name
    args.experiment.metrics = Namespace(training={"validation", "validation_controller"})
    # args.experiment.metrics = Namespace(training={"validation_analytical", "validation_controller_analytical"})

    configurations = [
        ("model", {
            "name": ["transformer_bias", "transformer_no_bias"],
            "model.bias": [True, False],
        }),
        (hp_name, {f"system.auxiliary.{hp_name}.train": hp_values})
    ]

    cache_fname = f"output/{output_dir}/{exp_name}/result_cache.pt"
    if os.path.exists(cache_fname):
        result_cache = torch.load(cache_fname, map_location=DEVICE)
    else:
        result_cache = run_experiments(args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True)
        torch.save(result_cache, cache_fname)
    result, systems, _ = result_cache

    lqg_params = systems.values[()].td().squeeze(1)
    lqg_list = [
        LTISystem(SHP.problem_shape, Namespace(**{hp_name: hp_value}), lqg_params)
        for hp_value in hp_values
    ]
    lqg = lqg_list[0]

    # LQG system visualization
    zero_controller_group = NNControllerGroup(problem_shape, *utils.stack_module_arr(utils.array_of(ZeroController(Namespace(problem_shape=problem_shape)))))
    batch_size, horizon = 128, 200

    datasets_cache_fname = f"output/{output_dir}/{exp_name}/datasets_cache.pt"
    if os.path.exists(datasets_cache_fname):
        _datasets = torch.load(datasets_cache_fname, map_location=DEVICE)
    else:
        _datasets = lqg.generate_dataset_with_controller_arr(np.concatenate([
            np.tile(np.array([zero_controller_group] + [lqg_.controller for lqg_ in lqg_list]), reps=(len(configurations[0][1]["name"]), 1)),
            utils.multi_map(
                lambda module_arr: NNControllerGroup(problem_shape, module_arr[0], module_arr[1].squeeze(1)),
                get_result_attr(result, "learned_kfs"), dtype=tuple
            )
        ], axis=1), batch_size, horizon)
        torch.save(_datasets, datasets_cache_fname)

    print(_datasets)
    raise Exception()
    
    # DONE: After running experiment, refresh LQG because the saved system overrides the one sampled at the start
    il_observation = lqg.irreducible_loss.environment.observation
    il_controller = lqg.irreducible_loss.controller.input

    # """
    # SECTION: Plot the training loss curve
    training_outputs = get_result_attr(result, "output")

    clip = 100
    sys_idx = 0
    for model_idx, model_name in enumerate(configurations[0][1]["name"]):
        fig, ax_observation = plt.subplots()
        ax_controller = ax_observation.twinx()
        for hp_value, training_output, color in zip(hp_values, training_outputs[model_idx], COLOR_LIST):
            out = training_output[sys_idx]
            tl = out["training"]
            vl = out["validation"].squeeze(-1)
            vcl = out["validation_controller"].squeeze(-1)

            def plot_with_clip(ax, y, **kwargs):
                return ax.plot(torch.arange(clip, len(y)).cpu(), y[clip:].cpu(), **kwargs)

            # plot_with_clip(ax_observation, (tl.median(dim=-2).values - il_observation[sys_idx]).detach(), color=0.7 * color, linestyle="--")
            plot_with_clip(ax_observation, (vl.median(dim=-2).values - il_observation[sys_idx]).detach(), color=color, linestyle="-", label=f"{hp_name}{hp_value}_validation")

            plot_with_clip(ax_observation, (vcl.median(dim=-2).values - il_controller[sys_idx]).detach(), color=0.7 * color, linestyle="--")

        plt.xlabel("epoch")
        ax_observation.set_yscale("log")
        ax_observation.set_ylabel("analytical_observation_loss")
        ax_observation.legend()

        ax_controller.set_yscale("log")
        ax_controller.set_ylabel("analytical_controller_loss")

        plt.title(f"{model_name}_training_curve")
        plt.show()
    # """


    optimal_states = datasets[1]["environment", "state"].flatten(1, -2)
    U, S, Vh = torch.linalg.svd(optimal_states, full_matrices=False)
    s0, s1 = S[..., :2] / ((batch_size * horizon) ** 0.5)
    compression = Vh.H[:, :2]

    # """
    # SECTION: Visualize trajectory generated using the learned controllers
    learned_controllers = [
        NNControllerGroup(SHP.problem_shape, reference_module, module_td.squeeze(0))
        for reference_module, module_td in get_result_attr(result, "learned_kfs")
    ]

    small_horizon = 20
    trace_idx, ensemble_idx = 0, 0

    trace = lqg.generate_dataset_with_controller_arr(
        [lqg.controller] + learned_controllers,
        batch_size, horizon
    )

    optimal_trajectory = trace[0]
    learned_trajectories = trace[1:]

    for hp_value, learned_trajectory, color in zip(hp_values, learned_trajectories, COLOR_LIST):
        trajectory = learned_trajectory["environment", "state"][ensemble_idx, trace_idx, :small_horizon] @ compression
        plt.plot(*trajectory.mT, color=color, marker=".", markersize="8", label=f"{hp_name}{hp_value}_learned_trajectory")
    plt.plot(*(optimal_trajectory["environment", "state"][ensemble_idx, trace_idx, :small_horizon] @ compression).mT, color="black", linestyle="--", marker="*", markersize="8", label="optimal_trajectory")

    plt.xlabel("$\\sigma_0$")
    # plt.xlim(left=-3 * s0, right=3 * s0)
    plt.ylabel("$\\sigma_1$")
    # plt.ylim(bottom=-3 * s0, top=3 * s0)
    plt.title("trajectory")

    plt.legend()
    plt.show()
    # """

    # """
    # SECTION: Plot covariances of sampled states
    indices = torch.randint(0, batch_size * horizon, (2000,))

    for hp_value, ds_, color in zip(hp_values, datasets, COLOR_LIST):
        compressed_states = ds_["environment", "state"].flatten(0, -2) @ compression
        plt.scatter(*compressed_states[indices].mT, s=3, color=color, alpha=0.15)
        utils.confidence_ellipse(
            *compressed_states.mT, plt.gca(),
            n_std=2.0, linewidth=2, linestyle='--', edgecolor=0.7 * color, label=f"{hp_name}{hp_value}_states", zorder=12
        )

    plt.xlabel("$\\sigma_0$")
    # plt.xlim(left=-3 * s0, right=3 * s0)
    plt.ylabel("$\\sigma_1$")
    # plt.ylim(bottom=-3 * s0, top=3 * s0)
    plt.title("state_covariance")

    plt.legend()
    plt.show()


    for hp_value, ds_, color in zip(hp_values, learned_trajectories, COLOR_LIST):
        compressed_states = ds_["environment", "state"].flatten(0, -2) @ compression
        plt.scatter(*compressed_states[indices].mT, s=3, color=color, alpha=0.15)
        utils.confidence_ellipse(
            *compressed_states.mT, plt.gca(),
            n_std=2.0, linewidth=2, linestyle='-', edgecolor=0.7 * color, label=f"{hp_name}{hp_value}_learned_states", zorder=12
        )

    compressed_optimal_states = optimal_trajectory["environment", "state"].flatten(0, -2) @ compression
    plt.scatter(*compressed_optimal_states[indices].mT, s=3, color="black", alpha=0.15)
    utils.confidence_ellipse(
        *compressed_optimal_states.mT, plt.gca(),
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

    for hp_value, lqg_, ds_, color in zip(hp_values, lqg_list, datasets, COLOR_LIST):
        l = loss(ds_, lqg_)
        plt.plot(torch.cumsum(l, dim=-1).median(dim=-2).values.detach(), color=color, label=f"{hp_name}{hp_value}_regret")

    plt.xlabel("horizon")
    plt.ylabel("loss")
    plt.title("cumulative_regret")

    plt.legend()
    plt.show()
    # """




