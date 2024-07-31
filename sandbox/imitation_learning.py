import math
import os
import sys
from argparse import Namespace

import numpy as np
import tensordict.utils
import torch
from matplotlib import colors
from matplotlib import pyplot as plt
from tensordict import TensorDict
from tensordict.utils import expand_as_right
from transformers import TransfoXLConfig

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader, utils
from infrastructure.experiment import *
from infrastructure.experiment.plotting import COLOR_LIST
from infrastructure.settings import DEVICE
from system.controller import NNControllerGroup
from system.linear_time_invariant import LTISystem, MOPDistribution
from model.zero_predictor import ZeroController
from model.sequential.rnn_controller import RnnController
from model.transformer.transformerxl_iccontroller import TransformerXLInContextController


if __name__ == "__main__":
    # Experiment setup
    exp_name = "ControlNoiseComparison"
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

    args.model.S_D = SHP.S_D
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
        max_lr=1e-2, min_lr=1e-9,
        weight_decay=0.0
    )
    args.training.scheduler = Namespace(
        type="exponential",
        warmup_duration=100,
        epochs=2000, lr_decay=0.995,
    )

    args.experiment.n_experiments = 5
    args.experiment.ensemble_size = 1
    args.experiment.exp_name = exp_name
    # args.experiment.metrics = Namespace(training={"validation", "validation_controller"})
    args.experiment.metrics = Namespace(training={"validation_analytical", "validation_controller_analytical"})

    configurations = [
        ("model", {
            "name": ["rnn", "transformer_no_bias", "transformer_bias"],
            "model": {
                "model": [RnnController, TransformerXLInContextController, TransformerXLInContextController],
                "bias": [None, False, True]
            },
            "training": {
                "optimizer": {
                    "max_lr": [1e-2, 3e-4, 3e-4],
                    "weight_decay": [0.0, 1e-2, 1e-2],
                },
                "scheduler": {
                    "epochs": [2000, 10000, 10000],
                    "lr_decay": [0.995, 0.9998, 0.9998],
                },
                "iterations_per_epoch": [20, 1, 1]
            },
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

    print(get_result_attr(result, "time"))
    raise Exception()

    lqg_params = systems.values[()].td().squeeze(1)
    lqg_list = [
        LTISystem(SHP.problem_shape, Namespace(**{hp_name: hp_value}), lqg_params)
        for hp_value in hp_values
    ]
    lqg = lqg_list[0]

    # DONE: After running experiment, refresh LQG because the saved system overrides the one sampled at the start
    il_observation = lqg.irreducible_loss.environment.observation
    il_controller = lqg.irreducible_loss.controller.input
    zl_observation = lqg.zero_predictor_loss.environment.observation
    zl_controller = lqg.zero_predictor_loss.controller.input

    sys_idx = 2

    # """
    # SECTION: Plot the training loss curve
    training_outputs = get_result_attr(result, "output")

    clip = 25
    plt.rcParams['figure.figsize'] = (18.0, 5.0)
    observation_fig, observation_axs = plt.subplots(nrows=1, ncols=2, sharey=True)
    controller_fig, controller_axs = plt.subplots(nrows=1, ncols=2, sharey=True)

    def plot_normalized_with_clip(ax_idx, out, color: np.ndarray, label: str):
        tl = out["training"]
        try:
            vl = out["validation_analytical"].squeeze(-1)
            vcl = out["validation_controller_analytical"].squeeze(-1)
        except KeyError:
            vl = out["validation"].squeeze(-1)
            vcl = out["validation_controller"].squeeze(-1)

        normalized_vl = ((vl - expand_as_right(il_observation, vl)) / expand_as_right(zl_observation - il_observation, vl)).median(dim=-2).values[sys_idx]
        normalized_vcl = ((vcl - expand_as_right(il_controller, vcl)) / expand_as_right(zl_controller - il_controller, vcl)).median(dim=-2).values[sys_idx]

        x = torch.arange(clip, out.shape[-1])
        observation_axs[ax_idx].plot(x.cpu(), normalized_vl[clip:].detach().cpu(), linestyle="-", color=color, label=label)
        controller_axs[ax_idx].plot(x.cpu(), normalized_vcl[clip:].detach().cpu(), linestyle="--", color=0.7 * color, label=label)

    for hp_index, (hp_value, color) in enumerate(zip(hp_values, COLOR_LIST)):
        rnn_out = training_outputs[0, hp_index]
        plot_normalized_with_clip(0, rnn_out, color, f"$\\alpha=${hp_value}")

        transformer_out = training_outputs[2, hp_index]
        plot_normalized_with_clip(1, transformer_out, color, f"$\\alpha=${hp_value}")

    for (ax, name) in zip(observation_axs, ("rnn", "transformer")):
        ax.set_xscale("log")
        ax.set_xlabel("training_epoch")
        ax.set_yscale("log")
        ax.set_ylabel("validation_error")

        ax.legend()
        ax.set_title(name)
    observation_fig.suptitle("observation_prediction_error")

    for (ax, name) in zip(controller_axs, ("rnn", "transformer")):
        ax.set_xscale("log")
        ax.set_xlabel("training_epoch")
        ax.set_yscale("log")
        ax.set_ylabel("validation_error")

        ax.legend()
        ax.set_title(name)
    controller_fig.suptitle("controller_imitation_error")

    plt.show()
    # """
    raise Exception()


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

    zero_trace = _datasets[:, 0:1]
    optimal_trace = _datasets[:, 1:2]
    exemplar_traces = _datasets[:, 1:len(hp_values) + 1]
    learned_traces = _datasets[:, len(hp_values) + 1:]

    optimal_states = optimal_trace["environment", "state"].flatten(-3, -2)
    U, S, Vh = torch.linalg.svd(optimal_states, full_matrices=False)
    s_max = S[0, 0, :, 0] / ((batch_size * horizon) ** 0.5)

    # DONE: Precompute aligned states
    alignment = Vh.mH[..., None, :, :]
    
    aligned_optimal_states = (optimal_trace["environment", "state"] @ alignment).squeeze(1)
    aligned_zero_states = (zero_trace["environment", "state"] @ alignment).squeeze(1)

    aligned_exemplar_states = exemplar_traces["environment", "state"] @ alignment
    aligned_learned_states = learned_traces["environment", "state"] @ alignment

    # DONE: Precompute compressed states
    compression = Vh.mH[..., :2][..., None, :, :]

    compressed_optimal_states = (optimal_trace["environment", "state"] @ compression).squeeze(1)
    compressed_zero_states = (zero_trace["environment", "state"] @ compression).squeeze(1)

    compressed_exemplar_states = exemplar_traces["environment", "state"] @ compression
    compressed_learned_states = learned_traces["environment", "state"] @ compression


    """
    # SECTION: Visualize trajectory generated using the learned controllers
    small_horizon = 12
    trace_idx = 0

    for model_idx, model_name in enumerate(configurations[0][1]["name"]):
        for hp_value, states, color in zip(hp_values, compressed_learned_states[model_idx], COLOR_LIST):
            trajectory = states[sys_idx, trace_idx, :small_horizon]
            plt.plot(*trajectory.mT, color=color, linewidth=0.5, marker=".", markersize=8, label=f"$\\alpha=${hp_value}_learned_trajectory")

        plt.plot(*compressed_zero_states[model_idx, sys_idx, trace_idx, :small_horizon].mT, color="gray", linewidth=0.5, linestyle="--", marker="o", markersize=4, label="zero_predictor_trajectory")
        plt.plot(*compressed_optimal_states[model_idx, sys_idx, trace_idx, :small_horizon].mT, color="black", linestyle="--", marker="*", markersize=8, label="optimal_trajectory")
        plt.plot(*compressed_zero_states[model_idx, sys_idx, trace_idx, 0], color="black", marker="*", markersize=16)

        plt.xlabel("$\\sigma_0$")
        plt.ylabel("$\\sigma_1$")
        plt.title(f"{model_name}_trajectories")

        plt.legend(fontsize=6)
        plt.show()
    """


    # """
    # SECTION: Plot covariances of exemplar states
    indices = torch.randint(0, batch_size * horizon, (2000,))

    sampled_compressed_optimal_states = compressed_optimal_states.flatten(-3, -2)[0, sys_idx, indices]
    sampled_compressed_zero_states = compressed_zero_states.flatten(-3, -2)[0, sys_idx, indices]

    def plot_covariance(states: torch.Tensor, color: np.ndarray, label: str) -> None:
        plt.scatter(*states.mT, s=3, color=color, alpha=0.15)
        utils.confidence_ellipse(
            *states.mT, plt.gca(),
            n_std=2.0, linewidth=2, linestyle="--", edgecolor=0.7 * color, label=label, zorder=12
        )

    for hp_value, states, color in zip(hp_values, compressed_exemplar_states[0], COLOR_LIST):
        plot_covariance(states.flatten(-3, -2)[sys_idx, indices], color, f"$\\alpha=${hp_value}_states")

    plot_covariance(sampled_compressed_optimal_states, np.array(colors.to_rgb("black")), "optimal_states")
    plot_covariance(sampled_compressed_zero_states, np.array(colors.to_rgb("gray")), "zero_predictor_states")


    plt.xlabel("$\\sigma_0$")
    plt.ylabel("$\\sigma_1$")
    plt.title(f"exemplar_state_covariance")

    plt.legend(fontsize=6)
    plt.savefig("images/paper/exemplar/state_covariance.png", bbox_inches="tight")
    plt.show()
    # """


    # """
    # SECTION: Plot excitations of exemplar states
    def plot_state_excitation(ax, states: torch.Tensor, color: np.ndarray, label: str) -> None:
        s = states.std()
        x = torch.linspace(-3 * s, 3 * s, 100)

        nbins = 20 * math.ceil(s * (batch_size * horizon) ** 0.5)
        ax.hist(states, density=True, color=color, bins=nbins, histtype="step", linewidth=0.3, label=label)
        ax.plot(x, torch.distributions.Normal(loc=0.0, scale=s).log_prob(x).exp(), color=0.5 * color, linewidth=1.0)

    fig, axs = plt.subplots(nrows=S_D, ncols=1, sharex=True, sharey=True)
    for i in range(S_D):
        for hp_value, states, color in zip(hp_values, aligned_exemplar_states[0], COLOR_LIST):
            plot_state_excitation(axs[i], states.flatten(-3, -2)[sys_idx, :, i], color, f"$\\alpha=${hp_value}_dist")

        # plot_state_excitation(aligned_optimal_states.flatten(-3, -2)[0, sys_idx, :, i], np.array(colors.to_rgb("black")), f"optimal_distribution")
        # plot_state_excitation(axs[i], aligned_zero_states.flatten(-3, -2)[0, sys_idx, :, i], np.array(colors.to_rgb("gray")), f"zero_predictor_dist")

        axs[i].set_xlabel(f"$\\sigma_{i}$")
        axs[i].set_xlim(left=-3.5 * s_max[sys_idx], right=3.5 * s_max[sys_idx])
        axs[i].set_ylabel("distribution")

        if i == 0:
            axs[i].set_title("exemplar_state_excitation")
            axs[i].legend(fontsize=6)

    plt.savefig("images/paper/exemplar/state_excitation.png", bbox_inches="tight")
    plt.show()
    # """


    """
    for model_idx, model_name in enumerate(configurations[0][1]["name"]):
        # SECTION: Plot covariances of learned states
        for hp_value, states, color in zip(hp_values, compressed_learned_states[model_idx], COLOR_LIST):
            plot_covariance(states.flatten(-3, -2)[sys_idx, indices], color, f"$\\alpha=${hp_value}_learned_states")

        plot_covariance(sampled_compressed_optimal_states, np.array(colors.to_rgb("black")), "optimal_states")
        plot_covariance(sampled_compressed_zero_states, np.array(colors.to_rgb("gray")), "zero_predictor_states")


        plt.xlabel("$\\sigma_0$")
        plt.ylabel("$\\sigma_1$")
        plt.title(f"{model_name}_learned_state_covariance")

        plt.legend(fontsize=6)
        plt.show()


        # SECTION: Plot excitations of learned states
        fig, axs = plt.subplots(nrows=S_D, ncols=1, sharex=True, sharey=True)
        for i in range(S_D):
            for hp_value, states, color in zip(hp_values, aligned_learned_states[model_idx], COLOR_LIST):
                plot_state_excitation(axs[i], states.flatten(-3, -2)[sys_idx, :, i], color, f"$\\alpha=${hp_value}_learned_dist")
    
            # plot_state_excitation(aligned_optimal_states.flatten(-3, -2)[0, sys_idx, :, i], np.array(colors.to_rgb("black")), f"optimal_distribution")
            # plot_state_excitation(axs[i], aligned_zero_states.flatten(-3, -2)[0, sys_idx, :, i], np.array(colors.to_rgb("gray")), f"zero_predictor_dist")
    
            axs[i].set_xlabel(f"$\\sigma_{i}$")
            axs[i].set_xlim(left=-3.5 * s_max[sys_idx], right=3.5 * s_max[sys_idx])
            axs[i].set_ylabel("distribution")
    
            if i == 0:
                axs[i].set_title(f"{model_name}_learned_state_excitation")
                axs[i].legend(fontsize=6)
    
        plt.show()
    """


    # """
    # SECTION: Plot exemplar cumulative loss over horizon
    def plot_cumulative_loss(dataset: TensorDict[str, torch.Tensor], color: np.ndarray, label: str, **kwargs) -> None:
        state_loss = (
                dataset["environment", "state"][..., None, :] @
                sum(lqg.controller.Q.values())[sys_idx] @
                dataset["environment", "state"][..., :, None]
        )
        control_loss = sum(
            (dataset["controller", k][..., None, :] @ v[sys_idx] @ dataset["controller", k][..., :, None])
            for k, v in lqg.controller.R.items()
        )
        loss = (state_loss + control_loss)[..., 0, 0]

        x = torch.arange(horizon)
        plt.plot(x, torch.cumsum(loss, dim=-1).median(dim=-2).values.detach(), color=color, label=label, **kwargs)
        plt.fill_between(
            x, *torch.cumsum(loss, dim=-1).quantile(torch.tensor([0.25, 0.75]), dim=-2).detach(),
            color=color, alpha=0.1
        )

    for hp_value, dataset, color in zip(hp_values, exemplar_traces[0], COLOR_LIST):
        plot_cumulative_loss(dataset[sys_idx], color, linewidth=0.5, label=f"$\\alpha=${hp_value}_loss")

    plot_cumulative_loss(optimal_trace[0, 0, sys_idx], np.array(colors.to_rgb("black")), linewidth=2.0, linestyle="--", label="optimal_loss")
    plot_cumulative_loss(zero_trace[0, 0, sys_idx], np.array(colors.to_rgb("gray")), linewidth=2.0, linestyle="--", label="zero_predictor_loss")


    plt.xlabel("horizon")
    plt.ylabel("loss")
    plt.title("exemplar_cumulative_loss")

    plt.legend(fontsize=6)
    plt.savefig("images/paper/exemplar/cumulative_loss.png", bbox_inches="tight")
    plt.show()
    # """


    """
    # SECTION: Plot learned cumulative loss over horizon
    for model_idx, model_name in enumerate(configurations[0][1]["name"]):
        for hp_value, dataset, color in zip(hp_values, learned_traces[model_idx], COLOR_LIST):
            plot_cumulative_loss(dataset[sys_idx], color, linewidth=0.5, label=f"$\\alpha=${hp_value}_learned_loss")

        plot_cumulative_loss(optimal_trace[0, 0, sys_idx], np.array(colors.to_rgb("black")), linewidth=2.0, linestyle="--", label="optimal_loss")
        plot_cumulative_loss(zero_trace[0, 0, sys_idx], np.array(colors.to_rgb("gray")), linewidth=2.0, linestyle="--", label="zero_predictor_loss")


        plt.xlabel("horizon")
        plt.ylabel("loss")
        plt.title(f"{model_name}_learned_cumulative_loss")

        plt.legend(fontsize=6)
        plt.show()
    """




