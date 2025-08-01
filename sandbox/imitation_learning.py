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
    exp_name = "ControlNoiseComparison_ZeroInit_NoWeightDecay"
    output_dir = "imitation_learning"
    output_fname = "result"

    # SECTION: Run imitation learning experiment across different control noises
    hp_name = "control_noise_std"
    hp_values = [round(a, 2) for a in torch.linspace(*torch.Tensor([0.0, 2.0]).exp(), 5).log().tolist()]

    S_D, I_D, O_D = 3, 2, 2
    problem_shape = Namespace(
        environment=Namespace(observation=I_D),
        controller=Namespace(input=O_D),
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
        type="AdamW",
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
    args.experiment.metrics = Namespace(training={"validation", "validation_controller"})
    # args.experiment.metrics = Namespace(training={"validation_analytical", "validation_controller_analytical"})

    configurations = [
        ("model", {
            "name": ["rnn", "transformer_bias", "transformer_no_bias"],
            "model": {
                "model": [RnnController, TransformerXLInContextController, TransformerXLInContextController],
                "bias": [None, True, False]
            },
            "training": {
                "optimizer": {
                    "max_lr": [1e-2, 1e-3, 1e-3]
                },
                "scheduler": {
                    "epochs": [2000, 10000, 10000],
                    "lr_decay": [0.995, 0.9997, 0.9997],
                },
                "iterations_per_epoch": [20, 1, 1]
            },
        }),
        (hp_name, {f"system.auxiliary.{hp_name}.train": hp_values})
    ]

    cache_fname = f"output/{output_dir}/{exp_name}/result_cache.pt"
    if os.path.exists(cache_fname):
        result_cache = utils.torch_load(cache_fname)
    else:
        result_cache = run_experiments(args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True)
        torch.save(result_cache, cache_fname)
    result, systems, _ = result_cache

    print(get_result_attr(result, "time"))

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
        _datasets = utils.torch_load(datasets_cache_fname)
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
    zl_observation = lqg.zero_predictor_loss.environment.observation
    zl_controller = lqg.zero_predictor_loss.controller.input

    sys_idx = 2

    """
    # SECTION: Plot the training loss curve
    training_outputs = get_result_attr(result, "output")

    clip = 80
    observation_fig, observation_ax = plt.subplots()
    controller_fig, controller_ax = plt.subplots()
    observation_legends, controller_legends = {}, {}

    def plot_normalized_with_clip(out, color: np.ndarray, label: str, **kwargs):
        tl = out["training"]
        try:
            vl = out["validation_analytical"].squeeze(-1)
            vcl = out["validation_controller_analytical"].squeeze(-1)
        except KeyError:
            vl = out["validation"].squeeze(-1)
            vcl = out["validation_controller"].squeeze(-1)

        normalized_vl = ((vl - expand_as_right(il_observation, vl)) / expand_as_right(zl_observation - il_observation, vl)).median(dim=-2).values[sys_idx]
        normalized_vcl = ((vcl - expand_as_right(il_controller, vcl)) / expand_as_right(zl_controller - il_controller, vcl)).median(dim=-2).values[sys_idx]

        x = torch.linspace(0, 1, out.shape[-1] - clip)

        key = label[:label.find("_")]
        observation_legends.setdefault(key, []).append(
            (observation_ax.plot(x.cpu(), normalized_vl[clip:].detach().cpu(), color=color, **kwargs)[0], label)
        )
        controller_legends.setdefault(key, []).append(
            (controller_ax.plot(x.cpu(), normalized_vcl[clip:].detach().cpu(), color=color, **kwargs)[0], label)
        )

    for hp_index, (hp_value, color) in enumerate(zip(hp_values, COLOR_LIST)):
        rnn_out = training_outputs[0, hp_index]
        plot_normalized_with_clip(rnn_out, color=0.5 * color, linestyle="--", label=f"rnn_$\\alpha=${hp_value}")

        transformer_out = training_outputs[2, hp_index]
        plot_normalized_with_clip(transformer_out, color=color, linestyle="-", label=f"transformer_$\\alpha=${hp_value}")

    for ax, fig, legends, task_name, title_name in zip(
            (observation_ax, controller_ax),
            (observation_fig, controller_fig),
            (observation_legends, controller_legends),
            ("prediction", "imitation"),
            ("observation_prediction_error", "controller_imitation_error")
    ):
        ax.set_xscale("log")
        ax.set_xlabel("training_progression")
        ax.set_ylabel("validation_error: $\\tilde{L}_{" + task_name + "}$")
        ax.set_yscale("log")

        for v in legends.values():
            ax.add_artist(ax.legend(*zip(*v)))
        ax.set_title(title_name)

        fig.savefig(f"images/paper/training/{title_name}.png", bbox_inches="tight")

    plt.show()
    """


    # LQG system visualization
    zero_controller_group = NNControllerGroup(problem_shape, *utils.stack_module_arr(utils.array_of(ZeroController(Namespace(problem_shape=problem_shape)))))
    batch_size, horizon = 128, 200

    datasets_cache_fname = f"output/{output_dir}/{exp_name}/datasets_cache.pt"
    if os.path.exists(datasets_cache_fname):
        _datasets = utils.torch_load(datasets_cache_fname)
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
    S = S / ((batch_size * horizon) ** 0.5)
    s0, s1 = S[0, 0, :, :2].mT

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
        plt.plot(*compressed_zero_states[model_idx, sys_idx, trace_idx, 0], color="black", marker=".", markersize=24, label="start_of_trajectory")
        for hp_value, states, color in zip(hp_values, compressed_learned_states[model_idx], COLOR_LIST):
            trajectory = states[sys_idx, trace_idx, :small_horizon]
            plt.plot(*trajectory.mT, color=color, linewidth=0.5, marker=".", markersize=8, label=f"$\\alpha=${hp_value}_learned_trajectory")

        plt.plot(*compressed_zero_states[model_idx, sys_idx, trace_idx, :small_horizon].mT, color="gray", linewidth=0.5, linestyle="--", marker="o", markersize=4, label="zero_predictor_trajectory")
        plt.plot(*compressed_optimal_states[model_idx, sys_idx, trace_idx, :small_horizon].mT, color="black", linestyle="--", marker="*", markersize=8, label="optimal_trajectory")

        plt.xlabel("$\\sigma_0$")
        plt.ylabel("$\\sigma_1$")
        plt.title(f"{model_name}_trajectory")

        plt.legend(fontsize=6)
        plt.savefig(f"images/paper/{model_name}/trajectory.png", bbox_inches="tight")
        plt.show()
    """

    def plot_covariance(states: torch.Tensor, color: np.ndarray, label: str) -> None:
        plt.scatter(*states.mT, s=3, color=color, alpha=0.15)
        utils.confidence_ellipse(
            *states.mT, plt.gca(),
            n_std=2.0, linewidth=2, linestyle="--", edgecolor=0.7 * color, label=label, zorder=12
        )

    indices = torch.randint(0, batch_size * horizon, (2000,))
    sampled_compressed_optimal_states = compressed_optimal_states.flatten(-3, -2)[0, sys_idx, indices]
    sampled_compressed_zero_states = compressed_zero_states.flatten(-3, -2)[0, sys_idx, indices]

    """
    # SECTION: Plot covariances of exemplar states
    for hp_value, states, color in zip(hp_values, compressed_exemplar_states[0], COLOR_LIST):
        plot_covariance(states.flatten(-3, -2)[sys_idx, indices], color, f"$\\alpha=${hp_value}_states")

    plot_covariance(sampled_compressed_optimal_states, np.array(colors.to_rgb("black")), "optimal_states")
    plot_covariance(sampled_compressed_zero_states, np.array(colors.to_rgb("gray")), "zero_predictor_states")

    plt.xlabel("$\\sigma_0$")
    plt.xlim(left=-3.5 * s0[sys_idx], right=3.5 * s0[sys_idx])
    plt.ylabel("$\\sigma_1$")
    plt.ylim(bottom=-3.5 * s1[sys_idx], top=3.5 * s1[sys_idx])
    plt.title(f"exemplar_state_covariance")

    plt.legend(fontsize=6)
    plt.savefig("images/paper/exemplar/state_covariance.png", bbox_inches="tight")
    plt.show()
    """

    def plot_state_excitation(ax, states: torch.Tensor, color: np.ndarray, label: str) -> None:
        s = states.std()
        x = torch.linspace(-3 * s, 3 * s, 100)

        nbins = 20 * math.ceil(s * (batch_size * horizon) ** 0.5)
        ax.hist(states, density=True, color=color, bins=nbins, histtype="step", linewidth=0.3, label=label)
        ax.plot(x, torch.distributions.Normal(loc=0.0, scale=s).log_prob(x).exp(), color=0.5 * color, linewidth=1.0)

    """
    # SECTION: Plot excitations of exemplar states
    fig, axs = plt.subplots(nrows=S_D, ncols=1, sharex=True, sharey=True)
    for i in range(S_D):
        for hp_value, states, color in zip(hp_values, aligned_exemplar_states[0], COLOR_LIST):
            plot_state_excitation(axs[i], states.flatten(-3, -2)[sys_idx, :, i], color, f"$\\alpha=${hp_value}_dist")

        plot_state_excitation(axs[i], aligned_optimal_states.flatten(-3, -2)[0, sys_idx, :, i], np.array(colors.to_rgb("black")), f"optimal_distribution")
        # plot_state_excitation(axs[i], aligned_zero_states.flatten(-3, -2)[0, sys_idx, :, i], np.array(colors.to_rgb("gray")), f"zero_predictor_dist")

        axs[i].set_xlabel(f"$\\sigma_{i}$")
        axs[i].set_xlim(left=-3.5 * s0[sys_idx], right=3.5 * s0[sys_idx])
        axs[i].set_ylabel("distribution")

        if i == 0:
            axs[i].set_title("exemplar_state_excitation")
            axs[i].legend(fontsize=6)

    plt.savefig("images/paper/exemplar/state_excitation.png", bbox_inches="tight")
    plt.show()
    """


    """
    for model_idx, model_name in enumerate(configurations[0][1]["name"]):
        # SECTION: Plot covariances of learned states
        for hp_value, states, color in zip(hp_values, compressed_learned_states[model_idx], COLOR_LIST):
            plot_covariance(states.flatten(-3, -2)[sys_idx, indices], color, f"$\\alpha=${hp_value}_learned_states")

        plot_covariance(sampled_compressed_optimal_states, np.array(colors.to_rgb("black")), "optimal_states")
        plot_covariance(sampled_compressed_zero_states, np.array(colors.to_rgb("gray")), "zero_predictor_states")


        plt.xlabel("$\\sigma_0$")
        plt.xlim(left=-3.5 * s0[sys_idx], right=3.5 * s0[sys_idx])
        plt.ylabel("$\\sigma_1$")
        plt.ylim(bottom=-3.5 * s1[sys_idx], top=3.5 * s1[sys_idx])
        plt.title(f"{model_name}_learned_state_covariance")

        plt.legend(fontsize=6)
        plt.savefig(f"images/paper/{model_name}/state_covariance.png", bbox_inches="tight")
        plt.show()


        # SECTION: Plot excitations of learned states
        fig, axs = plt.subplots(nrows=S_D, ncols=1, sharex=True, sharey=True)
        for i in range(S_D):
            for hp_value, states, color in zip(hp_values, aligned_learned_states[model_idx], COLOR_LIST):
                plot_state_excitation(axs[i], states.flatten(-3, -2)[sys_idx, :, i], color, f"$\\alpha=${hp_value}_learned_dist")
    
            plot_state_excitation(axs[i], aligned_optimal_states.flatten(-3, -2)[0, sys_idx, :, i], np.array(colors.to_rgb("black")), f"optimal_distribution")
            # plot_state_excitation(axs[i], aligned_zero_states.flatten(-3, -2)[0, sys_idx, :, i], np.array(colors.to_rgb("gray")), f"zero_predictor_dist")
    
            axs[i].set_xlabel(f"$\\sigma_{i}$")
            axs[i].set_xlim(left=-3.5 * s0[sys_idx], right=3.5 * s0[sys_idx])
            axs[i].set_ylabel("distribution")
    
            if i == 0:
                axs[i].set_title(f"{model_name}_learned_state_excitation")
                axs[i].legend(fontsize=6)

        plt.savefig(f"images/paper/{model_name}/state_excitation.png", bbox_inches="tight")
        plt.show()
    """

    def loss(dataset: TensorDict[str, torch.Tensor]) -> torch.Tensor:
        state_loss = (
                dataset["environment", "state"][..., None, :] @
                sum(lqg.controller.Q.values())[sys_idx] @
                dataset["environment", "state"][..., :, None]
        )
        control_loss = sum(
            (dataset["controller", k][..., None, :] @ v[sys_idx] @ dataset["controller", k][..., :, None])
            for k, v in lqg.controller.R.items()
        )
        return (state_loss + control_loss)[..., 0, 0]

    optimal_loss = loss(optimal_trace[0, 0, sys_idx])
    def plot_cumulative_loss(dataset: TensorDict[str, torch.Tensor], color: np.ndarray, label: str, **kwargs) -> None:
        l = loss(dataset)
        x = torch.arange(horizon) + 1
        plt.plot(x, torch.cumsum(l, dim=-1).median(dim=-2).values.detach(), color=color, label=label, **kwargs)
        plt.fill_between(
            x, *torch.cumsum(l, dim=-1).quantile(torch.tensor([0.25, 0.75]), dim=-2).detach(),
            color=color, alpha=0.1
        )

    clip = 2
    def plot_cumulative_regret(axs, dataset: TensorDict[str, torch.Tensor], color: np.ndarray, label: str, **kwargs) -> None:
        l = loss(dataset)
        r = l - optimal_loss

        x = torch.arange(horizon) + 1
        axs[0].plot(x, torch.cumsum(l, dim=-1).median(dim=-2).values.detach(), color=color, label=label, **kwargs)
        axs[0].fill_between(
            x, *torch.cumsum(l, dim=-1).quantile(torch.tensor([0.25, 0.75]), dim=-2).detach(),
            color=color, alpha=0.1
        )

        x = torch.arange(clip, horizon) + 1
        axs[1].plot(x, torch.cumsum(r, dim=-1).median(dim=-2).values[clip:].detach(), color=color, label=label, **kwargs)

    """
    # SECTION: Plot exemplar cumulative loss over horizon
    for hp_value, dataset, color in zip(hp_values, exemplar_traces[0], COLOR_LIST):
        plot_cumulative_loss(dataset[sys_idx], color, linewidth=2.0, label=f"$\\alpha=${hp_value}_loss")

    plot_cumulative_loss(optimal_trace[0, 0, sys_idx], np.array(colors.to_rgb("black")), linewidth=3.0, linestyle="--", label="optimal_loss")
    plot_cumulative_loss(zero_trace[0, 0, sys_idx], np.array(colors.to_rgb("gray")), linewidth=3.0, linestyle="--", label="zero_predictor_loss")


    plt.xlabel("horizon")
    plt.ylabel("loss")
    plt.title("exemplar_cumulative_loss")

    plt.legend(fontsize=6)
    plt.savefig("images/paper/exemplar/cumulative_loss.png", bbox_inches="tight")
    plt.show()
    """


    # """
    plt.rcParams["figure.figsize"] = (6.4, 7.0)
    for model_idx, model_name in enumerate(configurations[0][1]["name"]):
        # SECTION: Plot learned cumulative loss over horizon
        _, axs = plt.subplots(nrows=2, ncols=1)
        for hp_value, dataset, color in zip(hp_values, learned_traces[model_idx], COLOR_LIST):
            plot_cumulative_regret(axs, dataset[sys_idx], color, linewidth=2.0, label=f"$\\alpha=${hp_value}_loss")

        plot_cumulative_regret(axs, optimal_trace[0, 0, sys_idx], np.array(colors.to_rgb("black")), linewidth=3.0, linestyle="--", label="optimal_loss")
        plot_cumulative_regret(axs, zero_trace[0, 0, sys_idx], np.array(colors.to_rgb("gray")), linewidth=3.0, linestyle="--", label="zero_predictor_loss")

        axs[0].set_ylabel("loss")
        axs[0].set_title(f"{model_name}_cumulative_loss")
        axs[0].legend(fontsize=6)

        axs[1].set_xlabel("horizon")
        axs[1].set_xscale("log")
        axs[1].set_ylabel("regret")
        axs[1].set_yscale("log")
        axs[1].set_title(f"{model_name}_cumulative_regret")

        plt.savefig(f"images/paper/{model_name}/cumulative_regret.png", bbox_inches="tight")
        plt.show()
    plt.rcdefaults()
    # """


    """
    # SECTION: Statistical significance
    flattened_states = _datasets["environment", "state"].flatten(-3, -2)
    cov = (flattened_states.mT @ flattened_states) / (batch_size * horizon)

    zero_cov = cov[:, 0]
    optimal_cov = cov[:, 1]
    exemplar_covs = cov[:, 1:len(hp_values) + 1]
    learned_covs = cov[:, len(hp_values) + 1:]

    log_probs, kl_divs = [], []
    for hp_idx, hp_value,  in enumerate(hp_values):
        # [n_hypotheses x model x n_systems x S_D x S_D]
        hypothesis_covs = torch.stack([
            zero_cov, exemplar_covs[:, hp_idx], optimal_cov
        ], dim=0)
        dists = torch.distributions.MultivariateNormal(loc=torch.zeros(S_D), covariance_matrix=hypothesis_covs)

        # [model x hp_value x n_systems x batch_size x horizon x S_D]
        # -> [model x n_systems x batch_size x horizon x S_D]
        # -> [model x n_systems x BL x S_D]
        # -> [BL x model x n_systems x S_D]
        # -> [BL x 1 x model x n_systems x S_D]
        states = learned_traces["environment", "state"][:, hp_idx].flatten(-3, -2).permute(2, 0, 1, 3)[:, None]

        # [n_hypotheses x model x n_systems]
        log_probs.append(dists.log_prob(states).mean(dim=0))

        # [model x n_systems x S_D x S_D]
        learned_cov = learned_covs[:, hp_idx]
        # [n_hypotheses x model x n_systems]
        kl_divs.append(utils.kl_div(hypothesis_covs, learned_cov))

    # [n_hypotheses x model x hp_value x n_systems]
    # -> [model x n_hypotheses x hp_value x n_systems]
    log_probs = torch.stack(log_probs, dim=-2).transpose(dim0=0, dim1=1)
    log_probs[:, -1, 0, :] = -torch.finfo(torch.get_default_dtype()).max

    posterior = torch.softmax(log_probs, dim=1)
    log_posterior = torch.log_softmax(log_probs, dim=1)

    # [n_hypotheses x model x hp_value x n_systems]
    # -> [model x n_hypotheses x hp_value x n_systems]
    kl_divs = torch.stack(kl_divs, dim=-2).transpose(dim0=0, dim1=1)

    for model_idx, model_name in enumerate(configurations[0][1]["name"]):
        print(model_name + "\n" + "=" * 120)
        print(f"\tlog_probs: {log_probs[model_idx, ..., sys_idx]}")
        print(f"\tposterior: {posterior[model_idx, ..., sys_idx]}")
        print(f"\tlog_posterior: {log_posterior[model_idx, ..., sys_idx]}")
        print(f"\tkl_divs: {kl_divs[model_idx, ..., sys_idx]}")
        print()
    """


    """
    # SECTION: Efficacy of transformer bias
    x = torch.arange(horizon) + 1
    def plot_cumulative_regret_comparison(dataset1: TensorDict[str, torch.Tensor], dataset2: TensorDict[str, torch.Tensor], color: np.ndarray, label: str, **kwargs) -> None:
        l = loss(dataset1) - loss(dataset2)
        plt.plot(x, torch.cumsum(l, dim=-1).median(dim=-2).values.detach(), color=color, label=label, **kwargs)
        plt.fill_between(
            x, *torch.cumsum(l, dim=-1).quantile(torch.tensor([0.25, 0.75]), dim=-2).detach(),
            color=color, alpha=0.1
        )

    for hp_idx, (hp_value, color) in enumerate(zip(hp_values, COLOR_LIST)):
        plot_cumulative_regret_comparison(learned_traces[1, hp_idx, sys_idx], learned_traces[2, hp_idx, sys_idx], color, linewidth=2.0, label=f"$\\alpha=${hp_value}_loss_difference")

    plt.plot(x, torch.zeros_like(x), color="black", linewidth=3.0, linestyle="--")

    # plot_cumulative_regret_comparison(optimal_trace[0, 0, sys_idx], optimal_trace[0, 0, sys_idx], np.array(colors.to_rgb("black")), linewidth=3.0, linestyle="--", label="optimal_loss_difference")
    # plot_cumulative_regret_comparison(zero_trace[0, 0, sys_idx], zero_trace[0, 0, sys_idx], np.array(colors.to_rgb("gray")), linewidth=3.0, linestyle="--", label="zero_predictor_loss_difference")

    plt.xlabel("horizon")
    plt.ylabel("loss_difference")
    plt.title(f"transformer_cumulative_loss_difference")
    plt.legend(fontsize=6)

    plt.savefig("images/paper/transformer/cumulative_loss_difference.png", bbox_inches="tight")
    plt.show()
    """


    """
    # SECTION: Visualize encodings of observations and controls
    plt.rcParams["figure.figsize"] = (16.0, 4.8)

    learned_kfs_arr = get_result_attr(result, "learned_kfs")
    transformer_bias = learned_kfs_arr[2, 0][1].squeeze(-1)
    transformer_no_bias = learned_kfs_arr[1, 0][1].squeeze(-1)

    test_trace = optimal_trace[0, 0].view(args.experiment.n_experiments, -1)
    test_observations = test_trace["environment", "observation"]
    test_controls = test_trace["controller", "input"]

    transformer_bias_observation_embds = test_observations @ transformer_bias["observation_in"].mT + transformer_bias["observation_bias"][:, None]
    transformer_bias_control_embds = test_controls @ transformer_bias["input_in", "input"].mT + transformer_bias["input_bias"][:, None]

    transformer_no_bias_observation_embds = test_observations @ transformer_no_bias["observation_in"].mT
    transformer_no_bias_control_embds = test_controls @ transformer_no_bias["input_in", "input"].mT

    pca_fig, pca_axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    tsne_fig, tsne_axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    for ax_idx, (model_name, observation_embds, control_embds) in enumerate(zip(
        ["transformer", "transformer_no_bias"],
        [transformer_bias_observation_embds, transformer_no_bias_observation_embds],
        [transformer_bias_control_embds, transformer_no_bias_control_embds]
    )):
        # SECTION: Use PCA compression
        embds = torch.cat([observation_embds, control_embds], dim=-2)
        embds_mean = embds.mean(dim=-2, keepdims=True)

        embds_compression = torch.stack([
            torch.linalg.svd(observation_embds, full_matrices=False)[2].mH[..., 0],
            torch.linalg.svd(control_embds, full_matrices=False)[2].mH[..., 0],
        ], dim=-1)
        # U, S, Vh = torch.linalg.svd(embds - embds_mean, full_matrices=False)
        # embds_compression = Vh.mH[..., :2]

        pca_axs[ax_idx].scatter(*((observation_embds - embds_mean) @ embds_compression)[sys_idx].mT.detach(), color=COLOR_LIST[0], s=0.1, label="observation_embeddings")
        pca_axs[ax_idx].scatter(*((control_embds - embds_mean) @ embds_compression)[sys_idx].mT.detach(), color=COLOR_LIST[1], s=0.1, label="control_embeddings")

        pca_axs[ax_idx].set_xlabel("$\\sigma_{0, observation}$")
        pca_axs[ax_idx].set_ylabel("$\\sigma_{0, control}$")
        pca_axs[ax_idx].set_title(f"{model_name}_embeddings")
        pca_axs[ax_idx].legend()

        # SECTION: Use TSNE compression
        import seaborn
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)
        sample_size = 10000
        indices = torch.randint(0, batch_size * horizon, (sample_size,))
        X_tsne = tsne.fit_transform(torch.cat([
            observation_embds[:, indices],
            control_embds[:, indices]
        ], dim=-2)[sys_idx].detach())

        colors = seaborn.color_palette("hls", 10, as_cmap=True)(np.linspace(0, 1, 11))[:-1]
        tsne_axs[ax_idx].scatter(
            X_tsne[:sample_size, 0], X_tsne[:sample_size, 1],
            color=colors[0], label="observation_embeddings", s=0.1
        )
        tsne_axs[ax_idx].scatter(
            X_tsne[sample_size:, 0], X_tsne[sample_size:, 1],
            color=colors[1], label="control_embeddings", s=0.1
        )

        tsne_axs[ax_idx].set_title(f"{model_name}_embeddings")
        tsne_axs[ax_idx].legend()

    pca_fig.savefig("images/paper/transformer/pca_embedding_comparison.png", bbox_inches="tight")
    tsne_fig.savefig("images/paper/transformer/tsne_embedding_comparison.png", bbox_inches="tight")

    plt.show()
    plt.rcdefaults()
    """






