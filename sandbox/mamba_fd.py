#%%
import os
import sys
from argparse import Namespace
from typing import *

import einops
import numpy as np
import tensordict.utils
import torch
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tensordict import TensorDict
from transformers import (
    GPT2Config,
    TransfoXLConfig,
    Dinov2Config,
    MambaConfig,
    Mamba2Config,
)

# This line needs to be added since some terminals will not recognize the current directory
os.chdir("/home/wentinn/Desktop/KF_RNN/")
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader
from infrastructure import utils
from infrastructure.experiment import *
from infrastructure.settings import DEVICE
from infrastructure.utils import PTR
from model.convolutional import CnnLeastSquaresPredictor
from model.sequential import RnnAnalyticalPretrainPredictor, RnnComplexDiagonalPredictor
from model.transformer import (
    GPT2InContextPredictor,
    MambaInContextPredictor,
    Mamba2InContextPredictor,

    ObservableMambaConfig,
    ObservableMamba2InContextPredictor,
)
from model.zero_predictor import ZeroPredictor
from system.linear_time_invariant import (
    MOPDistribution,
    OrthonormalDistribution,
    ContinuousDistribution,
    ContinuousNoiselessDistribution,
)


if __name__ == "__main__":
    output_dir = "mamba_fd_model_comparison_test"
    output_fname = "result"
    # output_fname = "result_batch_sweep_exponential_lr"

    # dist = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
    # dist = ContinuousDistribution("gaussian", "gaussian", eps=0.1, W_std=0.1, V_std=0.1)
    dist = ContinuousNoiselessDistribution("gaussian", "gaussian", 0.001)
    # dist = OrthonormalDistribution()

    # S_D, O_D = 10, 5,
    S_D = O_D = 5
    SHP = Namespace(
        distribution=dist, S_D=S_D,
        problem_shape=Namespace(
            environment=Namespace(observation=O_D,),
            controller=Namespace(),
        ), auxiliary=Namespace(),
    )
    
    context_length = 1000 # 250
    n_train_systems = 10000 # 40000
    n_test_systems = 100
    n_valid_traces = 64
    n_test_traces = 64
    
    max_num_hidden_layers = 12
    n_layers = [*range(max_num_hidden_layers, 0, -2)]

    n_firs = 5
    rnn_increment = 1
    

    """ Transformer experiment """
    exp_name_transformer = "transformer"

    ARGS_TRANSFORMER = loader.generate_args(SHP)

    # SECTION: Transformer architecture hyperparameters
    hidden_size, num_heads, expand = 8, 8, 1, # 256, 8, 2,
    state_size = 256 // hidden_size
    ARGS_TRANSFORMER.model.bias = False
    ARGS_TRANSFORMER.model.adapter = True

    # ARGS_TRANSFORMER.model.gpt2 = GPT2Config(
    #     n_positions=context_length,
    #     n_embd=128,
    #     n_layer=max_num_hidden_layers,
    #     n_head=8,
    # )
    # ARGS_TRANSFORMER.model.model = MambaInContextPredictor
    # ARGS_TRANSFORMER.model.multi_mamba = MambaConfig(
    #     state_size=256,
    #     hidden_size=hidden_size,
    #     num_hidden_layers=num_hidden_layers,
    #     # expand=expand,
    # )
    ARGS_TRANSFORMER.model.model = Mamba2InContextPredictor
    ARGS_TRANSFORMER.model.mamba2 = Mamba2Config(
        state_size=state_size,
        hidden_size=hidden_size,
        num_hidden_layers=max_num_hidden_layers,
        num_heads=num_heads,
        head_dim=int(expand * hidden_size / num_heads),
        expand=expand,
    )
    ARGS_TRANSFORMER.model.model = ObservableMamba2InContextPredictor
    ARGS_TRANSFORMER.model.mamba = ObservableMambaConfig(
        state_size=state_size,
        hidden_size=hidden_size,
        num_hidden_layers=max_num_hidden_layers,
        num_heads=num_heads,
        head_dim=int(expand * hidden_size / num_heads),
        expand=expand,
        use_fast_conv_scan=True,
        chunk_size=256,
    )

    # SECTION: Dataset hyperparameters
    ARGS_TRANSFORMER.system.distribution.update(train=dist, valid=dist, test=dist)
    ARGS_TRANSFORMER.system.settings = Namespace(
        include_analytical=False,
    )
    
    ARGS_TRANSFORMER.dataset.n_systems.update(train=n_train_systems, valid=n_test_systems, test=n_test_systems)
    ARGS_TRANSFORMER.dataset.n_traces.update(train=1, valid=n_valid_traces, test=n_test_traces)
    ARGS_TRANSFORMER.dataset.total_sequence_length.update(train=context_length, valid=n_valid_traces * context_length, test=n_test_traces * context_length)

    # SECTION: Training hyperparameters
    ARGS_TRANSFORMER.training.loss = "fd_mse"
    ARGS_TRANSFORMER.training.ignore_initial = True
    ARGS_TRANSFORMER.training.sampling = Namespace(
        method=None, # "subsequence_padded",
        subsequence_length=None, # context_length,
        batch_size=256,
    )
    ARGS_TRANSFORMER.training.optimizer = Namespace(
        type="AdamW",
        max_lr=1e-2, min_lr=1e-6,
        weight_decay=1e-2, momentum=0.9,
    )
    ARGS_TRANSFORMER.training.scheduler = Namespace(
        type="reduce_on_plateau", factor=0.8, patience=3, warmup_duration=0,
        # type="exponential", lr_decay=0.982, warmup_duration=0,
        epochs=1, # 200,
    )

    ARGS_TRANSFORMER.experiment.n_experiments = 1
    ARGS_TRANSFORMER.experiment.ensemble_size = 1
    ARGS_TRANSFORMER.experiment.exp_name = exp_name_transformer
    ARGS_TRANSFORMER.experiment.metrics = Namespace(training={
        # "overfit",
        # "validation",
        "fd_noiseless_overfit",
        "fd_noiseless_validation",
    }, testing={
        # "nl", "al", "il", "neil",
        "fd_nl", # "fd_al", "fd_il", "fd_neil",
    })
    ARGS_TRANSFORMER.experiment.checkpoint_frequency = 5
    ARGS_TRANSFORMER.experiment.print_frequency = 1

    # configurations_transformer = []
    configurations_transformer = [
        # ("ignore_initial", {
        #     "training.ignore_initial": [False, True],
        # }),
        # ("batch_size", {
        #     # "training.sampling.batch_size": [16, 32, 64, 128,],
        #     # "training.sampling.batch_size": [64, 128,],
        #     "training.sampling.batch_size": [64, 64, 64,],
        # })
        # ("dataset_shape", {
        #     "dataset.n_systems.train": [80000, 40000, 20000, 10000,],
        #     "dataset.total_sequence_length.train": [125, 250, 500, 1000,],
        #     "training.sampling.batch_size": [512, 256, 128, 64,],
        # })
        # ("num_hidden_layers", {
        #     "model.mamba2.num_hidden_layers": n_layers,
        # }),
        ("model", {
            "model.model": [Mamba2InContextPredictor, ObservableMamba2InContextPredictor,],
        })
    ]
    result_transformer, info_dict = run_experiments(
        ARGS_TRANSFORMER, configurations_transformer, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True,
    )
    test_systems = info_dict["test"]["systems"].values[()]



    # """ Baseline experiment setup """
    # exp_name_cnn = "cnn"
    # exp_name_rnn = "rnn"

    # for _exp_name_baseline in (exp_name_cnn, exp_name_rnn):
    #     os.makedirs(f"output/{output_dir}/{_exp_name_baseline}/training", exist_ok=True)
    #     os.makedirs(f"output/{output_dir}/{_exp_name_baseline}/testing", exist_ok=True)

    #     if not all(map(os.path.exists, (
    #         f"output/{output_dir}/{_exp_name_baseline}/training/systems.pt",
    #         f"output/{output_dir}/{_exp_name_baseline}/testing/systems.pt",
    #     ))):
    #         baseline_systems = utils.multi_map(
    #             lambda lsg: LTISystem(lsg.hyperparameters, lsg.td().permute(1, 0)),
    #             systems, dtype=LTISystem,
    #         )
    #         torch.save({
    #             "train": baseline_systems,
    #         }, f"output/{output_dir}/{_exp_name_baseline}/training/systems.pt")
    #         torch.save({
    #             "test": baseline_systems,
    #         }, f"output/{output_dir}/{_exp_name_baseline}/testing/systems.pt")

    #     if not all(map(os.path.exists, (
    #         f"output/{output_dir}/{_exp_name_baseline}/training/dataset.pt",
    #         f"output/{output_dir}/{_exp_name_baseline}/testing/dataset.pt",
    #     ))):
    #         baseline_dataset = utils.multi_map(lambda dataset_: PTR(dataset_.obj.permute(2, 3, 0, 1, 4)), dataset, dtype=PTR)
    #         torch.save({
    #             "train": baseline_dataset,
    #             "valid": baseline_dataset,
    #         }, f"output/{output_dir}/{_exp_name_baseline}/training/dataset.pt")
    #         torch.save({
    #             "test": baseline_dataset,
    #         }, f"output/{output_dir}/{_exp_name_baseline}/testing/dataset.pt")



    # """ CNN Experiment """
    # ARGS_BASELINE_CNN = loader.generate_args(SHP)

    # ARGS_BASELINE_CNN.model.model = CnnLeastSquaresPredictor

    # ARGS_BASELINE_CNN.dataset.n_systems.reset(train=1)
    # ARGS_BASELINE_CNN.dataset.n_traces.reset(train=1)
    # ARGS_BASELINE_CNN.dataset.total_sequence_length.reset(train=context_length)

    # ARGS_BASELINE_CNN.training.sampling.batch_size = 1

    # ARGS_BASELINE_CNN.experiment.n_experiments = n_test_systems
    # ARGS_BASELINE_CNN.experiment.ensemble_size = n_test_traces
    # ARGS_BASELINE_CNN.experiment.metrics = Namespace(training={"validation_analytical"})
    # ARGS_BASELINE_CNN.experiment.backup_frequency = 250
    # ARGS_BASELINE_CNN.experiment.checkpoint_frequency = 10000
    # ARGS_BASELINE_CNN.experiment.print_frequency = 1

    # # SECTION: Make a copy for RNN args after setting shared parameters
    # ARGS_BASELINE_RNN = utils.deepcopy_namespace(ARGS_BASELINE_CNN)

    # # SECTION: Set CNN exclusive hyperparameters
    # ARGS_BASELINE_CNN.model.ridge = 1.0
    # ARGS_BASELINE_CNN.experiment.exp_name = exp_name_cnn

    # configurations_cnn = [
    #     ("model", {
    #         "model.ir_length": [*range(1, n_firs + 1)],
    #     }),
    # ]

    # result_cnn, _, _ = run_experiments(
    #     ARGS_BASELINE_CNN, configurations_cnn, {
    #         "dir": output_dir,
    #         "fname": output_fname,
    #     }, save_experiment=True,
    # )



    # """ RNN Experiment """
    # # SECTION: Set RNN exclusive hyperparameters
    # ARGS_BASELINE_RNN.model.S_D = SHP.S_D
    # ARGS_BASELINE_RNN.model.initial_state_scale = 0.00001

    # ARGS_BASELINE_RNN.training.sampling = Namespace(
    #     method=None, # "subsequence_padded",
    #     subsequence_length=None, # context_length,
    #     batch_size=None, # 64,
    # )
    # ARGS_BASELINE_RNN.training.optimizer = Namespace(
    #     type="SGD",
    #     max_lr=1e-3, min_lr=1e-7,
    #     weight_decay=0.0, momentum=0.9,
    # )
    # ARGS_BASELINE_RNN.training.scheduler = Namespace(
    #     type="reduce_on_plateau",
    #     factor=0.7, patience=10, warmup_duration=0,
    #     # type="exponential",
    #     # lr_decay=0.998, warmup_duration=100,
    #     epochs=2000, gradient_cutoff=1e-6,
    # )

    # ARGS_BASELINE_RNN.experiment.exp_name = exp_name_rnn
    # ARGS_BASELINE_RNN.experiment.checkpoint_frequency = 200
    # ARGS_BASELINE_RNN.experiment.print_frequency = 1

    # configurations_rnn = [
    #     ("total_trace_length", {
    #         "model.model": [ZeroPredictor] + [RnnComplexDiagonalPredictor] * (utils.ceildiv(context_length, rnn_increment) - 1),
    #         # "model.model": [ZeroPredictor] + [RnnAnalyticalPretrainPredictor] * (utils.ceildiv(context_length, rnn_increment) - 1),
    #         "dataset.total_sequence_length.train": [*range(0, context_length, rnn_increment),]
    #     })
    # ]

    # result_rnn, _, _ = run_experiments(
    #     ARGS_BASELINE_RNN, configurations_rnn, {
    #         "dir": output_dir,
    #         "fname": output_fname
    #     }, save_experiment=True,
    # )
    torch.cuda.empty_cache()



    M_transformer = get_metric_namespace_from_result(result_transformer)
    # M_cnn = get_metric_namespace_from_result(result_cnn)
    # M_rnn = get_metric_namespace_from_result(result_rnn)
    



    # print(M_transformer.nl.shape)
    # print(M_transformer.output.environment.observation.shape)
    training_log = utils.stack_tensor_arr(utils.multi_map(
        lambda p: p.obj, get_result_attr(result_transformer, "output"),
        dtype=TensorDict,
    ))[:, 0, 0]
    # training_log: TensorDict = result_transformer.values[()].output.obj[0, 0]

    # lr = torch.stack([td["learning_rate"] for td in training_log[:-1]], dim=0)
    # overfit_l = torch.mean(einops.rearrange(training_log["overfit"], "1 1 t b -> t b"), dim=-1)
    # validation_l = torch.mean(einops.rearrange(training_log["validation"], "1 1 t b -> t b"), dim=-1)


    hparam_name = configurations_transformer[0][0]
    hparam_values = [*configurations_transformer[0][1].values()][0]

    clist = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    nrows = len(hparam_values)
    plt.rcParams["figure.figsize"] = (10.0, 5.0 * nrows,)
    fig, axs = plt.subplots(nrows=len(hparam_values),)

    for i, (hparam_value, log_td) in enumerate(zip(hparam_values, training_log)):
        ax: Axes = axs[i]
        ax_lr = ax.twinx()

        for j, (k, v) in enumerate([
            ("fd_noiseless_overfit", "train",),
            ("fd_noiseless_validation", "valid",)
        ]):
            il = 0.0 # torch.mean(einops.rearrange(info_dict[v]["systems"].values[()].irreducible_loss.environment.observation, "1 b -> b",), dim=-1)
            l = torch.mean(log_td[k], dim=-1)
            ax.plot((l - il).numpy(force=True), label=f"{hparam_name}:{hparam_value}-{k}",)
            print((l - il).min())
        # ax_lr.plot(lr.numpy(force=True), color="green", linestyle="--", label="lr",)

        ax.set_xlabel("epochs", fontsize=10)
        ax.set_ylabel("loss", fontsize=10)
        ax.set_yscale("log")
        ax.legend(fontsize=10)

        ax_lr.set_ylabel("learning_rate")
        ax_lr.set_yscale("log")
        ax_lr.legend(fontsize=10)

    plt.show()
    plt.rcdefaults()
    
    
    # """ Result processing """
    # print("Result processing" + "\n" + "-" * 120)
    # lsg = systems.values[()]
    # systems = LTISystem(lsg.hyperparameters, lsg.td().squeeze(0))
    # dataset = dataset.values[()].obj.squeeze(1).squeeze(0)
    
    # def loss(observation_estimation: torch.Tensor) -> torch.Tensor:
    #     env = systems.environment
    #     reducible_error = (dataset["environment", "noiseless_observation"] - observation_estimation).norm(dim=-1) ** 2
    #     irreducible_error = utils.batch_trace(env.H @ env.S_W @ env.H.mT + env.S_V)[:, None, None]
    #     return reducible_error + irreducible_error
    
    # with torch.set_grad_enabled(False):
    #     zero_predictor_al = systems.zero_predictor_loss.environment.observation
    #     zero_predictor_l = loss(torch.zeros_like(dataset["environment", "observation"]))
    #     il = systems.irreducible_loss.environment.observation
    #     eil = loss(dataset["environment", "target_observation_estimation"])


    #     # [n_experiments x ensemble_size x n_test_systems x n_test_traces x context_length x O_D]
    #     # -> [n_test_systems x n_test_traces x context_length x O_D]
    #     mamba_output = einops.rearrange(M_transformer.output.environment.observation, "... n 1 1 b l d -> ... n b l d",)
    #     # -> [n_test_systems x n_test_traces x context_length]
    #     mamba_l = loss(mamba_output)


    #     # [n_firs x train.sequence_length x n_test_systems x n_test_traces x n_experiments x ensemble_size x context_length x O_D]
    #     # -> [n_firs x train.sequence_length x n_test_systems x n_test_traces x context_length x O_D]
    #     # -> [n_firs x n_test_systems x n_test_traces x O_D x context_length]
    #     # -> [n_firs x n_test_systems x n_test_traces x context_length x O_D]
    #     cnn_output = torch.diagonal(M_cnn.output.environment.observation.squeeze(5).squeeze(4), dim1=1, dim2=4).transpose(3, 4)
    #     # -> [n_firs x n_test_systems x n_test_traces x context_length]
    #     cnn_l = loss(cnn_output)
    #     # [n_firs x context_length x n_test_systems x n_test_traces x n_experiments x ensemble_size]
    #     # -> [n_firs x n_test_systems x n_test_traces x context_length]
    #     cnn_al = einops.rearrange(M_cnn.al, "r l s b 1 1 -> r s b l")


    #     # # [train.sequence_length x n_test_systems x n_test_traces x n_experiments x ensemble_size x context_length x O_D]
    #     # # -> [train.sequence_length x n_test_systems x n_test_traces x context_length x O_D]
    #     # # -> [train.sequence_length x n_test_systems x n_test_traces x O_D]
    #     # # -> [n_test_systems x n_test_traces x train.sequence_length x O_D]
    #     # rnn_sequence_lengths = [*range(0, context_length, rnn_increment),]
    #     # rnn_output = M_rnn.output.environment.observation.squeeze(4).squeeze(3)[torch.arange(len(rnn_sequence_lengths)), :, :, torch.tensor(rnn_sequence_lengths)].permute(1, 2, 0, 3)
    #     # # [train.sequence_length x n_test_systems x n_test_traces x n_experiments x ensemble_size]
    #     # # -> [n_test_systems x n_test_traces x train.sequence_length]
    #     # rnn_al = einops.rearrange(M_rnn.al, "l s b 1 1 -> s b l")
    
    
    #     # rnn_indices = torch.tensor(rnn_sequence_lengths)
    #     # padded_rnn_output = torch.zeros((n_test_systems, n_test_traces, context_length, SHP.problem_shape.environment.observation))
    #     # padded_rnn_output[:, :, rnn_indices] = rnn_output
    #     # # -> [n_test_systems x n_test_traces x context_length]
    #     # rnn_l = loss(padded_rnn_output)[:, :, rnn_indices]
    
    
    
    # # SECTION: Transformer impulse response    
    # def to_rgb(c: Any) -> np.ndarray:
    #     return np.array(colors.to_rgb(c))
    
    
    # # (gpt2_rm, gpt2_td), (transfoxl_rm, transfoxl_td) = get_result_attr(result_transformer, "learned_kfs")
    # # gpt2_td = gpt2_td.squeeze(1).squeeze(0)
    # # transfoxl_td = transfoxl_td.squeeze(1).squeeze(0)
    # #
    # # dataset_parameter = TensorDict.from_dict(dataset, batch_size=dataset.shape)
    # # dataset_parameter["environment"]["observation"] = nn.Parameter(dataset["environment"]["observation"])
    # #
    # # with torch.set_grad_enabled(True):
    # #     gpt2_response = Predictor.gradient(gpt2_rm, gpt2_td, dataset_parameter, split_size=1 << 17)
    # #     transfoxl_response = Predictor.gradient(transfoxl_rm, transfoxl_td, dataset_parameter, split_size=1 << 17)
    # #
    # # gpt2_gn = (gpt2_response["environment"]["observation"].norm(dim=-1) ** 2).mean(dim=1)
    # # for sys_idx in range(n_test_systems):
    # #     plt.plot(
    # #         cd(torch.arange(1, context_length + 1)),
    # #         cd(torch.flip(gpt2_gn[sys_idx].clamp_min(1e-4), dims=(0,))),
    # #         marker=".", label=f"gpt2_system{sys_idx}"
    # #     )
    # #
    # # plt.xscale("log")
    # # plt.xlabel("recency")
    # # plt.yscale("log")
    # # plt.ylabel("gradient_norm")
    # #
    # # plt.legend()
    # # plt.show()
    
    
    
    # # SECTION: Plotting code
    # def plot(system_idx: int, normalized: bool) -> None:
    #     x = torch.arange(1, context_length + 1)
    
    #     def plot_analytical(
    #             l: torch.Tensor, color,
    #             indices: torch.Tensor = torch.arange(context_length, dtype=torch.int),
    #             error_bars: bool = False, **kwargs
    #     ) -> None:
    #         assert l.ndim in (2, 3), f"Can only plot analytical loss of ndim 2 or 3 but got {l.ndim}."
    #         assert l.shape[0] == n_test_systems, f"First dimension of loss must match number of test systems."
    #         plt_kwargs = {
    #             "color": 0.6 * to_rgb(color),
    #             "linewidth": 3,
    #             "linestyle": "-."
    #         }
    #         plt_kwargs.update(kwargs)
    #         x_ = x[indices]

    #         if normalized:
    #             l = (l - tensordict.utils.expand_as_right(il, l))[system_idx]
    #         else:
    #             l = l[system_idx]

    #         if l.ndim == 1:
    #             plt.plot(x_.numpy(force=True), l.numpy(force=True), **plt_kwargs, zorder=12)
    #         else:
    #             # plt.plot(cd(x_), cd(l.mean(dim=0)), zorder=12, **plt_kwargs)
    #             plt.plot(x_.numpy(force=True), l.median(dim=0).values.numpy(force=True), zorder=12, **plt_kwargs)
    #             if error_bars:
    #                 plt.fill_between(s
    #                     x_.numpy(force=True), *torch.quantile(l, torch.tensor([0.25, 0.75]), dim=0).numpy(force=True),
    #                     alpha=0.1, **plt_kwargs
    #                 )
    
    #     def plot_empirical(
    #             l: torch.Tensor, name: str, color,
    #             indices: torch.Tensor = torch.arange(context_length, dtype=torch.int),
    #             error_bars: bool = False, **kwargs
    #     ) -> None:
    #         assert l.ndim == 3, f"Empirical loss must be ndim 3 but got {l.ndim}."
    #         assert l.shape[0] == n_test_systems, f"First dimension of loss must match number of test systems."
    #         plt_kwargs = {
    #             "color": to_rgb(color),
    #             "linewidth": 1
    #         }
    #         plt_kwargs.update(kwargs)
    #         x_ = x[indices]

    #         if normalized:
    #             l = (l - eil[:, :, indices])[system_idx]
    #         else:
    #             l = l[system_idx]

    #         plt.plot(x_.numpy(force=True), l.mean(dim=0).numpy(force=True), label=name, **plt_kwargs)
    #         if error_bars:
    #             plt.fill_between(
    #                 x_.numpy(force=True), *torch.quantile(l, torch.tensor([0.25, 0.75]), dim=0).numpy(force=True),
    #                 alpha=0.1, **plt_kwargs
    #             )
    
    #     # SECTION: Plot zero predictor
    #     plot_empirical(zero_predictor_l, "zero_predictor", "black")
    #     plot_analytical(zero_predictor_al[:, None].expand(n_test_systems, context_length), "black")

    #     # SECTION: Plot Kalman filter
    #     plot_empirical(eil, "kalman_filter", "black")
    #     plot_analytical(il[:, None].expand(n_test_systems, context_length), "black")
    
    #     # SECTION: Plot CNN baseline
    #     c_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    #     c_list[2] = "aquamarine"
    #     for fir_length in range(3):
    #         plot_empirical(cnn_l[fir_length], f"fir_length{fir_length + 1}", c_list[fir_length])
    #         plot_analytical(cnn_al[fir_length], c_list[fir_length])
    
    #     # # SECTION: Plot RNN baseline
    #     # plot_empirical(rnn_l, "iir", "crimson", indices=rnn_indices)
    #     # plot_analytical(rnn_al, "crimson", indices=rnn_indices)
    
    #     # SECTION: Plot transformers
    #     plot_empirical(mamba_l, "multi_mamba", "gold", error_bars=False, zorder=2)
    
    #     plt.title(f"InContext-GaussianA0.95/GaussianC: System {system_idx}")
    #     plt.xscale("log")
    #     plt.xlabel("context_length")
    #     plt.yscale("log")
    #     # plt.ylim(bottom=1e-3, top=2e0)
    #     plt.ylabel(r"normalized_loss: $|| F_\theta(\tau_t) - \tau_t ||^2 - || KF(\tau_t) - \tau_t ||^2$")
    
    #     plt.legend(framealpha=1.0)
    #     plt.show()
    
    # for sys_idx in range(n_test_systems):
    #     plot(sys_idx, False)









# %%
