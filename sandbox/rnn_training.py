import copy
import itertools
import os
import shutil
import sys
from argparse import Namespace

import tensordict.utils
import torch
from dimarray import DimArray
from matplotlib import pyplot as plt

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader, utils
from infrastructure.settings import DEVICE
from infrastructure.static import *
from infrastructure.utils import PTR
from infrastructure.experiment import *
from infrastructure.experiment.plotting import COLOR_LIST
from system.linear_time_invariant import LTISystem, MOPDistribution
from model.zero_predictor import ZeroPredictor
from model.sequential import RnnPredictor, RnnPredictorPretrainAnalytical


if __name__ == "__main__":
    output_dir = "in_context"
    output_fname = "result"

    context_length = 250
    n_test_systems = 3
    test_dataset_size = 256

    rnn_increment = 1

    S_D, O_D = 10, 5
    SHP = Namespace(S_D=S_D, problem_shape=Namespace(
        environment=Namespace(observation=O_D),
        controller=Namespace()
    ))
    exp_name_exemplar = "CDCReconstruction_rnn"
    result_exemplar, systems, _ = torch.load(f"output/{output_dir}/{exp_name_exemplar}/testing/result.pt", map_location=DEVICE)


    """ Chaining experiment setup """
    exp_name_chain_initialization = "Chaining_rnn"
    for dir_type in ("training", "testing"):
        os.makedirs(f"output/{output_dir}/{exp_name_chain_initialization}/{dir_type}", exist_ok=True)
        for info_type in INFO_DTYPE.names:
            shutil.copy(
                f"output/{output_dir}/{exp_name_exemplar}/{dir_type}/{info_type}.pt",
                f"output/{output_dir}/{exp_name_chain_initialization}/{dir_type}/{info_type}.pt"
            )

    # SECTION: Experiment hyperparameters
    ARGS_CHAIN_INITIALIZATION = loader.generate_args(SHP)
    ARGS_CHAIN_INITIALIZATION.model.model = RnnPredictor
    ARGS_CHAIN_INITIALIZATION.model.S_D = S_D

    ARGS_CHAIN_INITIALIZATION.dataset.n_systems.reset(train=1)
    ARGS_CHAIN_INITIALIZATION.dataset.dataset_size.reset(train=1)
    ARGS_CHAIN_INITIALIZATION.dataset.total_sequence_length.reset(valid=context_length, test=context_length)

    ARGS_CHAIN_INITIALIZATION.training.sampling = Namespace(method="full")
    ARGS_CHAIN_INITIALIZATION.training.optimizer = Namespace(
        type="SGD",
        # min_lr=0.0,
        weight_decay=0.0, momentum=0.9
    )
    ARGS_CHAIN_INITIALIZATION.training.scheduler = Namespace(
        # type="reduce_on_plateau",
        # factor=0.5, patience=10, warmup_duration=0,
        type="exponential",
        lr_decay=0.98, warmup_duration=0,
        epochs=200, gradient_cutoff=1e-6,
    )
    ARGS_CHAIN_INITIALIZATION.training.iterations_per_epoch = 200

    ARGS_CHAIN_INITIALIZATION.experiment.n_experiments = n_test_systems
    ARGS_CHAIN_INITIALIZATION.experiment.ensemble_size = test_dataset_size
    ARGS_CHAIN_INITIALIZATION.experiment.metrics = Namespace(training={"validation_analytical"})
    ARGS_CHAIN_INITIALIZATION.experiment.exp_name = exp_name_chain_initialization
    ARGS_CHAIN_INITIALIZATION.experiment.backup_frequency = 50

    # al_exemplar = get_metric_namespace_from_result(result_exemplar).al.reshape(utils.ceildiv(context_length, rnn_increment), n_test_systems, test_dataset_size, -1)
    # plt.plot(al_exemplar.mean(dim=-1).median(dim=-1).values.mean(dim=-1).cpu(), label="median")
    # plt.plot(al_exemplar.mean(dim=[-3, -2, -1]).cpu(), label="mean")
    # plt.legend()
    # plt.show()
    # print(al_exemplar.shape)
    # raise Exception()

    al_exemplar = get_metric_namespace_from_result(result_exemplar).al.flatten(3, -1).mean(dim=-1).median(dim=-1).values.mean(dim=-1)

    # SECTION: Chain initialization setup
    output_fname_formatter = "result_{0}"

    s = 1.0
    min_eqs = int(s * utils.ceildiv(S_D * (S_D + 2 * O_D), O_D))
    results_chain_initialization = [*map(DimArray, result_exemplar[:utils.ceildiv(min_eqs, rnn_increment)])]  # DimArray uses 1-indexing, include both the zero-predictor and the minimum number of observations to fully constrain RNN parameters

    scaling_lr, scaling_lr_decay = 1e-5, 0.97
    running_context_length = rnn_increment * (utils.ceildiv(min_eqs, rnn_increment) + 1)
    while running_context_length < context_length:
        print(f"Sequence length {running_context_length} target: {al_exemplar[running_context_length - 1].item()} -> {al_exemplar[running_context_length].item()} " + "-" * 120)
        args = utils.deepcopy_namespace(ARGS_CHAIN_INITIALIZATION)
        args.dataset.total_sequence_length.train = running_context_length
        args.training.optimizer.max_lr = args.training.optimizer.min_lr = scaling_lr

        initialization = utils.multi_map(
            lambda pair: PTR(pair[1]), DimArray(
                get_result_attr(results_chain_initialization[-1], "learned_kfs"),
                dims=results_chain_initialization[-1].dims
            ), dtype=PTR
        )

        results_chain_initialization.append(run_experiments(
            args, [], {
                "dir": output_dir,
                "fname": output_fname_formatter.format(running_context_length)
            }, initialization=initialization, save_experiment=True
        )[0])

        running_context_length += 1
        scaling_lr *= scaling_lr_decay

    raise Exception()


    """ Result processing """
    print("Result processing" + "\n" + "-" * 120)
    systems = LTISystem(SHP.problem_shape, systems.values[()].td().squeeze(0))
    dataset = dataset.values[()].obj.squeeze(1).squeeze(0)

    output_name = "output.environment.observation"
    analytical_initialization_output = utils.rgetattr(get_metric_namespace_from_result(result_analytical_initialization), output_name)
    chain_initialization_output = torch.stack([
        utils.rgetattr(get_metric_namespace_from_result(r), output_name)
        for r in results_chain_initialization
    ], dim=0)


    print(analytical_initialization_output.shape)
    print(chain_initialization_output.shape)
    raise Exception()




    def loss(observation_estimation: torch.Tensor) -> torch.Tensor:
        return (dataset["environment", "observation"] - observation_estimation).norm(dim=-1) ** 2

    with torch.set_grad_enabled(False):
        zero_predictor_al = utils.batch_trace(systems.S_observation_inf)
        zero_predictor_l = loss(torch.zeros_like(dataset["environment", "observation"]))
        il = utils.batch_trace(systems.S_prediction_err_inf)
        eil = loss(dataset["target"])


        # [n_experiments x ensemble_size x n_test_systems x test_dataset_size x context_length x O_D]
        # -> [n_test_systems x test_dataset_size x context_length x O_D]
        gpt2_output, transfoxl_output = M_transformer.output.observation_estimation.squeeze(2).squeeze(1)
        # -> [n_test_systems x test_dataset_size x context_length]
        gpt2_l, transfoxl_l = loss(gpt2_output), loss(transfoxl_output)


        # [n_firs x train.sequence_length x n_test_systems x test_dataset_size x n_experiments x ensemble_size x context_length x O_D]
        # -> [n_firs x train.sequence_length x n_test_systems x test_dataset_size x context_length x O_D]
        # -> [n_firs x n_test_systems x test_dataset_size x O_D x context_length]
        # -> [n_firs x n_test_systems x test_dataset_size x context_length x O_D]
        cnn_output = torch.diagonal(M_baseline_cnn.output.observation_estimation.squeeze(5).squeeze(4), dim1=1, dim2=4).transpose(3, 4)
        # -> [n_firs x n_test_systems x test_dataset_size x context_length]
        cnn_l = loss(cnn_output)
        # [n_firs x context_length x n_test_systems x test_dataset_size x n_experiments x ensemble_size]
        # -> [n_firs x context_length x n_test_systems x test_dataset_size]
        # -> [n_firs x n_test_systems x test_dataset_size x context_length]
        cnn_al = M_baseline_cnn.al.squeeze(5).squeeze(4).permute(0, 2, 3, 1)


        # [train.sequence_length x n_test_systems x test_dataset_size x n_experiments x ensemble_size x context_length x O_D]
        # -> [train.sequence_length x n_test_systems x test_dataset_size x context_length x O_D]
        # -> [train.sequence_length x n_test_systems x test_dataset_size x O_D]
        # -> [n_test_systems x test_dataset_size x train.sequence_length x O_D]
        rnn_sequence_lengths = [*range(0, context_length, rnn_increment),]
        rnn_output = M_baseline_rnn.output.observation_estimation.squeeze(4).squeeze(3)[torch.arange(len(rnn_sequence_lengths)), :, :, torch.tensor(rnn_sequence_lengths)].permute(1, 2, 0, 3)
        # [train.sequence_length x n_test_systems x test_dataset_size x n_experiments x ensemble_size]
        # -> [train.sequence_length x n_test_systems x test_dataset_size]
        # -> [n_test_systems x test_dataset_size x train.sequence_length]
        rnn_al = M_baseline_rnn.al.squeeze(4).squeeze(3).permute(1, 2, 0)


        rnn_indices = torch.tensor(rnn_sequence_lengths)
        padded_rnn_output = torch.zeros((n_test_systems, test_dataset_size, context_length, SHP.O_D))
        padded_rnn_output[:, :, rnn_indices] = rnn_output
        # -> [n_test_systems x test_dataset_size x context_length]
        rnn_l = loss(padded_rnn_output)[:, :, rnn_indices]




