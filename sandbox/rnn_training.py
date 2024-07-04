import copy
import os
import sys
from argparse import Namespace

import torch
from dimarray import DimArray

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader, utils
from infrastructure.settings import DEVICE
from infrastructure.utils import PTR
from infrastructure.experiment import *
from system.linear_time_invariant import LTISystem, MOPDistribution
from model.zero_predictor import ZeroPredictor
from model.sequential import RnnPredictor, RnnPredictorPretrainAnalytical


if __name__ == "__main__":
    output_dir = "in_context_rnn_chaining"
    output_fname = "result"

    context_length = 250
    n_test_systems = 3
    test_dataset_size = 16

    SHP = Namespace(S_D=10, problem_shape=Namespace(
        environment=Namespace(observation=5),
        controller=Namespace()
    ))

    os.makedirs(f"output/{output_dir}", exist_ok=True)
    shared_systems_fname, shared_dataset_fname = f"output/{output_dir}/systems.pt", f"output/{output_dir}/dataset.pt"
    if not os.path.exists(shared_systems_fname):
        dist = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
        systems = dist.sample(SHP, (n_test_systems, 1))
        torch.save(systems, shared_systems_fname)
    else:
        systems = torch.load(shared_systems_fname, map_location=DEVICE)

    if not os.path.exists(shared_dataset_fname):
        dataset = systems.generate_dataset(test_dataset_size, context_length).transpose(1, 2).unsqueeze(3)
        torch.save(dataset, shared_dataset_fname)
    else:
        dataset = torch.load(shared_dataset_fname, map_location=DEVICE)


    """ Baseline experiment setup """
    exp_name_analytical_initialization = "AnalyticalInitialization"
    exp_name_chain_initialization = "ChainInitialization"

    systems_dimarr = DimArray(utils.array_of(systems))
    dataset_dimarr = DimArray(utils.array_of(PTR(dataset)))

    for _exp_name_baseline in (exp_name_analytical_initialization, exp_name_chain_initialization):
        os.makedirs(f"output/{output_dir}/{_exp_name_baseline}/training", exist_ok=True)
        os.makedirs(f"output/{output_dir}/{_exp_name_baseline}/testing", exist_ok=True)

        for (exp_systems_fname, key) in [
            (f"output/{output_dir}/{_exp_name_baseline}/training/systems.pt", "train"),
            (f"output/{output_dir}/{_exp_name_baseline}/testing/systems.pt", "test")
        ]:
            if not os.path.exists(exp_systems_fname):
                torch.save({key: systems_dimarr}, exp_systems_fname)

        for (exp_dataset_fname, key) in [
            (f"output/{output_dir}/{_exp_name_baseline}/training/dataset.pt", "train"),
            (f"output/{output_dir}/{_exp_name_baseline}/testing/dataset.pt", "test")
        ]:
            if not os.path.exists(exp_dataset_fname):
                torch.save({key: dataset_dimarr}, exp_dataset_fname)



    # SECTION: RNN Experiment
    rnn_increment = 5

    ARGS_ANALYTICAL_INITIALIZATION = loader.generate_args(SHP)
    ARGS_ANALYTICAL_INITIALIZATION.model.S_D = SHP.S_D

    ARGS_ANALYTICAL_INITIALIZATION.dataset.train = Namespace(
        dataset_size=1,
        system=Namespace(n_systems=1)
    )
    ARGS_ANALYTICAL_INITIALIZATION.dataset.valid = ARGS_ANALYTICAL_INITIALIZATION.dataset.test = Namespace(
        total_sequence_length=context_length
    )

    ARGS_ANALYTICAL_INITIALIZATION.train.sampling.method = "full"
    ARGS_ANALYTICAL_INITIALIZATION.train.optimizer.max_lr = 1e-3
    ARGS_ANALYTICAL_INITIALIZATION.train.scheduler.epochs = 2000

    ARGS_ANALYTICAL_INITIALIZATION.experiment.n_experiments = n_test_systems
    ARGS_ANALYTICAL_INITIALIZATION.experiment.ensemble_size = test_dataset_size
    ARGS_ANALYTICAL_INITIALIZATION.experiment.metrics = {"validation_analytical"}
    ARGS_ANALYTICAL_INITIALIZATION.experiment.exp_name = exp_name_analytical_initialization

    rnn_sequence_lengths = [*range(rnn_increment, context_length, rnn_increment),]
    configurations_analytical_initialization = [
        ("total_trace_length", {
            "model.model": [ZeroPredictor] + [RnnPredictorPretrainAnalytical] * len(rnn_sequence_lengths),
            "dataset.train.total_sequence_length": [0] + rnn_sequence_lengths
        })
    ]

    result_analytical_initialization, systems, dataset = run_experiments(
        ARGS_ANALYTICAL_INITIALIZATION, configurations_analytical_initialization, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True
    )



    # SECTION: Chain initialization experiment setup
    ARGS_CHAIN_INITIALIZATION = copy.deepcopy(ARGS_ANALYTICAL_INITIALIZATION)
    ARGS_CHAIN_INITIALIZATION.model.model = RnnPredictor
    ARGS_CHAIN_INITIALIZATION.experiment.exp_name = exp_name_chain_initialization


    output_fname_formatter = "result_{0}"
    results_chain_initialization = [DimArray(result_analytical_initialization[0])]
    for idx, rnn_sequence_length in enumerate(rnn_sequence_lengths):
        args = copy.deepcopy(ARGS_CHAIN_INITIALIZATION)
        args.dataset.train.total_sequence_length = rnn_sequence_length

        if idx == 0:
            initialization = DimArray(utils.array_of(PTR(systems.values[()].td())))
        else:
            initialization = utils.multi_map(
                lambda pair: PTR(pair[1]),
                DimArray(get_result_attr(results_chain_initialization[-1], "learned_kfs")), dtype=PTR
            )

        results_chain_initialization.append(run_experiments(
            args, [], {
                "dir": output_dir,
                "fname": output_fname_formatter.format(rnn_sequence_length)
            }, initialization=initialization, save_experiment=True
        )[0])




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




