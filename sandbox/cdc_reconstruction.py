import copy
import os
from argparse import Namespace

import torch
from transformers import GPT2Config

from infrastructure import loader
from infrastructure import utils
from infrastructure.experiment import *
from infrastructure.settings import DEVICE
from infrastructure.utils import PTR
from model.convolutional import CnnKFLeastSquares
from model.linear_system import LinearSystemGroup
from model.linear_system_distribution import MOPDistribution
from model.sequential import RnnKFPretrainAnalytical
from model.transformer import GPT2InContextKF


if __name__ == "__main__":
    os.chdir("..")

    output_dir = "in_context"
    output_fname = "result"

    """ Transformer experiment """
    exp_name_transformer = "CDCReconstruction_transformer"

    SHP = Namespace(S_D=10, I_D=1, O_D=5, input_enabled=False)
    ARGS_TRANSFORMER = loader.generate_args(SHP)


    context_length = 250
    n_train_systems = 40000
    n_test_systems = 3
    valid_dataset_size = 2000
    test_dataset_size = 256


    ARGS_TRANSFORMER.model.model = GPT2InContextKF
    ARGS_TRANSFORMER.model.gpt2 = GPT2Config(
        n_positions=context_length,
        n_embd=256,
        n_layer=12,
        n_head=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    ARGS_TRANSFORMER.dataset.train = Namespace(
        dataset_size=1,
        total_sequence_length=context_length,
        system=Namespace(
            n_systems=n_train_systems,
            distribution=MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
        )
    )
    ARGS_TRANSFORMER.dataset.valid = Namespace(
        dataset_size=valid_dataset_size,
        total_sequence_length=valid_dataset_size * context_length,
        system=Namespace(
            n_systems=n_test_systems
        )
    )
    ARGS_TRANSFORMER.dataset.test = Namespace(
        dataset_size=test_dataset_size,
        total_sequence_length=test_dataset_size * context_length,
        system=Namespace(
            n_systems=n_test_systems
        )
    )

    del ARGS_TRANSFORMER.train.warmup_duration
    ARGS_TRANSFORMER.train.epochs = 40000
    ARGS_TRANSFORMER.train.subsequence_length = context_length
    ARGS_TRANSFORMER.train.batch_size = 28
    ARGS_TRANSFORMER.train.iterations_per_epoch = 1

    ARGS_TRANSFORMER.train.optim_type = "Adam"
    ARGS_TRANSFORMER.train.max_lr = 3e-4
    ARGS_TRANSFORMER.train.lr_decay = 1.0
    ARGS_TRANSFORMER.train.weight_decay = 1e-2

    ARGS_TRANSFORMER.experiment.n_experiments = 1
    ARGS_TRANSFORMER.experiment.ensemble_size = 1
    ARGS_TRANSFORMER.experiment.exp_name = exp_name_transformer
    ARGS_TRANSFORMER.experiment.metrics = set() # {"validation"}

    configurations_transformer = []

    result_transformer = run_experiments(
        ARGS_TRANSFORMER, configurations_transformer, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True
    )
    M = get_metric_namespace_from_result(result_transformer)
    print(M.output.shape)



    """ Baseline experiment setup """
    exp_name_baseline_cnn = "CDCReconstruction_baseline_cnn"
    exp_name_baseline_rnn = "CDCReconstruction_baseline_rnn"

    for _exp_name_baseline in (exp_name_baseline_cnn, exp_name_baseline_rnn):
        os.makedirs(f"output/{output_dir}/{_exp_name_baseline}/training", exist_ok=True)
        os.makedirs(f"output/{output_dir}/{_exp_name_baseline}/testing", exist_ok=True)


        if not all(map(os.path.exists, (
            f"output/{output_dir}/{_exp_name_baseline}/training/systems.pt",
            f"output/{output_dir}/{_exp_name_baseline}/testing/systems.pt"
        ))):
            systems = torch.load(f"output/{output_dir}/{exp_name_transformer}/testing/systems.pt", map_location=DEVICE)
            baseline_systems = utils.multi_map(
                lambda lsg: LinearSystemGroup(lsg.td().permute(1, 0), SHP.input_enabled),
                torch.load(f"output/{output_dir}/{exp_name_transformer}/testing/systems.pt", map_location=DEVICE)["test"], dtype=LinearSystemGroup
            )
            torch.save({
                "train": baseline_systems
            }, f"output/{output_dir}/{_exp_name_baseline}/training/systems.pt")
            torch.save({
                "test": baseline_systems
            }, f"output/{output_dir}/{_exp_name_baseline}/testing/systems.pt")


        if not all(map(os.path.exists, (
            f"output/{output_dir}/{_exp_name_baseline}/training/dataset.pt",
            f"output/{output_dir}/{_exp_name_baseline}/testing/dataset.pt"
        ))):
            dataset = torch.load(f"output/{output_dir}/{exp_name_transformer}/testing/dataset.pt", map_location=DEVICE)
            baseline_dataset = utils.multi_map(
                lambda dataset_: PTR(dataset_.obj.permute(2, 3, 0, 1, 4)),
                torch.load(f"output/{output_dir}/{exp_name_transformer}/testing/dataset.pt", map_location=DEVICE)["test"], dtype=PTR
            )
            torch.save({
                "train": baseline_dataset,
                "valid": baseline_dataset
            }, f"output/{output_dir}/{_exp_name_baseline}/training/dataset.pt")
            torch.save({
                "test": baseline_dataset,
            }, f"output/{output_dir}/{_exp_name_baseline}/testing/dataset.pt")



    """ CNN Experiment """
    ARGS_BASELINE_CNN = loader.generate_args(SHP)
    ARGS_BASELINE_CNN.dataset.train = Namespace(
        dataset_size=1,
        system=Namespace(
            n_systems=1,
            distribution=MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
        )
    )
    ARGS_BASELINE_CNN.dataset.valid = ARGS_BASELINE_CNN.dataset.test = Namespace(
        total_sequence_length=context_length
    )
    ARGS_BASELINE_CNN.experiment.n_experiments = n_test_systems
    ARGS_BASELINE_CNN.experiment.ensemble_size = test_dataset_size
    ARGS_BASELINE_CNN.experiment.metrics = {"validation_analytical"}

    # SECTION: Make a copy for RNN args after setting shared parameters
    ARGS_BASELINE_RNN = copy.deepcopy(ARGS_BASELINE_CNN)

    # SECTION: Set CNN exclusive hyperparameters
    ARGS_BASELINE_CNN.model.model = CnnKFLeastSquares
    ARGS_BASELINE_CNN.model.ridge = 1.0

    ARGS_BASELINE_CNN.experiment.exp_name = exp_name_baseline_cnn

    n_firs = 3
    configurations_baseline_cnn = [
        ("model", {
            "model.ir_length": [*range(1, n_firs + 1)],
        }),
        ("total_trace_length", {
            "dataset.train.total_sequence_length": [*range(1, context_length)]
        })
    ]

    result_baseline_cnn = run_experiments(
        ARGS_BASELINE_CNN, configurations_baseline_cnn, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True
    )
    M_baseline_cnn = get_metric_namespace_from_result(result_baseline_cnn)
    print(M_baseline_cnn.output.shape)



    """ RNN Experiment """
    # SECTION: Set RNN exclusive hyperparameters
    ARGS_BASELINE_RNN.model.model = RnnKFPretrainAnalytical
    ARGS_BASELINE_RNN.model.S_D = SHP.S_D

    ARGS_BASELINE_RNN.train.optim_type = "GD"
    ARGS_BASELINE_RNN.train.max_lr = 1e-3
    ARGS_BASELINE_RNN.train.epochs = 1200

    ARGS_BASELINE_RNN.experiment.exp_name = exp_name_baseline_rnn

    n_firs = 3
    configurations_baseline_rnn = [
        ("total_trace_length", {
            "dataset.train.total_sequence_length": [*range(5, context_length, 5)]
        })
    ]

    result_baseline_rnn = run_experiments(
        ARGS_BASELINE_RNN, configurations_baseline_rnn, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True
    )
    M_baseline_rnn = get_metric_namespace_from_result(result_baseline_rnn)
    print(M_baseline_rnn.output.shape)







