import os
import sys
from argparse import Namespace

import torch
from matplotlib import pyplot as plt
from transformers import GPT2Config, TransfoXLConfig

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader, utils
from infrastructure.settings import DEVICE
from infrastructure.experiment import *
from system.linear_time_invariant import MOPDistribution, LTISystem
from model.base import Predictor
from model.transformer import GPT2InContextPredictor, TransformerXLInContextPredictor


if __name__ == "__main__":
    output_dir = "in_context"
    output_fname = "result"

    dist = MOPDistribution("gaussian", "gaussian", 0.1, 0.1)
    SHP = Namespace(
        distribution=dist, S_D=10,
        problem_shape=Namespace(
            environment=Namespace(observation=5),
            controller=Namespace()
        ), auxiliary=Namespace()
    )
    
    context_length = 250
    n_train_systems = 40000
    n_test_systems = 3
    n_valid_traces = 256
    n_test_traces = 256
    
    n_firs = 5
    rnn_increment = 1
    

    """ Transformer experiment """
    exp_name_transformer = "CDCReconstruction_transformer"

    ARGS_TRANSFORMER = loader.generate_args(SHP)

    # SECTION: Transformer architecture hyperparameters
    d_embed = 256
    n_layer = 12
    n_head = 8
    d_inner = 4 * d_embed

    ARGS_TRANSFORMER.model.gpt2 = GPT2Config(
        n_positions=context_length,
        n_embd=d_embed,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=d_inner,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
    )
    ARGS_TRANSFORMER.model.transformerxl = TransfoXLConfig(
        d_model=d_embed,
        d_embed=d_embed,
        n_layer=n_layer,
        n_head=n_head,
        d_head=d_embed // n_head,
        d_inner=d_inner,
        dropout=0.0,
    )

    # SECTION: Dataset hyperparameters
    ARGS_TRANSFORMER.system.distribution.update(train=dist, valid=dist, test=dist)

    ARGS_TRANSFORMER.dataset.n_systems.update(train=n_train_systems, valid=n_test_systems, test=n_test_systems)
    ARGS_TRANSFORMER.dataset.n_traces.update(train=1, valid=n_valid_traces, test=n_test_traces)
    ARGS_TRANSFORMER.dataset.total_sequence_length.update(train=context_length, valid=n_valid_traces * context_length, test=n_test_traces * context_length)

    # SECTION: Training hyperparameters
    ARGS_TRANSFORMER.training.sampling = Namespace(
        method="subsequence_padded",
        subsequence_length=context_length,
        batch_size=32
    )
    ARGS_TRANSFORMER.training.optimizer = Namespace(
        type="AdamW",
        max_lr=3e-4, min_lr=1e-6,
        weight_decay=1e-2, momentum=0.9
    )
    ARGS_TRANSFORMER.training.scheduler = Namespace(
        type="exponential",
        epochs=2000, lr_decay=1.0,
    )
    ARGS_TRANSFORMER.training.iterations_per_epoch = 50


    ARGS_TRANSFORMER.experiment.n_experiments = 1
    ARGS_TRANSFORMER.experiment.ensemble_size = 1
    ARGS_TRANSFORMER.experiment.exp_name = exp_name_transformer
    ARGS_TRANSFORMER.experiment.metrics = Namespace(training={"noiseless_validation"}) # {"validation"}

    configurations_transformer = [
        ("model", {
            "name": ["gpt2", "transformerxl"],
            "model.model": [GPT2InContextPredictor, TransformerXLInContextPredictor],
        })
    ]

    result_transformer = utils.torch_load(f"output/{output_dir}/{exp_name_transformer}/testing/result.pt")

    output = get_result_attr(result_transformer, "output")
    time = get_result_attr(result_transformer, "time")

    valid_systems = utils.torch_load(f"output/{output_dir}/{exp_name_transformer}/training/systems.pt")["valid"].values[()]
    valid_systems = LTISystem(valid_systems.problem_shape, valid_systems.auxiliary, valid_systems.td().squeeze(0))
    valid_env = valid_systems.environment

    valid_dataset = utils.torch_load(f"output/{output_dir}/{exp_name_transformer}/training/dataset.pt")["valid"].values[()].obj
    valid_dataset = valid_dataset.reshape(*valid_dataset.shape[2:])

    noiseless_empirical_irreducible_loss = Predictor.evaluate_run(
        valid_dataset["environment", "target_observation_estimation"],
        valid_dataset, ("environment", "noiseless_observation")
    ) + utils.batch_trace(valid_env.H @ valid_env.S_W @ valid_env.H.mT + valid_env.S_V)

    expected_zero_predictor_loss = valid_systems.zero_predictor_loss.environment.observation
    expected_irreducible_loss = valid_systems.irreducible_loss.environment.observation

    print(expected_zero_predictor_loss)
    print(expected_irreducible_loss)
    print(get_result_attr(result_transformer, "time"))


    # SECTION: Plot over gradient descent steps
    x = ARGS_TRANSFORMER.training.iterations_per_epoch * torch.arange(ARGS_TRANSFORMER.training.scheduler.epochs) + 1
    clip = x >= 1000
    for model_name, output in zip(
            configurations_transformer[0][1]["name"],
            get_result_attr(result_transformer, "output")
    ):
        output = output.reshape(-1)
        noiseless_normalized_validation_error = (output["noiseless_validation"] - noiseless_empirical_irreducible_loss) / (expected_zero_predictor_loss - expected_irreducible_loss)

        # for sys_idx in range(n_test_systems):
        #     plt.plot(x, noiseless_normalized_validation_error[:, sys_idx].detach(), label=f"{model_name}_sys{sys_idx}")
        plt.plot(x[clip], noiseless_normalized_validation_error.mean(dim=-1).detach()[clip], label=model_name)

    plt.xlabel("gradient_descent_steps")
    plt.xscale("log")
    plt.ylabel("noiseless_normalized_validation_error")
    plt.yscale("log")

    plt.legend()
    plt.show()


    # SECTION: Plot over real elapsed time
    for model_name, output, time in zip(
            configurations_transformer[0][1]["name"],
            get_result_attr(result_transformer, "output"),
            get_result_attr(result_transformer, "time")
    ):
        x = torch.linspace(0, time, ARGS_TRANSFORMER.training.scheduler.epochs) + 1
        clip = x > 100
        output = output.reshape(-1)
        noiseless_normalized_validation_error = (output["noiseless_validation"] - noiseless_empirical_irreducible_loss) / (expected_zero_predictor_loss - expected_irreducible_loss)

        # for sys_idx in range(n_test_systems):
        #     plt.plot(x, noiseless_normalized_validation_error[:, sys_idx].detach(), label=f"{model_name}_sys{sys_idx}")
        plt.plot(x[clip], noiseless_normalized_validation_error.mean(dim=-1).detach()[clip], label=model_name)

    plt.xlabel("real_elapsed_time (s)")
    plt.xscale("log")
    plt.ylabel("noiseless_normalized_validation_error")
    plt.yscale("log")

    plt.legend()
    plt.show()




