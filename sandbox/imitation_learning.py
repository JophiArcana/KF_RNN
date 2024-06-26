import os
import sys
from argparse import Namespace
from matplotlib import pyplot as plt

import torch
from tensordict import TensorDict

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader, utils
from infrastructure.experiment import *
from model.base import Predictor
from system.linear_time_invariant import LinearSystemGroup


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    from transformers import TransfoXLConfig
    from system.linear_quadratic_gaussian import LQGDistribution
    from model.sequential.rnn_controller import RnnController, RnnControllerPretrainAnalytical
    from model.transformer.transformerxl_iccontroller import TransformerXLInContextController

    exp_name = "RnnControllerInitializationComparison_lowcontrolcoefficient"
    output_dir = "imitation_learning"
    output_fname = "result"

    d_embed = 256   # 256
    n_layer = 3     # 12
    n_head =  16     # 8
    d_inner = 4 * d_embed

    SHP = Namespace(S_D=3, I_D=2, O_D=2, input_enabled=True)
    args = loader.generate_args(SHP)

    args.model.model = RnnController
    args.model.S_D = SHP.S_D
    # args.model.model = TransformerXLInContextController
    # args.model.transformerxl = TransfoXLConfig(
    #     d_model=d_embed,
    #     d_embed=d_embed,
    #     n_layer=n_layer,
    #     n_head=n_head,
    #     d_head=d_embed // n_head,
    #     d_inner=d_inner,
    #     dropout=0.0,
    # )
    args.dataset.train = Namespace(
        dataset_size=1,
        total_sequence_length=2000,
        system=Namespace(
            n_systems=1,
            distribution=LQGDistribution("gaussian", "gaussian", 0.3, 0.1, 1.0, 1.0)
        )
    )
    args.dataset.valid = args.dataset.test = Namespace(
        dataset_size=10,
        total_sequence_length=100000,
    )
    args.train.sampling = Namespace(method="full")
    args.train.optimizer = Namespace(
        type="Adam",
        max_lr=1e-2, min_lr=1e-9,
        weight_decay=0.0
    )
    args.train.scheduler = Namespace(
        type="exponential",
        epochs=2000, lr_decay=0.995,
    )
    args.train.iterations_per_epoch = 1
    args.train.control_coefficient = 1e-3

    args.experiment.n_experiments = 1
    args.experiment.ensemble_size = 32
    args.experiment.exp_name = exp_name
    args.experiment.metrics = {"validation_analytical"}

    configurations = [
        ("model", {
            "model.model": [RnnController, RnnControllerPretrainAnalytical]
        })
    ]

    result, systems, dataset = run_experiments(args, configurations, {
        "dir": output_dir,
        "fname": output_fname
    }, save_experiment=True)

    # SECTION: Result analysis
    identity_training_output, analytical_training_output = map(
        lambda td: td.squeeze(0),
        get_result_attr(result, "output")
    )
    lsg = LinearSystemGroup(systems.values[()].td().squeeze(1).squeeze(0), SHP.input_enabled)
    print(lsg.irreducible_loss, lsg.zero_predictor_loss)

    plt.plot(identity_training_output["validation_analytical"].median(dim=0).values, label="identity_initialization")
    plt.plot(analytical_training_output["validation_analytical"].median(dim=0).values, label="analytical_initialization")
    plt.legend()
    plt.show()


    # def squeeze(t: torch.Tensor | TensorDict[str, torch.Tensor]) -> torch.Tensor | TensorDict[str, torch.Tensor]:
    #     return t.view(t.shape[3:])
    #
    # lsg = LinearSystemGroup(systems.values[()].td().squeeze(1).squeeze(0), SHP.input_enabled)
    # dataset = squeeze(dataset.values[()].obj)
    #
    # M = get_metric_namespace_from_result(result)
    # observation_estimation = squeeze(M.output.observation_estimation)
    # input_estimation = squeeze(M.output.input_estimation)
    #
    # print("Result processing" + "\n" + "-" * 120)
    # print("Irreducible loss:", lsg.irreducible_loss)
    # print("Zero predictor loss:", utils.batch_trace(lsg.H @ lsg.S_state_inf @ lsg.H.mT + lsg.S_V))
    #
    # print(Predictor.evaluate_run(observation_estimation, dataset, "observation"))
    # print(Predictor.evaluate_run(0, dataset, "observation"))
    # # print(Predictor.evaluate_run(input_estimation, dataset, "input"))
    #
    # print(observation_estimation.shape)
    # print(input_estimation.shape)
    # print(dataset)
    #
    # plt.plot(dataset["observation"][0, :100, 0], label="observation")
    # plt.plot(observation_estimation[0, :100, 0], label="observation_estimation")
    # plt.legend()
    # plt.show()





