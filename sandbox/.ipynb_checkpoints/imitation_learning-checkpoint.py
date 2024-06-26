import os
import sys
from argparse import Namespace

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from infrastructure import loader
from infrastructure.experiment import *

if __name__ == "__main__":
    from transformers import TransfoXLConfig
    from system.linear_quadratic_gaussian import LQGDistribution
    from model.sequential.rnn_controller import RnnController, RnnControllerPretrainAnalytical
    from model.transformer.transformerxl_iccontroller import TransformerXLInContextController

    exp_name = "RnnControllerInitalizationComparison"
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

    args.experiment.n_experiments = 1
    args.experiment.ensemble_size = 32
    args.experiment.exp_name = exp_name
    args.experiment.metrics = {"validation_analytical"}

    configurations = [
        ("model", {
            "model.model": [RnnController, RnnControllerPretrainAnalytical]
        })
    ]

    result, dataset = run_experiments(args, configurations, {
        "dir": output_dir,
        "fname": output_fname
    }, save_experiment=True)




