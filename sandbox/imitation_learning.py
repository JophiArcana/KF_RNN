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
    from model.transformer.transformerxl_iccontroller import TransformerXLInContextController

    exp_name = "TransformerXLImitationLearning"
    output_dir = "transformerxl"

    d_embed = 256
    n_layer = 12
    n_head = 8
    d_inner = 4 * d_embed

    SHP = Namespace(S_D=2, I_D=1, O_D=1, input_enabled=True)
    args = loader.generate_args(SHP)
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
    args.dataset.train = Namespace(
        dataset_size=1,
        total_sequence_length=200,
        system=Namespace(
            n_systems=1,
            distribution=LQGDistribution("gaussian", "gaussian", 0.1, 0.1, 1.0, 1.0)
        )
    )
    args.dataset.valid = args.dataset.test = Namespace(
        dataset_size=5,
        total_sequence_length=10000,
    )

    args.train.sampling = Namespace(type="full")
    args.train.optimizer = Namespace(
        type="SGD",
        max_lr=1e-4, min_lr=1e-6,
        weight_decay=0.0
    )
    args.train.scheduler = Namespace(
        type="exponential",
        epochs=2000, lr_decay=1.0,
    )
    args.train.iterations_per_epoch = 1

    args.experiment.n_experiments = 1
    args.experiment.ensemble_size = 1
    args.experiment.exp_name = exp_name
    args.experiment.metrics = {"validation"}

    result, dataset = run_experiments(args, [], {}, save_experiment=True)




