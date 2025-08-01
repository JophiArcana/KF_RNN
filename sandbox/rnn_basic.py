#%%
# This line needs to be added since some terminals will not recognize the current directory
import sys
sys.path.append("/home/wenliao/KF_RNN")

from argparse import Namespace

import torch
from matplotlib import pyplot as plt

from infrastructure import loader
from infrastructure.experiment import run_experiments, plot_experiment, get_result_attr
from model.sequential import RnnPredictor, RnnPredictorPretrainAnalytical


if __name__ == "__main__":
    base_exp_name = "basic"
    output_dir = "system6_iir"
    output_fname = "result"

    system2, args = loader.load_system_and_args("data/6dim_scalar_system_matrices")
    total_dataset_length = 10000
    
    args.model.S_D = args.system.S_D.default()
    args.dataset.total_sequence_length.reset(valid=total_dataset_length, test=total_dataset_length)
    args.training.sampling = Namespace(method="full")
    args.training.optimizer = Namespace(
        type="AdamW",
        max_lr=1e-3, min_lr=1e-7,
        weight_decay=0.0,
    )
    args.training.scheduler = Namespace(
        type="exponential",
        lr_decay=0.995, warmup_duration=100,
        epochs=2000, gradient_cutoff=1e-6,
    )
    args.training.iterations_per_epoch = 20
    args.experiment.metrics = Namespace(
        training={"validation_analytical"},
        testing={"al", "il"}
    )
    args.experiment.exp_name = base_exp_name

    configurations = [
        ("model", {
            "model.model": [
                RnnPredictor,
                RnnPredictorPretrainAnalytical,
            ]
        }),
        ("total_trace_length", {
            "dataset.total_sequence_length.train": [100, 200, 500, 1000, 2000, 5000, 10000]
        }),
    ]

    result, _, _ = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=True
    )
    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical", lstsq=False, xscale="linear")





# %%
