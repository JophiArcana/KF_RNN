#%%
import torch
from matplotlib import pyplot as plt

from kf_rnn.infrastructure import loader
from kf_rnn.infrastructure.config import MetricsConfig, OptimizerConfig, SamplingConfig, SchedulerConfig
from kf_rnn.infrastructure.experiment import run_experiments, plot_experiment, get_result_attr
from kf_rnn.model.sequential import RnnPredictor, RnnKalmanInitializedPredictor


if __name__ == "__main__":
    base_exp_name = "basic"
    output_dir = "system6_iir"
    output_fname = "result"

    system2, cfg = loader.load_system_and_args("6dim_scalar_system_matrices")
    total_dataset_length = 10000

    S_D = cfg.system.S_D
    cfg.dataset.total_sequence_length.reset(valid=total_dataset_length, test=total_dataset_length)
    cfg.training.sampling = SamplingConfig(method="full", batch_size=None, subsequence_length=None)
    cfg.training.optimizer = OptimizerConfig(
        type="AdamW",
        max_lr=1e-3, min_lr=1e-7,
        weight_decay=0.0,
    )
    cfg.training.scheduler = SchedulerConfig(
        type="exponential",
        lr_decay=0.995, warmup_duration=100,
        epochs=2000, gradient_cutoff=1e-6,
    )
    cfg.experiment.metrics = MetricsConfig(
        training={"validation_analytical"},
        testing={"al", "il"}
    )
    cfg.experiment.exp_name = base_exp_name

    configurations = [
        ("model", {
            "model": [
                RnnPredictor.Config(S_D=S_D),
                RnnKalmanInitializedPredictor.Config(S_D=S_D),
            ]
        }),
        ("total_trace_length", {
            "dataset.total_sequence_length.train": [100, 200, 500, 1000, 2000, 5000, 10000]
        }),
    ]

    result, _, _ = run_experiments(
        cfg, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=True
    )
    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical", lstsq=False, xscale="linear")





# %%
