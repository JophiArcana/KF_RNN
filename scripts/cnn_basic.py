#%%
from kf_rnn.infrastructure import loader
from kf_rnn.infrastructure.config import MetricsConfig
from kf_rnn.infrastructure.experiment import *
from kf_rnn.model.convolutional import *
from kf_rnn.model.sequential import *


if __name__ == "__main__":
    base_exp_name = "SingleTrace"
    output_dir = "system2_CNN"
    output_fname = "result"


    system2, cfg = loader.load_system_and_args("2dim_scalar_system_matrices")

    cfg.training.sampling.batch_size = 64
    cfg.training.sampling.subsequence_length = 32

    cfg.training.scheduler.epochs = 2000


    cfg.experiment.exp_name = base_exp_name
    cfg.experiment.ignore_metrics = MetricsConfig(training={"impulse_target"})

    configurations = [
        ("model", {
            "model": [CnnLeastSquaresPredictor.Config(ir_length=cfg.system.S_D)]
        }),
        ("total_trace_length", {
            "dataset.total_sequence_length.train": [100, 200, 500, 1000, 2000, 5000, 10000]
        })
    ]

    result, systems, dataset = run_experiments(
        cfg, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=False
    )

    M = get_metric_namespace_from_result(result)
    # print(M.output.shape)

    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical")





# %%
