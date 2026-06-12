# %%
from kf_rnn.infrastructure import loader
from kf_rnn.infrastructure.config import MetricsConfig
from kf_rnn.infrastructure.experiment import run_experiments, plot_experiment, get_result_attr
from kf_rnn.model.convolutional import CnnLeastSquaresPredictor, CnnAnalyticalPredictor, CnnAnalyticalLeastSquaresPredictor
from kf_rnn.model.sequential import RnnHoKalmanAnalyticalPredictor, RnnHoKalmanAnalyticalLeastSquaresPredictor
from kf_rnn.system.linear_time_invariant import OrthonormalDistribution, MOPDistribution


if __name__ == "__main__":
    base_exp_name = "ImpulseResponseLengthAnalytical"
    output_dir = "system6_CNN"
    output_fname = "result"

    system2, cfg = loader.load_system_and_args("6dim_scalar_system_matrices")

    context_length = 100
    n_train_traces = 1
    n_valid_traces = 100

    S_D = cfg.system.S_D
    cfg.dataset.n_traces.reset(train=n_train_traces, valid=n_valid_traces, test=n_valid_traces)
    cfg.dataset.total_sequence_length.reset(train=n_train_traces * context_length, valid=n_valid_traces * context_length, test=n_valid_traces * context_length)
    cfg.training.sampling.batch_size = 4096
    cfg.experiment.metrics = MetricsConfig(
        training={"validation_analytical",},
        testing={"al", "il", "l", "eil",},
    )
    cfg.experiment.checkpoint_frequency = 10000
    cfg.experiment.print_frequency = 1
    cfg.experiment.exp_name = base_exp_name

    max_ir_length = 40
    configurations = [
        ("model", {
            "model": [
                CnnLeastSquaresPredictor.Config(),
                CnnAnalyticalPredictor.Config(),
                CnnAnalyticalLeastSquaresPredictor.Config(),
                # RnnHoKalmanAnalyticalPredictor.Config(S_D=S_D),
                # RnnHoKalmanAnalyticalLeastSquaresPredictor.Config(S_D=S_D),
            ]
        }),
        ("ir_length", {
            "model.ir_length": list(range(1, max_ir_length + 1))
        })
    ]

    result, _ = run_experiments(
        cfg, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=False,
    )
    plot_experiment(
        f"{output_dir}/{base_exp_name}", configurations, result,
        loss_type="analytical", lstsq=False, xscale="linear",
    )




# %%
