#%%
# This line needs to be added since some terminals will not recognize the current directory
import sys
sys.path.append("/home/wenliao/KF_RNN")

from infrastructure import loader
from infrastructure.experiment import *
from model.convolutional import *
from model.sequential import *


if __name__ == "__main__":
    base_exp_name = "SingleTrace"
    output_dir = "system2_CNN"
    output_fname = "result"


    system2, args = loader.load_system_and_args("data/2dim_scalar_system_matrices")
    args.model.S_D = args.system.S_D

    args.training.sampling.batch_size = 64
    args.training.sampling.subsequence_length = 32

    # args.train.optimizer.type = "LBFGS"
    # args.train.optimizer.max_lr = 1.0

    # del args.train.scheduler.warmup_duration
    args.training.scheduler.epochs = 2000


    args.experiment.exp_name = base_exp_name
    args.experiment.ignore_metrics = {"impulse_target"}
    # args.experiment.metrics = {"validation_analytical"}

    configurations = [
        ("model", {
            # "model.model": [CnnPredictor, CnnPredictorLeastSquares]
            "model.model": [CnnPredictorLeastSquares]
        }),
        ("total_trace_length", {
            "dataset.train.total_sequence_length": [100, 200, 500, 1000, 2000, 5000, 10000]
        })
    ]

    result, systems, dataset = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, system2, save_experiment=False
    )

    M = get_metric_namespace_from_result(result)
    # print(M.output.shape)

    plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical")





# %%
