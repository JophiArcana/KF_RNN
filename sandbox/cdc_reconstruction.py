import os

from transformers import GPT2Config

from infrastructure import loader
from infrastructure.experiment import *
from model.linear_system_distribution import *
from model.transformer import GPT2InContextKF

if __name__ == "__main__":
    os.chdir("..")

    base_exp_name = "CDCReconstruction"
    output_dir = "in_context"
    output_fname = "result"

    SHP = Namespace(S_D=10, I_D=1, O_D=5, input_enabled=False)
    args = loader.generate_args(SHP)

    context_length = 250
    n_train_systems = 40000
    n_test_systems = 3
    test_dataset_size = 2000

    args.model.model = GPT2InContextKF
    args.model.gpt2 = GPT2Config(
        n_positions=context_length,
        n_embd=256,
        n_layer=12,
        n_head=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    args.dataset.train = Namespace(
        dataset_size=1,
        total_sequence_length=context_length,
        system=Namespace(
            n_systems=n_train_systems,
            distribution=Namespace(
                sample_func=get_mop_sample_func("gaussian", "gaussian", 0.1, 0.1)
            )
        )
    )
    args.dataset.valid = args.dataset.test = Namespace(
        dataset_size=test_dataset_size,
        total_sequence_length=test_dataset_size * context_length,
        system=Namespace(
            n_systems=n_test_systems
        )
    )

    del args.train.warmup_duration
    args.train.epochs = 40000
    args.train.subsequence_length = context_length
    args.train.batch_size = 28
    args.train.iterations_per_epoch = 1

    args.train.optim_type = "Adam"
    args.train.max_lr = 3e-4
    args.train.lr_decay = 1.0
    args.train.weight_decay = 1e-2

    args.experiment.n_experiments = 1
    args.experiment.ensemble_size = 1
    args.experiment.exp_name = base_exp_name
    args.experiment.metrics = set() # {"validation"}

    configurations = []

    result = run_experiments(
        args, configurations, {
            "dir": output_dir,
            "fname": output_fname
        }, save_experiment=True
    )

    # M = get_metric_namespace_from_result(result)
    # print(M.output.shape)
    #
    # plot_experiment(f"{output_dir}/{base_exp_name}", configurations, result, loss_type="analytical")




