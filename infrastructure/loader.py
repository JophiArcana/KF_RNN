from argparse import Namespace

import numpy as np
import torch
from dimarray import DimArray
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.settings import DEVICE
from system.linear_time_invariant import LTISystem


"""
Loading and generating args
"""
BaseDatasetArgs = Namespace(
    # Dataset
    train=Namespace(
        dataset_size=1,
        total_sequence_length=2000,
        system=Namespace(
            n_systems=1,
            distribution=...
        )
    ),
    valid=Namespace(
        dataset_size=100,
        total_sequence_length=20000
    ),
    test=Namespace(
        dataset_size=500,
        total_sequence_length=800000
    ),
    impulse_length=32
)
BaseTrainArgs = Namespace(
    # Batch sampling
    sampling=Namespace(
        method="subsequence_padded",
        batch_size=128,
        subsequence_length=16
    ),

    # Optimizer
    optimizer=Namespace(
        type="Adam",                        # {"SGD", "Adam"}
        max_lr=2e-2, min_lr=1e-6,
        weight_decay=0.0,

        momentum=0.9,                       # SECTION: Used for Adam and SGD
    ),

    # Scheduler
    scheduler=Namespace(
        type="exponential",                 # {"exponential", "cosine"}
        warmup_duration=100,

        epochs=2500, lr_decay=0.995,        # SECTION: Used for exponential scheduler
        T_0=10, T_mult=2, num_restarts=8,   # SECTION: Used for cosine scheduler
    ),

    # Iteration
    iterations_per_epoch=20,

    # Loss
    control_coefficient=1.0,
)
BaseExperimentArgs = Namespace(
    n_experiments=1,
    ensemble_size=32,
    backup_frequency=10
)

def load_system_and_args(folder: str):
    A = torch.Tensor(np.loadtxt(f"{folder}/A.out", delimiter=",")).to(DEVICE)
    B = torch.Tensor(np.loadtxt(f"{folder}/B.out", delimiter=","))[:, None].to(DEVICE)
    C = torch.Tensor(np.loadtxt(f"{folder}/C.out", delimiter=","))[None].to(DEVICE)
    input_enabled = not bool(torch.all(torch.isclose(B, torch.zeros_like(B))))

    S_D = A.shape[0]
    O_D = C.shape[0]
    I_D = B.shape[1]

    noise_block = torch.Tensor(np.loadtxt(f"{folder}/noise_block.out", delimiter=",")).to(DEVICE)
    W = noise_block[0:S_D, 0:S_D]
    V = noise_block[S_D:S_D + O_D, S_D:S_D + O_D]

    L_W, V_W = torch.linalg.eig(W)
    sqrt_W = torch.real(V_W @ torch.diag(torch.sqrt(L_W)) @ V_W.mT)
    L_V, V_V = torch.linalg.eig(V)
    sqrt_V = torch.real(V_V @ torch.diag(torch.sqrt(L_V)) @ V_V.mT)

    problem_shape = Namespace(
        environment=Namespace(observation=O_D),
        controller=Namespace(input=I_D) if input_enabled else Namespace(),
    )
    args = utils.deepcopy_namespace(Namespace(
        system=Namespace(
            S_D=S_D,
            problem_shape=problem_shape
        ),
        model=Namespace(problem_shape=problem_shape),
        train=BaseTrainArgs,
        dataset=BaseDatasetArgs,
        experiment=BaseExperimentArgs
    ))

    args.dataset.train.system.n_systems = 1
    args.experiment.n_experiments = 1

    system_group = LTISystem(args.system, TensorDict.from_dict({"environment": {
        "F": A, "B": TensorDict({"input": B}, batch_size=()), "H": C, "sqrt_S_W": sqrt_W, "sqrt_S_V": sqrt_V
    }}, batch_size=()).expand(args.dataset.train.system.n_systems, args.experiment.n_experiments))
    return {"train": DimArray(utils.array_of(system_group), dims=[])}, args

def generate_args(shp: Namespace) -> Namespace:
    return utils.deepcopy_namespace(Namespace(
        system=shp,
        model=Namespace(problem_shape=shp.problem_shape),
        train=BaseTrainArgs,
        dataset=BaseDatasetArgs,
        experiment=BaseExperimentArgs
    ))




