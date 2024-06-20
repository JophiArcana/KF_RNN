import copy
from argparse import Namespace

import numpy as np
import torch
from dimarray import DimArray
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.settings import DEVICE
from system.linear_time_invariant import LinearSystemGroup


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
        total_sequence_length=800000,
        sequence_buffer=50
    ),
    impulse_length=32
)
BaseTrainArgs = Namespace(
    # Batch sampling
    subsequence_length=16,
    batch_size=128,

    # Optimizer
    optim_type="Adam",          # {"GD", "SGD", "SGDMomentum", "Adam"}
    scheduler="exponential",    # {"exponential", "cosine"}
    momentum=0.9,
    min_lr=1.e-6,
    max_lr=2.e-2,
    T_0=10,
    T_mult=2,
    num_restarts=8,
    warmup_duration=100,
    weight_decay=0.0,
    lr_decay=0.995,

    # Iteration
    epochs=2500,
    iterations_per_epoch=20
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

    args = copy.deepcopy(Namespace(
        system=Namespace(
            S_D=S_D,
            I_D=I_D,
            O_D=O_D,
            SNR=None,
            input_enabled=input_enabled
        ),
        model=Namespace(
            I_D=I_D,
            O_D=O_D,
            input_enabled=input_enabled
        ),
        train=BaseTrainArgs,
        dataset=BaseDatasetArgs,
        experiment=BaseExperimentArgs
    ))

    args.dataset.train.system.n_systems = 1
    args.experiment.n_experiments = 1

    system_group = LinearSystemGroup(
        dict(TensorDict({
            "F": A, "B": B, "H": C, "sqrt_S_W": sqrt_W, "sqrt_S_V": sqrt_V
        }, batch_size=()).expand(args.dataset.train.system.n_systems, args.experiment.n_experiments)),
        input_enabled
    )
    return {"train": DimArray(utils.array_of(system_group), dims=[])}, args

def generate_args(shp: Namespace) -> Namespace:
    return copy.deepcopy(Namespace(
        system=shp,
        model=Namespace(
            I_D=shp.I_D,
            O_D=shp.O_D,
            input_enabled=shp.input_enabled
        ),
        train=BaseTrainArgs,
        dataset=BaseDatasetArgs,
        experiment=BaseExperimentArgs
    ))



