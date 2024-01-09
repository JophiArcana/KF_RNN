from argparse import Namespace
import copy
import numpy as np
import torch

from model.linear_system import LinearSystem


"""
Loading and generating args
"""
BaseTrainArgs = Namespace(
    # Dataset
    train_dataset_size=1,
    valid_dataset_size=100,
    total_train_sequence_length=2000,
    total_valid_sequence_length=20000,
    sequence_buffer=50,
    ir_length=32,

    # Batch sampling
    subsequence_length=16,
    batch_size=128,

    # Optimizer
    optim_type="Adam",  # {"GD", "SGD", "SGDMomentum", "Adam"}
    scheduler="cosine",
    momentum=0.9,
    min_lr=1.e-6,
    max_lr=2.e-2,
    T_0=10,
    T_mult=2,
    num_restarts=8,
    warmup_duration=100,
    weight_decay=0.1,
    lr_decay=0.995,

    # Iteration
    iterations_per_epoch=20
)
BaseExperimentArgs = Namespace(
    n_systems=1,
    ensemble_size=32,
    backup_frequency=5
)

def load_system_and_args(folder: str):
    A = torch.tensor(np.loadtxt(f'{folder}/A.out', delimiter=','))
    B = torch.tensor(np.loadtxt(f'{folder}/B.out', delimiter=','))[:, None]
    C = torch.tensor(np.loadtxt(f'{folder}/C.out', delimiter=','))[None]
    input_enabled = bool(torch.all(torch.isclose(B, torch.zeros_like(B))))

    S_D = A.shape[0]
    O_D = C.shape[0]
    I_D = B.shape[1]

    noise_block = torch.tensor(np.loadtxt(f'{folder}/noise_block.out', delimiter=','))
    W = noise_block[0:S_D, 0:S_D]
    V = noise_block[S_D:S_D + O_D, S_D:S_D + O_D]

    L_W, V_W = torch.linalg.eig(W)
    sqrt_W = torch.real(V_W @ torch.diag(torch.sqrt(L_W)) @ V_W.T)
    L_V, V_V = torch.linalg.eig(V)
    sqrt_V = torch.real(V_V @ torch.diag(torch.sqrt(L_V)) @ V_V.T)

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
        valid=Namespace(
            valid_dataset_size=500,
            valid_sequence_length=1600,
            sequence_buffer=50
        ),
        experiment=BaseExperimentArgs
    ))
    args.experiment.n_systems = 1
    return LinearSystem({'F': A, 'B': B, 'H': C, 'sqrt_S_W': sqrt_W, 'sqrt_S_V': sqrt_V}, input_enabled), args

def generate_systems_and_args(shp: Namespace, n_systems: int, **system_kwargs):
    systems = [LinearSystem.sample_stable_system(shp, **system_kwargs) for _ in range(n_systems)]
    args = copy.deepcopy(Namespace(
        system=shp,
        model=Namespace(
            I_D=shp.I_D,
            O_D=shp.O_D,
            input_enabled=shp.input_enabled
        ),
        train=BaseTrainArgs,
        valid=Namespace(
            valid_dataset_size=500,
            valid_sequence_length=1600,
            sequence_buffer=50
        ),
        experiment=BaseExperimentArgs
    ))
    args.experiment.n_systems = n_systems
    return systems, args




