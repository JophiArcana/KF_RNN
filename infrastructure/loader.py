import copy
import dataclasses
from argparse import Namespace

import numpy as np
import torch
from dimarray import DimArray
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.config import schema
from infrastructure.config.bridge import _split_namespace, config_to_namespace
from infrastructure.settings import DEVICE
from system.linear_time_invariant import LTISystem


"""
Loading and generating args

The base argument Namespaces below are derived from the typed dataclass schemas
in ``infrastructure.config.schema`` so the schema remains the single source of
truth for defaults, while the runtime pipeline keeps consuming Namespaces.
"""
def _dataset_args_from_schema() -> Namespace:
    dc = schema.DataConfig()
    return Namespace(
        n_systems=_split_namespace(dc.n_systems),
        n_traces=_split_namespace(dc.n_traces),
        total_sequence_length=_split_namespace(dc.total_sequence_length),
    )

def _training_args_from_schema() -> Namespace:
    tc = schema.TrainConfig()
    return Namespace(
        sampling=Namespace(**dataclasses.asdict(tc.sampling)),
        optimizer=Namespace(**dataclasses.asdict(tc.optimizer)),
        scheduler=Namespace(**dataclasses.asdict(tc.scheduler)),
        loss=tc.loss,
        control_coefficient=tc.control_coefficient,
        ignore_initial=tc.ignore_initial,
    )

def _experiment_args_from_schema() -> Namespace:
    rc = schema.RuntimeConfig()
    return Namespace(
        n_experiments=rc.n_experiments,
        ensemble_size=rc.ensemble_size,
        backup_frequency=rc.backup_frequency,
        checkpoint_frequency=rc.checkpoint_frequency,
        print_frequency=rc.print_frequency,
        debug=rc.debug,
        split_size=rc.split_size,
    )

BaseDatasetArgs = _dataset_args_from_schema()
BaseTrainingArgs = _training_args_from_schema()
BaseExperimentArgs = _experiment_args_from_schema()

def args_from(HP: Namespace):
    HP.system = utils.process_defaulting_roots(HP.system)
    HP.dataset = utils.process_defaulting_roots(HP.dataset)
    return HP

def args_from_config(cfg: schema.ExperimentConfig) -> Namespace:
    """Build a runtime Namespace HP from a typed ``ExperimentConfig``."""
    return args_from(config_to_namespace(cfg))

def load_system_and_args(folder: str):
    A = torch.Tensor(np.loadtxt(f"{folder}/A.out", delimiter=",")).to(DEVICE)
    B = torch.Tensor(np.loadtxt(f"{folder}/B.out", delimiter=","))[:, None].to(DEVICE)
    C = torch.Tensor(np.loadtxt(f"{folder}/C.out", delimiter=","))[None].to(DEVICE)
    input_enabled = not bool(torch.all(torch.isclose(B, torch.zeros_like(B))))

    S_D = A.shape[0]
    O_D = C.shape[0]
    I_D = B.shape[1]

    noise_block = torch.Tensor(np.loadtxt(f"{folder}/noise_block.out", delimiter=",")).to(DEVICE)
    W = noise_block[:S_D, :S_D]
    V = noise_block[S_D:, S_D:]

    sqrt_W, sqrt_V = utils.sqrtm(W), utils.sqrtm(V)

    problem_shape = Namespace(
        environment=Namespace(observation=O_D),
        controller=Namespace(input=I_D) if input_enabled else Namespace(),
    )
    auxiliary = Namespace()
    settings = Namespace(include_analytical=True,)
    args = utils.deepcopy_namespace(Namespace(
        system=Namespace(
            S_D=S_D,
            problem_shape=problem_shape,
            auxiliary=auxiliary,
            settings=settings,
        ),
        dataset=BaseDatasetArgs,
        model=Namespace(problem_shape=problem_shape),
        training=BaseTrainingArgs,
        experiment=BaseExperimentArgs
    ))
    args.dataset.n_systems.train = 1
    args.experiment.n_experiments = 1

    system_group = LTISystem(args.system, TensorDict.from_dict({"environment": {
        "F": A, "B": TensorDict({"input": B}, batch_size=()), "H": C, "sqrt_S_W": sqrt_W, "sqrt_S_V": sqrt_V
    }}, batch_size=()).expand(args.dataset.n_systems.train, args.experiment.n_experiments))
    return {"train": DimArray(utils.array_of(system_group), dims=[])}, args_from(copy.deepcopy(args))


def generate_args(shp: Namespace) -> Namespace:
    for k in ("auxiliary", "settings",):
        if not hasattr(shp, k):
            setattr(shp, k, Namespace())
    return args_from(utils.deepcopy_namespace(Namespace(
        system=shp,
        dataset=BaseDatasetArgs,
        model=Namespace(problem_shape=shp.problem_shape),
        training=BaseTrainingArgs,
        experiment=BaseExperimentArgs,
    )))




