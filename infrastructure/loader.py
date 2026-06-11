import copy
from argparse import Namespace

import numpy as np
import torch
from tensordict import TensorDict

from infrastructure.labeled_array import LabeledArray

from infrastructure import utils
from infrastructure.config import schema
from infrastructure.config.bridge import config_to_namespace
from infrastructure.settings import DEVICE
from system.linear_time_invariant import LTISystem


"""
Loading and generating args

The base argument Namespaces below are derived from a single default
``ExperimentConfig`` via the same ``config_to_namespace`` bridge the runtime uses,
so the typed schema in ``infrastructure.config.schema`` remains the single source
of truth for defaults while the pipeline keeps consuming Namespaces.
"""
_BASE_ARGS = config_to_namespace(schema.ExperimentConfig())

BaseDatasetArgs = _BASE_ARGS.dataset
BaseTrainingArgs = _BASE_ARGS.training
BaseExperimentArgs = _BASE_ARGS.experiment
# The historical base omitted ``exp_name`` (callers set it per experiment); drop
# the schema default so the shape matches and a missing override surfaces loudly.
if hasattr(BaseExperimentArgs, "exp_name"):
    delattr(BaseExperimentArgs, "exp_name")

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
    return {"train": LabeledArray(utils.array_of(system_group), ())}, args_from(copy.deepcopy(args))


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




