import os

import numpy as np
import torch
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.infrastructure.config.schema import (
    EnvironmentShape,
    ExperimentConfig,
    ProblemShape,
    Split,
    SystemConfig,
)
from ecliseutils.labeled_array import LabeledArray
from kf_rnn.infrastructure.settings import DEVICE, DATA_PATH
from kf_rnn.system.linear_time_invariant import LTISystem


def load_system_and_args(folder: str) -> tuple[dict[str, LabeledArray], ExperimentConfig]:
    """Load a saved LTI system (A/B/C + noise block) and a matching typed config.

    Returns ``(systems, cfg)`` where ``systems`` plugs into ``run_experiments``'s
    ``systems=`` argument and ``cfg`` is an ``ExperimentConfig`` whose problem
    shape matches the loaded system. The model branch is left for the caller.
    """
    # Resolve relative dataset folders under the repo-root ``data/`` directory so
    # loads are cwd-independent; absolute paths pass through unchanged.
    if not os.path.isabs(folder):
        folder = os.path.join(DATA_PATH, folder)
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

    sqrt_W, sqrt_V = eu.sqrtm(W), eu.sqrtm(V)

    cfg = ExperimentConfig(
        problem=ProblemShape(
            environment=EnvironmentShape(observation=O_D),
            controller={"input": I_D} if input_enabled else {},
        ),
        system=SystemConfig(S_D=S_D),
    )
    cfg.dataset.n_systems = Split(train=1)
    cfg.experiment.n_experiments = 1

    system_group = LTISystem(cfg.system, TensorDict.from_dict({"environment": {
        "F": A, "B": TensorDict({"input": B}, batch_size=()), "H": C, "sqrt_S_W": sqrt_W, "sqrt_S_V": sqrt_V
    }}, batch_size=()).expand(cfg.dataset.n_systems.train, cfg.experiment.n_experiments))
    return {"train": LabeledArray(eu.array_of(system_group), ())}, cfg
