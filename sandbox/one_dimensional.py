from argparse import Namespace

import torch
from matplotlib import pyplot as plt

from infrastructure import loader
from infrastructure import utils
from infrastructure.experiment import *
from infrastructure.settings import DEVICE
from infrastructure.utils import PTR
from model.convolutional import CnnLeastSquaresPredictor
from model.sequential import RnnKalmanInitializedPredictor
from model.transformer import (
    GPT2InContextPredictor,
    # GPT2AssociativeInContextPredictor,
    Dinov2AssociativeInContextPredictor,
    Mamba2InContextPredictor,
)
from model.zero_predictor import ZeroPredictor
from system.linear_time_invariant import LTISystem, MOPDistribution, OrthonormalDistribution


if __name__ == "__main__":

    dist = OrthonormalDistribution()
    S_D = O_D = 3
    SHP = Namespace(
        distribution=dist, S_D=S_D,
        problem_shape=Namespace(
            environment=Namespace(observation=O_D,),
            controller=Namespace(),
        ), auxiliary=Namespace(), settings=Namespace(include_analytical=False),
    )

    n_traces = 1
    context_length = 100

    sg = dist.sample(SHP, ())
    ds = sg.generate_dataset(n_traces, context_length)

    y = ds["environment", "observation"][0]
    t = torch.arange(context_length)
    for i in range(S_D):
        plt.plot(t.numpy(force=True), y[..., i].numpy(force=True))

    plt.show()




