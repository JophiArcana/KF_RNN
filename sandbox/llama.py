from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from transformers import LlamaConfig, Dinov2Config

from infrastructure import utils
from model.base import Predictor
from model.transformer.dinov2_icpredictor import Dinov2AssociativeInContextPredictor
from model.transformer.llama_icpredictor import LlamaInContextPredictor, LlamaAssociativeInContextPredictor


if __name__ == "__main__":
    torch.manual_seed(1212)
    O_D = 6
    problem_shape = Namespace(
        environment=Namespace(observation=O_D),
        controller=Namespace()
    )
    model_shape = ()

    configuration = Dinov2Config(
        hidden_size=256,
        num_hidden_layers=5,
        num_attention_heads=8,
    )
    # configuration = LlamaConfig(
    #     hidden_size=256,
    #     num_hidden_layers=5,
    #     num_attention_heads=8,
    # )
    n_sys, B, L = 3, 5, 12

    MHP = Namespace(problem_shape=problem_shape, dinov2=configuration)
    # MHP = Namespace(problem_shape=problem_shape, llama=configuration)
    models = utils.multi_map(
        lambda _: Dinov2AssociativeInContextPredictor(MHP),
        # lambda _: LlamaInContextPredictor(MHP),
        # lambda _: LlamaAssociativeInContextPredictor(MHP),
        np.empty(model_shape), dtype=object,
    )

    dataset = TensorDict({
        "environment": {
            "observation": nn.Parameter(torch.randn((*model_shape, n_sys, B, L, O_D)))
        },
        "controller": {}
    }, batch_size=(*model_shape, n_sys, B, L))

    out = Predictor.run(*utils.stack_module_arr(models), dataset)["environment", "observation"]
    print(torch.autograd.grad(
        out[:, :, -1, :].norm(),
        dataset["environment", "observation"]
    )[0][0, 0])
    raise Exception()

