import os
import sys
from argparse import Namespace
from typing import *

import numpy as np
import tensordict.utils
import torch
from matplotlib import colors
from matplotlib import pyplot as plt
from tensordict import TensorDict
from transformers import GPT2Config, TransfoXLConfig

from infrastructure import loader
from infrastructure import utils
from infrastructure.experiment import *
from infrastructure.settings import DEVICE
from infrastructure.utils import PTR
from model.base import Predictor
from model.convolutional import CnnPredictorLeastSquares
from model.sequential import RnnPredictorPretrainAnalytical
from model.transformer import GPT2InContextPredictor, TransformerXLInContextPredictor
from model.zero_predictor import ZeroPredictor
from system.linear_time_invariant import LTISystem, MOPDistribution


if __name__ == "__main__":
    system_type = "UpperTriangularA"
    weights = TensorDict({
        (*k.split("."),): v
        for k, v in torch.load(f"data/transformer_weights/{system_type}_state_dict.pt").items()
    }, batch_size=())

    S_D, O_D = 10, 5
    problem_shape = Namespace(
        environment=Namespace(observation=O_D),
        controller=Namespace()
    )

    # SECTION: Transformer architecture parameters
    context_length = 2048
    d_embed = 128
    n_layer = 12
    n_head = 8
    d_inner = 4 * d_embed

    model_hyperparameters = Namespace(
        problem_shape=problem_shape, gpt2=GPT2Config(
            n_positions=context_length,
            n_embd=d_embed,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=d_inner,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
        ), bias=True
    )
    model = GPT2InContextPredictor(model_hyperparameters)

    model.core.load_state_dict(utils.td_items(weights["_backbone"]))
    model.observation_in.data = weights["_read_in", "weight"]
    model.observation_bias.data = weights["_read_in", "bias"]
    model.observation_out.data = weights["_read_out", "weight"]
    model.observation_bias.data = weights["_read_out", "bias"]

    # SECTION: System setup
    n_adversarial_systems = 1
    test_dataset_size = 3
    lsg = MOPDistribution("gaussian", "gaussian", 0.1, 0.1).sample(Namespace(
        S_D=S_D, problem_shape=problem_shape, auxiliary=Namespace()
    ), (n_adversarial_systems,))

    optimizer = torch.optim.SGD(lsg.parameters(), lr=1e-4)
    for _ in range(100):
        ds = lsg.generate_dataset(test_dataset_size, context_length)
        out = TensorDict.from_dict(
            model(ds.flatten(0, 1).to_dict()),
            batch_size=(n_adversarial_systems * test_dataset_size, context_length)
        ).unflatten(0, (n_adversarial_systems, test_dataset_size))

        target = ("environment", "observation")
        loss = Predictor.evaluate_run(out[target], ds, target)
        zero_predictor_loss = utils.rgetattr(lsg.zero_predictor_loss, ".".join(target))
        irreducible_loss = utils.rgetattr(lsg.irreducible_loss, ".".join(target))

        normalized_loss = (loss - irreducible_loss) / (zero_predictor_loss - irreducible_loss)
        print(f"Normalized Loss: {normalized_loss.mean().item()}")

        optimizer.zero_grad()
        (-normalized_loss.sum()).backward()
        optimizer.step()

        lsg = LTISystem(lsg.problem_shape, lsg.auxiliary, lsg.td())

    raise Exception()



    for k, v in zip((p for p, _ in lsg.named_parameters()), torch.autograd.grad(
        normalized_loss.sum(),
        (*lsg.parameters(),)
    )):
        print(f"{k}: {v}")
        print()

    # out = utils.run_module_arr(
    #     *utils.stack_module_arr(utils.array_of(model)),
    #     ds.to_dict()
    # )

    valid_dataset_size = 256
    test_dataset_size = 256

    raise Exception()






