import os
import sys
import traceback
from argparse import Namespace
from typing import *

import numpy as np
import tensordict.utils
import torch
from matplotlib import colors
from matplotlib import pyplot as plt
from param import output
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
    output_dir = "adversarial_systems"
    os.makedirs(f"output/{output_dir}", exist_ok=True)

    system_type = "UpperTriangularA"
    save_fname = f"output/{output_dir}/{system_type}_log.pt"

    weights = TensorDict({
        (*k.split("."),): v
        for k, v in utils.torch_load(f"data/transformer_weights/{system_type}_state_dict.pt").items()
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

    if not os.path.exists(save_fname):
        n_test_traces = 64
        lsg = MOPDistribution("gaussian", "gaussian", 0.1, 0.1).sample(Namespace(
            S_D=S_D, problem_shape=problem_shape, auxiliary=Namespace()
        ), (n_adversarial_systems,))

        optimizer = torch.optim.SGD(lsg.parameters(), lr=1e-4)
        log = []
        for _ in range(100):
            try:
                ds = lsg.generate_dataset(n_test_traces, context_length)
                out = TensorDict.from_dict(
                    model(ds.flatten(0, 1).to_dict()),
                    batch_size=(n_adversarial_systems * n_test_traces, context_length)
                ).unflatten(0, (n_adversarial_systems, n_test_traces))

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
                log.append(TensorDict({
                    "system_params": lsg.td().apply(lambda t: t.detach().clone()),
                    "normalized_loss": normalized_loss
                }, batch_size=(n_adversarial_systems,)))

                torch.cuda.empty_cache()
            except RuntimeError:
                print(traceback.format_exc())
                break

        log = TensorDict.maybe_dense_stack(log, dim=0)
        torch.save(log, save_fname)
    else:
        log = utils.torch.load(save_fname)
    log = log.detach()


    # SECTION: Plotting code
    sys_idx = 0
    plt.rcParams['figure.figsize'] = (16.0, 18.0)  # set default size of plots
    fig, axs = plt.subplots(nrows=3, ncols=2)

    axs[0, 0].plot(log["normalized_loss"][:, sys_idx].cpu())
    axs[0, 0].set_title('Normalized analytical error')
    axs[0, 0].set_xlabel('iteration')
    axs[0, 0].set_ylabel(r'normalized_analytical_error: $\frac{\mathbb{E}_\tau[|| F_\theta(\tau) - \tau ||^2 - || KF(\tau) - \tau ||^2]}{\mathbb{E}_\tau[|| F_0(\tau) - \tau ||^2 - || KF(\tau) - \tau ||^2]} - 1$')
    axs[0, 0].set_yscale('log')


    F = log["system_params", "environment", "F"][:, sys_idx]
    F_eigenvalues = torch.linalg.eigvals(F)
    theta = torch.atan2(F_eigenvalues.imag, F_eigenvalues.real)
    theta = torch.linspace(
        theta.min() + -0.2 * (theta.max() - theta.min()),
        theta.min() + 1.2 * (theta.max() - theta.min()),
        100
    )
    axs[0, 1].plot(torch.cos(theta).cpu(), torch.sin(theta).cpu(), color='black', linestyle='--')

    colors = plt.cm.get_cmap("bone")(torch.linspace(0, 0.6, len(log)).cpu())
    for i in range(len(log)):
        axs[0, 1].scatter(F_eigenvalues[i].real.cpu(), F_eigenvalues[i].imag.cpu(), color=colors[i])
    axs[0, 1].set_title('State evolution eigenvalues')
    axs[0, 1].set_xlabel('real')
    axs[0, 1].set_ylabel('imag')


    H = log["system_params", "environment", "H"][:, sys_idx]
    axs[1, 0].plot(H.norm(dim=[-2, -1]).cpu(), color='black')
    axs[1, 0].set_title('Frobenius Norm of observation matrix')
    axs[1, 0].set_xlabel('iteration')
    axs[1, 0].set_ylabel(r'observation matrix norm: $|| H ||_F^2$')

    axs[1, 1].plot(torch.linalg.svdvals(H)[:, 0].cpu(), color='black')
    axs[1, 1].set_title('Spectral Norm of observation matrix')
    axs[1, 1].set_xlabel('iteration')
    axs[1, 1].set_ylabel(r'observation matrix norm: $|| H ||_2^2$')


    S_W = log["system_params", "environment", "S_W"][:, sys_idx]
    S_W_traces = utils.batch_trace(S_W)
    axs[2, 0].plot(S_W_traces.cpu(), color='black')
    axs[2, 0].set_title('Trace of evolution noise covariance')
    axs[2, 0].set_xlabel('iteration')
    axs[2, 0].set_ylabel(r'covariance_trace: $Tr(\Sigma_W)$')


    S_V = log["system_params", "environment", "S_V"][:, sys_idx]
    S_V_traces = utils.batch_trace(S_V)
    axs[2, 1].plot(S_V_traces.cpu(), color='black')
    axs[2, 1].set_title('Trace of observation noise covariance')
    axs[2, 1].set_xlabel('iteration')
    axs[2, 1].set_ylabel(r'covariance_trace: $Tr(\Sigma_V)$')

    plt.show()


    raise Exception()






