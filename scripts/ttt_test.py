import argparse
import os
import sys
import traceback
from typing import *

import numpy as np
import tensordict.utils
import torch
from matplotlib import colors
from matplotlib import pyplot as plt
from param import output
from tensordict import TensorDict
from transformers import GPT2Config, TransfoXLConfig

from kf_rnn.infrastructure import loader
import ecliseutils as eu
from kf_rnn.infrastructure.config import EnvironmentShape, ProblemShape, SystemConfig
from kf_rnn.infrastructure.experiment import *
from kf_rnn.infrastructure.settings import DEVICE, OUTPUT_PATH, DATA_PATH
from kf_rnn.model.base import Predictor
from kf_rnn.model.sequential import RnnInContextPredictor
from kf_rnn.system.linear_time_invariant import LTISystem, ContinuousDistribution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke test for RnnInContextPredictor: report the in-context loss "
                    "against the zero-predictor and irreducible baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_data = p.add_argument_group("data")
    g_data.add_argument("-N", "--traces", type=int, default=8, help="number of trajectories")
    g_data.add_argument("-L", "--length", type=int, default=100, help="trace length")
    g_data.add_argument("--seed", type=int, default=None, help="random seed (default: nondeterministic)")

    g_model = p.add_argument_group("model")
    g_model.add_argument("--s-d", type=int, default=6, help="state dimension")
    g_model.add_argument("--o-d", type=int, default=2, help="observation dimension")
    g_model.add_argument("--window", type=int, default=4, help="sliding-window length (>1 to train K)")
    g_model.add_argument("--n-steps", type=int, default=1, help="online SGD steps per timestep")
    g_model.add_argument("--step-size", type=float, default=None,
                         help="online learning rate (absolute); overrides --lr-scale if set")
    g_model.add_argument("--lr-scale", type=float, default=0.3,
                         help="online learning rate as a multiple of eps (used if --step-size unset)")
    g_model.add_argument("--initial-state-scale", type=float, default=1.0,
                         help="scale of the random initial state (only used in --train mode)")
    g_model.add_argument("--train", action="store_true",
                         help="run in training mode (random initial state) instead of eval (zeros)")

    g_sys = p.add_argument_group("system")
    g_sys.add_argument("--eps", type=float, default=0.1, help="continuous discretization step")
    g_sys.add_argument("--w-std", type=float, default=1.0, help="process-noise std (scaled by eps)")
    g_sys.add_argument("--v-std", type=float, default=1.0, help="observation-noise std (scaled by eps)")
    g_sys.add_argument("--f-mode", type=str, default="gaussian", choices=("gaussian", "uniform"),
                       help="F sampling mode")
    g_sys.add_argument("--h-mode", type=str, default="gaussian", choices=("gaussian", "uniform"),
                       help="H sampling mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    S_D, O_D = args.s_d, args.o_d
    eps = args.eps
    step_size = args.step_size if args.step_size is not None else args.lr_scale * eps

    problem_shape = ProblemShape(
        environment=EnvironmentShape(observation=O_D),
        controller={},
    )
    system_cfg = SystemConfig(S_D=S_D, problem_shape=problem_shape)

    # SECTION: Continuous (slowly-varying) system. Frequent observations of a
    # discretized continuous system keep consecutive observations stable, which is
    # the regime the in-context test-time update is meant for. The discretization
    # step ``eps`` also sets the natural scale of the online learning rate.
    distribution = ContinuousDistribution(args.f_mode, args.h_mode, eps, args.w_std, args.v_std)

    # SECTION: In-context predictor. ``window > 1`` is required for the Kalman gain
    # K to receive gradient (a 1-step window leaves K untrained, collapsing the
    # model onto the zero predictor), and the online step size is taken proportional
    # to the system's ``eps``.
    model_hyperparameters = RnnInContextPredictor.Config(
        problem_shape=problem_shape,
        S_D=S_D,
        n_steps=args.n_steps,
        step_size=step_size,
        window=args.window,
        initial_state_scale=args.initial_state_scale,
    )
    model = RnnInContextPredictor(model_hyperparameters)
    if args.train:
        model.train()
    else:
        model.eval()

    N, L = args.traces, args.length
    lsg = distribution.sample(system_cfg, ())
    ds = lsg.generate_dataset(N, L)

    out = TensorDict.from_dict(model(ds.to_dict()), batch_size=(N, L,))

    print(out)

    target = ("environment", "observation")
    loss = Predictor.evaluate_run(out[target], ds, target)
    zero_predictor_loss = eu.rgetattr(lsg.zero_predictor_loss, ".".join(target))
    irreducible_loss = eu.rgetattr(lsg.irreducible_loss, ".".join(target))

    loss_v = loss.item()
    zp_v = zero_predictor_loss.item()
    irr_v = irreducible_loss.item()

    print(f"ttt loss             {loss_v:.6e}")
    print(f"zero predictor loss  {zp_v:.6e}")
    print(f"irreducible loss     {irr_v:.6e}")

    # Where the in-context predictor sits between the zero-predictor ceiling and the
    # irreducible floor: 0% == matches zero predictor, 100% == matches irreducible.
    denom = zp_v - irr_v
    frac = (zp_v - loss_v) / denom if denom != 0 else float("nan")
    print(f"gap closed vs zero predictor: {100.0 * frac:.1f}%  "
          f"(beats zero predictor: {loss_v < zp_v})")