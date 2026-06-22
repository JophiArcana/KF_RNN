"""Impulse response of the learned in-context (TTT) filter vs. the ground-truth
Kalman filter's IIR.

The model's ``_forward`` is the Kalman recurrence
    x_t^- = F x_{t-1}^+ ,  y_hat_t = H x_t^- ,  x_t^+ = (I - K H) x_t^- + K y_t ,
so the observation->observation impulse response of the (adapted) filter is the
same closed form used by ``CnnAnalyticalPredictor``:
    IR[r] = H ((I - K H) F)^r ... = H (F (I - K H))^r (F K)            # [O_D x O_D]
evaluated either at the system's true (F, H, K) (ground truth Kalman) or at the
per-trajectory test-time-adapted (F, H, K) of the model (learned filter).
"""

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-kf-rnn")

import torch
from matplotlib import pyplot as plt
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.infrastructure.config import EnvironmentShape, ProblemShape, SystemConfig
from kf_rnn.infrastructure.settings import OUTPUT_PATH
from kf_rnn.model.sequential import RnnInContextPredictor
from kf_rnn.system.linear_time_invariant import ContinuousDistribution


def kalman_ir(F: torch.Tensor, H: torch.Tensor, K: torch.Tensor, R: int) -> torch.Tensor:
    """Observation->observation impulse response, ``[R x O_D x O_D]`` (lag, out, in).

    Same construction as ``CnnAnalyticalPredictor._analytical_initialization``:
    tap ``r`` is the linear map from ``y_{t-1-r}`` to ``y_hat_t``.
    """
    S_D = F.shape[-1]
    powers = eu.pow_series(F @ (torch.eye(S_D, device=F.device) - K @ H), R)     # [R x S_D x S_D]
    return H @ powers @ (F @ K)                                                  # [R x O_D x O_D]


@torch.no_grad()
def adapted_theta(model: RnnInContextPredictor,
                  state0: torch.Tensor,            # [S_D]
                  observations: torch.Tensor,      # [L x O_D]
) -> dict[str, torch.Tensor]:
    """Replay the online test-time loop for one trajectory and return the final
    adapted fast-weights. Mirrors ``RnnInContextPredictor.forward`` exactly."""
    theta = {k: v.detach().clone() for k, v in eu.td_items(eu.parameter_td(model)).items()}
    s = state0
    w = model.window
    ent_win = s[None].expand(w, *s.shape).clone()
    L = observations.shape[0]
    for t in range(L):
        out = torch.func.functional_call(model.cell, theta, (s, {}, observations[t]))
        s_post = out["environment", "state"]
        ent_win = torch.cat([ent_win[1:], s[None]], dim=0)
        s_start = ent_win[0].detach()
        w0 = max(0, t - w + 1)
        win_obs = observations[w0:t + 1]
        for _ in range(model.n_steps):
            grads = model._compute_grads(theta, s_start, {}, win_obs)
            theta = model._optimizer_step(theta, grads)
        s = s_post
    return theta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot the learned in-context (TTT) filter's impulse response "
                    "vs. the ground-truth Kalman IIR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_data = p.add_argument_group("data")
    g_data.add_argument("-N", "--traces", type=int, default=16, help="number of trajectories")
    g_data.add_argument("-L", "--length", type=int, default=200, help="trace length")
    g_data.add_argument("-R", "--lags", type=int, default=32, help="number of impulse-response lags plotted")
    g_data.add_argument("--seed", type=int, default=0, help="random seed")

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

    g_out = p.add_argument_group("output")
    g_out.add_argument("--out", type=str, default=os.path.join(OUTPUT_PATH, "ttt_impulse_response.png"),
                       help="output image path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    S_D, O_D = args.s_d, args.o_d
    eps = args.eps
    step_size = args.step_size if args.step_size is not None else args.lr_scale * eps

    problem_shape = ProblemShape(environment=EnvironmentShape(observation=O_D), controller={})
    system_cfg = SystemConfig(S_D=S_D, problem_shape=problem_shape)

    distribution = ContinuousDistribution(args.f_mode, args.h_mode, eps, args.w_std, args.v_std)
    lsg = distribution.sample(system_cfg, ())

    model = RnnInContextPredictor(RnnInContextPredictor.Config(
        problem_shape=problem_shape, S_D=S_D, n_steps=args.n_steps, step_size=step_size,
        window=args.window, initial_state_scale=args.initial_state_scale,
    ))
    if args.train:
        model.train()
    else:
        model.eval()

    N, L, R = args.traces, args.length, args.lags
    ds = lsg.generate_dataset(N, L)
    observations = ds["environment", "observation"]                   # [N x L x O_D]
    state0 = (args.initial_state_scale * torch.randn(N, S_D)) if args.train else torch.zeros(N, S_D)

    # Ground-truth steady-state Kalman filter.
    F_t, H_t, K_t = lsg.environment.F, lsg.environment.H, lsg.environment.K
    truth_ir = kalman_ir(F_t, H_t, K_t, R).detach().cpu()            # [R x O_D x O_D]

    # Learned filter: per-trajectory adapted weights -> per-trajectory IR.
    learned_irs = []
    for n in range(N):
        theta = adapted_theta(model, state0[n], observations[n])
        learned_irs.append(kalman_ir(theta["F"], theta["H"], theta["K"], R).cpu())
    learned_ir = torch.stack(learned_irs, dim=0)                      # [N x R x O_D x O_D]
    learned_mean = learned_ir.mean(dim=0)                            # [R x O_D x O_D]
    learned_std = learned_ir.std(dim=0)                             # [R x O_D x O_D]

    lags = torch.arange(1, R + 1).cpu().numpy()                       # tap r maps y_{t-1-r}, i.e. lag r+1

    fig, axes = plt.subplots(O_D, O_D, figsize=(4.2 * O_D, 3.4 * O_D), sharex=True, squeeze=False)
    for o in range(O_D):
        for i in range(O_D):
            ax = axes[o][i]
            t = truth_ir[:, o, i].numpy()
            m = learned_mean[:, o, i].numpy()
            sd = learned_std[:, o, i].numpy()
            ax.plot(lags, t, color="C1", marker="o", ms=3, lw=1.6, label="ground-truth Kalman")
            ax.plot(lags, m, color="C0", marker=".", ms=4, lw=1.6, label="learned (TTT)")
            ax.fill_between(lags, m - sd, m + sd, color="C0", alpha=0.2, label="learned ±1 std")
            ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
            ax.set_title(f"obs out[{o}] <- in[{i}]")
            if o == O_D - 1:
                ax.set_xlabel("lag")
            if i == 0:
                ax.set_ylabel("IR coefficient")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Impulse response: learned in-context filter vs. ground-truth Kalman\n"
        f"ContinuousDistribution(eps={eps}), window={model.window}, "
        f"step_size={model.step_size:.3g}, N={N}, L={L}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print("saved", out_path)

    # Quick scalar agreement summary.
    rel = (learned_mean - truth_ir).norm() / truth_ir.norm()
    print(f"relative IR error (||learned_mean - truth|| / ||truth||): {rel.item():.4f}")
