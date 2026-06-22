"""Animate the learned in-context (TTT) filter's impulse response as it adapts,
one frame per online step, against the fixed ground-truth Kalman IIR.

Same IR construction as ``CnnAnalyticalPredictor`` / ``scripts/ttt_impulse.py``;
here we snapshot the per-trajectory adapted ``(F, H, K)`` after *every* step of
``RnnInContextPredictor.forward``'s online loop and render one frame per step.
"""

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-kf-rnn")

import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import animation

# Prefer a bundled ffmpeg binary (so mp4 export works without a system ffmpeg).
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import ecliseutils as eu
from kf_rnn.infrastructure.config import EnvironmentShape, ProblemShape, SystemConfig
from kf_rnn.infrastructure.settings import OUTPUT_PATH
from kf_rnn.model.sequential import RnnInContextPredictor
from kf_rnn.system.linear_time_invariant import ContinuousDistribution


def kalman_ir(F: torch.Tensor, H: torch.Tensor, K: torch.Tensor, R: int) -> torch.Tensor:
    """Observation->observation impulse response, ``[R x O_D x O_D]`` (lag, out, in)."""
    S_D = F.shape[-1]
    powers = eu.pow_series(F @ (torch.eye(S_D, device=F.device) - K @ H), R)     # [R x S_D x S_D]
    return H @ powers @ (F @ K)                                                  # [R x O_D x O_D]


@torch.no_grad()
def adapted_ir_history(model: RnnInContextPredictor,
                       state0: torch.Tensor,        # [S_D]
                       observations: torch.Tensor,  # [L x O_D]
                       R: int,
) -> torch.Tensor:                                   # [L x R x O_D x O_D]
    """Replay the online loop for one trajectory, returning the learned filter's
    IR snapshot after each step. Mirrors ``RnnInContextPredictor.forward``."""
    theta = {k: v.detach().clone() for k, v in eu.td_items(eu.parameter_td(model)).items()}
    s = state0
    w = model.window
    ent_win = s[None].expand(w, *s.shape).clone()
    L = observations.shape[0]
    irs = []
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
        irs.append(kalman_ir(theta["F"], theta["H"], theta["K"], R))
    return torch.stack(irs, dim=0)                                               # [L x R x O_D x O_D]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Animate the learned in-context (TTT) filter's impulse response, "
                    "one frame per online step, vs. the ground-truth Kalman IIR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # SECTION: data / trajectory shape
    g_data = p.add_argument_group("data")
    g_data.add_argument("-N", "--traces", type=int, default=16, help="number of trajectories")
    g_data.add_argument("-L", "--length", type=int, default=1000, help="trace length (== #frames)")
    g_data.add_argument("-R", "--lags", type=int, default=32, help="number of impulse-response lags plotted")
    g_data.add_argument("--seed", type=int, default=0, help="random seed")

    # SECTION: model / test-time-training hyperparameters
    g_model = p.add_argument_group("model")
    g_model.add_argument("--s-d", type=int, default=6, help="state dimension")
    g_model.add_argument("--o-d", type=int, default=2, help="observation dimension")
    g_model.add_argument("--window", type=int, default=8, help="sliding-window length (>1 to train K)")
    g_model.add_argument("--n-steps", type=int, default=1, help="online SGD steps per timestep")
    g_model.add_argument("--step-size", type=float, default=None,
                         help="online learning rate (absolute); overrides --lr-scale if set")
    g_model.add_argument("--lr-scale", type=float, default=0.3,
                         help="online learning rate as a multiple of eps (used if --step-size unset)")
    g_model.add_argument("--initial-state-scale", type=float, default=1.0,
                         help="scale of the random initial state (only used in --train mode)")
    g_model.add_argument("--train", action="store_true",
                         help="run in training mode (random initial state) instead of eval (zeros)")

    # SECTION: continuous system distribution
    g_sys = p.add_argument_group("system")
    g_sys.add_argument("--eps", type=float, default=0.1, help="continuous discretization step")
    g_sys.add_argument("--w-std", type=float, default=1.0, help="process-noise std (scaled by eps)")
    g_sys.add_argument("--v-std", type=float, default=1.0, help="observation-noise std (scaled by eps)")
    g_sys.add_argument("--f-mode", type=str, default="gaussian", choices=("gaussian", "uniform"),
                       help="F sampling mode")
    g_sys.add_argument("--h-mode", type=str, default="gaussian", choices=("gaussian", "uniform"),
                       help="H sampling mode")

    # SECTION: rendering / output
    g_out = p.add_argument_group("output")
    g_out.add_argument("--fps", type=int, default=20, help="animation frames per second")
    g_out.add_argument("--bitrate", type=int, default=2400, help="mp4 bitrate")
    g_out.add_argument("--out", type=str, default=os.path.join(OUTPUT_PATH, "ttt_impulse_response.mp4"),
                       help="output video path (.mp4; falls back to .gif if ffmpeg is missing)")
    g_out.add_argument("--no-montage", action="store_true", help="skip the static sample-frame montage")

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

    F_t, H_t, K_t = lsg.environment.F, lsg.environment.H, lsg.environment.K
    truth_ir = kalman_ir(F_t, H_t, K_t, R).detach().cpu()            # [R x O_D x O_D]

    learned = torch.stack([
        adapted_ir_history(model, state0[n], observations[n], R) for n in range(N)
    ], dim=0)                                                        # [N x L x R x O_D x O_D]
    learned_mean = learned.mean(dim=0).cpu()                        # [L x R x O_D x O_D]
    learned_std = learned.std(dim=0).cpu()                          # [L x R x O_D x O_D]

    lags = torch.arange(1, R + 1).cpu().numpy()

    # Fixed per-subplot y-limits (truth + learned mean band across all frames).
    lo = torch.minimum((learned_mean - learned_std).min(dim=0).values, truth_ir)  # [R x O_D x O_D]
    hi = torch.maximum((learned_mean + learned_std).max(dim=0).values, truth_ir)
    lo = lo.amin(dim=0)                                              # [O_D x O_D]
    hi = hi.amax(dim=0)

    fig, axes = plt.subplots(O_D, O_D, figsize=(4.2 * O_D, 3.4 * O_D), sharex=True, squeeze=False)
    artists = {}
    for o in range(O_D):
        for i in range(O_D):
            ax = axes[o][i]
            ax.plot(lags, truth_ir[:, o, i].numpy(), color="C1", marker="o", ms=3, lw=1.6,
                    label="ground-truth Kalman")
            (line,) = ax.plot([], [], color="C0", marker=".", ms=4, lw=1.6, label="learned (TTT)")
            band = ax.fill_between(lags, lo[o, i].item(), hi[o, i].item(), color="C0", alpha=0.0)
            artists[(o, i)] = (line, band)
            ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
            pad = 0.08 * (hi[o, i].item() - lo[o, i].item() + 1e-9)
            ax.set_ylim(lo[o, i].item() - pad, hi[o, i].item() + pad)
            ax.set_title(f"obs out[{o}] <- in[{i}]")
            if o == O_D - 1:
                ax.set_xlabel("lag")
            if i == 0:
                ax.set_ylabel("IR coefficient")
    axes[0][0].legend(fontsize=8, loc="best")

    suptitle = fig.suptitle("", fontsize=11)

    def update(t: int):
        changed = [suptitle]
        for o in range(O_D):
            for i in range(O_D):
                line, band = artists[(o, i)]
                m = learned_mean[t, :, o, i].numpy()
                sd = learned_std[t, :, o, i].numpy()
                line.set_data(lags, m)
                band.remove()
                new_band = axes[o][i].fill_between(lags, m - sd, m + sd, color="C0", alpha=0.2)
                artists[(o, i)] = (line, new_band)
                changed += [line, new_band]
        suptitle.set_text(
            f"Impulse response over test-time adaptation  -  step {t + 1}/{L}\n"
            f"ContinuousDistribution(eps={eps}), window={model.window}, "
            f"step_size={model.step_size:.3g}, N={N}"
        )
        return changed

    anim = animation.FuncAnimation(fig, update, frames=L, interval=1000 // max(args.fps, 1), blit=False)

    out_path = args.out
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(out_path)[0]
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # Verification montage: a few representative frames spread across the trajectory.
    if not args.no_montage:
        sample_ts = sorted({int(f * (L - 1)) for f in (0.0, 0.01, 0.05, 0.15, 0.4, 1.0)})
        mfig, maxes = plt.subplots(len(sample_ts), O_D * O_D,
                                   figsize=(2.4 * O_D * O_D, 1.9 * len(sample_ts)), squeeze=False)
        for row, t in enumerate(sample_ts):
            for c, (o, i) in enumerate((o, i) for o in range(O_D) for i in range(O_D)):
                ax = maxes[row][c]
                ax.plot(lags, truth_ir[:, o, i].numpy(), color="C1", lw=1.2)
                ax.plot(lags, learned_mean[t, :, o, i].numpy(), color="C0", lw=1.2)
                ax.axhline(0.0, color="0.8", lw=0.6)
                ax.set_ylim(lo[o, i].item(), hi[o, i].item())
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"o{o}<-i{i}", fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"step {t + 1}", fontsize=8)
        mfig.suptitle("Learned (blue) vs ground-truth Kalman (orange) IR at sample steps", fontsize=10)
        mfig.tight_layout(rect=(0, 0, 1, 0.96))
        montage_path = f"{stem}_frames.png"
        mfig.savefig(montage_path, dpi=140)
        print("saved", montage_path)

    try:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=args.fps, bitrate=args.bitrate))
        print("saved", out_path)
    except Exception as exc:  # ffmpeg unavailable -> fall back to gif
        print(f"ffmpeg writer failed ({exc!r}); falling back to gif")
        out_path = f"{stem}.gif"
        anim.save(out_path, writer=animation.PillowWriter(fps=args.fps))
        print("saved", out_path)
