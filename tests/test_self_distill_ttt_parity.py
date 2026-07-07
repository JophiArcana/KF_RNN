"""Numerical parity: the new ``SelfDistillTTTStage`` (static-forward
``RnnSelfDistillTTTPredictor``) reproduces, step for step, the online
self-distillation loop of the original fast-weight ``RnnSelfDistillPredictor``.

The reference loop below is the same online recurrence the (now deleted) driver
``_adapt_and_measure`` ran: predict with the current iterate, slide the window,
take ``n_steps`` gradient steps of ``RnnSelfDistillPredictor._window_loss`` /
``_optimizer_step``, and Polyak-tail-average the reported filter. Both models are
built with identical config and identical initial ``(F, H, K)``, so the reported
filter after every online step must match to tight tolerance.

Run: PYTHONPATH=. python tests/test_self_distill_ttt_parity.py
"""
import warnings
warnings.filterwarnings("ignore")

import torch

import ecliseutils as eu
from kf_rnn.infrastructure.config.schema import EnvironmentShape, ProblemShape, TrainConfig
from kf_rnn.infrastructure.experiment.stages import TrainingContext
from kf_rnn.infrastructure.experiment.training import SelfDistillTTTStage, _build_model_ensemble
from kf_rnn.model.sequential.rnn_ttt import RnnSelfDistillPredictor, RnnSelfDistillTTTPredictor
from types import SimpleNamespace
from tensordict import TensorDict


def _reference_trajectory(ref_model, init_td, observations, L):
    """Original fast-weight online loop; returns the reported (Polyak) filter after
    each online step as a list of ``{F,H,K}`` dicts (length ``L``)."""
    theta = {k: init_td[k].detach().clone() for k in ("F", "H", "K")}
    S_D = theta["F"].shape[-1]
    s = torch.zeros(S_D)
    w = ref_model.window
    ent_win = s[None].expand(w, S_D).clone()
    n0 = int(ref_model.polyak_burnin * L) if ref_model.polyak_burnin >= 0.0 else None
    theta_sum, count = None, 0
    reported = []
    for t in range(L):
        out = torch.func.functional_call(ref_model.cell, theta, (s, {}, observations[t]))
        s_post = out["environment", "state"]
        ent_win = torch.cat([ent_win[1:], s[None]], dim=0)
        s_start = ent_win[0].detach()
        w0 = max(0, t - w + 1)
        win_obs = observations[w0:t + 1]
        for _ in range(ref_model.n_steps):
            grads = ref_model._compute_grads(theta, s_start, {}, win_obs)
            theta = ref_model._optimizer_step(theta, grads, t)
        s = s_post
        if n0 is not None and t >= n0:
            if theta_sum is None:
                theta_sum = {k: v.clone() for k, v in theta.items()}
            else:
                for k, v in theta.items():
                    theta_sum[k] = theta_sum[k] + v
            count += 1
            bar = {k: theta_sum[k] / count for k in ("F", "H", "K")}
        else:
            bar = theta
        reported.append({k: bar[k].detach().clone() for k in ("F", "H", "K")})
    return reported


def _stage_trajectory(cfg, init_td, observations, L):
    """New stage online loop; returns the reported (model) filter after each step."""
    O_D = observations.shape[-1]
    model_pair = _build_model_ensemble(cfg, (1, 1), init_td)
    _, stacked = model_pair
    train_ds = TensorDict.from_dict(
        {"environment": {"observation": observations.reshape(1, 1, 1, 1, L, O_D)}, "controller": {}},
        batch_size=(1, 1, 1, 1, L),
    )
    exclusive = SimpleNamespace(train_info=SimpleNamespace(dataset=train_ds))
    stage = SelfDistillTTTStage()
    ctx = TrainingContext(TrainConfig(), exclusive, model_pair)
    reported = []
    while not stage.is_done(ctx):
        stage.step(ctx)
        reported.append({k: stacked[k][0, 0].detach().clone() for k in ("F", "H", "K")})
        stage.t += 1
    return reported


def _run_case(name, method_cfg, S_D=3, O_D=2, L=18, window=3, seed=0):
    torch.manual_seed(seed)
    problem_shape = ProblemShape(environment=EnvironmentShape(observation=O_D), controller={})
    common = dict(problem_shape=problem_shape, S_D=S_D, n_steps=1, step_size=0.05, window=window)

    ttt_cfg = RnnSelfDistillTTTPredictor.Config(**common, **method_cfg)
    # Build the TTT model to obtain the shared initial (F, H, K) (incl. any M1 H=0 /
    # frozen-K tweaks applied in __init__), then mirror it into the reference model.
    torch.manual_seed(seed)
    ttt_model = ttt_cfg.cls(ttt_cfg)
    init_td = eu.parameter_td(ttt_model).detach().clone()

    ref_cfg = RnnSelfDistillPredictor.Config(**common, **{
        k: v for k, v in method_cfg.items()
        if k not in ("h_init", "frozen_k_random")
    })
    ref_model = RnnSelfDistillPredictor(ref_cfg)
    ref_model.eval()

    observations = torch.randn(L, O_D)
    ref_traj = _reference_trajectory(ref_model, init_td, observations, L)
    torch.manual_seed(seed)  # re-seed: _build_model_ensemble draws (overwritten by init_td)
    stage_traj = _stage_trajectory(ttt_cfg, init_td, observations, L)

    max_err = 0.0
    for t, (a, b) in enumerate(zip(ref_traj, stage_traj)):
        for k in ("F", "H", "K"):
            d = (a[k] - b[k]).abs().max().item()
            max_err = max(max_err, d)
    assert max_err < 1e-6, f"{name}: reported-filter mismatch, max|Δ|={max_err:.3e}"
    print(f"OK  {name:32s} max|Δreported|={max_err:.2e}")


def test_cadence_and_metric() -> None:
    """Drive the stage through the real engine (`_run_training`) and check the
    configurable metric cadence + guarded ``sd_analytical`` / ``sd_frac_stable``
    metrics: rows are recorded only every ``metric_frequency`` steps (+ a final
    row), each carries the online ``step`` index, and the analytical error is
    per-trajectory (one value per ensemble member)."""
    import numpy as np
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    from kf_rnn.infrastructure.config.schema import (
        ExperimentConfig, RuntimeConfig, MetricsConfig, SystemConfig,
    )
    from kf_rnn.infrastructure.experiment.results import InfoGrid
    from kf_rnn.infrastructure.experiment.training import _run_training
    from kf_rnn.system.linear_time_invariant import ContinuousDistribution
    from ecliseutils.labeled_array import LabeledArray

    S_D, O_D, N, L, k = 3, 2, 4, 12, 4
    problem_shape = ProblemShape(environment=EnvironmentShape(observation=O_D), controller={})
    lsg = ContinuousDistribution("gaussian", "gaussian", 0.1, 1.0, 1.0).sample(
        SystemConfig(S_D=S_D, problem_shape=problem_shape), (1, 1))
    obs = lsg.generate_dataset(N, L)["environment", "observation"]      # [1 x 1 x N x L x O_D]
    train_ds = TensorDict.from_dict(
        {"environment": {"observation": obs.reshape(1, N, 1, 1, L, O_D)}, "controller": {}},
        batch_size=(1, N, 1, 1, L),
    )

    def wrap(o):
        a = np.empty((), dtype=object)
        a[()] = o
        return LabeledArray(a, ())

    grid = InfoGrid(systems=wrap(lsg), system_params=wrap(None), dataset=wrap(train_ds))
    info = SimpleNamespace(train=grid, valid=grid)
    exclusive = SimpleNamespace(info=info, train_info=info.train[()], n_train_systems=1)

    cfg = RnnSelfDistillTTTPredictor.Config(
        problem_shape=problem_shape, S_D=S_D, window=3, step_size=0.05,
        alpha=0.0, beta0=1.0, adapt_keys=("F", "H", "K"),
    )
    init_td = eu.parameter_td(cfg.cls(cfg)).detach().clone()
    ref, stacked = _build_model_ensemble(cfg, (1, N), init_td)
    # stack_module_arr may land params on the settings device (cuda); pin to cpu.
    model_pair = (ref.to("cpu"), stacked.to("cpu"))

    HP = ExperimentConfig(
        model=cfg,
        experiment=RuntimeConfig(
            n_experiments=1, ensemble_size=N, metric_frequency=k,
            metrics=MetricsConfig(training={"sd_analytical", "sd_frac_stable"}),
        ),
    )
    output = _run_training(HP, exclusive, model_pair, [])

    assert "sd_analytical" in output and "step" in output, "expected sd metrics + step axis in output"
    steps = output["step"][0, 0].tolist()
    expected = [float(t) for t in range(0, L, k)] + [float(L)]
    assert steps == expected, f"cadence mismatch: {steps} != {expected}"
    # sd_analytical is per-trajectory (ensemble dim preserved): [N_exp x E x n_rec x S].
    assert output["sd_analytical"].shape[:2] == (1, N), output["sd_analytical"].shape
    frac = output["sd_frac_stable"]
    assert bool(((frac == 0.0) | (frac == 1.0) | frac.isnan()).all()), "frac_stable must be a 0/1 indicator"
    print(f"OK  cadence+metric  steps={steps}  sd_analytical.shape={tuple(output['sd_analytical'].shape)}")


def test_parity() -> None:
    torch.set_default_dtype(torch.float64)
    # M4: pure a-priori, adapt F,H,K (stable).
    _run_case("M4 a-priori", dict(alpha=0.0, beta0=1.0, beta2=0.0, adapt_keys=("F", "H", "K")))
    # M3+post: latent SD + anchor + a-posteriori + weight decay + ladder depth 2.
    _run_case("M3+post ladder2 wd", dict(alpha=1.0, beta0=0.05, beta2=0.05,
                                         adapt_keys=("F", "H", "K"), weight_decay=1e-3, sd_horizon=2))
    # Polyak tail averaging + decaying step.
    _run_case("M4 decay+polyak", dict(alpha=0.0, beta0=1.0, beta2=0.0, adapt_keys=("F", "H", "K"),
                                      step_decay=0.5, polyak_burnin=0.3))
    # M1 raw-injection: H=0, adapt only F (exercises the h_init tweak + frozen H/K).
    _run_case("M1 raw-injection", dict(alpha=1.0, beta0=0.0, beta2=0.0, adapt_keys=("F",),
                                       h_init="zero"), L=10)


if __name__ == "__main__":
    test_parity()
    test_cadence_and_metric()
    print("ALL_OK")
