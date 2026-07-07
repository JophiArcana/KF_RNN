"""Correctness of the LTISystem fast (parallel-scan) data generator.

``LTISystem.generate_dataset`` replaces the generic per-step rollout loop with a
single batched linear-Gaussian scan whenever the closed loop is linear (the LQE /
optimal-LQR case). Two checks:

1. Exact, same-noise parity: re-draw the exact noise the fast path consumed (seed
   replay) and run an independent naive Kalman/LQR loop with that noise; every
   output field must match to float64 tolerance. Covers no-control (LQE),
   with-control (the augmented BL / action math), multiple leading batch dims and
   the length-one edge case.
2. Statistical sanity: the empirical a-priori observation error of the generated
   data matches the closed-form irreducible floor, for both the fast path and the
   generic loop.

Run: PYTHONPATH=. python tests/test_fast_lti_generation.py
"""
import warnings
warnings.filterwarnings("ignore")

import torch
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.infrastructure.config.schema import (
    EnvironmentShape, ProblemShape, SystemConfig, SystemSettings, controller_dims,
)
from kf_rnn.system.base import SystemGroup
from kf_rnn.system.linear_time_invariant import ContinuousDistribution, MOPDistribution


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
torch.set_default_device("cpu")

_FIELDS = (
    "state", "observation", "noiseless_observation",
    "target_state_estimation", "target_observation_estimation",
)


def _lqe_system(S_D, O_D, group_shape):
    ps = ProblemShape(environment=EnvironmentShape(observation=O_D), controller={})
    cfg = SystemConfig(S_D=S_D, problem_shape=ps)
    dist = ContinuousDistribution("gaussian", "gaussian", 0.1, 1.0, 1.0)
    return dist.sample(cfg, group_shape)


def _controlled_system(S_D, O_D, I_D, group_shape):
    """A controlled LQG system with the analytical Kalman gain enabled manually.

    ``setup_analytical`` asserts no control channels, so controlled systems are
    built with ``include_analytical=False``; we add the steady-state Kalman gain
    afterwards so the fast path's controlled branch (BL feedback + actions) runs.
    """
    ps = ProblemShape(environment=EnvironmentShape(observation=O_D), controller={"input": I_D})
    cfg = SystemConfig(S_D=S_D, problem_shape=ps, settings=SystemSettings(include_analytical=False))
    dist = MOPDistribution("gaussian", "gaussian", 1.0, 1.0)
    lsg = dist.sample(cfg, group_shape)

    env = lsg.environment
    F, H, S_W, S_V = env.F, env.H, env.S_W, env.S_V
    Sii = eu.solve_discrete_are(F.mT, H.mT, S_W, S_V)
    S_pred = H @ Sii @ H.mT + S_V
    K = torch.linalg.solve(S_pred, (Sii @ H.mT).mT).mT
    env.register_buffer("K", K)
    env.include_analytical = True
    return lsg


def _apply_mat(t, M):
    # Apply M ([g x p x q]) to the last axis of t ([g x B x T x q]), broadcasting
    # over the trace axis -- identical to the fast path's per-timestep helper.
    return t @ M.reshape(*M.shape[:-2], 1, *M.shape[-2:]).mT


def _replay_noise(lsg, B, L):
    """Re-draw the exact noise tensors the fast path consumes, in the same order."""
    env = lsg.environment
    g = env.F.shape[:-2]
    S_D, O_D = env.S_D, env.O_D
    kw = dict(device=env.F.device, dtype=env.F.dtype)

    sqrt_S_state_inf = eu.sqrtm(env.S_state_inf)
    x0 = env.initial_state_scale * (torch.randn((*g, B, S_D), **kw) @ sqrt_S_state_inf.mT)
    v0 = torch.randn((*g, B, O_D), **kw) @ env.sqrt_S_V.mT
    if L > 1:
        W = _apply_mat(torch.randn((*g, B, L - 1, S_D), **kw), env.sqrt_S_W)
        Vn = _apply_mat(torch.randn((*g, B, L - 1, O_D), **kw), env.sqrt_S_V)
    else:
        W = Vn = None
    return x0, v0, W, Vn


def _naive_rollout(lsg, x0, v0, W, Vn):
    """Independent per-step Kalman/LQR loop mirroring environment.step + controller.act."""
    env = lsg.environment
    F, H, K = env.F, env.H, env.K
    cdims = controller_dims(lsg.hyperparameters.problem_shape)
    L = 1 if W is None else W.shape[-2] + 1

    y0 = x0 @ H.mT + v0
    xhat0 = y0 @ K.mT
    zero_y = torch.zeros_like(y0)
    cols = {"state": [x0], "observation": [y0], "noiseless_observation": [zero_y],
            "target_state_estimation": [xhat0], "target_observation_estimation": [zero_y]}
    acts = {k: [torch.zeros((*x0.shape[:-1], d), dtype=x0.dtype, device=x0.device)]
            for k, d in cdims.items()}

    x_prev, xhat_prev = x0, xhat0
    for t in range(1, L):
        w, v = W[..., t - 1, :], Vn[..., t - 1, :]
        action = {k: -(xhat_prev @ getattr(lsg.controller.L, k).mT) for k in cdims}
        u = sum((action[k] @ env.B[k].mT for k in cdims), start=torch.zeros_like(x_prev))
        x = x_prev @ F.mT + u + w
        y = x @ H.mT + v
        nobs = (x_prev @ F.mT + u) @ H.mT
        xhat_pred = xhat_prev @ F.mT + u
        yhat = xhat_pred @ H.mT
        xhat = xhat_pred + (y - yhat) @ K.mT
        cols["state"].append(x); cols["observation"].append(y)
        cols["noiseless_observation"].append(nobs)
        cols["target_state_estimation"].append(xhat)
        cols["target_observation_estimation"].append(yhat)
        for k in cdims:
            acts[k].append(action[k])
        x_prev, xhat_prev = x, xhat

    env_td = {k: torch.stack(v, dim=-2) for k, v in cols.items()}
    ctrl_td = {k: torch.stack(v, dim=-2) for k, v in acts.items()}
    return env_td, ctrl_td


def _check_parity(name, lsg, B, L, seed=0):
    torch.manual_seed(seed)
    ds = lsg.generate_dataset(B, L)
    torch.manual_seed(seed)
    x0, v0, W, Vn = _replay_noise(lsg, B, L)
    env_ref, ctrl_ref = _naive_rollout(lsg, x0, v0, W, Vn)

    worst = 0.0
    for f in _FIELDS:
        err = (ds["environment", f] - env_ref[f]).abs().max().item()
        assert err < 1e-6, f"{name}: field {f} mismatch {err}"
        worst = max(worst, err)
    for k in ctrl_ref:
        err = (ds["controller", k] - ctrl_ref[k]).abs().max().item()
        assert err < 1e-6, f"{name}: control {k} mismatch {err}"
        worst = max(worst, err)
    # Shape / key parity with the generic loop output.
    assert ds.batch_size == (*lsg.group_shape, B, L), f"{name}: batch {ds.batch_size}"
    print(f"OK  parity {name:34s} max_err={worst:.2e}")


def test_fast_matches_naive_same_noise():
    _check_parity("lqe-scalar-batch", _lqe_system(6, 2, ()), B=8, L=40)
    _check_parity("lqe-group-dims", _lqe_system(5, 3, (2, 3)), B=4, L=25)
    _check_parity("lqe-length-one", _lqe_system(4, 2, ()), B=8, L=1)
    _check_parity("lqe-length-two", _lqe_system(4, 2, ()), B=8, L=2)
    _check_parity("ctrl-scalar-batch", _controlled_system(5, 2, 2, ()), B=8, L=40)
    _check_parity("ctrl-group-dims", _controlled_system(4, 2, 2, (2,)), B=4, L=30)


def test_fast_statistics_match_floor():
    lsg = _lqe_system(6, 2, ())
    floor = lsg.irreducible_loss.environment.observation.item()
    N, L = 6000, 60

    def empirical_apriori_error(ds):
        # a-priori one-step observation error at the steady-state tail.
        err = (ds["environment", "observation"] - ds["environment", "target_observation_estimation"])
        return (err[..., L // 2:, :] ** 2).sum(-1).mean().item()

    torch.manual_seed(1)
    fast = empirical_apriori_error(lsg.generate_dataset(N, L))
    torch.manual_seed(1)
    loop = empirical_apriori_error(SystemGroup.generate_dataset(lsg, N, L))

    rel_fast, rel_loop = abs(fast - floor) / floor, abs(loop - floor) / floor
    assert rel_fast < 0.05, f"fast empirical {fast:.4f} vs floor {floor:.4f} (rel {rel_fast:.3f})"
    assert rel_loop < 0.05, f"loop empirical {loop:.4f} vs floor {floor:.4f} (rel {rel_loop:.3f})"
    print(f"OK  stats floor={floor:.4f} fast={fast:.4f} ({rel_fast:.3f}) "
          f"loop={loop:.4f} ({rel_loop:.3f})")


if __name__ == "__main__":
    test_fast_matches_naive_same_noise()
    test_fast_statistics_match_floor()
    print("ALL_OK")
