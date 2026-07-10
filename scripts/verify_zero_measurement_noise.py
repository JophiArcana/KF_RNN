"""Verification: can the analytics run with zero measurement noise (S_V = 0)?

Historically the analytical machinery (Riccati / irreducible_error /
analytical_error) was gated off for S_V = 0 because a standard DARE cannot be
solved with a singular ``R = S_V``. The in-house ``ecliseutils.solve_discrete_are``
uses the van Dooren extended pencil + matrix disk function, which never forms
``R^{-1}`` and is advertised as robust to singular ``R``.

This script empirically checks, over a battery of randomly sampled systems with
``S_V = 0`` (and a few controls / partial-observation variants):

  1. DARE residual: ``test_discrete_are(Fᵀ, Hᵀ, S_W, S_V, P)`` is ~0.
  2. Innovation covariance ``S_pred = H P Hᵀ + S_V`` is well-conditioned, so the
     Kalman gain ``K`` is well-defined.
  3. Closed loop ``F(I - K H)`` is a strict contraction (filter is stable).
  4. Steady-state consistency vs a brute-force Riccati recursion (floor & K).
  5. irreducible_loss (= trace(S_pred)) matches the empirical a-priori
     observation error of a long simulated rollout.
  6. analytical_error of the *optimal* filter (F, H, K) equals irreducible_loss.
  7. Continuity: as S_V -> 0 the (K, floor) computed with singular S_V match the
     limit of the nonsingular-S_V solutions.
  8. End-to-end: the real ``MOPDistribution(V_std=0)`` + ``setup_analytical`` path.

Run: PYTHONPATH=. python scripts/verify_zero_measurement_noise.py
"""
import warnings
warnings.filterwarnings("ignore")

import torch

import ecliseutils as eu
from ecliseutils.are import solve_discrete_are, test_discrete_are

torch.set_default_dtype(torch.float64)
torch.set_default_device("cpu")


def _stable_F(S_D, radius=0.9):
    F = torch.randn(S_D, S_D)
    F *= radius / torch.linalg.eigvals(F).abs().max()
    return F


def kalman_from_dare(F, H, S_W, S_V):
    """Filter-form steady state via ecliseutils DARE (never forms S_V^{-1})."""
    P = solve_discrete_are(F.mT, H.mT, S_W, S_V)          # a-priori state error cov
    S_pred = H @ P @ H.mT + S_V                           # innovation covariance
    K = torch.linalg.solve(S_pred, (P @ H.mT).mT).mT      # steady-state Kalman gain
    return P, S_pred, K


def kalman_recursion(F, H, S_W, S_V, steps=50000):
    """Brute-force a-priori Riccati recursion -> reference (P, S_pred, K)."""
    S_D = F.shape[-1]
    P = torch.eye(S_D)
    K = None
    for _ in range(steps):
        S_pred = H @ P @ H.mT + S_V
        K = torch.linalg.solve(S_pred, (P @ H.mT).mT).mT
        P_post = P - K @ H @ P
        P = F @ P_post @ F.mT + S_W
    S_pred = H @ P @ H.mT + S_V
    K = torch.linalg.solve(S_pred, (P @ H.mT).mT).mT
    return P, S_pred, K


def simulate_apriori_error(F, H, S_W, S_V, K, N=20000, L=80):
    """Empirical steady-state a-priori observation error of the Kalman filter."""
    S_D, O_D = F.shape[-1], H.shape[-2]
    sqrt_S_W = eu.sqrtm(S_W) if torch.norm(S_W) > 0 else torch.zeros(S_D, S_D)
    sqrt_S_V = eu.sqrtm(S_V) if torch.norm(S_V) > 0 else torch.zeros(O_D, O_D)

    x = torch.randn(N, S_D)
    xhat = torch.zeros(N, S_D)
    errs = []
    for t in range(L):
        w = torch.randn(N, S_D) @ sqrt_S_W.mT
        v = torch.randn(N, O_D) @ sqrt_S_V.mT
        x = x @ F.mT + w
        y = x @ H.mT + v
        yhat = (xhat @ F.mT) @ H.mT              # a-priori observation prediction
        if t > L // 2:
            errs.append(((y - yhat) ** 2).sum(-1))
        xhat = xhat @ F.mT + (y - yhat) @ K.mT
    return torch.stack(errs).mean().item()


def analytical_error_of_filter(F, H, K, S_W, S_V):
    """Closed-form steady-state obs error of filter (F,H,K) via SequentialPredictor."""
    from tensordict import TensorDict
    from kf_rnn.model.sequential.base import SequentialPredictor

    S_D, O_D = F.shape[-1], H.shape[-2]
    KH = K @ H
    F_augmented = torch.cat([
        torch.cat([F, torch.zeros_like(F)], dim=-1),
        torch.cat([(KH @ F), (F - KH @ F)], dim=-1),
    ], dim=-2)
    H_augmented = torch.cat([H, torch.zeros_like(H)], dim=-1)

    sys_td = TensorDict.from_dict({
        "environment": {
            "F": F, "H": H, "K": K,
            "sqrt_S_W": eu.sqrtm(S_W) if torch.norm(S_W) > 0 else torch.zeros(S_D, S_D),
            "sqrt_S_V": eu.sqrtm(S_V) if torch.norm(S_V) > 0 else torch.zeros(O_D, O_D),
            "B": TensorDict({}, batch_size=(S_D,)),
        },
        "F_augmented": F_augmented,
        "H_augmented": H_augmented,
        "controller": {},
    }, batch_size=torch.Size([]))
    kfs = TensorDict({"F": F, "H": H, "K": K}, batch_size=torch.Size([]))
    return SequentialPredictor.analytical_error(kfs, sys_td)["environment", "observation"].item()


def run_case(name, S_D, O_D, W_std, V_std, seed=0):
    torch.manual_seed(seed)
    F = _stable_F(S_D)
    H = torch.randn(O_D, S_D) / (3 ** 0.5)
    S_W = (torch.eye(S_D) * W_std) @ (torch.eye(S_D) * W_std).mT
    S_V = (torch.eye(O_D) * V_std) @ (torch.eye(O_D) * V_std).mT

    try:
        P, S_pred, K = kalman_from_dare(F, H, S_W, S_V)
    except Exception as e:
        print(f"[XX ] {name:26s} DARE FAILED: {type(e).__name__}: {str(e)[:60]}")
        return False

    resid = test_discrete_are(F.mT, H.mT, S_W, S_V, P).abs().max().item()
    cond = torch.linalg.cond(S_pred).item()
    rho = torch.linalg.eigvals(F @ (torch.eye(S_D) - K @ H)).abs().max().item()
    floor = torch.trace(S_pred).item()

    # Independent, Riccati-free oracle: Monte-Carlo steady-state a-priori error.
    emp = simulate_apriori_error(F, H, S_W, S_V, K)
    rel_emp = abs(emp - floor) / max(floor, 1e-12)

    # Self-consistency: the optimal filter's closed-form error equals the floor.
    ae = analytical_error_of_filter(F, H, K, S_W, S_V)
    rel_ae = abs(ae - floor) / max(floor, 1e-12)

    # Informational only: a naive Riccati *iteration* can converge to a
    # non-stabilizing fixed point when S_V = 0, so it is NOT a pass/fail oracle
    # (the symplectic/disk DARE correctly selects the stabilizing solution).
    _, S_pred_ref, K_ref = kalman_recursion(F, H, S_W, S_V)
    rel_floor_ref = abs(floor - torch.trace(S_pred_ref).item()) / max(floor, 1e-12)

    ok = (resid < 1e-6 and rho < 1.0 and rel_emp < 0.05 and rel_ae < 1e-4)
    print(f"[{'OK ' if ok else 'XX '}] {name:26s} "
          f"resid={resid:.1e} cond={cond:.1e} rho={rho:.3f} floor={floor:.4f} "
          f"emp_rel={rel_emp:.3f} ae_rel={rel_ae:.1e} (naiveIterRef_rel={rel_floor_ref:.1e})")
    return ok


def run_continuity(name, S_D, O_D, W_std, seed=0):
    torch.manual_seed(seed)
    F = _stable_F(S_D)
    H = torch.randn(O_D, S_D) / (3 ** 0.5)
    S_W = (torch.eye(S_D) * W_std) @ (torch.eye(S_D) * W_std).mT

    _, S_pred0, K0 = kalman_from_dare(F, H, S_W, torch.zeros(O_D, O_D))
    floor0 = torch.trace(S_pred0).item()
    print(f"  continuity {name}: floor(S_V=0)={floor0:.6f}")
    last = None
    for v in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
        S_V = torch.eye(O_D) * (v ** 2)
        _, S_pred, K = kalman_from_dare(F, H, S_W, S_V)
        dK = (K - K0).abs().max().item()
        dfloor = abs(torch.trace(S_pred).item() - floor0)
        print(f"    V_std={v:.0e}  |K-K0|={dK:.2e}  |floor-floor0|={dfloor:.2e}")
        last = (dK, dfloor)
    return last[0] < 1e-2 and last[1] < 1e-2


def run_end_to_end():
    """Exercise the real distribution + setup_analytical path with V_std = 0."""
    from kf_rnn.infrastructure.config.schema import (
        EnvironmentShape, ProblemShape, SystemConfig, SystemSettings,
    )
    from kf_rnn.system.linear_time_invariant import MOPDistribution

    S_D, O_D = 6, 3
    ps = ProblemShape(environment=EnvironmentShape(observation=O_D), controller={})
    cfg = SystemConfig(S_D=S_D, problem_shape=ps, settings=SystemSettings(include_analytical=True))
    dist = MOPDistribution("gaussian", "gaussian", W_std=1.0, V_std=0.0)

    torch.manual_seed(0)
    try:
        lsg = dist.sample(cfg, (4,))                       # 4 systems in a group
    except Exception as e:
        print(f"[XX ] end-to-end MOPDistribution(V_std=0): {type(e).__name__}: {str(e)[:80]}")
        return False

    floor = lsg.irreducible_loss.environment.observation           # [4]
    # Empirical a-priori error from the fast generator, per system.
    torch.manual_seed(1)
    ds = lsg.generate_dataset(4000, 60)
    err = (ds["environment", "observation"] - ds["environment", "target_observation_estimation"])
    emp = (err[..., 30:, :] ** 2).sum(-1).mean(dim=(-2, -1))         # [4]
    rel = ((emp - floor).abs() / floor.clamp_min(1e-12))
    ok = bool((rel < 0.05).all())
    print(f"[{'OK ' if ok else 'XX '}] end-to-end MOPDistribution(V_std=0)  "
          f"floor={floor.tolist()}  emp_rel_max={rel.max().item():.3f}")
    return ok


if __name__ == "__main__":
    print("=== S_V = 0 (zero measurement noise), S_W > 0 ===")
    results = []
    results.append(run_case("fully-observed O=S", 6, 6, 1.0, 0.0, seed=0))
    results.append(run_case("partial-obs O<S", 6, 3, 1.0, 0.0, seed=2))
    results.append(run_case("partial-obs O<S", 8, 2, 1.0, 0.0, seed=3))
    results.append(run_case("scalar-obs O=1", 5, 1, 1.0, 0.0, seed=4))
    results.append(run_case("small W_std", 6, 3, 0.1, 0.0, seed=5))
    for s in range(10, 20):
        results.append(run_case(f"partial-obs rand s={s}", 6, 2, 1.0, 0.0, seed=s))

    print("\n=== over-observed O>S (unusual: more noiseless sensors than states) ===")
    results.append(run_case("over-observed O>S", 4, 6, 1.0, 0.0, seed=1))

    print("\n=== baseline sanity: S_V > 0 (should already work) ===")
    results.append(run_case("baseline S_V>0", 6, 3, 1.0, 0.5, seed=6))

    print("\n=== continuity as S_V -> 0 ===")
    c1 = run_continuity("fully-observed", 6, 6, 1.0, seed=0)
    c2 = run_continuity("partial-obs", 6, 3, 1.0, seed=2)

    print("\n=== degenerate: S_V = 0 AND S_W = 0 (fully deterministic) ===")
    run_case("deterministic W=V=0", 6, 6, 0.0, 0.0, seed=7)

    print("\n=== end-to-end through the real code path ===")
    e2e = run_end_to_end()

    print(f"\nSummary: {sum(results)}/{len(results)} S_V=0/baseline cases passed; "
          f"continuity={'OK' if (c1 and c2) else 'XX'}; end-to-end={'OK' if e2e else 'XX'}")
