# Zero measurement noise (`S_V = 0`) and the analytics

Historically the analytical machinery — the filter-form Riccati solve, the
`irreducible_loss` floor, and the closed-form `analytical_error` — was assumed to
require a nonsingular measurement-noise covariance `S_V` (the `R` of the standard
DARE), because a textbook DARE solver forms `R^{-1}` and cannot run with singular
`R`. This note records a verification that the in-house
`ecliseutils.solve_discrete_are` lifts that restriction, so `S_V = 0` is a
first-class, supported regime for the analytics.

## Why it works

`ecliseutils.are.solve_discrete_are` does **not** solve the DARE by forming
`R^{-1}`. It builds the van Dooren **extended pencil** `L - λM` (size `2m + n`),
deflates the control columns via a QR onto their orthogonal complement (never
inverting `R` nor `H P Hᵀ`), and extracts the stabilizing invariant subspace with
the matrix **disk function** `(L̂ + M̂)^{-1}(L̂ − M̂)`. A singular / zero `R`
merely places closed-loop eigenvalues at `μ = 0` (with symplectic partners at
`μ = ∞`), which the disk form handles cleanly. See the module docstring in
`.../site-packages/ecliseutils/are.py`.

Downstream, nothing else inverts `S_V`:

- `environment.py` uses `K = solve(H P Hᵀ + S_V, P Hᵀ)`. With `S_V = 0` the
  innovation covariance `H P Hᵀ` is still nonsingular as long as `H` has full row
  rank (`O_D ≤ S_D`) and `S_W > 0` drives the observed subspace.
- `SequentialPredictor._analytical_error_and_cache` only ever uses `sqrt_S_V`
  (never its inverse), so it is inherently `S_V = 0`-safe.

## Verification

`scripts/verify_zero_measurement_noise.py` (float64) checks, over ~16 randomly
sampled systems plus the real distribution path:

1. **DARE residual** `test_discrete_are(Fᵀ, Hᵀ, S_W, S_V, P)` ≈ `1e-15`.
2. **Filter stability** — closed loop `F(I − KH)` is a strict contraction (`ρ < 1`).
3. **Monte-Carlo oracle** — the empirical steady-state a-priori observation error
   of a long simulated Kalman rollout matches `trace(S_pred)` to ~0.1% (this
   oracle uses no Riccati at all).
4. **Self-consistency** — `analytical_error` of the optimal filter `(F, H, K)`
   equals `irreducible_loss` to ~1e-14.
5. **Continuity** — `K` and the floor from the singular-`S_V` solve are the exact
   `O(V_std²)` limit of the nonsingular-`S_V` solves.
6. **End-to-end** — `MOPDistribution(V_std=0)` with `include_analytical=True`
   builds via `setup_analytical` and reproduces the floor to 0.5%.

Result: **16/17** cases pass; continuity and end-to-end pass.

Bonus finding: a naive Riccati *iteration* can converge to a **non-stabilizing**
fixed point when `S_V = 0`; the symplectic/disk solver correctly selects the
stabilizing (deadbeat) solution. So the pencil method is not just tolerant of
`S_V = 0` — it is more correct than iterating.

## Scope / limitations (do NOT feed these to the analytics)

The DARE (matrix-sign iteration) genuinely breaks on two degenerate inputs, both
of which have a pencil eigenvalue **on** the unit circle (the standard
non-existence boundary for a stabilizing solution):

- **Fully deterministic** `S_W = 0 AND S_V = 0`. `P = 0`, the innovation is
  singular, and there is no meaningful steady-state filter. This is exactly what
  `LTIZeroNoiseEnvironment` and its distributions (`Orthonormal`,
  `ContinuousNoiseless`, `Periodic`) produce — which is why that class keeps its
  `assert not include_analytical` gate. That gate is **correct** and unchanged.
- **Over-observed** `O_D > S_D` (more noiseless sensors than states).

## Bottom line

No gate needed relaxing: the standard `LTISystem` / `LTIEnvironment` path already
supports `S_V = 0` (with `S_W > 0`, `O_D ≤ S_D`) with `include_analytical=True`,
and the analytics are exact there. The only remaining gate —
`LTIZeroNoiseEnvironment` forbidding analytics — guards the genuinely degenerate
fully-deterministic case and is left in place.
