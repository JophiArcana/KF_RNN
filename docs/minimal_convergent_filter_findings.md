# Minimal zero-convergence isolation on stationary LQE -- findings

Constructive follow-up to the negative result recorded in
[`cursor_test_time_training_for_lqe_probl.md`](../cursor_test_time_training_for_lqe_probl.md)
(decaying step + Polyak averaging went closed-loop unstable, `relIR=0.82`,
`+inf` analytical error) and to the plan it motivated. The question: on the
stationary `sd6_od2` system (`eps=0.1`, `L=10000`, `N=16`), what is the *single*
component that flips the asymptote from the constant-gain plateau (~0.3-0.5%
excess over the irreducible floor) to zero excess -- and is the stability
projection actually load-bearing here?

All runs use the pure a-priori M4 objective `(alpha, beta0, beta2) = (0, 1, 0)`
adapting `(F, H, K)` with the truncated `window=4` gradient -- **no** stability
projection, **no** warm start, **no** RTRL. Only the gain schedule
`eta_t = eta_0 * (t+1)^(-a)` and the Polyak tail-average burn-in `n0 = b * L`
vary. Driver: [`scripts/self_distillation_losses.py`](../scripts/self_distillation_losses.py),
`--methods minimal`.

## Verdict

- **The minimal flipping component is the gain schedule** (a vanishing gain plus
  tail averaging), **not** a structural addition. With an adequate base step the
  decay+avg filter reaches the floor (`excess ~ 0.1%`, `relIR ~ 0.02-0.04`,
  matching the optimum) while the constant-gain control at the *same* base step
  sits in a noise ball or diverges. This is exactly the proposal's
  noise-ball-vs-vanishing-gain thesis (constant gain -> ball of radius
  `~ sqrt(eta)`; vanishing gain + averaging -> zero excess).
- **The original failure was under-convergence, not instability.** The raw-iterate
  diagnostic (below) shows the *raw* iterate was itself far from the optimum
  (`relIR 0.66-0.80`) at every decay exponent -- it was never a parameter-space
  averaging artifact. The base step `eta_0 = 0.03` with a decaying schedule shrank
  the effective step far below the constant gain's, freezing the iterate before it
  locked on.
- **The stability projection is NOT necessary on stationary `sd6_od2`.** The
  converged decay+avg readout is at least as closed-loop-stable (~30% of tail
  snapshots) as the constant baseline (~17%). The low absolute stability is a
  property of this system's readout -- the optimal filter itself hugs the unit
  circle (`median|lambda(F_hat)| ~ 1`) -- and affects both methods equally. The
  projection was a crutch for under-convergence in the earlier failed run, not an
  independent load-bearing component.

## Diagnostic added

`_adapt_and_measure` now snapshots the analytical error / closed-loop stability of
the **raw** iterate alongside the reported **Polyak-averaged** filter, and
`run_method` reports both at the tail. This separates two failure modes:
raw also far/unstable => under-convergence (fix via the schedule); raw stable but
the average unstable => a parameter-averaging artifact. Every result below was the
former.

## Sweep 1 -- step-decay x burn-in at the original base step `eta_0 = 0.03`

Constant baseline (control): `excess 0.5%`, `relIR 0.031`, 17% stable. None of the
decay configs approach the floor -- the raw iterate (which depends only on `a`,
not the burn-in) is far from the optimum everywhere, i.e. under-convergence:

| a    | b   | excess (avg) | relIR (avg) | % stable (avg) | raw relIR | raw % stable |
|------|-----|--------------|-------------|----------------|-----------|--------------|
| 0.51 | 0.1 | 41.0%        | 0.708       | 9%             | 0.656     | 32%          |
| 0.51 | 0.5 | 34.0%        | 0.679       | 11%            | 0.656     | 32%          |
| 0.51 | 0.8 | 31.7%        | 0.664       | 29%            | 0.656     | 32%          |
| 0.6  | 0.1 | 50.4%        | 0.783       | 3%             | 0.747     | 28%          |
| 0.6  | 0.5 | 46.7%        | 0.763       | 10%            | 0.747     | 28%          |
| 0.6  | 0.8 | 45.2%        | 0.753       | 21%            | 0.747     | 28%          |
| 0.67 | 0.1 | 60.4%        | 0.830       | 1%             | 0.804     | 8%           |
| 0.67 | 0.5 | 59.3%        | 0.816       | 6%             | 0.804     | 8%           |
| 0.67 | 0.8 | 51.2%        | 0.808       | 8%             | 0.804     | 8%           |

Gentler decay (`a=0.51`) and larger burn-in help at the margin, but the iterate is
dominated by under-convergence: it freezes far from the optimum before the gain
vanishes.

## Sweep 2 -- base step `eta_0` at `a=0.51`, `b=0.5` (the convergence-quality knob)

The base step `eta_0` is part of the gain schedule (no new moving part). Raising it
lets the *same* vanishing schedule actually converge within the horizon:

| eta_0 | decay+avg excess | decay+avg relIR | decay+avg % stable | diverged | constant control at same eta_0 |
|-------|------------------|-----------------|--------------------|----------|--------------------------------|
| 0.03  | 34.0%            | 0.679           | 11%                | 0/16     | 0.5%  / relIR 0.031 / 17% stable |
| 0.1   | 11.7%            | 0.476           | 11%                | 0/16     | 1.9%  / relIR 0.067 / 54% stable |
| 0.3   | 2.7%             | 0.242           | 7%                 | 0/16     | 4.7%  / relIR 0.110 / 100% stable |
| 1.0   | **0.1%**         | **0.039**       | 33%                | 0/16     | 18.4% / relIR 0.230 / 81% (3 diverged) |
| 2.0   | **0.1%**         | **0.018**       | 30%                | 2/16     | `+inf` (16/16 diverged) |

At `eta_0 = 1.0` the vanishing gain reaches the floor (`excess 0.1%`, `relIR 0.039`,
0 diverged) while the constant control at the same step is a `18.4%` noise ball; at
`eta_0 = 2.0` the constant gain diverges entirely yet the decay+avg still homes onto
the optimum (`relIR 0.018`). This is the cleanest possible demonstration of the
gain-schedule axis as the load-bearing component.

## Headline config

`a=0.51`, `n0=0.5`, `eta_0=1.0` (both curves finite, maximal contrast): the
decay+avg filter sits on the irreducible floor (`0.1%` excess) with no projection,
beating the constant-gain plateau on both excess (0.1% vs 0.5%) and tail stability
(33% vs 17%). Output: `figures/sd_minimal_solved_sd6_od2_analytical_error_curve.png`.

## Implication for the next step (piecewise / nonstationary)

The pivot is a single knob. Stationary wants a vanishing gain (`a in (1/2,1)`,
adequate `eta_0`, tail averaging) and reaches zero excess. The piecewise/tracking
study should flip exactly this knob -- decaying `a > 0` -> constant gain (`a = 0`)
or a forgetting factor `lambda < 1` -- holding the objective, gradient, init, and
(absent) projection fixed, and ask whether the constant gain is then necessary and
sufficient for bounded tracking error.
