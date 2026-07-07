# Constant-gain mechanism study: does self-distillation shrink the noise ball?

Constructive follow-up to
[`minimal_convergent_filter_findings.md`](minimal_convergent_filter_findings.md).
That study showed the *stationary* zero-excess winner is a vanishing gain +
Polyak averaging. But a vanishing gain cannot track a moving target, so for the
eventual non-stationary / video goal it is a **stationary-only oracle**, not a
deployable filter. This study therefore asks the deployable question, among
**constant-gain** schemes only:

> When each method is tuned to its **own best step** (fair comparison: the
> constant and decaying schedules have opposite optimal-step regimes), does the
> latent self-distillation objective with a light a-priori anchor (the "M3
> bridge", `alpha=1, beta0=0.05`) achieve a **smaller steady noise ball** and/or
> **better tail stability** than a plain a-priori constant-gain filter?

## Setup

`--methods mech` ([`scripts/self_distillation_losses.py`](../scripts/self_distillation_losses.py)),
three arms, all adapting `(F, H, K)` with `beta2=0`, `window=4`, no projection /
warm start / RTRL:

- **M4 constant** -- pure a-priori `(0, 1, 0)`, constant gain (the deployable baseline).
- **SD+anchor constant** -- `(alpha, beta0, beta2) = (1, 0.05, 0)`, constant gain (the candidate).
- **M4 decay+avg oracle** -- pure a-priori, vanishing gain `eta_t = eta_0 (t+1)^{-0.51}` + Polyak tail-average (`n0=0.5L`): the stationary lower bound.

Per-method step sweep `--step-size in {0.01, 0.03, 0.1, 0.3, 1.0}`, `sd6_od2`,
`eps=0.1`, `N=16`, `seed=0`, on the same CUDA-drawn system as `sd_m3m4_L30000`
(floor 0.0546). The **headline is the formal `L=100000` sweep**
(`output/sd_mech_L100000/ss*`); an earlier `L=30000` sweep
(`output/sd_mech/ss*`) is kept only to document the horizon artifact it produced
at the smallest step. Sweep driver:
[`scripts/submit_mech_sweep.sh`](../scripts/submit_mech_sweep.sh); envelope reader:
[`scripts/analyze_mech_sweep.py`](../scripts/analyze_mech_sweep.py).

`excess` = median analytical error **among the stable trajectories**, relative to
the floor. `%stable` = fraction of the 16 trajectories whose readout filter is
closed-loop stable in the tail. They MUST be read together: at the low-excess
small steps only a few of 16 trajectories stay stable, so `excess` there is noisy
(`%stable` swings run-to-run); the robust comparisons are the steps where a method
is 100% stable.

## Verdict

**SD+anchor is the better deployable constant-gain filter here.** It has **lower
excess than plain a-priori at every step**, it is **not biased**, and at the
fully-stable operating point it **strictly dominates** a-priori. Plain a-priori's
only residual advantage is slightly higher stability at the smallest, noisiest
steps.

- **Lower excess at every matched step** (formal `L=100000`): eta=0.01 0.036% vs
  0.085%; 0.03 0.073% vs 0.287%; 0.1 0.249% vs 0.882%; 0.3 0.735% vs 2.339%;
  1.0 2.278% vs 8.726%.
- **Strict Pareto win at the robust point.** At eta=0.3 *both* methods are 100%
  stable, and there SD is **0.735% vs M4's 2.339%** -- ~3x lower excess at
  identical (perfect) stability. Even restricting to fully-stable configs, SD's
  best (0.735% @ eta=0.3) beats M4's best (0.882% @ eta=0.1).
- **SD is essentially unbiased** (converged `relIR` 0.009-0.014, on par with or
  below M4's 0.010-0.017). An earlier draft mis-read SD's *under-converged*
  L=30000 eta=0.01 `relIR` of 0.102 as a fixed-`alpha` bias; the converged value
  is ~0.01.
- **M4's only edge is small-step stability** (eta=0.01: 21% vs 5% stable; eta=0.03:
  33% vs 3%): it offers more stable trajectories at modestly higher excess. These
  counts are noisy (this run's M4@eta=0.01 is 21% vs 12% in a separate L=100k run).

## The formal L=100000 sweep

floor = 0.0546; `%stbl` = tail fraction stable; `rad` = median |lambda(F_hat)|;
each method's lowest-excess step in **bold**.

| method | step | excess | %stbl | ndiv | rad |
|---|---|---|---|---|---|
| M4 constant | **0.01** | **0.085%** | 0.21 | 0 | 1.00 |
| M4 constant | 0.03 | 0.287% | 0.33 | 0 | 1.00 |
| M4 constant | 0.1 | 0.882% | 1.00 | 0 | 0.99 |
| M4 constant | 0.3 | 2.339% | 1.00 | 0 | 0.93 |
| M4 constant | 1.0 | 8.726% | 1.00 | 0 | 0.73 |
| SD+anchor | **0.01** | **0.036%** | 0.05 | 0 | 1.00 |
| SD+anchor | 0.03 | 0.073% | 0.03 | 0 | 1.00 |
| SD+anchor | 0.1 | 0.249% | 0.29 | 0 | 1.00 |
| SD+anchor | 0.3 | 0.735% | 1.00 | 0 | 0.99 |
| SD+anchor | 1.0 | 2.278% | 0.69 | 5 | 0.81 |
| decay+avg oracle | 0.01 | inf | 0.00 | 0 | 1.00 |
| decay+avg oracle | 0.03 | inf | 0.00 | 0 | 1.00 |
| decay+avg oracle | 0.1 | inf | 0.00 | 0 | 1.00 |
| decay+avg oracle | 0.3 | 0.032% | 0.34 | 0 | 1.00 |
| decay+avg oracle | **1.0** | **0.006%** | 0.06 | 0 | 1.00 |

Trend: both constant methods' excess falls monotonically as the step shrinks, and
SD sits *below* M4 at every step. (In the earlier `L=30000` sweep SD@eta=0.01 was
an outlier at 0.555%; that was under-convergence -- a tiny step with only a weak
`beta0=0.05` anchor -- and it vanishes here, SD@eta=0.01 -> 0.036%.)

### Curve: three methods, each at its own best step

![Three-method envelope curve over context length](../output/sd_mech_L100000_envelope_curve.png)

Standard two-panel readout (left: trajectory-median analytical error vs online
step, with the irreducible floor and zero-predictor ceiling; right: excess over
the floor, log-y), each method drawn from the step that minimises its steady
excess -- M4 and SD both at `eta=0.01`, the oracle at `eta=1.0` (its raw,
pre-averaging iterate is the faint dashed line). SD+anchor (orange) sits below M4
constant (blue) throughout, and the decay+avg oracle (green) is lowest. The
`eta=0.01` curves are visibly noisy because only ~5-21% of trajectories stay
stable there; for the clean, fully-stable comparison see `eta=0.3` in the table
above (SD 0.735% vs M4 2.339%, both 100% stable). Generated by
[`scripts/plot_mech_envelope_curve.py`](../scripts/plot_mech_envelope_curve.py).

## Reading

1. **Matched-step accuracy favors SD -- the variance-reduction view holds.** At a
   fixed constant step the smooth model-generated SD target gives a lower steady
   excess than the raw-observation a-priori gradient at every step (the "smaller
   ball at the same eta" hypothesis), and at the fully-stable eta=0.3 this is a
   clean ~3x win with no stability penalty.

2. **Stability is the one axis where a-priori can lead, and only at tiny steps.**
   The optimal filter on this system hugs the unit circle (`rad ~ 1.00`), so
   absolute stability is low for everyone at the smallest steps; there M4 keeps a
   few more trajectories stable than SD. But by eta=0.3 SD is also 100% stable
   (and still far lower excess), so the trade disappears at the sensible operating
   point. SD does diverge on 5/16 at eta=1.0 (vs M4's 0), i.e. SD is the one to
   watch at *large* steps.

3. **The decay+avg oracle is the stationary-only lower bound.** Lowest excess
   (0.006% at its eta=1.0 recipe) but 6% stable and `inf` (unstable averaged
   readout) at every step below 0.3 -- and *worse* at L=100000 than L=30000,
   since more steps give the averaged iterate more time to drift over the unit
   circle. Not deployable, and off the table for tracking anyway.

## Implication for the non-stationary / video next step

The transferable property we hoped for -- self-distillation reaching a lower
steady error than plain a-priori at a *given constant step* -- holds cleanly, SD
is unbiased, and at the fully-stable operating point SD strictly dominates. That
is a strong point in SD's favor for the video goal, where a constant gain is
mandatory. Two things to carry into the deferred tracking study: (a) confirm SD's
matched-step accuracy edge survives non-stationarity (does the smooth target also
reduce *tracking* lag/variance, not just the stationary ball); and (b) watch SD's
stability at larger steps (its 5/16 divergences at eta=1.0), where a cheap
stability projection (proposal section 1.4) may be the price of using SD with an
aggressive constant gain. Concretely: a piecewise / drifting `sd6_od2` system,
constant a-priori vs constant SD+anchor (+/- stability projection), judged on
tracking error and fraction-stable.
