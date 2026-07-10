# a-posteriori self-distillation: sweeping the direct-K observation loss

Follow-up to
[`mechanism_constant_gain_sd_findings.md`](mechanism_constant_gain_sd_findings.md)
and
[`nstep_sd_ladder_launch_findings.md`](nstep_sd_ladder_launch_findings.md). Those
studies settled the SD base config -- latent SD + a light a-priori anchor
(`alpha=1, beta0=0.05`), and, for the launch gradient, that **detaching** it
(`keep_launch=False`) buys a lower asymptotic floor while deeper `n`/the launch
gradient buy early convergence speed. The one section-6 term left uncharacterized
is the **a-posteriori observation loss** `beta2`. An early result
(`output/sd_m3m4_L30000`) flagged that it *improves convergence but hurts the
asymptote due to bias*; this study sweeps `beta2` to pin down that tradeoff.

The three-term window objective (see
[`src/kf_rnn/model/sequential/rnn_ttt.py`](../src/kf_rnn/model/sequential/rnn_ttt.py)):

```
beta0 * 0.5||H F x_{t-1}^+ - y||^2                              # a-priori obs (anchor)
+ beta2 * 0.5||H x_t^+ - y||^2                                  # a-posteriori obs  <- swept
+ alpha * sum_{k=1}^{n} 0.5||sg(x_t^+) - F^k x_{t-k}^+||^2      # latent SD ladder
```

## Setup

`--methods post`
([`scripts/self_distillation_losses.py`](../scripts/self_distillation_losses.py)):
one arm per `beta2` on a **fixed** base -- latent SD + anchor (`alpha=1,
beta0=0.05`), **detached launch** (`keep_launch=False`), the `n = window = 4`
ladder (`sd_horizon=4`), all adapting `(F, H, K)`, constant gain, no projection /
warm start / RTRL / weight decay. The only independent variable is `beta2`:

```
beta2 in {0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}
```

with `beta2=0.0` the **control** (the detached `n=4` SD+anchor base itself).
Per-arm step sweep `--step-size in {0.01, 0.03, 0.1, 0.3, 1.0}`, `sd6_od2`,
`eps=0.1`, `N=16`, `seed=0`, on the same CUDA-drawn system as the mechanism /
depth studies (floor 0.0546), at the **converged** `L=300000` horizon.

Sharded one-arm-per-job layout (each `beta2` is its own `--post-grid <value>`
run): `output/sd_post_L300k/ss<step>/b<beta2>/sd6_od2_save.pt`. Driver:
[`scripts/submit_post_sweep.sh`](../scripts/submit_post_sweep.sh); readers:
[`scripts/analyze_mech_sweep.py`](../scripts/analyze_mech_sweep.py) (envelope
table), [`scripts/plot_post_sweep.py`](../scripts/plot_post_sweep.py) (steady
grid), [`scripts/plot_post_convergence.py`](../scripts/plot_post_convergence.py)
(per-step convergence overlay). The two grid readers glob recursively and parse
the step from the `ss*` path token, so they handle both this sharded layout and
the flat `ss*/save.pt` layout of the earlier sweeps.

As before, `excess` (median analytical error among stable trajectories, relative
to floor) and `%stbl` (tail fraction closed-loop stable) MUST be read together:
at the low-excess small steps only a fraction of the 16 trajectories are stable,
so `excess` there is noisy; the robust comparisons are the fully-stable steps.

> **Scope caveat.** The base here is the detached `n=4` ladder (the
> *fast-convergence* detached corner), **not** the globally-best floor config
> (`keep=False, n=1, eta=0.01`, 0.018% from the launch study). `n=4` was chosen
> deliberately: `beta2` is a convergence-accelerator candidate, so it is isolated
> on the base that already favors speed. So the absolute numbers below are
> `beta2`-vs-`beta2` at fixed `n=4`, not claims about the best deployable filter.

## Verdict

**The a-posteriori term is a pure asymptote liability at the converged horizon,
and a genuine early-convergence accelerator in the transient -- the same
speed/floor trade as the launch gradient and ladder depth.**

1. **`beta2=0` wins the converged floor decisively.** The control reaches
   **0.068% excess** (`eta=0.01`); the best of *any* `beta2>0` arm is 0.806%
   (`beta2=0.01`), and the `beta2>=0.05` arms all plateau at ~1.07-1.15%. There
   is no weight at which the posteriori term lowers the steady floor.
2. **But `beta2` descends much faster early.** At `eta=0.01` the large-`beta2`
   arms drop out of the initial ~50-100% region by step ~10^2-10^3, while the
   control is the *slowest* early and only catches up by ~10^4, overtaking
   everyone around step ~3x10^4-10^5. Classic fan-then-crossover.
3. **Bias saturates by `beta2 ~ 0.05`.** For `beta2 in {0.1, 0.2, 0.5, 1.0}` the
   per-arm-best excess is ~1.07% flat -- past ~0.05 the a-posteriori term already
   dominates the objective, so adding more neither helps nor hurts the endpoint.

Sanity check: the `beta2=0` control (detached `n=4`) lands at 0.068% @ `eta=0.01`,
matching the depth study's `keep=False, n=4` arm (0.066%) -- the base reproduces.

## Per-arm best (each at its own best step; L=300000)

floor = 0.0546. Each arm's lowest-excess step in the table below.

| beta2 | best excess | best step | %stbl @ best |
|---|---|---|---|
| **0.0 (control)** | **0.068%** | 0.01 | 0.97 |
| 0.01 | 0.806% | 0.01 | 0.76 |
| 0.02 | 1.622% | 0.01 | 0.77 |
| 0.05 | 1.146% | 0.1 | 1.00 |
| 0.1 | 1.081% | 0.1 | 1.00 |
| 0.2 | 1.082% | 0.1 | 1.00 |
| 0.5 | 1.072% | 0.1 | 1.00 |
| 1.0 | 1.073% | 0.1 | 1.00 |

## Steady excess: the control is lowest at every step, and `beta2` shifts the optimum right

`ndiv` = diverged trajectories out of 16. Steady excess (%) per `(beta2, step)`:

| beta2 | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 (ndiv) |
|---|---|---|---|---|---|
| **0.0 (control)** | **0.068** | 0.188 | 0.609 | 1.626 | 3.563 (1) |
| 0.01 | 0.806 | 0.969 | 1.518 | 3.096 | 5.324 (4) |
| 0.02 | 1.622 | 1.753 | 1.797 | 3.352 | 5.763 (6) |
| 0.05 | 2.832 | 2.749 | 1.146 | 2.932 | 5.291 (3) |
| 0.1 | 3.515 | 3.026 | 1.081 | 2.730 | 4.872 (4) |
| 0.2 | 3.939 | 3.187 | 1.082 | 2.699 | 4.868 (6) |
| 0.5 | 4.225 | 3.040 | 1.072 | 2.682 | 4.757 (5) |
| 1.0 | 4.329 | 3.067 | 1.073 | 2.688 | 4.817 (5) |

Two readings of this grid:

- **At the accuracy-optimal small step (`eta=0.01`) the bias is monotone in
  `beta2`:** 0.068 -> 0.806 -> 1.622 -> 2.832 -> ... -> 4.329. Every unit of
  posteriori weight is a direct floor penalty there.
- **`beta2` pushes the optimal step right.** The control and tiny `beta2`
  (0.01, 0.02) are best at `eta=0.01`; `beta2>=0.05` can no longer use the small
  step (too biased) and are best at the noisier `eta=0.1` -- where they collapse
  onto a common ~1.07% plateau, still above the control's 0.609% there.

### Steady grid

Left: steady excess vs step size, one line per `beta2`; right: per-arm best.

![beta2 steady sweep](../output/sd_post_L300k_post_sweep.png)

## Convergence: `beta2` is faster early, the control bottoms out lower

One panel per step size, each overlaying every `beta2` arm's excess vs online
step:

![beta2 convergence per step](../output/sd_post_L300k_convergence.png)

- **`beta2` is a strong early accelerator.** At `eta=0.01` / `0.03` the high-`beta2`
  arms (yellow) leave the initial ~50-100% region ~1-2 orders of magnitude sooner
  than the control (gray dashed), which is the slowest curve early.
- **Then each `beta2>0` arm plateaus and the control overtakes it.** The control
  keeps descending past all of them around step ~3x10^4-10^5 and ends lowest
  (~0.07% / ~0.19%), while the posteriori arms flatten at ~1-4%. Larger `beta2`
  crosses earlier and settles higher; `beta2 in {0.01, 0.02}` cross latest and
  settle closest to the control.
- **The advantage window shrinks with step size.** By `eta=0.1` the early gap is
  small and mid-context is noise-dominated; at `eta=0.3` / `1.0` the arms are
  indistinguishable in the noise with no persistent asymptote separation (and
  `beta2>0` diverges on more trajectories at `eta=1.0`: 3-6/16 vs the control's
  1/16).

## Reading

1. **The a-posteriori observation loss is the third face of the speed/floor
   trade.** Like the launch gradient (`keep_launch=True`) and ladder depth `n`, a
   larger `beta2` accelerates early descent but raises the asymptotic floor. At a
   converged stationary horizon it is a net loss -- `beta2=0` is the lower floor
   at every matched step -- so it should be **off** for a deployable stationary
   constant-gain filter.
2. **The bias is a direct-K over-correction that saturates.** Fitting `x_t^+` to
   `y` pulls the filter toward the noisy observation rather than the smooth latent
   dynamics; past `beta2 ~ 0.05` this over-correction dominates the objective and
   the endpoint stops moving (~1.07% plateau). The relative-IR error in the earlier
   `sd_m3m4` redo told the same story (`beta2=1` arms had relIR ~0.33 vs <=0.03 for
   `beta2=0`).
3. **Its only value is transient.** The one thing `beta2` reliably does is get the
   filter down fast early -- exactly the property that is invisible once `L=300000`
   has converged.

## Implication for the non-stationary / video next step

This closes the section-6 loss survey on the stationary problem: for a *converged*
deployable filter, keep `beta2=0` (as with `keep_launch=False` and `n=1`). But the
decisive axis here is again the *convergence horizon*, which non-stationarity
removes. `beta2` joins `keep_launch=True` and deeper `n` as a **convergence
accelerator whose stationary penalty may reverse under drift**, where fast
re-adaptation matters more than the asymptotic ball. So the deferred tracking study
should sweep `beta2` (small, e.g. `<=0.05`) alongside the launch variant and depth
on a piecewise / drifting `sd6_od2`, judged on tracking error and fraction-stable
-- with the expectation that a small posteriori weight could pay for itself
precisely when the horizon is too short to reach the floor.
