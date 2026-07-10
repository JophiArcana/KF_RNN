# n-step self-distillation ladder x launch gradient, on the A-init

Follow-up to [`nstep_sd_ladder_launch_findings.md`](nstep_sd_ladder_launch_findings.md)
and [`init_pathway_findings.md`](init_pathway_findings.md). The first study swept the
SD ladder depth `n = 1..4` and the launch gradient (`keep_launch` True/False) on the
**current default init** (`fI_k0`: `F=(1-eps)I, K=0`) and concluded: `n=1` wins the
converged floor, and detaching the launch dominates once converged (lower excess at
every matched step *and* far better stability, its only cost being slower
convergence). The second study then found a better *initialization* -- the A-init
`f0_kpinv` (`F=0, K=H^+`, the toy `K=A` replace/high-gain end) -- which converges
faster, is 100% stable at every step, and lands in a **contractive** basin (`|λ| ≈
0.80`, inside the true spectrum) at the *same* floor, versus the default init's
unit-circle-hugging marginal basin.

This study re-runs the full depth x launch grid **on the A-init**, to ask whether
the depth/launch verdicts survive when the whole grid starts from `f0_kpinv`. Every
arm (including the M4 a-priori control) is put on the A-init, for an apples-to-apples
comparison at the new init; the default-init numbers from the first study are the
reference.

## Setup

`--methods nstep` with `--f-init zero --k-init pinv`
([`scripts/self_distillation_losses.py`](../scripts/self_distillation_losses.py)):
a pure a-priori control plus one SD+anchor arm per ladder depth `n = 1..4`
(`window=4`), all adapting `(F, H, K)` with `beta2=0`, constant gain, no projection
/ warm start / RTRL, **and all initialized at `f0_kpinv`**. The two new CLI flags
`--f-init {default,zero}` / `--k-init {zero,pinv}` set the init for the whole
`nstep` grid at once; the defaults reproduce the original default-init sweep exactly
(so `n=1, keep_launch=True` is still bit-identical to the previous single-step SD
term at the default init).

Per-arm step sweep `--step-size in {0.01, 0.03, 0.1, 0.3, 1.0}`, `sd6_od2`,
`eps=0.1`, `N=16`, `seed=0`, on the same CUDA-drawn system as the mechanism / depth
/ init studies (floor 0.0546). Two sweeps at the converged `L=300000` horizon,
identical except for the launch gradient:

- `output/sd_depth_ainit_L300k/ss*` -- `keep_launch=True`
- `output/sd_depth_ainit_detach_L300k/ss*` -- `keep_launch=False`

Driver: [`scripts/submit_sd_depth_ainit_sweep.sh`](../scripts/submit_sd_depth_ainit_sweep.sh)
(loops both launch variants x step sizes). Readers (all prefix-parametrized, reused
unchanged): [`scripts/analyze_mech_sweep.py`](../scripts/analyze_mech_sweep.py)
(envelope table), [`scripts/plot_sd_depth_sweep.py`](../scripts/plot_sd_depth_sweep.py)
(depth grid), [`scripts/plot_sd_launch_convergence.py`](../scripts/plot_sd_launch_convergence.py)
(keep-vs-detach convergence overlay),
[`scripts/plot_sd_depth_convergence.py`](../scripts/plot_sd_depth_convergence.py)
(per-depth convergence).

As before, `excess` (median analytical error among stable trajectories, relative to
floor) and `%stbl` (tail fraction closed-loop stable) MUST be read together -- but on
the A-init this caveat almost evaporates: every cell here is essentially 100% stable
(see the stability section), so `excess` is read off full populations, not the
marginal ones that made the default-init small-step numbers noisy.

## Verdict

Four findings at the converged `L=300000` horizon. The depth verdict survives intact;
the launch verdict is reshaped by the A-init's contractive basin.

1. **`n=1` still wins the converged floor; depth is still a speed/floor knob.**
   Exactly as at the default init, `n=1` is the lowest-floor SD arm in both launch
   variants, and each extra autonomous step raises the steady floor. A deeper ladder
   still converges *faster* early (at `eta=0.03`, ctx=3k: `n=4` 3.47% vs `n=1` 6.48%)
   and is overtaken by `n=1` around ctx ~30-100k. The `F^k` ladder is not worth
   building past a single step on this stationary problem -- unchanged.

2. **The A-init dissolves the launch-gradient *stability* problem, so `keep_launch`
   is now safe -- and fastest.** The entire reason to detach in the first study was
   stability: at the default init `keep_launch=True` diverged **9-15 of 16**
   trajectories at `eta=1.0` (the deeper the ladder, the worse). On the A-init the
   grid is **100% stable everywhere** (`ndiv=0`) except the single `n=4, eta=1.0`
   corner (1/16), for *both* launch variants -- because the contractive `F=0`-grown
   basin (`|λ(F_hat)| ≈ 0.82` at usable gains, inside the true spectrum) never lets
   the launch gradient push `F` over the unit circle. With stability no longer a
   reason to detach, `keep_launch=True`'s faster convergence makes it the better
   operating point at this horizon.

3. **Detach still reaches a slightly lower floor where it has converged, but its
   only remaining advantage is that floor.** At matched *converged* cells detach is
   lower (e.g. `eta=0.03, n=1`: detach 0.036% vs keep 0.074%; and detach is lower at
   every depth at `eta=1.0`). So the "detach bottoms out lower" half of the original
   tradeoff holds; the "detach is also more stable" half does not -- the A-init
   already supplied the stability.

4. **The floor is unchanged by the init (as the init study predicted).** All the
   converged best cells sit in the same 0.02-0.04% ball as the default-init study
   (keep `n=1` @`eta=0.01`: 0.024-0.027%; detach `n=1` @`eta=0.03`: 0.036%). The
   A-init changes the *trajectory and the basin*, not the destination.

**Best configuration found (converged at 300k): `keep_launch=True, n=1, eta=0.01`,
0.027% excess at 100% stability** -- vs the default-init study's best (detached,
`n=1`, `eta=0.01`, 0.018%, but read off a marginal-stability population). See the
convergence caveat below: the detached `eta=0.01` column is still under-converged at
300k here (as it was at the default init, which needed `L=1000000` to resolve it), so
the absolute floor race between keep and detach at the smallest gain is not settled at
this horizon -- but it no longer matters operationally, because keep is now stable.

## Per-arm best (each at its own best step; L=300000)

floor = 0.0546. `%stbl` at the best step in parentheses.

| arm | keep=True best | detach best |
|---|---|---|
| M4 (a-priori control) | 0.089% @0.01 (100%) | 0.092% @0.01 (100%) |
| SD+anchor n=1 | **0.027% @0.01 (100%)** | 0.044% @0.03 (100%) |
| SD+anchor n=2 | 0.055% @0.01 (100%) | 0.068% @0.01 (100%) |
| SD+anchor n=3 | 0.081% @0.01 (100%) | 0.072% @0.01 (100%) |
| SD+anchor n=4 | 0.095% @0.01 (100%) | 0.083% @0.01 (100%) |

Note the detached `n=1` best falls at `eta=0.03`, not `eta=0.01`: the detached
`eta=0.01` cell (0.257%) is still descending at 300k (convergence caveat below), so
its own-best-step lands one gain higher.

## Steady excess grids: 100% stable everywhere, contractive spectrum

Steady excess (%) per `(arm, step)`, tail `%stable` in parentheses. `ndiv=0` for
every cell except `n=4, eta=1.0` (`ndiv=1`, 94% stable) in both variants.

### keep_launch=True (`sd_depth_ainit_L300k`)

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.089 (100%) | 0.269 (100%) | 0.777 (100%) | 2.197 (100%) | 8.832 (100%) |
| SD+anchor n=1 | **0.027 (100%)** | 0.080 (100%) | 0.263 (100%) | 0.685 (100%) | 2.167 (100%) |
| SD+anchor n=2 | 0.055 (100%) | 0.165 (100%) | 0.461 (100%) | 1.094 (100%) | 3.418 (100%) |
| SD+anchor n=3 | 0.081 (100%) | 0.230 (100%) | 0.592 (100%) | 1.346 (100%) | 3.982 (100%) |
| SD+anchor n=4 | 0.095 (100%) | 0.269 (100%) | 0.669 (100%) | 1.487 (100%) | 4.229 (94%) |

### keep_launch=False / detached (`sd_depth_ainit_detach_L300k`)

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.092 (100%) | 0.268 (100%) | 0.779 (100%) | 2.197 (100%) | 8.832 (100%) |
| SD+anchor n=1 | 0.257* (100%) | **0.044 (100%)** | 0.144 (100%) | 0.444 (100%) | 1.457 (100%) |
| SD+anchor n=2 | 0.068 (100%) | 0.106 (100%) | 0.357 (100%) | 1.026 (100%) | 2.614 (100%) |
| SD+anchor n=3 | 0.072 (100%) | 0.172 (100%) | 0.566 (100%) | 1.476 (100%) | 3.181 (100%) |
| SD+anchor n=4 | 0.083 (100%) | 0.214 (100%) | 0.702 (100%) | 1.744 (100%) | 3.513 (94%) |

`*` detached `n=1, eta=0.01` is under-converged at 300k (still descending; see below).

Median `|λ(F_hat)|` reads **0.82-0.83 at `eta ≤ 0.1`** and drifts down to 0.67-0.77
at the bruising `eta=1.0` in both variants -- i.e. every cell sits *inside* the true
spectrum `|eig(F)| ∈ [0.778, 0.900]` or contractive of it, never on the unit circle.
This is the A-init's contractive basin holding across the entire depth x launch grid.
Contrast the default-init study, where the `fI` init drifted to `|λ| ≈ 1.00` under SD
self-consistency pressure and stability at small gains was marginal (35-67% even at
1M) -- none of that fragility appears here.

At `eta=1.0`, detach still has lower excess than keep at every depth (n=1: 1.457% vs
2.167%; n=4: 3.513% vs 4.229%), the same "detach lower at aggressive gain" ordering
as the original -- but now *both* are stable rather than detach being stable and keep
blowing up.

![keep_launch=True depth sweep](../output/sd_depth_ainit_L300k_depth_sweep.png)

![keep_launch=False depth sweep](../output/sd_depth_ainit_detach_L300k_depth_sweep.png)

## Convergence: keep is faster, detach's low-gain column is under-converged

At `n=1`, excess vs context length (%) -- the same speed/floor split as the original,
now with keep the more attractive side because its stability cost is gone:

| ctx | keep @0.01 | detach @0.01 | keep @0.03 | detach @0.03 |
|---|---|---|---|---|
| 3,000 | 19.82 | 22.88 | 6.48 | 7.05 |
| 10,000 | 5.80 | 6.17 | 2.13 | 3.52 |
| 30,000 | 2.02 | 3.42 | 0.462 | 1.90 |
| 100,000 | 0.259 | 1.671 | 0.076 | 0.199 |
| 200,000 | 0.028 | 0.577 | 0.056 | 0.053 |
| 300,000 | **0.024** | 0.170 | 0.074 | **0.036** |

- **keep @0.01 is essentially converged by 200k** (0.028% → 0.024%), and is the
  lowest-excess cell at this horizon.
- **detach @0.01 is still descending at 300k** (0.577% → 0.170%): the exact
  under-convergence signature the first study flagged at the default init, which only
  resolved at `L=1000000` (detach @0.01 there: 0.257% at 300k → 0.018% at 1M). So the
  keep-vs-detach floor race at the *smallest* gain is not settled at 300k; to pin the
  detached `eta=0.01` floor, escalate that column with `-L 1000000`.
- **at `eta=0.03`, both are converged and detach is lower** (0.036% vs 0.074%) --
  the "detach bottoms out lower" tradeoff, cleanly visible where both have converged.

![keep vs detach convergence, A-init](../output/sd_depth_ainit_launch_convergence.png)

Solid = keep, dashed = detach, color = depth. Left `eta=0.01`, right `eta=0.03`. Keep
descends faster throughout; the depths fan out early (deeper faster) and re-order to
`n=1`-lowest at the tail.

### Depth is the same speed/floor tradeoff

Excess vs context (%) at `eta=0.03`, `keep_launch=True`, per depth:

| ctx | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| 3,000 | 6.48 | 4.12 | 3.57 | 3.47 |
| 10,000 | 2.13 | 1.33 | 1.07 | 1.04 |
| 30,000 | 0.462 | 0.238 | 0.276 | 0.266 |
| 100,000 | 0.076 | 0.131 | 0.215 | 0.265 |
| 300,000 | **0.074** | 0.158 | 0.244 | 0.297 |

Early (ctx ≤ 10k) the deeper ladders lead by ~2x (richer multi-horizon gradient),
the curves cross around ctx ~30-100k, and by the tail the ordering fully inverts to
`n=1` lowest -- identical to the default-init study.

![A-init depth convergence, eta=0.03](../output/sd_depth_ainit_depth_convergence_eta0p03.png)

Left = keep, right = detach; color = depth.

## Stability: the launch-gradient instability is gone

Diverged trajectories (out of 16) at `eta=1.0`, `L=300000`, A-init vs the default
init (from the first study):

| arm | default keep | default detach | A-init keep | A-init detach |
|---|---|---|---|---|
| SD+anchor n=1 | 11 | 0 | **0** | **0** |
| SD+anchor n=2 | 9 | 0 | **0** | **0** |
| SD+anchor n=3 | 15 | 0 | **0** | **0** |
| SD+anchor n=4 | 15 | 2 | **1** | **1** |

At the default init, detaching the launch was the fix for the recursive-gradient
blow-up that tipped `keep_launch=True` over the unit circle at aggressive gains. The
A-init makes that fix redundant: growing `F`'s dynamics *up from zero* keeps the
learned `F` contractive, so the launch gradient has no near-unit-circle `F` to
destabilize. Both launch variants are now essentially fully stable across the whole
grid; the only casualty anywhere is one trajectory in the extreme `n=4, eta=1.0`
corner, in both variants alike.

## Reading

1. **Depth verdict: unchanged.** `n=1` wins the converged floor; deeper ladders buy
   early convergence speed at the cost of the asymptotic floor. The ladder is not
   worth building past one step on this stationary problem, at either init.

2. **Launch verdict: reshaped.** The original launch tradeoff was
   speed (keep) vs floor+stability (detach) -- detach won two of three. On the A-init
   the *stability* leg moves to the init: both variants are stable everywhere, so the
   launch gradient reduces to a pure speed/floor knob. Keep converges faster and is
   now safe; detach still reaches a slightly lower floor where it converges. At a
   finite deployable horizon, **`keep_launch=True` is the better default on the
   A-init** -- the opposite of the default-init recommendation, and for a clean
   reason: the thing detach was buying (stability) is already paid for by the init.

3. **The A-init is a free accelerator here too.** Just as the init study found the
   A-init gives β2-like early descent at zero floor cost, this study finds it gives
   the *launch gradient* back for free: `keep_launch=True`'s convergence speedup no
   longer carries a stability tax. The best converged cell (keep, `n=1`, `eta=0.01`,
   0.027%) matches the original floor ball while being 100% stable throughout, not
   marginally stable.

## Caveats / follow-ups

- **Detached `eta=0.01` is under-converged at 300k** (0.170%, still descending). The
  exact keep-vs-detach floor at the smallest gain needs `L=1000000` on that column
  (as the init study needed 1M to resolve the same cell). It does not change the
  operational verdict (keep is stable and fastest), only the absolute detach floor.
  Escalate with: `LAUNCHES=detach STEPS=0.01 OUT_PREFIX=sd_depth_ainit_L1M bash
  scripts/submit_sd_depth_ainit_sweep.sh -L 1000000`.
- **Gain migration per depth** (the `‖K‖`, `‖KH − H^+H‖` A→K diagnostic from the init
  study, [`scripts/plot_init_gain_migration.py`](../scripts/plot_init_gain_migration.py))
  was not run here; it would show whether deeper ladders / the launch gradient change
  how far the gain walks from the replace limit. Optional, if the migration picture
  across the ladder is wanted.
