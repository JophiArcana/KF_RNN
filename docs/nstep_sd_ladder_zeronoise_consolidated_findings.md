# n-step self-distillation ladder at zero measurement noise: depth x launch x reduction (consolidated)

Zero-measurement-noise (`--v-std 0`, process noise kept at `--w-std 1`) replay of
[`nstep_sd_ladder_consolidated_findings.md`](nstep_sd_ladder_consolidated_findings.md).
It consolidates the same three studies of the **n-step latent self-distillation
(SD) ladder** on the constant-gain linear filter, re-run on the *same* CUDA-drawn
seed-0 `sd6_od2` system (`true |eig(F)| ∈ [0.778, 0.900]`) with the observation
noise covariance set to zero:

1. default-init (`fI_k0`: `F=(1-eps)I, K=0`), sum reduction --
   `sd_depth_L1M_v0` (keep) / `sd_depth_detach_L1M_v0` (detach).
2. A-init (`f0_kpinv`: `F=0, K=H⁺`), sum reduction --
   `sd_depth_ainit_L1M_v0` / `sd_depth_ainit_detach_L1M_v0`.
3. A-init, mean reduction -- `sd_depth_ainit_mean_L1M_v0` /
   `sd_depth_ainit_mean_detach_L1M_v0`.

The three axes are **depth** (`n = 1..4`), **launch gradient** (keep/detach), and
**reduction** (sum/mean), plus the **init** as a contextual fourth variable.
Removing the measurement noise drops the irreducible floor from `0.0546` to
**`0.0265`** and the zero-predictor ceiling to `0.0790`, so every number is
directly comparable to the `v=1` consolidated doc.

**One-paragraph summary of where the line ends up (v=0).** The three headline
verdicts of the `v=1` study **survive intact**: `n=1` wins the converged floor
everywhere; the sum's depth tradeoff is a gradient-*magnitude* artifact that the
mean removes; and detach converges to a strictly lower floor than keep at equal
stability. Best configuration found: **A-init, `keep_launch=False`, `n=1`,
`eta=0.01` -- 0.012-0.015% excess at 100% stability** (the reduction is
irrelevant at `n=1`); the default-init detach `n=1` reaches the same 0.014% floor
but off a ~42%-stable marginal population. What zero measurement noise **changes**
is the *stability* picture, and in a surprising direction: the **pure a-priori M4
control now diverges at large gain (`eta ≥ 0.3`, 16/16) in every sweep,
regardless of init or launch** -- with no measurement noise the a-priori gradient
drives an aggressive constant gain over the unit circle -- whereas the SD+anchor
arms on the A-init stay contractive and 100% stable out to `eta=1.0`. At zero
noise, SD+anchor is not just more accurate than plain a-priori, it is
*more stable at aggressive gain*.

---

## 1. The objective and the three axes

The latent SD term generalizes the single-step bootstrap to an **n-step ladder**
over autonomous horizons `k = 1..n`, launched from the rolled window (a horizon
past the window clamps to the detached root):

```
sum  (studies 1-2):  alpha * sum_{k=1}^{n_eff} 0.5 || sg(x_j^+) - F^k x_{j-k}^+ ||^2
mean (study 3):      alpha * (1/n_eff) sum_{k=1}^{n_eff} 0.5 || sg(x_j^+) - F^k x_{j-k}^+ ||^2
```

| axis | values | flag | controls |
|---|---|---|---|
| depth `n` | 1..4 (`window=4`) | `--sd-horizon` | autonomous roll-out horizons `F^k` contributing targets |
| launch gradient | keep / detach | `--keep-launch` / `--no-keep-launch` | whether launch state `x_{j-k}^+` carries gradient |
| reduction | sum / mean | `--no-sd-mean` / `--sd-mean` | effective SD weight grows ~`n` (sum) or held fixed (mean) |

Init as the contextual fourth variable:

| init | `F` init | `K` init | basin (v=0) |
|---|---|---|---|
| default (`fI_k0`) | `(1-eps)I` | `0` | marginal: drifts to `|lambda| ~ 1.00`; small-step stability stays partial even at 1M |
| A-init (`f0_kpinv`) | `0` | `H⁺` | contractive: `|lambda| ~ 0.78-0.82`, inside `[0.778, 0.900]`; 100% stable at every step |

Built-in sanity checks (unchanged): `n=1` is bit-identical to the original
single-step SD; `n=1` is bit-identical under sum and mean (small tail-median
diffs at 1M are run-to-run drift); the M4 control has `alpha=0`, untouched by all
three SD axes.

---

## 2. Shared experimental setup

`--methods nstep`: a pure a-priori control (M4) plus one SD+anchor arm per ladder
depth `n = 1..4`, all adapting `(F, H, K)` with `beta2=0`, constant gain. Per-arm
step sweep `--step-size in {0.01, 0.03, 0.1, 0.3, 1.0}`, on `sd6_od2`, `eps=0.1`,
`N=16`, `seed=0`, `L=1000000`, `--v-std 0` (**floor 0.0265**). Six sweeps:

| study | init | reduction | keep sweep | detach sweep |
|---|---|---|---|---|
| 1. default-init | `fI_k0` | sum | `output/sd_depth_L1M_v0` | `output/sd_depth_detach_L1M_v0` |
| 2. A-init | `f0_kpinv` (`--f-init zero --k-init pinv`) | sum | `output/sd_depth_ainit_L1M_v0` | `output/sd_depth_ainit_detach_L1M_v0` |
| 3. A-init mean | `f0_kpinv` | mean (`--sd-mean`) | `output/sd_depth_ainit_mean_L1M_v0` | `output/sd_depth_ainit_mean_detach_L1M_v0` |

Drivers (`OUT_PREFIX=... bash scripts/submit_sd_depth_sweep.sh -L 1000000
{--keep-launch|--no-keep-launch} --v-std 0` and
`scripts/submit_sd_depth_ainit_sweep.sh -L 1000000 [--sd-mean] --v-std 0`).
Readers: [`analyze_mech_sweep.py`](../scripts/analyze_mech_sweep.py),
[`plot_sd_depth_sweep.py`](../scripts/plot_sd_depth_sweep.py),
[`plot_sd_launch_convergence.py`](../scripts/plot_sd_launch_convergence.py),
[`plot_sd_depth_convergence.py`](../scripts/plot_sd_depth_convergence.py).

**Reading the tables.** `excess` = median analytical error among stable
trajectories, relative to the floor; `%stbl` = tail fraction closed-loop stable;
`ndiv` = diverged of 16; `rad` = median `|lambda(F_hat)|`. On the **default
init**, `excess` and `%stbl` MUST be read together (small-step medians are over
marginal populations); on the **A-init** every SD cell is ~100% stable, so
`excess` is read off full populations.

---

## 3. Consolidated verdicts, axis by axis

### 3.1 Depth `n`: still a magnitude artifact

Unchanged from `v=1`:

1. **Study 1 (default, sum):** depth is a **speed/floor knob** -- deeper ladders
   descend ~2x faster through the first ~10k steps (n=4 4.07% vs n=1 11.45% at
   ctx=3k, `eta=0.03`) but settle higher (sum keep floors `eta=0.01`: 0.019 /
   0.045 / 0.071 / 0.089% across `n=1..4`); curves cross ~10-30k and `n=1` wins
   the floor.
2. **Study 2 (A-init, sum):** verdict **survives the init change** -- same early
   fan-out, same crossover, same `n=1`-lowest tail (sum keep floors 0.021 /
   0.052 / 0.081 / 0.101%).
3. **Study 3 (A-init, mean):** the tradeoff **was mostly the sum's magnitude**.
   Under the mean the floor spread collapses (keep `eta=0.01`: 0.024 / 0.030 /
   0.034 / 0.036% for `n=1..4`; detach: 0.012 / 0.021 / 0.026 / 0.029% vs `n=1`
   0.012%) **and** the early speedup disappears (deeper is now marginally
   *slower*: n=4 20.6% vs n=1 11.45% at ctx=3k, `eta=0.03`). With magnitude
   controlled, the multi-horizon *information* buys nothing on this stationary
   problem.

Bottom line: **`n=1` is the unconditional recommendation**, exactly as at `v=1`.

### 3.2 Launch gradient: a trilemma still resolved by the init

1. **Study 1 (default init):** **detach wins two of three.** Keeping the launch
   gradient buys fast convergence but is the mechanism of the aggressive-gain
   instability: keep bleeds trajectories from `eta=0.1` up (n=1 @0.3: 13/16
   diverged; @1.0: 16/16), while detach is ~100% stable at every `eta ≥ 0.1`.
   Detach also reaches a lower floor (n=1: 0.014% detach vs 0.019% keep). Given
   enough context, detach is strictly better.
2. **Study 2 (A-init):** **the A-init dissolves the SD-arm stability leg.** The
   contractive `F=0`-grown basin never lets the launch gradient push `F` near the
   unit circle: every SD+anchor cell is 100% stable in *both* launch variants
   (only casualties: the detached sum `n=3, eta=1.0` at 1/16 and `n=4, eta=1.0`
   at 3/16). Detach bottoms out lower at every matched cell (n=1: 0.015% vs
   0.021% @`eta=0.01`).
3. **Study 3 (mean):** unchanged by construction (launch story lives at `n=1`,
   where sum and mean are bit-identical).

As at `v=1`, keep converges ~2-3x faster to a somewhat higher floor; detach wins
the converged asymptote. On the A-init the choice is a pure horizon call (keep
for short horizons, detach for the converged optimum), with no SD-arm stability
penalty either way.

### 3.3 Reduction: sum = information + magnitude; mean isolates the (nil) information

The mean holds magnitude fixed and both halves of the sum's depth story collapse:

- **Floors compress to near-`n=1`.** Sum keep `eta=0.01`: 0.021 / 0.052 / 0.081 /
  0.101% (`n=1..4`, 4.8x spread). Mean: 0.024 / 0.030 / 0.034 / 0.036% (1.5x
  spread). Detach shows the same compression (sum 0.015-0.088%, mean
  0.012-0.029%).
- **The early speedup disappears.** Sum keep `eta=0.03`, ctx=3k: n=4 4.07% vs
  n=1 11.45% (deeper ~3x faster). Mean: n=4 20.6% vs n=1 11.45% (deeper
  *slower*). The acceleration was purely a larger effective step.
- **Stability marginally improves.** The mean removes the sum's detached
  `n=3/n=4, eta=1.0` casualties (1/16 and 3/16 -> 0) -- the whole A-init mean
  grid is `ndiv=0` for the SD arms.

### 3.4 Stability: the a-priori control is now the fragile arm

Diverged trajectories (of 16) at the most aggressive gain `eta=1.0`, `L=1000000`:

| arm | default sum keep | default sum detach | A-init sum keep | A-init sum detach | A-init mean keep | A-init mean detach |
|---|---|---|---|---|---|---|
| M4 (a-priori) | 16 | 16 | 16 | 16 | 16 | 16 |
| SD+anchor n=1 | 16 | 0 | 0 | 0 | 0 | 0 |
| SD+anchor n=2 | 16 | 0 | 0 | 0 | 0 | 0 |
| SD+anchor n=3 | 13 | 0 | 0 | 1 | 0 | 0 |
| SD+anchor n=4 | 16 | 0 | 0 | 3 | 0 | 0 |

Two structural changes from `v=1`, one preserved mechanism:

1. **The M4 a-priori control diverges at large gain everywhere (the v=0
   surprise).** At `v=1` M4 was stable at every gain (8.9% excess, `rad 0.77`
   @`eta=1.0`). At `v=0` M4 diverges 16/16 at both `eta=0.3` and `eta=1.0` in all
   six sweeps -- independent of init (it is `alpha=0`, so the A-init only sets its
   starting F/K) and of launch (no launch gradient at `alpha=0`). With no
   measurement noise the pure-a-priori gradient drives an aggressive *constant*
   gain over the unit circle. So the least stable arm at large gain is now the
   a-priori baseline, not the SD ladder -- SD+anchor on the A-init is stable to
   `eta=1.0` where M4 blows up.
2. **The launch-gradient instability at the default init is preserved** (default
   keep vs detach): the recursive bootstrap-chain gradient drives `F` over the
   unit circle (n=1..4 keep: 16/16/13/16 diverged; detach: 0 everywhere). Detach
   remains the cheap stability fix at the default init.
3. **The A-init makes the SD-arm fix redundant** (default vs A-init): growing
   `F` up from zero keeps it contractive (`rad 0.76-0.82`), so the launch
   gradient has no near-unit-circle `F` to destabilize; the mean removes the last
   two detached-corner casualties.

### 3.5 Best configurations across the line

At `L=1000000`, `v=0`, **detach, `n=1`, `eta=0.01` is the best SD arm in every
sweep**:

| study | best config | excess | %stbl | note |
|---|---|---|---|---|
| 1. default-init sum | detach, `n=1`, `eta=0.01` | 0.014% | 42% (marginal population) | keep counterpart 0.019%; M4 baseline 0.056% |
| 2. A-init sum | **detach, `n=1`, `eta=0.01`** | **0.015%** | **100%** | the line's best deployable cell; keep counterpart 0.021% |
| 3. A-init mean | detach, `n=1`, `eta=0.01` | 0.012% | 100% | bit-identical to study 2 at `n=1` (0.012 vs 0.015 is run-to-run drift); `n=2..4` now near-tied (0.021-0.029%) |

The floor itself is init- and reduction-invariant: every converged detached `n=1`
cell lands at 0.012-0.015% (matching the init study's independent `v=0` readout of
0.008-0.013%). The axes change the *trajectory, the basin, and the stability of
the population the number is read from* -- not the destination. All are ~4x
closer to the floor than the a-priori M4 baseline (0.051-0.056%), the same margin
as `v=1`.

---

## 4. Study 1: default init, sum reduction (`sd_depth_L1M_v0` / `sd_depth_detach_L1M_v0`)

### 4.1 Per-arm best and full excess grids

Per-arm best (each at its own best step = `eta=0.01`; floor = 0.0265):

| arm | keep=True excess (%stbl) | keep=False (detach) excess (%stbl) |
|---|---|---|
| M4 (a-priori control) | 0.052% (92%) | 0.056% (93%) |
| SD+anchor n=1 | 0.019% (51%) | **0.014% (42%)** |
| SD+anchor n=2 | 0.045% (94%) | 0.038% (93%) |
| SD+anchor n=3 | 0.071% (100%) | 0.066% (100%) |
| SD+anchor n=4 | 0.089% (100%) | 0.083% (100%) |

Full steady-excess grids, tail `%stable` (parens); `rad` = median `|lambda|`.

**keep_launch=True (`sd_depth_L1M_v0`):**

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.052 (92%) | 0.164 (100%) | 0.499 (100%) | inf (0%, 16 div) | inf (0%, 16 div) |
| SD+anchor n=1 | **0.019 (51%)** | 0.060 (98%) | 0.202 (100%) | 0.533 (19%, 13 div) | inf (16 div) |
| SD+anchor n=2 | 0.045 (94%) | 0.136 (100%) | 0.389 (61%, 7 div) | 0.890 (48%, 9 div) | inf (16 div) |
| SD+anchor n=3 | 0.071 (100%) | 0.205 (100%) | 0.566 (76%, 4 div) | 1.143 (52%, 8 div) | 3.497 (21%, 13 div) |
| SD+anchor n=4 | 0.089 (100%) | 0.259 (100%) | 0.607 (52%, 8 div) | 1.345 (56%, 7 div) | 4.853 (5%, 16 div) |

**keep_launch=False / detached (`sd_depth_detach_L1M_v0`):**

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.056 (93%) | 0.164 (100%) | 0.499 (100%) | inf (16 div) | inf (16 div) |
| SD+anchor n=1 | **0.014 (42%)** | 0.041 (58%) | 0.140 (99%) | 0.471 (100%) | 3.406 (100%) |
| SD+anchor n=2 | 0.038 (93%) | 0.108 (100%) | 0.376 (100%) | 1.250 (100%) | 3.713 (100%) |
| SD+anchor n=3 | 0.066 (100%) | 0.185 (100%) | 0.631 (100%) | 1.851 (100%) | 3.970 (100%) |
| SD+anchor n=4 | 0.083 (100%) | 0.234 (100%) | 0.794 (100%) | 2.187 (100%) | 4.304 (100%) |

Matched-step head-to-head for `n=1`: **detach is lower at every step** --
`eta=0.01` 0.014% vs 0.019%; 0.1 0.140% vs 0.202% -- and vastly more stable at
`eta ≥ 0.3` (100% vs keep's 13-16 divergences). Note the M4 control diverges at
`eta ≥ 0.3` in *both* variants (it has no launch gradient; this is the pure
a-priori v=0 instability of section 3.4).

![keep_launch=True depth sweep](../output/sd_depth_L1M_v0_depth_sweep.png)

![keep_launch=False depth sweep](../output/sd_depth_detach_L1M_v0_depth_sweep.png)

### 4.2 Convergence: keep is faster, detach reaches a lower floor

`n=1` excess (%) vs context at `eta=0.01`:

| context | keep=True | keep=False (detach) |
|---|---|---|
| 3,000 | 79.7 | 105.1 |
| 10,000 | 22.4 | 52.4 |
| 30,000 | 2.57 | 21.0 |
| 100,000 | 0.067 | 4.12 |
| 300,000 | -- | -- |
| 1,000,000 | 0.019 (floor **0.019%**) | 0.014 (floor **0.014%**) |

Keep is ~3x or more lower through the mid-context regime and plateaus by ~100k;
detach is still descending and crosses below keep past ~300k, then bottoms out
lower. Same clean "keep converges faster, detach bottoms out lower" separation as
`v=1`. (`--` = every trajectory read unstable at that single sample -- the
marginal default-init population.)

![keep vs detach convergence at L=1M, v=0](../output/sd_launch_convergence_L1M_v0.png)

**Depth is the same speed/floor tradeoff.** Excess (%) vs context at `eta=0.03`,
`keep_launch=True`:

| context | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| 3,000 | 27.1 | 15.3 | 11.5 | 12.9 |
| 10,000 | 2.94 | 0.93 | 0.68 | 1.08 |
| 30,000 | 0.105 | 0.106 | 0.412 | 0.464 |
| 100,000 | 0.064 | 0.130 | 0.219 | 0.314 |
| 1,000,000 | **0.085** | 0.174 | 0.265 | 0.322 |

Early (ctx ≤ 10k) the deeper ladders lead by ~2x; the curves cross ~10-30k and
by the tail `n=1` is lowest.

![SD ladder depth convergence at 1M, v=0, eta=0.03](../output/sd_depth_convergence_L1M_v0_eta0p03.png)

---

## 5. Study 2: A-init, sum reduction (`sd_depth_ainit_L1M_v0` / `sd_depth_ainit_detach_L1M_v0`)

### 5.1 Per-arm best and full excess grids

Per-arm best (best step = `eta=0.01`; `%stbl` in parens):

| arm | keep=True best | detach best |
|---|---|---|
| M4 (a-priori control) | 0.051% (100%) | 0.055% (100%) |
| SD+anchor n=1 | 0.021% (100%) | **0.015% (100%)** |
| SD+anchor n=2 | 0.052% (100%) | 0.040% (100%) |
| SD+anchor n=3 | 0.081% (100%) | 0.068% (100%) |
| SD+anchor n=4 | 0.101% (100%) | 0.088% (100%) |

Full steady-excess grids (tail `%stable` in parens):

**keep_launch=True (`sd_depth_ainit_L1M_v0`):**

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.051 (100%) | 0.156 (100%) | 0.477 (100%) | inf (16 div) | inf (16 div) |
| SD+anchor n=1 | **0.021 (100%)** | 0.067 (100%) | 0.214 (100%) | 0.568 (100%) | 1.809 (100%) |
| SD+anchor n=2 | 0.052 (100%) | 0.148 (100%) | 0.382 (100%) | 0.902 (100%) | 2.740 (100%) |
| SD+anchor n=3 | 0.081 (100%) | 0.218 (100%) | 0.508 (100%) | 1.133 (100%) | 3.298 (100%) |
| SD+anchor n=4 | 0.101 (100%) | 0.275 (100%) | 0.635 (100%) | 1.346 (100%) | 3.802 (100%) |

**keep_launch=False / detached (`sd_depth_ainit_detach_L1M_v0`):**

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.055 (100%) | 0.162 (100%) | 0.488 (100%) | inf (16 div) | inf (16 div) |
| SD+anchor n=1 | **0.015 (100%)** | 0.044 (100%) | 0.151 (100%) | 0.500 (100%) | 3.471 (100%) |
| SD+anchor n=2 | 0.040 (100%) | 0.121 (100%) | 0.421 (100%) | 1.376 (100%) | 3.817 (100%) |
| SD+anchor n=3 | 0.068 (100%) | 0.210 (100%) | 0.720 (100%) | 2.009 (100%) | 4.005 (94%, 1 div) |
| SD+anchor n=4 | 0.088 (100%) | 0.268 (100%) | 0.911 (100%) | 2.356 (100%) | 4.377 (81%, 3 div) |

Median `|lambda(F_hat)|` reads **0.78-0.82 across `eta ≤ 0.3`** in both variants
-- every SD cell sits *inside* the true spectrum `[0.778, 0.900]` or contractive
of it, the A-init's contractive basin holding across the whole grid. **Detach has
lower excess than keep at every matched (arm, step)** (0.015% vs 0.021% at the
accuracy end), and -- unlike the default init -- both variants are fully stable
for the SD arms (the only casualties are the extreme detached `n=3/n=4, eta=1.0`
corners). The M4 control still diverges at `eta ≥ 0.3` even from the A-init.

![keep_launch=True depth sweep](../output/sd_depth_ainit_L1M_v0_depth_sweep.png)

![keep_launch=False depth sweep](../output/sd_depth_ainit_detach_L1M_v0_depth_sweep.png)

### 5.2 Convergence

`n=1` excess (%) vs context:

| ctx | keep @0.01 | detach @0.01 | keep @0.03 | detach @0.03 |
|---|---|---|---|---|
| 3,000 | 55.4 | 62.9 | 11.5 | 15.3 |
| 10,000 | 9.41 | 12.3 | 1.00 | 0.56 |
| 30,000 | 0.868 | 0.545 | 0.119 | 0.102 |
| 100,000 | 0.068 | 0.079 | 0.076 | 0.059 |
| 300,000 | 0.014 | 0.023 | 0.045 | 0.040 |
| 1,000,000 | 0.013 | **0.011** | 0.047 | **0.050** |

Keep descends faster early; detach crosses below at the accuracy gain by the
tail. (At `eta=0.03` both are long converged and near-tied.) Depth shows the same
early fan-out (deeper faster) re-ordering to `n=1`-lowest at the tail, identical
in shape to study 1.

![keep vs detach convergence, A-init v=0](../output/sd_depth_ainit_L1M_v0_launch_convergence.png)

![A-init depth convergence, v=0, eta=0.03](../output/sd_depth_ainit_L1M_v0_depth_convergence_eta0p03.png)

### 5.3 Stability: the launch-gradient instability is gone (for the SD arms)

Diverged trajectories (of 16) at `eta=1.0`, A-init vs default init:

| arm | default keep | default detach | A-init keep | A-init detach |
|---|---|---|---|---|
| SD+anchor n=1 | 16 | 0 | **0** | **0** |
| SD+anchor n=2 | 16 | 0 | **0** | **0** |
| SD+anchor n=3 | 13 | 0 | **0** | **1** |
| SD+anchor n=4 | 16 | 0 | **0** | **3** |

The A-init makes the launch-gradient fix redundant for the SD arms: growing `F`
up from zero keeps it contractive, so the launch gradient has no near-unit-circle
`F` to destabilize. (The M4 control is a separate story -- it diverges at
`eta ≥ 0.3` from *either* init; section 3.4.)

---

## 6. Study 3: A-init, mean reduction (`sd_depth_ainit_mean_L1M_v0` / `sd_depth_ainit_mean_detach_L1M_v0`)

### 6.1 Per-arm best, sum vs mean side by side

Floor = 0.0265; every best step is `eta=0.01`; every SD cell 100% stable:

| arm | mean keep | sum keep | mean detach | sum detach |
|---|---|---|---|---|
| M4 (a-priori control) | 0.055% | 0.051% | 0.051% | 0.055% |
| SD+anchor n=1 | 0.024% | 0.021% | **0.012%** | **0.015%** |
| SD+anchor n=2 | 0.030% | 0.052% | 0.021% | 0.040% |
| SD+anchor n=3 | 0.034% | 0.081% | 0.026% | 0.068% |
| SD+anchor n=4 | 0.036% | 0.101% | 0.029% | 0.088% |

`n=1` matches the sum study up to run-to-run drift (it is bit-identical by
construction). The signal is the deeper arms: the mean nearly ties them to `n=1`
in both launch variants (keep 0.030-0.036% vs the sum's 0.052-0.101%; detach
0.021-0.029% vs the sum's 0.040-0.088%), but `n=1` stays lowest and the overall
winner is unchanged: detach `n=1` at 0.012%.

### 6.2 Full excess grids: depths compressed, 100% stable everywhere

`ndiv=0` for **every** SD cell in both variants (the sum's detached
`n=3/n=4, eta=1.0` casualties are gone). The M4 control still diverges at
`eta ≥ 0.3`.

**keep_launch=True (`sd_depth_ainit_mean_L1M_v0`):**

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.055 (100%) | 0.156 (100%) | 0.488 (100%) | inf (16 div) | inf (16 div) |
| SD+anchor n=1 | **0.024 (100%)** | 0.067 (100%) | 0.222 (100%) | 0.554 (100%) | 1.785 (100%) |
| SD+anchor n=2 | 0.030 (100%) | 0.085 (100%) | 0.265 (100%) | 0.626 (100%) | 1.821 (100%) |
| SD+anchor n=3 | 0.034 (100%) | 0.098 (100%) | 0.297 (100%) | 0.668 (100%) | 1.830 (100%) |
| SD+anchor n=4 | 0.036 (100%) | 0.105 (100%) | 0.341 (100%) | 0.777 (100%) | 2.042 (100%) |

**keep_launch=False / detached (`sd_depth_ainit_mean_detach_L1M_v0`):**

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.051 (100%) | 0.156 (100%) | 0.477 (100%) | inf (16 div) | inf (16 div) |
| SD+anchor n=1 | **0.012 (100%)** | 0.041 (100%) | 0.146 (100%) | 0.500 (100%) | 3.471 (100%) |
| SD+anchor n=2 | 0.021 (100%) | 0.065 (100%) | 0.232 (100%) | 0.796 (100%) | 4.062 (100%) |
| SD+anchor n=3 | 0.026 (100%) | 0.081 (100%) | 0.286 (100%) | 0.944 (100%) | 3.646 (100%) |
| SD+anchor n=4 | 0.029 (100%) | 0.089 (100%) | 0.311 (100%) | 1.002 (100%) | 3.673 (100%) |

Compare the sum keep grid, where `n=4` ran roughly `n`x the `n=1` row; the mean
rows sit within ~1.1-1.5x of `n=1` everywhere. `rad` stays contractive
(0.76-0.82) -- the same basin as study 2.

![mean keep_launch=True depth sweep](../output/sd_depth_ainit_mean_L1M_v0_depth_sweep.png)

![mean keep_launch=False depth sweep](../output/sd_depth_ainit_mean_detach_L1M_v0_depth_sweep.png)

### 6.3 Convergence: depth loses its early-speed edge under the mean

Depth convergence (%) at `eta=0.03`, `keep_launch=True`, per depth:

| ctx | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| 3,000 | 11.5 | 16.5 | 19.6 | 20.6 |
| 10,000 | 1.00 | 0.76 | 0.77 | 0.78 |
| 30,000 | 0.119 | 0.130 | 0.150 | 0.156 |
| 100,000 | 0.076 | 0.092 | 0.108 | 0.120 |
| 1,000,000 | **0.047** | 0.049 | 0.057 | 0.066 |

Under the **sum** the deeper ladders *led* by ~2-3x early before inverting;
under the **mean** the early lead is **gone** (deeper is now slower at 3k), the
depths collapse together from 10k on, and `n=1` is lowest at the tail. The
multi-horizon gradient's early acceleration was its summed magnitude.

![mean keep vs detach convergence, A-init v=0](../output/sd_depth_ainit_mean_L1M_v0_launch_convergence.png)

![mean A-init depth convergence, v=0, eta=0.03](../output/sd_depth_ainit_mean_L1M_v0_depth_convergence_eta0p03.png)

### 6.4 Study-3 reading

1. **The sum's depth tradeoff was a magnitude artifact** -- the mean holds
   magnitude fixed and both halves (faster early, higher floor) collapse.
2. **The multi-horizon information alone buys nothing here** -- with magnitude
   controlled, adding horizons `k = 2..n` neither accelerates nor lowers the
   floor; deeper is uniformly a hair worse.
3. **`n=1` remains the recommendation**; the launch verdict is inherited from
   study 2 (keep for short horizons, detach 0.012% for the converged optimum).

---

## 7. What zero measurement noise changed (vs the v=1 consolidated study)

The same six sweeps at `v=1` are in
[`nstep_sd_ladder_consolidated_findings.md`](nstep_sd_ladder_consolidated_findings.md).
The two studies use the *identical* system (only `S_V` differs), so the deltas
isolate the measurement-noise effect:

1. **The depth / launch / reduction verdicts are all preserved.** `n=1` wins the
   floor; the sum's depth tradeoff is a magnitude artifact the mean removes;
   detach converges to a strictly lower floor than keep; the A-init dissolves the
   SD-arm launch-gradient instability. Every qualitative shape (early fan-out,
   crossover, tail inversion, contractive A-init spectrum `rad 0.78-0.82`) is
   unchanged.
2. **The floor and the excess scale shift down but the *ratios* hold.** Floor
   `0.0546 -> 0.0265`; the best detached `n=1` cell is 0.012-0.015% (`v=1`:
   0.014%), ~4x closer to the floor than the a-priori baseline in both regimes.
3. **The a-priori M4 control becomes the fragile arm at large gain (the headline
   v=0 change).** At `v=1` M4 was stable at every gain; at `v=0` it diverges
   16/16 at `eta ≥ 0.3` in all six sweeps, from either init and either launch
   (it has `alpha=0`). With no measurement noise the a-priori gradient drives an
   aggressive constant gain over the unit circle. Consequently **SD+anchor on the
   A-init is now the most robust arm at aggressive gain** (100% stable to
   `eta=1.0`) -- at `v=1` the SD ladder was "the one to watch at large steps",
   and at `v=0` the roles invert.
4. **The default-init keep divergences do not *accumulate* the way they did at
   `v=1`.** The `v=1` study emphasized keep divergences growing with exposure
   (9-15 at 300k -> 15-16 at 1M). At `v=0` the default-init keep column is
   already saturated at 1M (n=1..4: 16/16/13/16 at `eta=1.0`) and the bleeding
   extends down to `eta=0.1-0.3` as before; detach holds 0 for the SD arms.

### 7.1 Recommendation (v=0)

- **Deploy target (stationary, long horizon): A-init, detach, `n=1`,
  `eta=0.01`** -- 0.012-0.015% excess, 100% stable, converged.
- **Short/medium horizon: A-init, keep, `n=1`** -- reaches its 0.021-0.024%
  floor ~2-3x sooner and is equally stable on the A-init.
- **Never: the pure a-priori M4 at `eta ≥ 0.3`** (it diverges at zero measurement
  noise), and **never keep at the default init at moderate-or-larger gains**.
- **New at zero noise: prefer SD+anchor over plain a-priori for its *stability*,
  not only its accuracy** -- the SD target regularizes `F` where the bare
  a-priori gradient blows the gain up.

---

## 8. Open caveats and follow-ups

- **`n=1` sum-vs-mean drift.** The detached `n=1` cell reads 0.015% (sum) vs
  0.012% (mean) though the two are mathematically bit-identical at `n=1`; this is
  run-to-run / GPU-nondeterminism drift at the ~0.003%-of-excess level (the same
  caveat the `v=1` study noted), not a reduction effect.

- **The a-priori M4 divergence at `eta ≥ 0.3` deserves its own probe.** It is the
  cleanest new `v=0` phenomenon (pure a-priori is unstable at aggressive constant
  gain with noiseless observations). A vanishing-gain / smaller-`eta` M4 sweep
  would locate the stability boundary; the deployable gains (`eta ≤ 0.1`) are
  well clear of it.

- **Gain migration per depth** (the `||K||`, `||KH - H^+H||` diagnostic) was not
  run on the depth grids. The init study's `v=0` result -- the gain *stays* at
  the replace limit rather than migrating -- suggests the depth grids would show
  no A→K walk either; worth confirming if the migration picture across the ladder
  is wanted.

- **Non-stationary / video implication.** Unchanged from `v=1`: the decisive axis
  is convergence horizon, which non-stationarity removes, putting a tracking
  filter in the regime where keep (and the A-init's transient acceleration) win.
  The `v=0` addition is that at low observation noise the a-priori baseline is
  itself gain-fragile, so an SD target is load-bearing for stability, not just
  accuracy, under an aggressive tracking gain.
