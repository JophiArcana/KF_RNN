# n-step self-distillation ladder: depth x launch gradient x reduction (consolidated)

This document consolidates three sequential studies of the **n-step latent
self-distillation (SD) ladder** on the constant-gain linear filter:

1. [`nstep_sd_ladder_launch_findings.md`](nstep_sd_ladder_launch_findings.md) --
   ladder depth `n = 1..4` x launch gradient (`keep_launch` True/False), on the
   **default init** (`fI_k0`: `F=(1-eps)I, K=0`), sum reduction.
2. [`nstep_sd_ladder_launch_ainit_findings.md`](nstep_sd_ladder_launch_ainit_findings.md) --
   the same depth x launch grid re-run on the **A-init** (`f0_kpinv`: `F=0,
   K=H^+`), sum reduction.
3. [`nstep_sd_ladder_launch_ainit_mean_findings.md`](nstep_sd_ladder_launch_ainit_mean_findings.md) --
   the A-init grid again, with the ladder reduced by a **mean** instead of a sum
   (magnitude-controlled depth).

Together they sweep three axes of the SD objective -- **depth** (how many
autonomous bootstrap horizons), **launch gradient** (whether the launch state
carries gradient), and **reduction** (sum vs mean over the ladder's horizon
terms) -- plus, via the second study, the interaction with the
**initialization** established in
[`init_pathway_findings.md`](init_pathway_findings.md). The parent of the whole
line is
[`mechanism_constant_gain_sd_findings.md`](mechanism_constant_gain_sd_findings.md),
which established the base configuration: 1-step latent SD + a light a-priori
anchor ("SD+anchor", `alpha=1, beta0=0.05`) as a better deployable constant-gain
filter than plain a-priori.

All six sweeps were subsequently **escalated from `L=300000` to `L=1000000`**
(section 7), which resolves the one caveat every study carried -- the
under-converged detached `eta=0.01` column -- and settles the keep-vs-detach
floor race.

**One-paragraph summary of where the line ends up.** `n=1` wins the converged
floor everywhere, and the third study shows why: the sum's apparent depth
tradeoff (deeper = faster early, higher floor) was almost entirely a
gradient-*magnitude* artifact -- summing `n` horizon terms applies a `~n`x
larger effective SD weight -- and with magnitude held fixed (mean) the extra
horizons buy nothing. The launch-gradient verdict is a speed/floor/stability
trilemma whose resolution depends on init and horizon: at the default init
detaching is the stability fix; on the A-init both variants are 100% stable, so
the knob reduces to pure speed (keep) vs floor (detach). At `L=300000` the
detached small-gain column was under-converged, making keep the operational
winner there; the `L=1000000` escalation settles it -- **detach converges to a
strictly lower floor at equal (100%) stability on the A-init**. Best
configuration found: **A-init, `keep_launch=False`, `n=1`, `eta=0.01` -- 0.014%
excess at 100% stability** (the reduction is irrelevant at `n=1`, where sum and
mean are bit-identical); `keep_launch=True` (0.024-0.027%) remains the
faster-converging choice when the horizon is short.

---

## 1. The objective and the three axes

The latent SD term generalizes the single-step bootstrap to an **n-step
ladder** over autonomous horizons `k = 1..n`, launched from the same rolled
window (a horizon past the window clamps to the detached root `s_start`):

```
sum reduction (studies 1-2, and every prior study):
    alpha * sum_{k=1}^{n_eff} 0.5 || sg(x_j^+) - F^k x_{j-k}^+ ||^2

mean reduction (study 3):
    alpha * (1/n_eff) sum_{k=1}^{n_eff} 0.5 || sg(x_j^+) - F^k x_{j-k}^+ ||^2 ,
        n_eff = min(n, j+1)
```

The three axes:

| axis | values | flag | what it controls |
|---|---|---|---|
| depth `n` | 1..4 (`window=4` is the ceiling) | `--sd-horizon` (per arm in the `nstep` grid) | how many autonomous roll-out horizons `F^k` contribute distillation targets |
| launch gradient | keep / detach | `--keep-launch` / `--no-keep-launch` | whether the launch state `x_{j-k}^+` carries gradient (keep = the original single-step behavior) or is detached |
| reduction | sum / mean | `--no-sd-mean` (default) / `--sd-mean` | whether the `n_eff` horizon terms are summed (effective SD weight grows ~linearly with depth) or averaged (magnitude held fixed across depth) |

Plus the init as a fourth, contextual variable (from the init study):

| init | `F` init | `K` init | basin |
|---|---|---|---|
| default (`fI_k0`) | `(1-eps)I` | `0` | marginal: drifts to `|lambda(F_hat)| ~ 1.00` under SD self-consistency pressure; small-step stability stays partial even at 1M |
| A-init (`f0_kpinv`) | `0` | `H^+` (replace / high-gain limit, the toy `K=A`) | contractive: `|lambda| ~ 0.80-0.83`, inside the true spectrum `[0.778, 0.900]`; ~100% stable at every step |

Key identities used as built-in sanity checks throughout:

- At `n=1, keep_launch=True, sum`, the ladder is **bit-identical** to the
  original single-step SD term (verified: loss and gradient match to float64
  precision).
- `n_eff = 1` whenever a target only has the one-step horizon, so **`n=1` is
  bit-identical under sum and mean**; every launch conclusion, which lives at
  the winning `n=1` arm, is reduction-independent.
- The M4 a-priori control has `alpha=0`, so it is untouched by all three axes
  (its cells match across sweeps up to stored-run/code drift -- a check that
  nothing else moved).

---

## 2. Shared experimental setup

All sweeps use `--methods nstep`
([`scripts/self_distillation_losses.py`](../scripts/self_distillation_losses.py)):
a pure a-priori control (M4) plus one SD+anchor arm per ladder depth `n = 1..4`
(`window=4`), all adapting `(F, H, K)` with `beta2=0`, constant gain, no
projection / warm start / RTRL. Per-arm step sweep
`--step-size in {0.01, 0.03, 0.1, 0.3, 1.0}`, on the partially-observed
`sd6_od2` system, `eps=0.1`, `N=16`, `seed=0`, the same CUDA-drawn system as
the mechanism / init studies (**floor 0.0546**). Each study runs two sweeps at
the converged `L=300000` horizon, identical except for the launch gradient.

| study | init | reduction | keep sweep | detach sweep |
|---|---|---|---|---|
| 1. default-init | `fI_k0` (default) | sum | `output/sd_depth_L300k/ss*` | `output/sd_depth_detach_L300k/ss*` |
| 2. A-init | `f0_kpinv` (`--f-init zero --k-init pinv`) | sum | `output/sd_depth_ainit_L300k/ss*` | `output/sd_depth_ainit_detach_L300k/ss*` |
| 3. A-init mean | `f0_kpinv` | mean (`--sd-mean`) | `output/sd_depth_ainit_mean_L300k/ss*` | `output/sd_depth_ainit_mean_detach_L300k/ss*` |

All six sweeps were later re-run in full at **`L=1000000`** (section 7), under
the same prefixes with `L1M` in place of `L300k`
(`output/sd_depth_L1M`, `sd_depth_detach_L1M`, `sd_depth_ainit_L1M`,
`sd_depth_ainit_detach_L1M`, `sd_depth_ainit_mean_L1M`,
`sd_depth_ainit_mean_detach_L1M`).

Study 1 also retains shorter `L=30000` / `L=100000` sweeps (`sd_depth_L30k`,
`sd_depth_L100k`, `sd_depth_detach_L100k`) purely to document convergence-rate
differences; the `L=100000` sweep famously **mis-ranked** detach because it
stopped before the keep/detach crossover (see section 4.2).

Drivers: [`scripts/submit_sd_depth_sweep.sh`](../scripts/submit_sd_depth_sweep.sh)
(study 1), [`scripts/submit_sd_depth_ainit_sweep.sh`](../scripts/submit_sd_depth_ainit_sweep.sh)
(studies 2-3; loops both launch variants x step sizes, forwards `--sd-mean` and
`OUT_PREFIX` for study 3). Readers (all prefix-parametrized, reused unchanged
across the studies):
[`scripts/analyze_mech_sweep.py`](../scripts/analyze_mech_sweep.py) (envelope
table), [`scripts/plot_sd_depth_sweep.py`](../scripts/plot_sd_depth_sweep.py)
(depth grid), [`scripts/plot_sd_launch_convergence.py`](../scripts/plot_sd_launch_convergence.py)
(keep-vs-detach convergence overlay),
[`scripts/plot_sd_depth_convergence.py`](../scripts/plot_sd_depth_convergence.py)
(per-depth convergence).

**Reading the tables.** `excess` = median analytical error among the stable
trajectories, relative to the floor; `%stbl` = tail fraction of the 16
trajectories closed-loop stable; `ndiv` = diverged trajectories out of 16;
`rad` = median `|lambda(F_hat)|`. On the **default init**, `excess` and `%stbl`
MUST be read together -- at the low-excess small steps only a few trajectories
are stable, so `excess` there is a noisy median over a marginal population. On
the **A-init** this caveat essentially evaporates: every cell is ~100% stable,
so `excess` is read off full populations.

---

## 3. Consolidated verdicts, axis by axis

### 3.1 Depth `n`: from "speed/floor tradeoff" to "magnitude artifact"

The depth story evolves across the three studies and ends demystified:

1. **Study 1 (default init, sum):** depth is a **speed/floor knob**. Deeper
   ladders descend 2-3x faster through the first ~10k steps (richer
   multi-horizon gradient) but settle on a higher asymptotic floor (noisier
   model-generated targets at longer horizons enlarge the noise ball); curves
   cross around ctx ~10-30k and `n=1` wins the converged floor in both launch
   variants.
2. **Study 2 (A-init, sum):** the verdict **survives the init change intact**.
   Same early fan-out (deeper faster, ~2x at ctx <= 10k), same crossover
   (~30-100k), same tail inversion to `n=1`-lowest.
3. **Study 3 (A-init, mean):** the tradeoff **was mostly the sum's magnitude**.
   Summing the ladder applies a `~n`x larger effective SD weight at depth `n`;
   that single fact produced *both* halves of the depth verdict -- the faster
   early convergence (bigger step) and the higher floor (bigger noise ball).
   Under the mean the floor spread collapses (keep @`eta=0.01`: `n=4` goes from
   0.095% under the sum to 0.033%, vs `n=1` 0.027%) **and** the early speedup
   disappears (deeper is now marginally *slower* at ctx=3k). With magnitude
   controlled, the multi-horizon *information* buys nothing on this stationary
   problem.

Bottom line: **`n=1` is the unconditional recommendation.** Under the sum it
won the converged floor but conceded early speed (a real tradeoff a tracking
task might have wanted); under the mean it wins essentially everywhere with no
tradeoff surrendered. The `F^k` ladder's only real lever was its summed
gradient magnitude, which a larger `alpha` / gain at `n=1` supplies more
directly.

### 3.2 Launch gradient: a trilemma resolved by the init

The launch gradient is a **speed / floor / stability** knob, and which side to
take depends on the init:

1. **Study 1 (default init):** **detach wins two of three.** Keeping the launch
   gradient buys fast convergence (3-13x lower excess through the whole
   mid-context regime) but at the default init it is the mechanism of the
   aggressive-gain instability: the recursive gradient path through the
   bootstrap chain pushes `F` toward self-consistency and over the unit circle
   (9-15 of 16 trajectories diverge at `eta=1.0`). Detaching removes that path
   -- lower excess at every matched step once converged, ~100% stable at every
   gain -- at the sole cost of slower convergence. Verdict: given enough
   context, detach is the strictly better operating point.
2. **Study 2 (A-init):** **the A-init dissolves the stability leg, and the
   verdict flips.** The contractive `F=0`-grown basin never lets the launch
   gradient push `F` near the unit circle: the entire grid is ~100% stable in
   *both* launch variants (only casualty: 1/16 in the extreme `n=4, eta=1.0`
   corner). With stability paid for by the init, the launch gradient reduces to
   a pure speed/floor knob -- keep converges faster and is now safe; detach
   still bottoms out slightly lower where both have converged (e.g.
   `eta=0.03, n=1`: detach 0.036% vs keep 0.074%). At a finite deployable
   horizon **`keep_launch=True` is the better default on the A-init** -- the
   opposite of the default-init recommendation, for a clean reason: the thing
   detach was buying is already supplied.
3. **Study 3 (mean):** unchanged by construction -- the launch story lives at
   `n=1`, where sum and mean are bit-identical.

A persistent caveat ran through all three studies: **the detached `eta=0.01`
column was under-converged at `L=300000`** (still descending; the same
signature the init study needed `L=1000000` to resolve), so the absolute
keep-vs-detach floor race at the smallest gain was not settled at 300k.
**The `L=1000000` escalation (section 7) settles it: detach converges to a
strictly lower floor at every depth, init, and reduction** (n=1: 0.014% vs
keep's 0.022-0.027%) -- on the A-init at equal, 100% stability. So study 2's
"keep is the better default" was a horizon statement: keep wins when the
deployable horizon is short of detach's convergence; detach wins the asymptote
outright.

### 3.3 Reduction: sum = information + magnitude; mean isolates the information (which is nil)

The sum confounds the multi-horizon *information* (more distillation targets)
with its *magnitude* (a `~n`x larger effective SD weight at depth `n`). The
mean holds magnitude fixed and shows the information alone is worthless here:

- **Floors compress to near-`n=1`.** Sum keep @`eta=0.01`: 0.027 / 0.055 /
  0.081 / 0.095% across `n=1..4` (3.5x spread). Mean: 0.027 / 0.031 / 0.033 /
  0.033% (1.2x spread). The deeper targets are not intrinsically much noisier;
  the sum was just weighting them harder.
- **The early speedup disappears.** Sum keep @`eta=0.03`, ctx=3k: `n=4` 3.47%
  vs `n=1` 6.48% (deeper 2x faster). Mean: `n=4` 9.06% vs `n=1` 6.48% (deeper
  slightly slower). The acceleration was purely a larger step.
- **Stability marginally improves.** The mean removes the sum's lone `n=4,
  eta=1.0` casualty (1/16 in both variants) -- the whole mean grid is `ndiv=0`
  -- because the smaller effective SD magnitude at depth never stresses `F`.

Practical corollary: **if a future (e.g. drift/tracking) study wants the
ladder's early speed, it should use the sum (or just tune the gain), not the
mean** -- a deeper *summed* ladder and a larger `alpha`/gain at `n=1` are
likely interchangeable accelerators, and the mean removes depth as an
independent knob.

### 3.4 Stability: three mechanisms, one picture

Diverged trajectories (out of 16) at the most aggressive gain `eta=1.0`,
`L=300000`, across all six sweeps:

| arm | default sum keep | default sum detach | A-init sum keep | A-init sum detach | A-init mean keep | A-init mean detach |
|---|---|---|---|---|---|---|
| SD+anchor n=1 | 11 | 0 | 0 | 0 | 0 | 0 |
| SD+anchor n=2 | 9 | 0 | 0 | 0 | 0 | 0 |
| SD+anchor n=3 | 15 | 0 | 0 | 0 | 0 | 0 |
| SD+anchor n=4 | 15 | 2 | 1 | 1 | 0 | 0 |

Read left to right, the three axes each remove a layer of instability:

1. **The instability is the launch gradient** (default keep vs default detach):
   the recursive gradient path through the bootstrap chain drives `F` toward
   self-consistency and over the unit circle. Detaching removes it -- this
   resolves the mechanism study's caveat that SD was "the one to watch at large
   steps".
2. **The A-init makes the fix redundant** (default vs A-init): growing `F`'s
   dynamics up from zero keeps the learned `F` contractive
   (`|lambda| ~ 0.82` at usable gains, inside the true spectrum), so the launch
   gradient has no near-unit-circle `F` to destabilize.
3. **The mean removes the last corner** (sum vs mean at `n=4, eta=1.0`): the
   smaller effective SD magnitude at depth no longer stresses `F` even at the
   most aggressive gain.

Depth's own stability effect (study 1) is **gain-dependent and
launch-gradient-mediated**: at small/moderate gains (`eta <= 0.1`) deeper is
*more* stable (faster convergence settles the filter sooner, so more
trajectories are closed-loop stable at the tail); at aggressive gains
(`eta >= 0.3`) deeper is *less* stable, but only with the launch gradient on --
the detached sweep has essentially zero divergences everywhere.

The `L=1000000` escalation (section 7) sharpens the left column: default-init
keep divergences *accumulate with horizon* (`eta=1.0` climbs to 15-16 of 16 by
1M, including two fully-diverged cells), while every detached and every A-init
cell holds its 300k count. The launch-gradient instability at the default init
is not a transient -- more exposure means more casualties.

### 3.5 Best configurations across the line

At the original `L=300000` horizon:

| study | best config | excess | %stbl | note |
|---|---|---|---|---|
| 1. default-init sum | detach, `n=1`, `eta=0.01` | 0.018% | 12% (marginal population) | ~4.8x closer to floor than the a-priori baseline (0.087%); beats the 1-step keep headline (0.029%) |
| 2. A-init sum | keep, `n=1`, `eta=0.01` | 0.027% | 100% | same floor ball, read off a fully-stable population; detach @0.01 under-converged at 300k |
| 3. A-init mean | keep, `n=1`, `eta=0.01` | 0.027% | 100% | identical to study 2 (bit-identical at `n=1`); the mean's contribution is that `n=2..4` are now near-tied instead of 2-3.5x worse |

At the converged `L=1000000` horizon (section 7), the three studies agree:
**detach, `n=1`, `eta=0.01` is the best arm in every sweep** -- 0.014% excess
in all three, at 100% stability on the A-init (24% marginal-population
stability at the default init). The keep counterpart reads 0.022-0.027%.

The floor itself is init- and reduction-invariant: every converged detached
`n=1` cell lands at 0.014% (matching the init study's independent 1M readout of
0.013-0.015%). The axes change the *trajectory, the basin, and the stability of
the population the number is read from* -- not the destination.

---

## 4. Study 1: default init, sum reduction (`sd_depth_L300k` / `sd_depth_detach_L300k`)

Source: [`nstep_sd_ladder_launch_findings.md`](nstep_sd_ladder_launch_findings.md).
Two questions: does bootstrapping the latent target over a longer autonomous
roll-out `F^k x_{t-k}^+` help or hurt; and should the launch state carry
gradient (`keep_launch=True`, the original behavior) or be detached.

### 4.1 Per-arm best and full excess tables

Per-arm best (each at its own best step, which is `eta=0.01` for every arm;
floor = 0.0546):

| arm | keep=True excess | keep=False (detach) excess |
|---|---|---|
| M4 (a-priori control) | 0.087% | 0.087% |
| SD+anchor n=1 | 0.029% | **0.018%** |
| SD+anchor n=2 | 0.049% | 0.036% |
| SD+anchor n=3 | 0.068% | 0.050% |
| SD+anchor n=4 | 0.087% | 0.066% |

(The M4 control is bit-identical across the two sweeps -- `alpha=0`, so
`keep_launch` has no effect there.)

Selected rows of the full grids (each arm's lowest-excess step in **bold**):

**keep_launch=True (`sd_depth_L300k`):**

| arm | step | excess | %stbl | ndiv | rad |
|---|---|---|---|---|---|
| M4 constant | **0.01** | **0.087%** | 0.35 | 0 | 1.00 |
| M4 constant | 0.1 | 0.780% | 1.00 | 0 | 0.97 |
| M4 constant | 1.0 | 8.884% | 1.00 | 0 | 0.76 |
| SD+anchor n=1 | **0.01** | **0.029%** | 0.01 | 0 | 1.00 |
| SD+anchor n=1 | 0.1 | 0.234% | 1.00 | 0 | 1.00 |
| SD+anchor n=1 | 0.3 | 0.684% | 0.89 | 2 | 0.97 |
| SD+anchor n=1 | 1.0 | 2.490% | 0.31 | 11 | 0.76 |
| SD+anchor n=2 | **0.01** | **0.049%** | 0.29 | 0 | 1.00 |
| SD+anchor n=3 | **0.01** | **0.068%** | 0.46 | 0 | 1.00 |
| SD+anchor n=4 | **0.01** | **0.087%** | 0.66 | 0 | 1.00 |

**keep_launch=False / detached (`sd_depth_detach_L300k`):**

| arm | step | excess | %stbl | ndiv | rad |
|---|---|---|---|---|---|
| M4 constant | **0.01** | **0.087%** | 0.35 | 0 | 1.00 |
| M4 constant | 0.1 | 0.780% | 1.00 | 0 | 0.97 |
| M4 constant | 1.0 | 8.884% | 1.00 | 0 | 0.76 |
| SD+anchor n=1 | **0.01** | **0.018%** | 0.12 | 0 | 1.00 |
| SD+anchor n=1 | 0.1 | 0.149% | 0.20 | 0 | 1.00 |
| SD+anchor n=1 | 0.3 | 0.414% | 1.00 | 0 | 1.00 |
| SD+anchor n=1 | 1.0 | 1.562% | 1.00 | 0 | 0.95 |
| SD+anchor n=2 | **0.01** | **0.036%** | 0.00 | 0 | 1.00 |
| SD+anchor n=3 | **0.01** | **0.050%** | 0.56 | 0 | 1.00 |
| SD+anchor n=4 | **0.01** | **0.066%** | 0.99 | 0 | 1.00 |

Matched-step head-to-head for the best arm (n=1): **detach is lower at every
step** -- `eta=0.01` 0.018% vs 0.029%; 0.03 0.039% vs 0.069%; 0.1 0.149% vs
0.234%; 0.3 0.414% vs 0.684%; 1.0 1.562% vs 2.490%.

![keep_launch=True depth sweep](../output/sd_depth_L300k_depth_sweep.png)

![keep_launch=False depth sweep](../output/sd_depth_detach_L300k_depth_sweep.png)

### 4.2 Convergence: keep is faster, detach reaches a lower floor

At the best step `eta=0.01`, n=1 excess vs context length:

| context | keep=True | keep=False (detach) |
|---|---|---|
| 3,000 | 26.0% | 36.3% |
| 10,000 | 5.3% | 15.5% |
| 30,000 | 0.42% | 4.27% |
| 100,000 | 0.061% | 0.323% |
| 200,000 | 0.013% | 0.037% |
| 300,000 | 0.022% | 0.021% (tail floor **0.018%** vs keep **0.029%**) |

`keep_launch=True` is 3-13x lower excess through the entire mid-context regime
-- the launch gradient is a strong convergence accelerator. It plateaus by
~100k; the detached variant is still descending and only crosses below keep at
the very tail. So the two effects are cleanly separated: **keep converges
faster, detach bottoms out lower.**

![keep vs detach convergence at L=300k](../output/sd_launch_convergence_L300k.png)

Solid = keep, dashed = detach, same color per depth. The earlier `L=100000`
sweep mis-ranked detach precisely because it stopped before this crossover: at
100k, detach@`eta=0.01` read 0.436% (under-converged) vs keep's 0.040%; the
300k run resolves it to 0.018% vs 0.029%.

**Depth is the same speed/floor tradeoff.** Excess vs context length at
`eta=0.03`, `keep_launch=True`, per depth:

| context | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| 3,000 | 6.24% | 2.68% | 2.10% | 2.30% |
| 10,000 | 0.51% | 0.20% | 0.28% | 0.25% |
| 30,000 | 0.103% | 0.151% | 0.297% | 0.262% |
| 100,000 | 0.084% | 0.186% | 0.248% | 0.298% |
| 300,000 | **0.080%** | 0.130% | 0.200% | 0.227% |

Early (ctx <= 10k) the deeper ladders lead by 2-3x; the curves cross around
ctx ~10-30k, and by the tail the ordering fully inverts to `n=1` lowest.

![SD ladder depth convergence, eta=0.03](../output/sd_depth_convergence_eta0p03.png)

Left = `keep_launch=True`, right = detach; color = depth. The effect is
pronounced with the launch gradient on (clear early fan-out then inversion) and
muted when detached (depths track together early since detaching already
weakens the gradient signal), but the tail floor ordering (`n=1` lowest) holds
in both.

### 4.3 Stability grids

Full grid, `L=300000`, each cell `tail %stable | ndiv (of 16)`:

**keep_launch=True:**

| depth | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| n=1 | 0.01 \| 0 | 0.36 \| 0 | 1.00 \| 0 | 0.89 \| 2 | 0.31 \| 11 |
| n=2 | 0.29 \| 0 | 0.99 \| 0 | 1.00 \| 0 | 0.69 \| 5 | 0.44 \| 9 |
| n=3 | 0.46 \| 0 | 1.00 \| 0 | 0.99 \| 1 | 0.62 \| 6 | 0.06 \| 15 |
| n=4 | 0.66 \| 0 | 1.00 \| 0 | 1.00 \| 1 | 0.81 \| 3 | 0.06 \| 15 |

**keep_launch=False (detach):**

| depth | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| n=1 | 0.12 \| 0 | 0.14 \| 0 | 0.20 \| 0 | 1.00 \| 0 | 1.00 \| 0 |
| n=2 | 0.00 \| 0 | 0.15 \| 0 | 1.00 \| 0 | 1.00 \| 0 | 1.00 \| 0 |
| n=3 | 0.56 \| 0 | 1.00 \| 0 | 1.00 \| 0 | 1.00 \| 0 | 1.00 \| 0 |
| n=4 | 0.99 \| 0 | 1.00 \| 0 | 1.00 \| 0 | 1.00 \| 0 | 0.88 \| 2 |

Three observations:

1. **Small/moderate gain (eta <= 0.1): deeper is *more* stable** (keep
   `eta=0.03`: 0.36 -> 0.99 -> 1.00 -> 1.00, zero divergences) -- faster
   convergence settles the filter into a stable configuration sooner.
2. **Aggressive gain (eta >= 0.3): deeper is *less* stable, but only with the
   launch gradient on** (`eta=1.0`: 11 -> 9 -> 15 -> 15 divergences) -- the
   deeper `F^k` roll-out drives `F` harder toward self-consistency.
3. **Detach removes the depth-driven instability** -- zero divergences
   everywhere except the extreme `n=4`/`eta=1.0` corner (2/16).

(The small-`eta` %stable values are noisy at this init -- the optimal filter
hugs the unit circle, `rad ~ 1.00`, so stability is marginal for everyone there
-- but the depth trend is consistent across both sweeps.)

### 4.4 Study-1 reading

1. **Depth is a speed/floor knob, losing only at the stationary limit** -- and
   its early-convergence benefit is exactly the property that could matter
   under non-stationarity.
2. **The launch gradient is a speed/floor/stability trilemma, and detach wins
   two of three.** Given enough context, the detached variant is the strictly
   better operating point at this init.
3. **Best configuration: detached, single-step, small-gain, long-run**
   (`keep_launch=False, n=1, eta=0.01`, 0.018%).

---

## 5. Study 2: A-init, sum reduction (`sd_depth_ainit_L300k` / `sd_depth_ainit_detach_L300k`)

Source: [`nstep_sd_ladder_launch_ainit_findings.md`](nstep_sd_ladder_launch_ainit_findings.md).
The init study ([`init_pathway_findings.md`](init_pathway_findings.md)) found
that the A-init `f0_kpinv` (`F=0, K=H^+`, the toy `K=A` replace/high-gain end)
converges faster, is 100% stable at every step, and lands in a **contractive**
basin at the *same* floor as the default init. This study re-runs the full
depth x launch grid with every arm (including the M4 control) on the A-init
(`--f-init zero --k-init pinv`); the two init CLI flags default to reproducing
the study-1 sweep exactly.

### 5.1 Per-arm best and full excess grids

Per-arm best (each at its own best step; `%stbl` at the best step in
parentheses):

| arm | keep=True best | detach best |
|---|---|---|
| M4 (a-priori control) | 0.089% @0.01 (100%) | 0.092% @0.01 (100%) |
| SD+anchor n=1 | **0.027% @0.01 (100%)** | 0.044% @0.03 (100%) |
| SD+anchor n=2 | 0.055% @0.01 (100%) | 0.068% @0.01 (100%) |
| SD+anchor n=3 | 0.081% @0.01 (100%) | 0.072% @0.01 (100%) |
| SD+anchor n=4 | 0.095% @0.01 (100%) | 0.083% @0.01 (100%) |

Note the detached `n=1` best falls at `eta=0.03`, not `eta=0.01`: the detached
`eta=0.01` cell (0.257%) is still descending at 300k (section 5.2), so its
own-best-step lands one gain higher.

Full steady-excess grids, tail `%stable` in parentheses; `ndiv=0` for every
cell except `n=4, eta=1.0` (`ndiv=1`, 94% stable) in both variants:

**keep_launch=True (`sd_depth_ainit_L300k`):**

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.089 (100%) | 0.269 (100%) | 0.777 (100%) | 2.197 (100%) | 8.832 (100%) |
| SD+anchor n=1 | **0.027 (100%)** | 0.080 (100%) | 0.263 (100%) | 0.685 (100%) | 2.167 (100%) |
| SD+anchor n=2 | 0.055 (100%) | 0.165 (100%) | 0.461 (100%) | 1.094 (100%) | 3.418 (100%) |
| SD+anchor n=3 | 0.081 (100%) | 0.230 (100%) | 0.592 (100%) | 1.346 (100%) | 3.982 (100%) |
| SD+anchor n=4 | 0.095 (100%) | 0.269 (100%) | 0.669 (100%) | 1.487 (100%) | 4.229 (94%) |

**keep_launch=False / detached (`sd_depth_ainit_detach_L300k`):**

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.092 (100%) | 0.268 (100%) | 0.779 (100%) | 2.197 (100%) | 8.832 (100%) |
| SD+anchor n=1 | 0.257* (100%) | **0.044 (100%)** | 0.144 (100%) | 0.444 (100%) | 1.457 (100%) |
| SD+anchor n=2 | 0.068 (100%) | 0.106 (100%) | 0.357 (100%) | 1.026 (100%) | 2.614 (100%) |
| SD+anchor n=3 | 0.072 (100%) | 0.172 (100%) | 0.566 (100%) | 1.476 (100%) | 3.181 (100%) |
| SD+anchor n=4 | 0.083 (100%) | 0.214 (100%) | 0.702 (100%) | 1.744 (100%) | 3.513 (94%) |

`*` detached `n=1, eta=0.01` is under-converged at 300k (still descending).

Median `|lambda(F_hat)|` reads **0.82-0.83 at `eta <= 0.1`** and drifts down to
0.67-0.77 at the bruising `eta=1.0` in both variants -- every cell sits
*inside* the true spectrum `|eig(F)| in [0.778, 0.900]` or contractive of it,
never on the unit circle. This is the A-init's contractive basin holding across
the entire depth x launch grid. Contrast study 1, where the `fI` init drifted
to `|lambda| ~ 1.00` under SD self-consistency pressure and stability at small
gains was marginal.

At `eta=1.0`, detach still has lower excess than keep at every depth (n=1:
1.457% vs 2.167%; n=4: 3.513% vs 4.229%) -- the same "detach lower at
aggressive gain" ordering as study 1, but now *both* are stable rather than
detach being stable and keep blowing up.

![keep_launch=True depth sweep](../output/sd_depth_ainit_L300k_depth_sweep.png)

![keep_launch=False depth sweep](../output/sd_depth_ainit_detach_L300k_depth_sweep.png)

### 5.2 Convergence: keep is faster, detach's low-gain column is under-converged

At `n=1`, excess vs context length (%):

| ctx | keep @0.01 | detach @0.01 | keep @0.03 | detach @0.03 |
|---|---|---|---|---|
| 3,000 | 19.82 | 22.88 | 6.48 | 7.05 |
| 10,000 | 5.80 | 6.17 | 2.13 | 3.52 |
| 30,000 | 2.02 | 3.42 | 0.462 | 1.90 |
| 100,000 | 0.259 | 1.671 | 0.076 | 0.199 |
| 200,000 | 0.028 | 0.577 | 0.056 | 0.053 |
| 300,000 | **0.024** | 0.170 | 0.074 | **0.036** |

- **keep @0.01 is essentially converged by 200k** (0.028% -> 0.024%), the
  lowest-excess cell at this horizon.
- **detach @0.01 is still descending at 300k** (0.577% -> 0.170%): the same
  under-convergence signature study 1 flagged, which only resolved at
  `L=1000000` on the default init (detach @0.01 there: 0.257% at 300k -> 0.018%
  at 1M). The keep-vs-detach floor race at the smallest gain is not settled at
  300k.
- **At `eta=0.03`, both are converged and detach is lower** (0.036% vs 0.074%)
  -- the "detach bottoms out lower" half of the tradeoff, cleanly visible where
  both have converged.

![keep vs detach convergence, A-init](../output/sd_depth_ainit_launch_convergence.png)

Solid = keep, dashed = detach, color = depth. Left `eta=0.01`, right
`eta=0.03`. Keep descends faster throughout; the depths fan out early (deeper
faster) and re-order to `n=1`-lowest at the tail.

**Depth is the same speed/floor tradeoff.** Excess vs context (%) at
`eta=0.03`, `keep_launch=True`, per depth:

| ctx | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| 3,000 | 6.48 | 4.12 | 3.57 | 3.47 |
| 10,000 | 2.13 | 1.33 | 1.07 | 1.04 |
| 30,000 | 0.462 | 0.238 | 0.276 | 0.266 |
| 100,000 | 0.076 | 0.131 | 0.215 | 0.265 |
| 300,000 | **0.074** | 0.158 | 0.244 | 0.297 |

Early (ctx <= 10k) the deeper ladders lead by ~2x, the curves cross around
ctx ~30-100k, and by the tail the ordering fully inverts to `n=1` lowest --
identical in shape to study 1.

![A-init depth convergence, eta=0.03](../output/sd_depth_ainit_depth_convergence_eta0p03.png)

Left = keep, right = detach; color = depth.

### 5.3 Stability: the launch-gradient instability is gone

Diverged trajectories (out of 16) at `eta=1.0`, `L=300000`, A-init vs default
init:

| arm | default keep | default detach | A-init keep | A-init detach |
|---|---|---|---|---|
| SD+anchor n=1 | 11 | 0 | **0** | **0** |
| SD+anchor n=2 | 9 | 0 | **0** | **0** |
| SD+anchor n=3 | 15 | 0 | **0** | **0** |
| SD+anchor n=4 | 15 | 2 | **1** | **1** |

At the default init, detaching the launch was the fix for the
recursive-gradient blow-up. The A-init makes that fix redundant: growing `F`'s
dynamics up from zero keeps the learned `F` contractive, so the launch gradient
has no near-unit-circle `F` to destabilize.

### 5.4 Study-2 reading

1. **Depth verdict: unchanged** from study 1 at either init.
2. **Launch verdict: reshaped.** The stability leg of the trilemma moves to the
   init; the launch gradient reduces to a pure speed/floor knob, and
   **`keep_launch=True` becomes the better default on the A-init** -- the
   opposite of the study-1 recommendation.
3. **The A-init is a free accelerator here too**: it gives the launch gradient
   back for free -- `keep_launch=True`'s convergence speedup no longer carries
   a stability tax. The best converged cell (keep, `n=1`, `eta=0.01`, 0.027%)
   matches the study-1 floor ball while being 100% stable throughout.
4. **The floor is unchanged by the init** (as the init study predicted): the
   A-init changes the trajectory and the basin, not the destination.

---

## 6. Study 3: A-init, mean reduction (`sd_depth_ainit_mean_L300k` / `sd_depth_ainit_mean_detach_L300k`)

Source: [`nstep_sd_ladder_launch_ainit_mean_findings.md`](nstep_sd_ladder_launch_ainit_mean_findings.md).
Identical to the study-2 sweep except for the new `--sd-mean` flag, which
divides each target's ladder contribution by `n_eff = min(sd_horizon, j+1)`
(the number of distinct horizons that reach a buffered launch; horizons past
the detached root collapse to one term and are already excluded). Run only on
the A-init, the basin the sum study settled on; the study-2 numbers are the
reference throughout. Because `n=1` is bit-identical under sum and mean
(verified in a parity test and smoke run), the study genuinely only re-prices
depth `n >= 2`.

### 6.1 Per-arm best, sum vs mean side by side

Floor = 0.0546; `%stbl` at the best step in parentheses:

| arm | mean keep | sum keep | mean detach | sum detach |
|---|---|---|---|---|
| M4 (a-priori control) | 0.089% @0.01 (100%) | 0.089% @0.01 | 0.089% @0.01 (100%) | 0.092% @0.01 |
| SD+anchor n=1 | **0.027% @0.01 (100%)** | **0.027% @0.01** | 0.040% @0.03 (100%) | 0.044% @0.03 |
| SD+anchor n=2 | 0.031% @0.01 (100%) | 0.055% @0.01 | 0.060% @0.03 (100%) | 0.068% @0.01 |
| SD+anchor n=3 | 0.033% @0.01 (100%) | 0.081% @0.01 | 0.059% @0.01 (100%) | 0.072% @0.01 |
| SD+anchor n=4 | 0.033% @0.01 (100%) | 0.095% @0.01 | 0.056% @0.01 (100%) | 0.083% @0.01 |

`n=1` matches the sum study by construction (the tiny M4/`n=1` reference diffs
vs sum are stored-run/code drift, not the reduction). The signal is the deeper
arms: under keep the mean nearly ties them to `n=1` (0.031-0.033% vs the sum's
0.055-0.095%); under detach the mean even lets deeper arms edge below the
under-converged `n=1` @`eta=0.01`, but `n=1` at its own best step (`eta=0.03`,
0.040%) still wins.

### 6.2 Full excess grids: depths compressed, 100% stable everywhere

`ndiv=0` for **every** cell in both variants (the sum's lone `n=4, eta=1.0`
casualty is gone).

**keep_launch=True (`sd_depth_ainit_mean_L300k`):**

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.089 (100%) | 0.269 (100%) | 0.777 (100%) | 2.234 (100%) | 8.844 (100%) |
| SD+anchor n=1 | **0.027 (100%)** | 0.080 (100%) | 0.263 (100%) | 0.692 (100%) | 2.125 (100%) |
| SD+anchor n=2 | 0.031 (100%) | 0.092 (100%) | 0.298 (100%) | 0.736 (100%) | 2.195 (100%) |
| SD+anchor n=3 | 0.033 (100%) | 0.101 (100%) | 0.314 (100%) | 0.760 (100%) | 2.193 (100%) |
| SD+anchor n=4 | 0.033 (100%) | 0.103 (100%) | 0.326 (100%) | 0.791 (100%) | 2.212 (100%) |

**keep_launch=False / detached (`sd_depth_ainit_mean_detach_L300k`):**

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.089 (100%) | 0.269 (100%) | 0.777 (100%) | 2.234 (100%) | 8.859 (100%) |
| SD+anchor n=1 | 0.251* (100%) | **0.040 (100%)** | 0.144 (100%) | 0.439 (100%) | 1.470 (100%) |
| SD+anchor n=2 | 0.077 (100%) | 0.060 (100%) | 0.208 (100%) | 0.628 (100%) | 1.965 (100%) |
| SD+anchor n=3 | 0.059 (100%) | 0.073 (100%) | 0.253 (100%) | 0.738 (100%) | 2.124 (100%) |
| SD+anchor n=4 | 0.056 (100%) | 0.078 (100%) | 0.267 (100%) | 0.776 (100%) | 2.219 (100%) |

`*` detached `n=1, eta=0.01` is under-converged at 300k (same signature as the
sum study).

Compare the sum keep grid, where `n=4` ran 0.095 / 0.269 / 0.669 / 1.487 /
4.229 across the same steps -- roughly `n`x the `n=1` row. The mean rows sit
within ~1.05-1.2x of `n=1` everywhere. Median `|lambda(F_hat)|` reads **0.82 at
`eta <= 0.1`** across all depths and drifts to 0.74-0.80 at `eta=1.0` -- the
same contractive basin as study 2, unchanged by the reduction.

![mean keep_launch=True depth sweep](../output/sd_depth_ainit_mean_L300k_depth_sweep.png)

![mean keep_launch=False depth sweep](../output/sd_depth_ainit_mean_detach_L300k_depth_sweep.png)

### 6.3 Convergence: depth loses its early-speed edge under the mean

`n=1` excess vs context (%) -- identical to study 2 by construction (keep @0.01
converged by 200k and lowest at this horizon; detach @0.01 still descending):

| ctx | keep @0.01 | detach @0.01 | keep @0.03 | detach @0.03 |
|---|---|---|---|---|
| 3,000 | 19.82 | 21.35 | 6.48 | 6.68 |
| 10,000 | 5.80 | 6.13 | 2.13 | 3.51 |
| 30,000 | 2.02 | 3.42 | 0.462 | 1.95 |
| 100,000 | 0.259 | 1.674 | 0.076 | 0.213 |
| 200,000 | 0.028 | 0.573 | 0.056 | 0.031 |
| 300,000 | **0.024** | 0.179 | 0.074 | **0.040** |

![mean keep vs detach convergence, A-init](../output/sd_depth_ainit_mean_launch_convergence.png)

The depth story is where the mean diverges from the sum. Excess vs context (%)
at `eta=0.03`, `keep_launch=True`, per depth:

| ctx | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| 3,000 | 6.48 | 7.70 | 8.74 | 9.06 |
| 10,000 | 2.13 | 2.11 | 2.20 | 2.24 |
| 30,000 | 0.462 | 0.413 | 0.456 | 0.467 |
| 100,000 | 0.076 | 0.083 | 0.078 | 0.077 |
| 300,000 | **0.074** | 0.080 | 0.086 | 0.086 |

Under the **sum**, this same table had the deeper ladders *leading* by ~2x
early (n=4 3.47 vs n=1 6.48 at 3k) before inverting to `n=1`-lowest at the tail
-- a clean speed/floor fan-out. Under the **mean**, the early lead is **gone**:
deeper is marginally *slower* at 3k, the depths are essentially on top of each
other from 10k onward, and `n=1` is (barely) lowest at the tail. The
multi-horizon gradient's early acceleration was its summed magnitude; average
it away and the extra horizons add nothing on this stationary problem.

![mean A-init depth convergence, eta=0.03](../output/sd_depth_ainit_mean_depth_convergence_eta0p03.png)

Left = keep, right = detach; color = depth. (Detach @`eta=0.03` shows the
deeper arms reaching the mid-context floor a bit sooner -- 30k: n=4 1.37 vs n=1
1.95 -- because detaching weakens `n=1`'s gradient more than it weakens the
averaged deep ladder; but `n=1` still wins the tail, 0.040 vs 0.065.)

### 6.4 Stability: the mean removes the last casualty

Diverged trajectories (out of 16) at `eta=1.0`, `L=300000`:

| arm | A-init sum keep | A-init sum detach | A-init mean keep | A-init mean detach |
|---|---|---|---|---|
| SD+anchor n=1 | 0 | 0 | **0** | **0** |
| SD+anchor n=2 | 0 | 0 | **0** | **0** |
| SD+anchor n=3 | 0 | 0 | **0** | **0** |
| SD+anchor n=4 | 1 | 1 | **0** | **0** |

The A-init already made both launch variants essentially fully stable; the mean
removes the `n=4, eta=1.0` corner casualty by shrinking the effective SD
magnitude at depth. `rad` stays contractive (0.74-0.82) everywhere.

### 6.5 Study-3 reading

1. **The sum's depth tradeoff was a magnitude artifact.** The mean holds the
   magnitude fixed and both halves collapse: floors compress to near-`n=1`, and
   the early speedup disappears.
2. **The multi-horizon information alone buys nothing here.** With magnitude
   controlled, adding autonomous horizons `k = 2..n` neither accelerates early
   descent nor lowers the floor -- deeper is uniformly a hair worse.
3. **`n=1` remains the recommendation, now for a simpler reason** -- it wins
   essentially everywhere with no tradeoff surrendered. The launch verdict is
   inherited unchanged from study 2 (it lives entirely at `n=1`).

---

## 7. The L=1000000 escalation: the floor race settled

All six sweeps (not just the flagged detach @`eta=0.01` column) were re-run in
full at `L=1000000`, "for completion":

```bash
OUT_PREFIX=sd_depth_L1M \
  bash scripts/submit_sd_depth_sweep.sh -L 1000000 --keep-launch
OUT_PREFIX=sd_depth_detach_L1M \
  bash scripts/submit_sd_depth_sweep.sh -L 1000000 --no-keep-launch
OUT_PREFIX=sd_depth_ainit_L1M \
  bash scripts/submit_sd_depth_ainit_sweep.sh -L 1000000
OUT_PREFIX=sd_depth_ainit_mean_L1M \
  bash scripts/submit_sd_depth_ainit_sweep.sh -L 1000000 --sd-mean
```

Outputs land under the same layout with `L1M` prefixes; the readers are reused
unchanged with the new prefixes.

### 7.1 Headline: detach wins the converged floor everywhere, and on the A-init it costs nothing

Per-arm best (each at its own best step, which is `eta=0.01` for every arm in
every sweep; `%stbl` at the best step in parentheses; floor = 0.0546):

| arm | default sum keep | default sum detach | A-init sum keep | A-init sum detach | A-init mean keep | A-init mean detach |
|---|---|---|---|---|---|---|
| M4 (a-priori control) | 0.091% (100%) | 0.090% (100%) | 0.089% (100%) | 0.089% (100%) | 0.089% (100%) | 0.090% (100%) |
| SD+anchor n=1 | 0.024% (54%) | 0.014% (24%) | 0.027% (100%) | **0.014% (100%)** | 0.027% (100%) | **0.014% (100%)** |
| SD+anchor n=2 | 0.047% (97%) | 0.030% (74%) | 0.054% (100%) | 0.033% (100%) | 0.031% (100%) | 0.020% (100%) |
| SD+anchor n=3 | 0.069% (100%) | 0.047% (99%) | 0.077% (100%) | 0.054% (100%) | 0.033% (100%) | 0.024% (100%) |
| SD+anchor n=4 | 0.082% (100%) | 0.059% (100%) | 0.090% (100%) | 0.068% (100%) | 0.034% (100%) | 0.026% (100%) |

Four things resolve at once:

1. **The under-convergence caveat is gone and detach wins the floor race.** The
   detached `eta=0.01` column, which read 0.170-0.257% at 300k, converges to
   **0.014%** at `n=1` in every sweep -- exactly the init study's independent
   1M readout (0.013-0.015%). Keep bottoms out at 0.022-0.027%. The 300k
   verdict "keep is the better operating point on the A-init" was therefore a
   horizon statement: keep reaches its (higher) floor by ~200k, detach needs
   ~1M to reach its (lower) one. Given the horizon, **detach is the strictly
   better converged configuration at every depth, init, and reduction** --
   and on the A-init it is 100% stable, so nothing is traded away.
2. **The depth ordering is unchanged at full convergence.** Sum floors still
   scale roughly with depth (detach A-init: 0.014 / 0.033 / 0.054 / 0.068%);
   the mean still compresses them toward `n=1` (detach mean: 0.014 / 0.020 /
   0.024 / 0.026%). `n=1` is lowest in every column -- the study-3 verdict
   (depth buys nothing once magnitude is controlled) holds at 1M.
3. **The default-init marginal basin does not heal.** Its small-step `%stbl`
   improves with horizon (keep `n=1` @0.01: 1% at 300k -> 54% at 1M; detach:
   12% -> 24%) but plateaus far short of 100% -- the same pattern the init
   study saw for every `fI` arm. The A-init columns read 100% everywhere at
   `eta <= 0.3`. So the two 0.014% detach cells are *not* equivalent: the
   default-init one is the median of a ~24%-stable population, the A-init one
   of a fully-stable one.
4. **The best configuration on the line is now: A-init, `keep_launch=False`,
   `n=1`, `eta=0.01` -- 0.014% excess at 100% stability** (sum or mean,
   bit-identical at `n=1`).

### 7.2 Convergence: the keep/detach crossover, finally on-screen

A-init `n=1` excess vs context (%), now with the 1M row (sum sweep; the mean
sweep matches at `n=1` to sampling noise):

| ctx | keep @0.01 | detach @0.01 | keep @0.03 | detach @0.03 |
|---|---|---|---|---|
| 3,000 | 20.16 | 21.96 | 6.31 | 6.65 |
| 10,000 | 5.81 | 6.14 | 2.10 | 3.51 |
| 30,000 | 2.02 | 3.43 | 0.426 | 1.88 |
| 100,000 | 0.280 | 1.69 | 0.101 | 0.227 |
| 300,000 | 0.019 | 0.167 | 0.055 | 0.027 |
| 1,000,000 | 0.022 | **0.013** | 0.067 | **0.039** |

Keep @0.01 is flat from ~300k (0.019 -> 0.022, converged); detach @0.01 drops
another ~13x between 300k and 1M and crosses below keep somewhere past 300k --
the same crossover shape study 1 caught at the default init between 200k and
300k, shifted out because the A-init's contractive basin (like detaching)
weakens the SD gradient path. At `eta=0.03`, where both were already converged
at 300k, the 1M rows confirm the ordering (detach 0.039 vs keep 0.067).

![keep vs detach convergence, A-init, L=1M](../output/sd_depth_ainit_L1M_launch_convergence.png)

![mean keep vs detach convergence, A-init, L=1M](../output/sd_depth_ainit_mean_L1M_launch_convergence.png)

(Default-init counterpart: `output/sd_launch_convergence_L1M.png`. Its
small-step curves are visibly noisier -- the marginal `fI` population again --
but tell the same story: detach `n=1` @0.01 tail-averages 0.014% vs keep's
0.024%.)

Per-depth tails at `eta=0.03` (excess %, ctx = 1M):

| sweep | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| A-init sum keep | **0.067** | 0.108 | 0.141 | 0.162 |
| A-init mean keep | **0.093** | 0.106 | 0.105 | 0.104 |
| A-init mean detach | **0.038** | 0.055 | 0.069 | 0.075 |

Sum: the depth fan stays inverted (`n=1` lowest, spread ~2.4x). Mean: depths
essentially tied under keep, and `n=1`-lowest with a mild spread under detach.
No re-ordering appears at the longer horizon.

![A-init depth convergence at 1M, eta=0.03](../output/sd_depth_ainit_L1M_depth_convergence_eta0p03.png)

![mean A-init depth convergence at 1M, eta=0.03](../output/sd_depth_ainit_mean_L1M_depth_convergence_eta0p03.png)

### 7.3 Stability: longer exposure punishes default-init keep, and no one else

Diverged trajectories (out of 16) at `eta=1.0`, 300k vs 1M:

| arm | default keep 300k -> 1M | default detach 300k -> 1M | A-init (all four sweeps) 300k -> 1M |
|---|---|---|---|
| SD+anchor n=1 | 11 -> 15 | 0 -> 0 | 0 -> 0 |
| SD+anchor n=2 | 9 -> 16 | 0 -> 0 | 0 -> 0 |
| SD+anchor n=3 | 15 -> 15 | 0 -> 0 | 0 -> 0 |
| SD+anchor n=4 | 15 -> 16 | 2 -> 2 | <=1 -> <=1 |

Default-init keep loses *more* trajectories at 1M (two cells reach 16/16 --
fully diverged, `excess = inf`), and its mid-gain cells degrade too (`n=1`
@`eta=0.3`: 2 -> 9 divergences). Divergence under the launch gradient at the
default init is an absorbing state: every extra step of exposure is another
chance to tip over the unit circle, and no trajectory comes back. Detach and
the A-init hold their counts exactly (the A-init grid remains `ndiv = 0`
everywhere but the familiar `n=4, eta=1.0` corner at <=1, and the mean grid is
clean even there). `rad` is unchanged from 300k: ~1.00 for live default-init
cells at small gains, 0.81-0.82 across the A-init grid.

### 7.4 Updated recommendation

- **Deploy target (stationary, long horizon): A-init, detach, `n=1`,
  `eta=0.01`** -- 0.014% excess, 100% stable, converged.
- **Short/medium horizon (or wherever <~300k adaptation steps are available):
  A-init, keep, `n=1`** -- reaches its 0.022-0.027% floor ~3x sooner and is
  equally stable on the A-init; this was the 300k-era recommendation and it
  remains correct *for that horizon*.
- **Never: keep at the default init at aggressive gains** -- its divergences
  grow with exposure.

---

## 8. Open caveats and follow-ups

- ~~Detached `eta=0.01` under-converged at `L=300000`~~ -- **resolved by the
  section-7 escalation**: the column converges to 0.014% at `n=1` in every
  sweep, and detach wins the converged floor outright.

- **Gain migration per depth** (the `||K||`, `||KH - H^+H||` A->K diagnostic
  from the init study,
  [`scripts/plot_init_gain_migration.py`](../scripts/plot_init_gain_migration.py))
  was not run on the depth grids; it would show whether deeper ladders / the
  launch gradient change how far the gain walks from the replace limit.
  Optional, if the migration picture across the ladder is wanted.

- **Non-stationary / video implication.** The stationary picture is settled:
  `n=1`, A-init, detached launch at long horizons (keep for short ones). But
  the decisive axis throughout was *convergence horizon*, which is exactly
  what non-stationarity removes -- a tracking filter never gets 300k
  stationary steps, which puts it squarely in the regime where keep (and the
  A-init's transient acceleration) win. Study 1 flagged the early-speed
  advantages of `keep_launch=True` and deeper `n` as the properties that could
  reverse the verdict under drift. Study 3 sharpens that: depth's early speed
  is purely a gradient-magnitude effect, so under non-stationarity a deeper
  *summed* ladder and a larger `alpha`/gain at `n=1` are likely
  interchangeable accelerators, and the mean reduction removes depth as an
  independent knob. A drift study wanting the ladder's early speed should use
  the sum (or tune the gain), not the mean -- and should sweep the launch
  variant, since `keep_launch=False` remains the cheap stability fix when an
  aggressive constant gain is required at the default init (where, per 7.3,
  keep's divergences compound with exposure).
