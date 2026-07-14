# A-init pathway at zero measurement noise: initializing at the high-gain (K = A) end

Zero-measurement-noise (`--v-std 0`, process noise kept at `--w-std 1`) replay of
[`init_pathway_findings.md`](init_pathway_findings.md). The system is the *same*
CUDA-drawn seed-0 `sd6_od2` (`true |eig(F)| ∈ [0.778, 0.900]`); only the
observation noise covariance is set to zero, so every number here is directly
comparable to the original `v=1` study. Removing the measurement noise drops the
irreducible floor from `0.0546` to **`0.0265`** (only process noise now
contributes to the observable-subspace prediction error) and the zero-predictor
ceiling to `0.0790`.

`K = H⁺` is the exact toy realization of "**K = A**". At `F = 0` the innovation is
the raw observation, so the corrector is literally the encoder (`corrector ≡
encoder`); with a general `F` it is the REPLACE / high-gain limit of the single
gain update (DESIGN.md §3.1) -- the observable subspace is replaced by `H⁺y` and
`null(H)` coasts:

```
x_t^+ = x_t^- + K(y - H x_t^-)  =  (I - KH) x_t^- + K y
      ->  (I - H⁺H) x_t^-  +  H⁺ y      when  K = H⁺   (replace limit)
```

The study **initializes at that high-gain end** and lets training anneal the gain
down, versus the current low-gain (`K = 0`) init growing the gain up.

**Why re-run at zero measurement noise.** The `v=1` study's two most delicate
findings both hinged on the *noise* that is now gone: (finding 2) `beta2` fitting
the a-posteriori to the *noisy* observation raised the floor, and the A-init was
credited with *subsuming* that biased accelerator; and (the gain-migration
section) the learned gain "walked away" from the replace limit toward a Kalman
gain that trades off measurement noise. With `v = 0` the observation *is* the
noiseless signal in the observable subspace, so the a-priori/a-posteriori
distinction collapses there and the **replace limit `K = H⁺` becomes (near-)
optimal**. The central question of this replay: does the A-init still help when
the thing it was annealing toward is now the thing it started at?

**Central hypothesis (restated for v=0).** With no measurement noise the
high-gain end is no longer a mere accelerator to be annealed away -- it is close
to the optimum. If so, (a) the free-accelerator claim should survive, (b) β2's
floor penalty should *vanish* (making finding 2's "subsumption" moot -- there is
nothing biased left to subsume), and (c) the gain should *stay* at the replace
limit rather than migrate. All three are confirmed below; the contractive-basin
Pareto win (finding 3) survives unchanged.

## Setup

`--methods init`
([`scripts/self_distillation_losses.py`](../scripts/self_distillation_losses.py)):
one arm per named init token on the settled base -- latent SD + anchor (`alpha=1,
beta0=0.05`), detached launch (`keep_launch=False`), the single-step ladder (`n=1`,
`sd_horizon=1`), all adapting `(F, H, K)`, constant gain, no projection / warm
start / RTRL / weight decay. The independent variables are the init pair
`(f_init, k_init)` and, for two arms, the a-posteriori weight `beta2`.

The init 2×2 (`beta2=0`) plus two β2-substitution arms:

| arm token | `F` init | `K` init | `beta2` | role |
|---|---|---|---|---|
| `fI_k0` | `(1-eps)I` | `0` | 0 | current init (**control**) |
| `fI_kpinv` | `(1-eps)I` | `H⁺` | 0 | hybrid: observation-ZOH predictor from step 0 |
| `f0_kpinv` | `0` | `H⁺` | 0 | the literal `K = A` proposal |
| `f0_k0` | `0` | `0` | 0 | axis-attribution control (expected worst) |
| `fI_k0_b0p05` | `(1-eps)I` | `0` | 0.05 | β2 substitution on the control init |
| `f0_kpinv_b0p05` | `0` | `H⁺` | 0.05 | β2 substitution / stacking on the `K=A` init |

Per-arm step sweep `--step-size in {0.01, 0.03, 0.1, 0.3, 1.0}`, `sd6_od2`,
`eps=0.1`, `N=16`, `seed=0`, `L=1000000`, `--v-std 0`. Total: 6 arms × 5 steps =
30 runs, sharded one-arm-per-job:
`output/sd_init_1M_v0/ss<step>/<arm>/sd6_od2_save.pt`. Driver:
`OUT_PREFIX=sd_init_1M_v0 bash scripts/submit_init_sweep.sh -L 1000000 --v-std 0`.
Readers: [`scripts/analyze_mech_sweep.py`](../scripts/analyze_mech_sweep.py)
(`--prefix sd_init_1M_v0`),
[`scripts/plot_init_convergence.py`](../scripts/plot_init_convergence.py), and
[`scripts/plot_init_gain_migration.py`](../scripts/plot_init_gain_migration.py).

As before, `excess` (median analytical error among stable trajectories, relative
to floor) and `%stbl` (tail fraction closed-loop stable) MUST be read together: at
the low-excess small steps only a fraction of the 16 `fI`-init trajectories are
stable, so `excess` there is a noisy median; the robust comparisons are the
fully-stable steps and the `f0` arms (100% stable everywhere). `ndiv = 0` for
every cell (the detached launch holds; "unstable" here is marginal closed-loop
tails, not blow-ups).

## Verdict

Four findings at the converged `L=1000000` horizon, plus the two that the zero
measurement noise *changes* relative to the `v=1` study:

1. **The free-accelerator hypothesis is confirmed, unchanged.** Both A-init arms
   accelerate the transient at zero floor cost. At the best step `eta=0.01` all
   live arms converge to the **same floor** (0.008-0.013% at the final sample,
   relIR 0.002-0.005 -- unbiased) -- but the hybrid starts at the
   observation-ZOH excess (**10.6%** at step 100, vs the control's 192%),
   leading through the whole mid-transient, and `f0_kpinv` descends ~2-4x faster
   than the control from 3k to 30k (61% vs 103% at 3k; 12% vs 53% at 10k). The
   init changes the starting point, not the destination. Note the hybrid's head
   start is *much larger* than at `v=1` (10.6% here vs 26% there): a
   zero-order-hold on a **noiseless** observation is a genuinely good predictor,
   so the copy/ZOH baseline sits far closer to the floor.
2. **β2's bias is gone -- so there is nothing left to "subsume".** This is the
   headline change from `v=1`. There, β2 arms plateaued at ~2.86% excess (relIR
   0.34) and the A-init's virtue was *subsuming* that biased accelerator. At
   `v=0` the a-posteriori term fits the posterior to the **noiseless** signal in
   the observable subspace, so it is no longer biased: both β2 arms reach the
   *same* floor as their β2=0 counterparts (`fI_k0_b0p05` 0.011%, `f0_kpinv_b0p05`
   0.012%) with *lower* relIR than β2=0 (0.0017-0.0018 vs 0.004-0.005). β2 is
   now simply harmless-to-mildly-helpful rather than a floor-raiser. The `v=1`
   "displacement" verdict is therefore moot at zero noise: the accelerator it
   displaced no longer costs anything.
3. **The contractive-basin Pareto win survives intact.** `F=0, K=H⁺` lands in a
   contractive basin -- **100% stable at every step** -- while every `fI` arm
   finishes at `|λ(F_hat)| ≈ 1.00` (unit-circle hugging: only 39-48% of
   trajectories stable at the small steps even after 10⁶ steps). `f0_kpinv`
   finishes at `|λ| ≈ 0.79-0.82`, **inside the true spectrum [0.778, 0.900]**.
   At `eta=0.01` its 0.012% matches the control's 0.011% -- but the control's
   readout is the median of a 42%-stable population while `f0_kpinv`'s is a
   100%-stable one: the same excess with *nothing* sacrificed. The fully-stable
   Pareto gap is ~13x: `f0_kpinv` 0.012% @ `eta=0.01` (100% stable) vs the
   control's best fully-stable 0.137% @ `eta=0.1`.
4. **The interaction is essential (axis attribution).** `f0_k0` is still an
   *exact gradient saddle*: `F=0 ∧ K=0 ⇒ z ≡ 0`, so nothing trains -- it sits at
   the zero-predictor ceiling (**198.542%** excess, `‖K‖ = 0`, relIR 1.0)
   forever, at every step, even after 10⁶ steps. `F=0` alone is fatal; `K=H⁺`
   alone (the hybrid) accelerates but inherits the marginal unit-circle basin.
   Only the combination finds the contractive basin.

The two findings the zero measurement noise **rewrites**:

- **(finding 2, above) β2's replace-limit pinning is now a feature, not a bug.**
- **(gain migration) The gain does not migrate -- the replace limit is optimal.**
  At `v=1` every live arm walked to `‖KH − H⁺H‖ ≈ 0.53` (a Kalman gain well away
  from replace). At `v=0` every live arm *stays* at the replace limit:
  `‖KH − H⁺H‖ ≈ 0.006-0.012` at `eta ≤ 0.1` (see the gain section). With no
  measurement noise to reject, full replacement of the observable subspace is
  (near-)optimal, so there is no A→K migration to perform.

## Per-arm best (each at its own best step; L=1000000, v=0)

floor = 0.0265; relIR at the best step. From
`python scripts/analyze_mech_sweep.py --prefix sd_init_1M_v0`.

| arm | best excess | best step | %stbl @ best | relIR |
|---|---|---|---|---|
| `fI_k0` (control) | 0.011% | 0.01 | 0.42 | 0.005 |
| `fI_kpinv` (hybrid) | 0.013% | 0.01 | 0.39 | 0.004 |
| `f0_kpinv` (K=A) | **0.012%** | 0.01 | **1.00** | 0.004 |
| `f0_k0` | 198.542% | (any) | 1.00 | 1.000 |
| `fI_k0_b0p05` | 0.011% | 0.01 | 0.46 | 0.002 |
| `f0_kpinv_b0p05` | 0.012% | 0.01 | 1.00 | 0.002 |

The three live init identities land within ~0.002% of one another at the floor;
the robust comparison is `f0_kpinv`'s 0.012% at **perfect** stability vs the
control's 0.011% read off a 42%-stable population (its best *fully-stable* cell is
0.137% @ `eta=0.1`).

## Steady excess grid: matched excess, unmatched stability

Steady excess (%) per `(arm, step)`, tail `%stable` in parentheses; `ndiv = 0`
for every cell.

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| `fI_k0` (control) | 0.011 (42%) | 0.039 (48%) | 0.137 (100%) | 0.468 (100%) | 3.412 (100%) |
| `fI_kpinv` | 0.013 (39%) | 0.043 (44%) | 0.151 (100%) | 0.516 (100%) | 3.859 (100%) |
| `f0_kpinv` | **0.012 (100%)** | 0.041 (100%) | 0.146 (100%) | 0.498 (100%) | 3.471 (100%) |
| `f0_k0` | 198.542 (100%) | 198.542 (100%) | 198.542 (100%) | 198.542 (100%) | 198.552 (100%) |
| `fI_k0_b0p05` | 0.011 (46%) | 0.037 (27%) | 0.132 (100%) | 0.450 (100%) | 2.878 (100%) |
| `f0_kpinv_b0p05` | 0.012 (100%) | 0.039 (100%) | 0.138 (100%) | 0.464 (100%) | 2.592 (100%) |

Median `|λ(F_hat)|` splits cleanly by F-init: every live `fI` cell at
`eta ≤ 0.3` reads 0.95-1.00; every live `f0` cell reads 0.78-0.82 (true
`|eig(F)|` ∈ [0.778, 0.900]). Note the control's F init `(1-eps)I = 0.9·I` starts
*below* the true spectral radius and still drifts up to 1.00 under the SD
self-consistency pressure; `F = 0` grows only as much dynamics as the data
demands and stops inside the true spectrum.

The two `beta2=0.05` arms are **no longer a raised-floor plateau** (the `v=1`
signature). They are indistinguishable-to-slightly-better than their β2=0
counterparts at every step (`fI_k0_b0p05` 0.011% vs `fI_k0` 0.011%;
`f0_kpinv_b0p05` 0.012% vs `f0_kpinv` 0.012%, and both β2 arms are *lower* at the
bruising `eta=1.0`: 2.878/2.592% vs 3.412/3.471%). With a noiseless observation,
the a-posteriori fit is a legitimate grounding signal, not a bias source.

The `eta=0.01` adapted-`F` spectra make finding 3 visual: blue `×` are the true
`|eig(F)| ∈ [0.778, 0.900]`, the red `○` the converged mean `F_hat`. `f0_kpinv`
lands its spectrum *inside* the true one (`|λ| = 0.82`), while the control's
drifts *outward* onto the unit circle (`|λ| = 1.00`), past the true dynamics.

| `f0_kpinv` (K=A): converged inside the spectrum | `fI_k0` (control): drifted onto the unit circle |
|---|---|
| ![f0_kpinv adapted F spectrum, converged inside the true spectrum at |lambda|=0.82](../output/sd_init_1M_v0/ss0p01/f0_kpinv/sd6_od2_eig_complex_plane.png) | ![fI_k0 control adapted F spectrum, drifted onto the unit circle at |lambda|=1.00](../output/sd_init_1M_v0/ss0p01/fI_k0/sd6_od2_eig_complex_plane.png) |

## Convergence: A-init leaves the transient early, at no floor cost

One panel per step size, each overlaying every init arm's excess vs online step
(the control `fI_k0` drawn dashed/gray):

![A-init convergence per step](../output/sd_init_1M_v0_convergence.png)

Excess (%) vs online step at the best step `eta=0.01`:

| arm | 100 | 1k | 3k | 10k | 30k | 100k | 300k | 1M |
|---|---|---|---|---|---|---|---|---|
| `fI_k0` (control) | 192.23 | 147.75 | 102.76 | 52.99 | 21.21 | 4.02 | 0.18 | 0.008 |
| `fI_kpinv` | **10.55** | **7.58** | **4.11** | **1.06** | **0.09** | -- | -- | 0.010 |
| `f0_kpinv` | 190.66 | 131.39 | 61.50 | 12.34 | 0.50 | 0.068 | 0.015 | 0.009 |
| `fI_k0_b0p05` | 181.58 | 105.55 | 59.04 | 21.94 | 4.44 | -- | -- | 0.004 |
| `f0_kpinv_b0p05` | 190.66 | 131.40 | 61.54 | 12.44 | 0.51 | 0.014 | 0.008 | 0.008 |

(`--` = every trajectory read unstable at that single sampled step -- the
marginal-population noise of the `fI` basin, which is exactly the point of
finding 3.)

- **All live arms end at 0.004-0.010%** -- the init changes the trajectory, not
  the destination. The init identities are visible at step 100: the hybrid
  starts at the observation-ZOH (~10.6%), the `f0` arms and the control at the
  zero-predictor ceiling (~190-192%).
- **The hybrid owns the early transient** (an order of magnitude below the
  control through 30k): the noiseless observation-ZOH prediction baseline is a
  pure -- and here, large -- head start.
- **`f0_kpinv` is the mid-transient accelerant** (~2-4x under the control from
  3k to 30k) despite starting at the ceiling -- informative `H⁺y` states make the
  SD and anchor gradients live immediately.
- **β2 tracks its β2=0 base the whole way down** (`f0_kpinv_b0p05` is bit-close
  to `f0_kpinv`; `fI_k0_b0p05` tracks the control) and, unlike `v=1`, never peels
  off into a plateau -- it descends to the same 0.004-0.008% floor.

## Gain migration: the gain stays at the replace limit (no A→K walk)

`‖KH − H⁺H‖` is exactly 0 at the `K = H⁺` replace limit and grows as the learned
gain walks toward a Kalman gain that trades off measurement noise; `‖K‖` is the
gain magnitude:

![A-init gain migration](../output/sd_init_1M_v0_gain_migration.png)

- **No migration at zero noise (the inversion of the `v=1` result).** All five
  live arms sit *at* the replace limit across `eta ≤ 0.1`: `‖KH − H⁺H‖ ≈
  0.006-0.012`, `‖K‖ ≈ 0.80-0.86` (near `‖H⁺‖`), whether the gain grew up from 0
  (`fI_k0`, `fI_k0_b0p05`) or started at replace (`f0`/`fI` `kpinv` arms). Compare
  `v=1`, where the same arms converged to `‖KH − H⁺H‖ ≈ 0.53` -- a gain far from
  replace. With no measurement noise to reject, full replacement of the
  observable subspace *is* (near-)optimal, so both ends converge *on* the replace
  limit rather than away from it. The distance only grows at the bruising
  `eta=1.0` (`‖KH − H⁺H‖ ≈ 0.09-0.22`), an SGD-noise-ball effect, not a
  migration.
- **β2 pins even tighter to replace -- and that is now correct.** Both β2 arms
  read `‖KH − H⁺H‖ ≈ 0.002-0.004` at `eta ≤ 0.1` (vs ~0.006-0.012 for β2=0),
  i.e. β2 holds the filter at the copy/replace end `HK ≈ I`. At `v=1` this was
  *the* mechanism of β2's bias (it blocked the A→K migration); at `v=0` there is
  no beneficial migration to block, so pinning at replace simply lands the arm on
  the optimum -- which is exactly why the β2 arms match the floor (finding 2) and
  post the lowest relIR (0.0017-0.0018).
- The `f0_k0` saddle reads `‖K‖ = 0`, `‖KH − H⁺H‖ = ‖H⁺H‖ = √2` (rank-2
  projector), as expected.

## Real-case translation (deferred design notes)

The toy's single `K = H⁺` gain maps to the routed **K/A pipeline**, not the K
network. The zero-measurement-noise regime sharpens some of the `v=1` design
reads and reverses others.

- **The high-gain-end init is now doubly motivated.** In the `v=1` study the
  A-init was a *free accelerator* whose target (a Kalman gain away from replace)
  the training still had to find. At `v=0` the replace limit is itself
  (near-)optimal, so initializing there is not just a good starting point but
  close to the destination -- the "anneal the gain down" framing collapses to
  "start at the answer". For a low-observation-noise real regime, the
  gate-open / high-gain init is the natural operating point, not merely a
  curriculum trick.
- **β2 during K's onset is safe here.** The `v=1` verdict flagged a small β2
  during K's onset window as a residual risk (it pins the gain at replace, a
  "never funnel A→K" mechanism). At `v=0` the replace limit is where you *want*
  to be, so a small β2 is a legitimate grounding term throughout, not something
  to anneal out. The real-case caution is now scoped to the *measurement-noise
  level*: β2's pinning is a bug only when the observation is noisy enough that
  the optimal gain sits well inside the replace limit.
- **Contractive-F-init translation (unchanged).** The stable basin still came
  from growing `F`'s dynamics *up from zero* rather than starting at marginal
  stability; the control starts at `0.9·I` and still drifts to the unit circle
  under SD self-consistency pressure. A *contractive* velocity init (e.g.
  `v = -γ·z`, small `γ > 0`) remains the faithful video translation, and it is
  independent of the noise level -- finding 3 held identically at both `v=1` and
  `v=0`.
- **Inverted curriculum recommendation (unchanged).** Initialize at `τ₂ ≈ 0` /
  gate open: the model at step 0 *is* the E1 frame-AE, and routing migrates
  regions to K as F earns them. At low observation noise this is even more
  benign, since the high-gain start is near-optimal.
- **Scope statement.** The toy informs: the high-gain-end init as (at low noise)
  a near-optimal operating point rather than an accelerator to anneal away; that
  β2's replace-limit pinning is bug-or-feature depending purely on the
  measurement-noise level; and the contractive-vs-marginal F-init basin (finding
  3), which is noise-independent. The toy cannot inform: the two-network
  handoff / deadlock, trainable-A target drift, the position / vector asymmetry.
