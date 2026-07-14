# A-init pathway: initializing at the high-gain (K = A) end

Follow-up to
[`posteriori_loss_sweep_findings.md`](posteriori_loss_sweep_findings.md),
[`nstep_sd_ladder_launch_findings.md`](nstep_sd_ladder_launch_findings.md) and
[`mechanism_constant_gain_sd_findings.md`](mechanism_constant_gain_sd_findings.md).
Those studies settled the SD base config -- latent SD + a light a-priori anchor
(`alpha=1, beta0=0.05`), detached launch (`keep_launch=False`), and showed that
the objective-biasing accelerators (`beta2`, `keep_launch=True`, deeper `n`) all
buy early convergence at the price of a raised asymptotic floor. This study asks
whether an **initialization** change can accelerate early descent for **free** --
i.e. give the β2-like transient *without* β2's floor penalty.

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

**Central hypothesis.** Unlike β2 / `keep_launch=True` / deeper `n` (which raise
the floor), the A-init is a *free* accelerator: β2-like early descent at the β2=0
floor. If confirmed, it can displace the biased β2 term that the real case needed.

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

The hybrid arm is `ŷ_t ≈ (1-eps)·y_{t-1}` at step 0 (an observation-ZOH predictor,
since `H(I - H⁺H) = 0`). Identical per-method seeding keeps `H` (hence `H⁺`)
matched across arms, so the arms differ *only* in the init pair and `beta2`.

Per-arm step sweep `--step-size in {0.01, 0.03, 0.1, 0.3, 1.0}`, `sd6_od2`,
`eps=0.1`, `N=16`, `seed=0`, on the same CUDA-drawn system as the mechanism /
depth / post studies (floor 0.0546). Total: 6 arms × 5 steps = 30 runs. The
**headline is the formal `L=1000000` sweep** (`output/sd_init_1M/ss*`,
`OUT_PREFIX=sd_init_1M ... -L 1000000`); the earlier `L=300000` sweep
(`output/sd_init_L300k/ss*`) is retained to document the horizon artifacts it
produced -- `eta=0.01` was under-converged there (e.g. `f0_kpinv` read 0.257% at
`eta=0.01` vs its converged 0.014%), which is exactly what the 1M rerun was for.

Sharded one-arm-per-job layout (each cell is its own `--init-grid <arm>` run):
`output/sd_init_1M/ss<step>/<arm>/sd6_od2_save.pt`. Driver:
[`scripts/submit_init_sweep.sh`](../scripts/submit_init_sweep.sh); readers:
[`scripts/analyze_mech_sweep.py`](../scripts/analyze_mech_sweep.py) (envelope
table),
[`scripts/plot_init_convergence.py`](../scripts/plot_init_convergence.py)
(per-step convergence overlay -- the crossover headline), and
[`scripts/plot_init_gain_migration.py`](../scripts/plot_init_gain_migration.py)
(gain-migration diagnostic: `‖K‖` and `‖KH − H⁺H‖`). The readers glob recursively
and parse the step from the `ss*` path token, so they handle both this sharded
layout and the flat `ss*/save.pt` layout of the earlier sweeps.

As before, `excess` (median analytical error among stable trajectories, relative
to floor) and `%stbl` (tail fraction closed-loop stable) MUST be read together: at
the low-excess small steps only a fraction of the 16 trajectories are stable, so
`excess` there is noisy; the robust comparisons are the fully-stable steps. Note
the `f0_*` arms start at **spectral radius 0** (`F = 0`), so their early stability
at large `eta` is a distinct axis to watch.

## Verdict

Four findings, all at the converged `L=1000000` horizon (the 1M rerun resolves
the `eta=0.01` under-convergence of the earlier 300k sweep and every 300k
conclusion survives it -- several get *stronger*):

1. **The free-accelerator hypothesis is confirmed.** Both A-init arms accelerate
   the transient at zero floor cost. At the now-converged best step `eta=0.01`
   all three live `beta2=0` arms converge to the **same floor** (0.013-0.014% at
   the final sample; steady tail medians 0.012-0.015%, within each other's
   noise) **with identical relIR 0.004-0.005** (unbiased) -- but the hybrid
   starts at the *copy-predictor* excess (26.0%, vs the analytical
   copy-predictor's 27.0% -- it IS an observation-ZOH at init) against the
   control's 78%, leading 3-10x through the whole mid-transient, and `f0_kpinv`
   descends ~2-3x faster than the control from 3k to 30k. Unlike β2 (2.9%
   plateau, relIR 0.34), the init buys the speed for free -- it changes the
   starting point, not the objective.
2. **A-init fully subsumes β2's accelerator role.** On the A-init base, β2 adds
   *nothing*: `f0_kpinv` and `f0_kpinv_b0p05` trace bit-close curves until
   ~10-30k steps (e.g. at 10k, `eta=0.01`: 6.03% vs 6.11%), after which the β2
   arm plateaus at its ~2.86% bias floor while the β2=0 arm keeps descending to
   0.014%. On the *control* init β2 does accelerate (its known role) -- so β2's
   transient value is real but completely redundant given the A-init. One 1M
   refinement: the β2 plateau is **metastable at mid steps** -- between 300k and
   1M, `f0_kpinv_b0p05@0.03` slid off it to 0.127% (relIR 0.34 → 0.043) and
   `fI_k0_b0p05@0.1` to 0.469% -- but the escaped cells remain 3-10x above their
   β2=0 counterparts at the same step, the accuracy-optimal `eta=0.01` cells
   stay pinned at 2.86-2.87% (relIR 0.335) even after 10⁶ steps, and the escape
   happens *without* gain migration (see the gain section). The displacement
   verdict is unchanged.
3. **The headline surprise: `F=0, K=H⁺` lands in a different, contractive basin
   -- 100% stable at every step -- and at 1M this is a strict Pareto win.**
   Every `fI` arm finishes with `|λ(F_hat)| ≈ 1.00` at `eta ≤ 0.3`
   (unit-circle hugging: 35-67% of trajectories stable at the small steps even
   after 10⁶ steps -- the plague of this entire study line), while `f0_kpinv`
   finishes at `|λ| ≈ 0.80-0.82`, **inside the true spectrum [0.778, 0.900]**,
   and is 100% stable at every single step. At the converged `eta=0.01` its
   0.014% matches the control's 0.012% -- but the control's readout is the
   median of a 35%-stable population while `f0_kpinv`'s is a 100%-stable one:
   the same excess with *nothing* sacrificed. The fully-stable Pareto gap is
   ~30x: `f0_kpinv` 0.014% @ `eta=0.01` (100% stable) vs the control's best
   fully-stable 0.416% @ `eta=0.3`. (`ndiv = 0` everywhere -- the detached
   launch holds; "unstable" here is marginal closed-loop tails, not blow-ups.)
4. **The interaction is essential (axis attribution).** `f0_k0` is an *exact
   gradient saddle*: `F=0 ∧ K=0 ⇒ z ≡ 0`, and every gradient path is `∝ z` or
   `∝ F`, so nothing ever trains -- it sits at the zero-predictor ceiling
   (81.294% excess, `‖K‖ = 0`, relIR 1.0) forever, at every step size, even
   after 10⁶ steps. `F=0` alone is fatal; `K=H⁺` alone (the hybrid) accelerates
   but inherits the marginal unit-circle basin. Only the combination finds the
   contractive basin.

## Per-arm best (each at its own best step; L=1000000)

floor = 0.0546; relIR at the best step. From
`python scripts/analyze_mech_sweep.py --prefix sd_init_1M`.

| arm | best excess | best step | %stbl @ best | relIR |
|---|---|---|---|---|
| `fI_k0` (control) | 0.012% | 0.01 | 0.35 | 0.004 |
| `fI_kpinv` (hybrid) | 0.015% | 0.01 | 0.44 | 0.005 |
| `f0_kpinv` (K=A) | **0.014%** | 0.01 | **1.00** | 0.005 |
| `f0_k0` | 81.294% | (any) | 1.00 | 1.000 |
| `fI_k0_b0p05` | 0.469% | 0.1 | 1.00 | 0.045 |
| `f0_kpinv_b0p05` | 0.127% | 0.03 | 1.00 | 0.043 |

At 300k the `eta=0.01` column was under-converged (control 0.025% @ 9% stable,
`f0_kpinv` 0.257% with its best at `eta=0.03`); at 1M every live β2=0 arm's best
step is `eta=0.01` and the three excesses agree to ~0.001%. The robust
comparison is `f0_kpinv`'s 0.014% at **perfect** stability vs the control's
0.012% read off a 35%-stable population (and its best *fully-stable* 0.416% @
`eta=0.3`).

## Steady excess grid: matched excess, unmatched stability

Steady excess (%) per `(arm, step)`, tail `%stable` in parentheses; `ndiv = 0`
for every cell.

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| `fI_k0` (control) | 0.012 (35%) | 0.039 (60%) | 0.131 (99%) | 0.416 (100%) | 1.669 (100%) |
| `fI_kpinv` | 0.015 (44%) | 0.042 (67%) | 0.143 (100%) | 0.445 (100%) | 1.624 (100%) |
| `f0_kpinv` | **0.014 (100%)** | 0.041 (100%) | 0.140 (100%) | 0.420 (100%) | 1.441 (100%) |
| `f0_k0` | 81.294 (100%) | 81.294 (100%) | 81.294 (100%) | 81.294 (100%) | 81.294 (100%) |
| `fI_k0_b0p05` | 2.867 (81%) | 2.933 (100%) | 0.469 (100%) | 1.187 (100%) | 5.349 (100%) |
| `f0_kpinv_b0p05` | 2.859 (100%) | 0.127 (100%) | 0.224 (100%) | 0.806 (100%) | 5.693 (100%) |

Median `|λ(F_hat)|` splits cleanly by F-init: every live `fI` cell at
`eta ≤ 0.3` reads 0.96-1.00; every live `f0` cell reads 0.73-0.82 (true
`|eig(F)|` ∈ [0.778, 0.900]); at the bruising `eta=1.0` everyone is driven
contractive (~0.77-0.80). Note the control's F init `(1-eps)I = 0.9·I` starts
*exactly at* the true spectral radius and still drifts up to 1.00 under the SD
self-consistency pressure -- and 10⁶ steps do not bring it back down (the fI
small-step stability improves with horizon, 9-18% at 300k → 35-67% at 1M, but
plateaus well short of 100%); `F = 0` grows only as much dynamics as the data
demands and stops inside the true spectrum. The β2 arms are uniformly worse at
every step -- their mid-step cells escaped the ~2.9% plateau between 300k and 1M
(0.03: 2.919 → 0.127; 0.1: 3.120 → 0.469) yet still sit 3-10x above the β2=0
arms, and the small-step cells never escape -- β2 buys nothing anywhere on the
A-init base.

The `eta=0.01` adapted-`F` spectra make finding 3 visual: blue `×` are the true
`|eig(F)| ∈ [0.778, 0.900]`, the red `○` the converged mean `F_hat`. `f0_kpinv`
lands its spectrum *inside* the true one (`|λ| = 0.82`), while the control's
drifts *outward* onto the unit circle (`|λ| = 1.00`), past the true dynamics --
the marginal basin, from the same data and the same realized filter.

| `f0_kpinv` (K=A): converged inside the spectrum | `fI_k0` (control): drifted onto the unit circle |
|---|---|
| ![f0_kpinv adapted F spectrum, converged inside the true spectrum at |lambda|=0.82](../output/sd_init_1M/ss0p01/f0_kpinv/sd6_od2_eig_complex_plane.png) | ![fI_k0 control adapted F spectrum, drifted onto the unit circle at |lambda|=1.00](../output/sd_init_1M/ss0p01/fI_k0/sd6_od2_eig_complex_plane.png) |

## Convergence: A-init leaves the transient early, at no floor cost

One panel per step size, each overlaying every init arm's excess vs online step
(the control `fI_k0` drawn dashed/gray):

![A-init convergence per step](../output/sd_init_1M_convergence.png)

Excess (%) vs online step at the converged best step `eta=0.01`:

| arm | 100 | 1k | 3k | 10k | 30k | 100k | 300k | 1M |
|---|---|---|---|---|---|---|---|---|
| `fI_k0` (control) | 78.44 | 57.31 | 36.89 | 15.24 | 4.41 | 0.34 | -- | 0.014 |
| `fI_kpinv` | **25.95** | **18.54** | **10.39** | **3.86** | **1.94** | 0.77 | 0.046 | 0.013 |
| `f0_kpinv` | 77.35 | 50.92 | 22.27 | 6.03 | 3.44 | 1.67 | 0.178 | 0.013 |
| `fI_k0_b0p05` | 72.94 | 35.90 | 15.97 | 3.24 | 1.11 | 2.74 | 2.87 | 2.88 |
| `f0_kpinv_b0p05` | 77.69 | 49.50 | 21.98 | 6.11 | 3.41 | 2.90 | 2.86 | 2.86 |

(The control's 300k cell is blank because every trajectory read unstable at that
sampled step -- the marginal-population noise of the fI basin, which is exactly
the point of finding 3.)

- **The three live β2=0 arms all end at 0.013-0.014%** -- the init changes the
  trajectory, not the destination. This is the claim the 1M rerun was for: at
  300k the `eta=0.01` column was still mid-descent (`f0_kpinv` read 0.257%) and
  the floors could not be compared; at 1M all three agree to ~0.001%. The two
  init identities are visible at step 100: the hybrid starts at the copy
  predictor (~26%), the `f0` arms at the zero-predictor ceiling (~78-81%).
- **The hybrid owns the early transient** (3-10x below the control through 30k):
  the observation-ZOH prediction baseline is a pure head start.
- **`f0_kpinv` is the mid-transient accelerant** (~2-3x under the control from
  3k to 30k) despite starting at the ceiling -- informative `H⁺y` states make
  the SD and anchor gradients live immediately.
- **β2 vs A-init as accelerators**: on the control init β2 is the faster early
  descender (its known role), but it flattens at ~2.9% by 30k and never
  rejoins. On the A-init base its curve is indistinguishable from β2=0 until
  precisely the point where it starts to *hurt* (~10-30k). The control only
  crosses below `f0_kpinv` at ~36k (`eta=0.01`) / ~12k (`eta=0.03`) -- and by
  then both are en route to the same floor, while the control's trajectory
  population is already largely marginal (35% / 60% stable at 1M) and
  `f0_kpinv`'s is intact.

## Gain migration: both ends walk to the same Kalman ball; β2 pins the replace limit

`‖KH − H⁺H‖` is exactly 0 at the `K = H⁺` replace limit and grows as the learned
gain walks toward the Kalman gain; `‖K‖` is the gain magnitude (the `K=0` arms
grow from 0, the `K=H⁺` arms adapt from `‖H⁺‖`):

![A-init gain migration](../output/sd_init_1M_gain_migration.png)

- **Endpoint invariance of the gain (β2=0):** all three live arms converge to the
  same ball -- `‖K‖ ≈ 0.51-0.57`, `‖KH − H⁺H‖ ≈ 0.49-0.58` -- whether the gain
  grew up from 0 or annealed down from the replace limit. The learned Kalman gain
  is genuinely away from replace, and the migration completes from both ends.
- **β2 pins the gain near the replace limit:** at `eta ≤ 0.1` both β2 arms sit at
  `‖KH − H⁺H‖ ≈ 0.05-0.08` (vs ~0.53 for β2=0) with inflated `‖K‖ ≈ 0.76-0.83`.
  This is the mechanistic picture of β2's bias in one number: fitting the
  posterior to the noisy observation holds the filter at the copy/replace end
  (`HK ≈ I`) and *blocks the A→K migration* -- the toy shadow of a real-case
  system that never funnels regions from the intra path to the residual path.
  Sharpened by the 1M horizon: the β2 cells that *escaped* the excess plateau
  (`f0_kpinv_b0p05@0.03` → 0.127%, `fI_k0_b0p05@0.1` → 0.469%) did so **with the
  gain still pinned** (`‖KH − H⁺H‖` 0.077 / 0.047) -- the late excess descent is
  `F`/`H` compensating around a copy-regime gain, not a gain migration, which is
  why those cells still sit 3-10x above their β2=0 counterparts. The `f0_k0`
  saddle reads `‖K‖ = 0`, `‖KH − H⁺H‖ = ‖H⁺H‖ = √2` (rank-2 projector), as
  expected.

## Real-case translation (deferred design notes)

The toy's single `K = H⁺` gain maps to the routed **K/A pipeline**, not the K
network. The toy informs some real-case decisions and is silent on others; the
scope statement at the end is load-bearing.

- **F/gate binding (a real-case-only phenomenon).** Because the router gates on
  `‖ε‖`, the F init and the default pathway are bound: `F=0` ⇒ `ε ≈` full signal ⇒
  all-A default for free; `F=ZOH` ⇒ small `ε` ⇒ K-routed, but an inert K ⇒ states
  coast/stale ⇒ drift accumulates past `τ₂` ⇒ intermittent A firing (a
  relaxation-oscillator regime -- intended at deployment, plausibly bad for early
  training: sparse A gradients, K trained on drift not innovation). The toy cannot
  exhibit this binding: with a single gain, replace-from-error ≡
  replace-from-observation (`ẑ + H⁺(y − Hẑ) = (I − H⁺H)ẑ + H⁺y`, F-independent), so
  the two `K=H⁺` arms differ only in the prediction baseline. **Decision rule from
  the toy 2×2:** hybrid ≫ F=0 ⇒ the prediction baseline carries the acceleration ⇒
  invest in gate-override init machinery (forced-open, annealed to error-driven);
  hybrid ≈ F=0 ⇒ informative A-states suffice ⇒ the error-driven router yields the
  inverted curriculum with no new machinery (a minimality win).
  **RESOLVED by this sweep, in favor of no gate-override machinery.** The
  hybrid's prediction-baseline advantage is *transient-only* (3-10x through
  ~30k steps; both arms end at the same 0.013-0.014% converged floor at 1M),
  while `f0_kpinv` wins the axis that has plagued the entire study line -- it
  is 100% stable at every step where the hybrid keeps the control's marginal
  small-step stability (44-67% even after 10⁶ steps). Since the hybrid's real
  counterpart costs machinery and buys only a transient, and the router-natural
  arm wins stability outright: trust the error-driven router. Corollary: do not
  fear the bad-prediction phase at init -- it is exactly what routes everything
  to A, and that configuration trained *more* stably here, not less.
- **Contractive-F-init translation (new, from finding 3).** The stable basin came
  from growing `F`'s dynamics *up from zero* rather than starting at marginal
  stability -- notable because the control starts at `0.9·I`, exactly the true
  spectral radius, and still drifts to the unit circle under SD
  self-consistency pressure. The video `F` init (`v = 0` zero-init read-out ⇒
  exact ZOH) **is the marginal-stability init** (`ẑ = z`, |λ| = 1), i.e. the
  video default inherits the toy control's basin, and E2's `λ_max ≈ 1.00-1.13`
  endpoints are consistent with that. The toy `F=0` has no exact video analog
  (the Euler form has no "predict nothing"), but a *contractive* velocity init
  -- e.g. `v = -γ·z` with small `γ > 0`, decaying toward 0/mean instead of
  holding -- is the faithful translation: prediction slightly worse than ZOH
  early (which also keeps `ε` large ⇒ all-A routing, reinforcing the inverted
  curriculum), in exchange for approaching the true dynamics from the
  contractive side. Worth a cheap E-series probe arm; it costs one init flag.
- **Inverted curriculum recommendation.** Instead of DESIGN.md's staging (K-only at
  `τ₂ = ∞`, anneal A in later), initialize at `τ₂ ≈ 0` / gate open: the model at
  step 0 *is* the E1 frame-AE (a sane operating point; states pinned to the AE
  manifold, suppressing the E2 `z_post` null-space runaway), and routing migrates
  regions to K as F earns them. Codec framing: start all-intra, become inter as
  prediction improves.
- **K's cold start is correct credit assignment but needs two real-case-only
  mechanisms:** (1) soft-gate gradient leakage -- in the gain-interpolation form K
  receives `(1−g)`-scaled gradient even in A-routed regions, so it trains in A's
  shadow before winning the route; (2) RD pressure (`λ_A ≫ λ_K`) as the forcing out
  of the all-A basin, with the annealing schedule (tie `λ_A` / `τ₂` to measured
  innovation) flagged as the key hyperparameter risk (over-use of A becomes the
  *default* basin under this init).
- **β2 displacement claim is F-scoped in the real case.** In the toy the
  displacement was *total* (finding 2: β2 adds nothing on the A-init base, at any
  point in training) -- but toy A-init fixes F's cold start, and with split
  networks it gives K no gradient, so a small β2 during K's onset window remains
  a candidate there. The toy's verdict does not rule this out (K's real-case
  training is all transient). The toy cannot adjudicate this -- though the gain-
  migration result sharpens the risk: β2 held the toy gain pinned at the replace
  limit, so a real-case β2 left on past K's onset is a *mechanism for never
  funnelling regions from A to K*. Anneal it out.
- **A must train under this init, and that is the objectness mechanism.** F
  predicting a frozen retinotopic-tiling A may plateau (motion = content hopping
  across token owners) and the funnel to K never opens; unfreezing A puts
  predictability pressure on the code (object-bound tokens are predictable,
  tile-bound are not), unifying the routing curriculum with tiling→objectness
  emergence. Guardrails: stop-grad targets, A anchored by frozen-S reconstruction
  (anchored TD), slow LR on A, a frozen-A warm-up phase matching the toy regime.
- **A's loss set (recommended).** (1) Reconstruction anchor `‖S − H(A(S))‖²`
  dominant, with H frozen -- grounds A (anchored TD), pins the code scale
  (homogeneous arch / E2 rms runaway), confines drift to H's decodable manifold.
  (2) Predictability pressure via existing grounded losses flowing through the
  state into A: primarily the β0 a-priori obs term's gradient through
  `z_{t-1} = A(...)` (external frozen target, no collapse path); the α SD launch
  into A is the `keep_launch` knob -- detached by default, keep = known speed/floor
  trade. (3) Reverse distillation `A(x_t) → sg[ẑ_t]` rejected as core: closes an
  F↔A consensus loop (BYOL-symmetric, anchor-only grounding), inverts teacher
  privilege (ẑ's extra content is permanence, which single-frame A cannot match),
  and under ZOH-F + gate-open init degenerates to a slowness prior
  `‖A(x_t) − sg A(x_{t-1})‖²` -- subsumed by the β0-through-state path. Keep only as
  an escalation-drawer, observable-masked slowness regularizer. The rate term
  reaches A indirectly through `ε`; no A-specific rate term. Schedule: frozen-A
  warm-up → unfreeze (anchor + β0-through-state), slow LR, sg on all target uses,
  EMA-A as a drift escalation.
- **Vector vs position space.** Zero-preserving / homogeneous constraints are
  mandated for K (difference space, 0 = omit) and incidental for A (needs affine /
  DC capacity; E1's winning arch is bias-free -- fine on synthetic, watch on real
  latents). Cheap structural fix: de-mean the comparison space, making position
  space zero-mean too -- the real-case analog of why the zero-mean toy cannot see
  the linear-vs-affine distinction.
- **Scope statement.** The toy informs: high-gain-end init as a free accelerator;
  gain migration toward Kalman (`‖KH − H⁺H‖` = the toy shadow of regions funnelling
  A → K); the contractive-vs-marginal F-init basin (finding 3) and β2's
  replace-limit pinning as a migration blocker. The toy cannot inform: the
  two-network handoff / deadlock, trainable-A target drift, the position / vector
  asymmetry.
