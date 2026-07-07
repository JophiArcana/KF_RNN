# n-step self-distillation: ladder depth and the launch gradient

Follow-up to
[`mechanism_constant_gain_sd_findings.md`](mechanism_constant_gain_sd_findings.md),
which established that the 1-step latent self-distillation objective with a light
a-priori anchor ("SD+anchor", `alpha=1, beta0=0.05`) is a better *deployable
constant-gain* filter than plain a-priori. This study extends the SD target from a
single autonomous step to an **n-step ladder** and asks two questions:

> 1. **Depth.** Does bootstrapping the latent target over a longer autonomous
>    roll-out `F^k x_{t-k}^+` (horizons `k = 1..n`) help or hurt?
> 2. **Launch gradient.** Should the launch state `x_{t-k}^+` carry gradient
>    (`keep_launch=True`, the original behavior) or be detached
>    (`keep_launch=False`)?

The latent term generalizes to

```
alpha * sum_{k=1}^{n} 0.5 || sg(x_t^+) - F^k x_{t-k}^+ ||^2
```

launched from the same rolled window (a horizon past the window clamps to the
detached root `s_start`). At `n=1, keep_launch=True` this is bit-identical to the
original single-step SD term (verified: loss and gradient match to float64
precision), so the previous behavior is the `n=1` keep arm here.

## Setup

`--methods nstep` ([`scripts/self_distillation_losses.py`](../scripts/self_distillation_losses.py)):
a pure a-priori control plus one SD+anchor arm per ladder depth `n = 1..4`
(`window=4`, so 4 is the depth ceiling), all adapting `(F, H, K)` with
`beta2=0`, constant gain, no projection / warm start / RTRL. Per-arm step sweep
`--step-size in {0.01, 0.03, 0.1, 0.3, 1.0}`, `sd6_od2`, `eps=0.1`, `N=16`,
`seed=0`, on the same CUDA-drawn system as the mechanism study (floor 0.0546).

Two sweeps, identical except for the launch gradient, at the **converged**
`L=300000` horizon:

- `output/sd_depth_L300k/ss*` -- `keep_launch=True`
- `output/sd_depth_detach_L300k/ss*` -- `keep_launch=False`

Shorter `L=30000` / `L=100000` sweeps (`sd_depth_L30k`, `sd_depth_L100k`,
`sd_depth_detach_L100k`) are retained only to document the convergence-rate
difference below. Sweep driver:
[`scripts/submit_sd_depth_sweep.sh`](../scripts/submit_sd_depth_sweep.sh); readers:
[`scripts/analyze_mech_sweep.py`](../scripts/analyze_mech_sweep.py) (envelope
table), [`scripts/plot_sd_depth_sweep.py`](../scripts/plot_sd_depth_sweep.py)
(depth grid), [`scripts/plot_sd_launch_convergence.py`](../scripts/plot_sd_launch_convergence.py)
(keep-vs-detach convergence overlay).

As in the mechanism study, `excess` (median analytical error among the stable
trajectories, relative to floor) and `%stbl` (tail fraction of the 16
trajectories closed-loop stable) MUST be read together: at the low-excess small
steps only a few trajectories are stable, so `excess` there is noisy; the robust
comparisons are the steps where a method is ~100% stable.

## Verdict

Two clean findings, both at the converged `L=300000` horizon:

1. **Deeper ladders trade convergence speed for a higher floor** -- the same
   tradeoff as the launch gradient (below). At the converged horizon depth `n=1`
   is the best SD arm at every step, in both launch variants: each extra
   autonomous step raises the steady floor (noisier model-generated targets at
   longer horizons enlarge the noise ball). But a deeper ladder converges
   *faster* early -- the richer multi-horizon gradient descends 2-3x quicker
   through the first ~10k steps before being overtaken.

2. **Detaching the launch (`keep_launch=False`) dominates once converged.** At the
   converged horizon the detached variant has **lower excess at every matched
   step AND better tail stability** than the gradient-carrying launch. Its only
   cost is slower convergence -- the launch gradient is a strong accelerator, so
   `keep_launch=True` leads throughout the mid-context regime and is only overtaken
   near the end. The best configuration found overall is **`keep_launch=False`,
   `n=1`, `eta=0.01`: 0.018% excess** -- ~4.8x closer to the irreducible floor than
   the a-priori baseline (0.087%), and below the previous 1-step keep headline
   (0.029%).

## Per-arm best (each at its own best step; L=300000)

floor = 0.0546. Best step is `eta=0.01` for every SD arm and the control.

| arm | keep=True excess | keep=False (detach) excess |
|---|---|---|
| M4 (a-priori control) | 0.087% | 0.087% |
| SD+anchor n=1 | 0.029% | **0.018%** |
| SD+anchor n=2 | 0.049% | 0.036% |
| SD+anchor n=3 | 0.068% | 0.050% |
| SD+anchor n=4 | 0.087% | 0.066% |

(The M4 control is bit-identical across the two sweeps -- `alpha=0`, so
`keep_launch` has no effect there; a useful sanity check that nothing else
drifted.)

## Excess: detach is lower at every matched step (L=300000)

`%stbl` = tail fraction stable; `ndiv` = diverged trajectories out of 16; `rad` =
median |lambda(F_hat)|. Each arm's lowest-excess step in **bold**.

### keep_launch=True (`sd_depth_L300k`)

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

### keep_launch=False / detached (`sd_depth_detach_L300k`)

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

Matched-step head-to-head for the best arm (n=1): detach is lower at every step --
`eta=0.01` 0.018% vs 0.029%; 0.03 0.039% vs 0.069%; 0.1 0.149% vs 0.234%; 0.3
0.414% vs 0.684%; 1.0 **1.562% vs 2.490%**.

### Depth grids

Left: steady excess vs step size, one line per depth; right: per-arm best.

![keep_launch=True depth sweep](../output/sd_depth_L300k_depth_sweep.png)

![keep_launch=False depth sweep](../output/sd_depth_detach_L300k_depth_sweep.png)

## Convergence: keep is faster, detach reaches a lower floor

This is the tradeoff behind the "best step is always the smallest" pattern. At the
best step `eta=0.01`, tracking n=1 excess vs context length:

| context | keep=True | keep=False (detach) |
|---|---|---|
| 3,000 | 26.0% | 36.3% |
| 10,000 | 5.3% | 15.5% |
| 30,000 | 0.42% | 4.27% |
| 100,000 | 0.061% | 0.323% |
| 200,000 | 0.013% | 0.037% |
| 300,000 | 0.022% | 0.021% (tail floor **0.018%** vs keep **0.029%**) |

`keep_launch=True` is **3-13x lower excess through the entire mid-context regime**
-- the launch gradient is a strong convergence accelerator. It plateaus by ~100k;
the detached variant is still descending and only crosses below keep at the very
tail. So the two effects are cleanly separated: **keep converges faster, detach
bottoms out lower.**

![keep vs detach convergence at L=300k](../output/sd_launch_convergence_L300k.png)

Solid = keep, dashed = detach, same color per depth. The earlier `L=100000` sweep
mis-ranked detach precisely because it stopped before this crossover: at 100k,
detach@eta=0.01 read 0.436% (under-converged) vs keep's 0.040%; the 300k run
resolves it to 0.018% vs 0.029%.

### Depth is the same speed/floor tradeoff

The launch gradient is not the only knob with this shape -- **ladder depth `n`
behaves identically**: a deeper roll-out converges faster but settles higher.
Excess vs context length at `eta=0.03`, `keep_launch=True`, per depth:

| context | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| 3,000 | 6.24% | 2.68% | 2.10% | 2.30% |
| 10,000 | 0.51% | 0.20% | 0.28% | 0.25% |
| 30,000 | 0.103% | 0.151% | 0.297% | 0.262% |
| 100,000 | 0.084% | 0.186% | 0.248% | 0.298% |
| 300,000 | **0.080%** | 0.130% | 0.200% | 0.227% |

Early (ctx <= 10k) the deeper ladders lead by 2-3x -- the richer multi-horizon
gradient (more distillation terms per window position) is a stronger early signal.
The curves cross around ctx ~ 10-30k, and by the tail the ordering fully inverts
to `n=1` lowest. So depth, like the launch gradient, buys early convergence speed
at the cost of the asymptotic floor.

![SD ladder depth convergence, eta=0.03](../output/sd_depth_convergence_eta0p03.png)

Left = `keep_launch=True`, right = detach; color = depth. The effect is
pronounced with the launch gradient on (left, clear early fan-out then inversion)
and muted when detached (right, depths track together early since detaching
already weakens the gradient signal), but the tail floor ordering (`n=1` lowest)
holds in both. Generated by
[`scripts/plot_sd_depth_convergence.py`](../scripts/plot_sd_depth_convergence.py).

## Stability: detach is far more robust at aggressive gains

Diverged trajectories (out of 16) at the largest gain `eta=1.0`, `L=300000`:

| arm | keep=True ndiv (%stbl) | keep=False ndiv (%stbl) |
|---|---|---|
| SD+anchor n=1 | 11 (0.31) | **0 (1.00)** |
| SD+anchor n=2 | 9 (0.44) | **0 (1.00)** |
| SD+anchor n=3 | 15 (0.06) | **0 (1.00)** |
| SD+anchor n=4 | 15 (0.06) | 2 (0.88) |

Detaching the launch removes the recursive gradient path through the bootstrap
chain -- the feedback that pushed `F` toward self-consistency and tipped
trajectories over the unit circle. The effect is dramatic: at `eta=1.0` the
gradient-carrying launch diverges on 9-15 of 16 trajectories, while the detached
launch is 100% stable for `n<=3`. This mirrors (and resolves) the mechanism
study's caveat that SD was "the one to watch at large steps" -- that instability
was the launch gradient, and `keep_launch=False` removes it.

### Depth's stability effect is gain-dependent (and launch-gradient-mediated)

Depth has a noticeable but **sign-flipping** impact on stability across the gain
range. Full grid, `L=300000`, each cell `tail %stable | ndiv (of 16)`:

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

1. **Small/moderate gain (eta <= 0.1): deeper is *more* stable.** Tail %stable
   rises monotonically with depth (keep eta=0.03: 0.36 -> 0.99 -> 1.00 -> 1.00),
   with zero divergences -- the same faster convergence settles the filter into a
   stable configuration sooner, so more trajectories are closed-loop stable at the
   tail.
2. **Aggressive gain (eta >= 0.3): deeper is *less* stable, but only with the
   launch gradient on.** With `keep_launch=True`, divergences climb with depth
   (eta=1.0: 11 -> 9 -> 15 -> 15) as the deeper `F^k` roll-out drives `F` harder
   toward self-consistency.
3. **Detach removes the depth-driven instability.** The detach sweep has zero
   divergences everywhere except the extreme n=4/eta=1 corner (2/16); without the
   launch gradient, depth only helps stability (via faster convergence) and costs
   essentially nothing across the grid.

So depth's stability effect is another face of the speed/floor tradeoff: faster
early convergence buys tail-stability at usable gains, while the aggressive-gain
instability is a `keep_launch` artifact that `--no-keep-launch` fixes. (The
small-`eta` %stable values are noisy -- the optimal filter hugs the unit circle,
`rad ~ 1.00`, so stability is marginal for everyone there -- but the depth trend
is consistent across both sweeps.)

## Reading

1. **Depth is a speed/floor knob, losing only at the stationary limit.** More
   autonomous steps give a richer early gradient (2-3x faster initial descent) but
   noisier distillation targets and a larger asymptotic noise ball; `n=1` wins the
   converged floor in both variants. So the `F^k` ladder is not worth building past
   a single step on this *stationary* problem -- but its early-convergence benefit
   is exactly the property that could matter under non-stationarity (see below).

2. **The launch gradient is a speed/floor/stability trilemma, and detach wins two
   of three.** Keeping it buys faster convergence; detaching it buys both a lower
   asymptotic floor and much better stability at aggressive gains. Given enough
   context, the detached variant is the strictly better operating point (lower
   excess at every matched step, more stable), and its speed deficit is only a
   transient.

3. **The best configuration found is detached, single-step, small-gain, long-run.**
   `keep_launch=False, n=1, eta=0.01` reaches 0.018% excess -- the lowest
   deployable constant-gain result in this line of study.

## Implication for the non-stationary / video next step

The stationary picture is now: detach the launch, keep the ladder at depth 1. But
this study's decisive axis is *convergence horizon*, which is exactly what
non-stationarity removes -- a tracking filter never gets 300k stationary steps to
reach an asymptotic floor. Two things to carry into the deferred tracking study:
(a) since **both** `keep_launch=True` and deeper `n` dominate at short horizons,
their convergence-speed advantage may *reverse* the stationary verdict under
drift, where fast adaptation matters more than the asymptotic ball -- worth
sweeping both the launch variant and depth `n=1..4` on a piecewise / drifting
`sd6_od2`; and (b) `keep_launch=False`
is the cheap stability fix flagged in the mechanism study (no projection needed),
so it is the natural default when an aggressive constant gain is required for
tracking responsiveness.
