# n-step self-distillation ladder x launch, on the A-init: mean vs sum reduction

Follow-up to [`nstep_sd_ladder_launch_ainit_findings.md`](nstep_sd_ladder_launch_ainit_findings.md)
(the same depth x launch grid on the A-init, but reducing the ladder by a **sum**)
and [`init_pathway_findings.md`](init_pathway_findings.md). The n-step latent SD
target is a ladder over autonomous horizons `k = 1..n`; every prior study
**summed** those horizon terms:

```
alpha * sum_{k=1}^{n} 0.5 || sg(x_j^+) - F^k x_{j-k}^+ ||^2
```

so the *effective* SD weight grows ~linearly with depth. That confounds two
things the depth sweep tried to separate: the multi-horizon *information* (more
distillation targets) and its *magnitude* (a bigger gradient at deeper `n`). This
study swaps the sum for a **mean** over the ladder's active horizons,

```
alpha * (1/n_eff) sum_{k=1}^{n_eff} 0.5 || sg(x_j^+) - F^k x_{j-k}^+ ||^2 ,
    n_eff = min(n, j+1)
```

holding the SD gradient *magnitude* roughly fixed across depth, and asks whether
the "deeper = higher floor, faster early" tradeoff survives once the magnitude
confound is removed. As specified, the study is run **only on the A-init**
(`f0_kpinv`: `F=0, K=H^+`), the contractive/100%-stable basin the sum study
settled on; the sum-reduction A-init numbers are the reference throughout.

## Setup

`--methods nstep --f-init zero --k-init pinv --sd-mean`
([`scripts/self_distillation_losses.py`](../scripts/self_distillation_losses.py)):
identical to the sum A-init sweep except for the new `--sd-mean` flag. A pure
a-priori control plus one SD+anchor arm per ladder depth `n = 1..4` (`window=4`),
all adapting `(F, H, K)` with `beta2=0`, constant gain, no projection / warm start
/ RTRL, all initialized at `f0_kpinv`. The ladder reduction is the *only* change:

- `--no-sd-mean` (default) -- sum the `n_eff` horizon terms (every prior study).
- `--sd-mean` -- average them, dividing each target's ladder contribution by
  `n_eff = min(sd_horizon, j+1)` (the number of distinct horizons that reach a
  buffered launch; horizons past the detached root collapse to one term and are
  already excluded).

Because `n_eff = 1` whenever a target only has the one-step horizon, **`n=1` is
bit-identical under sum and mean**, and the M4 control (`alpha=0`) is untouched --
so those two arms are a built-in sanity check, and every launch conclusion, which
lives at the winning `n=1` arm, carries over unchanged from the sum study.

Per-arm step sweep `--step-size in {0.01, 0.03, 0.1, 0.3, 1.0}`, `sd6_od2`,
`eps=0.1`, `N=16`, `seed=0`, same CUDA-drawn system as the mechanism / depth /
init studies (floor 0.0546). Two sweeps at the converged `L=300000` horizon,
identical except for the launch gradient:

- `output/sd_depth_ainit_mean_L300k/ss*` -- `keep_launch=True`
- `output/sd_depth_ainit_mean_detach_L300k/ss*` -- `keep_launch=False`

Driver: the existing [`scripts/submit_sd_depth_ainit_sweep.sh`](../scripts/submit_sd_depth_ainit_sweep.sh)
with `--sd-mean` forwarded and `OUT_PREFIX=sd_depth_ainit_mean_L300k`. Readers
(all prefix-parametrized, reused unchanged):
[`scripts/analyze_mech_sweep.py`](../scripts/analyze_mech_sweep.py),
[`scripts/plot_sd_depth_sweep.py`](../scripts/plot_sd_depth_sweep.py),
[`scripts/plot_sd_launch_convergence.py`](../scripts/plot_sd_launch_convergence.py),
[`scripts/plot_sd_depth_convergence.py`](../scripts/plot_sd_depth_convergence.py).

As before `excess` (median analytical error among stable trajectories, relative to
floor) and `%stbl` are read together -- but on the A-init this caveat evaporates:
every cell here is 100% stable (`ndiv=0` everywhere), so `excess` is read off full
populations.

## Verdict

Three findings at the converged `L=300000` horizon.

1. **The "deeper = higher floor" penalty was mostly a magnitude artifact of the
   sum: the mean collapses it.** Under the sum, each extra autonomous step raised
   the steady floor sharply (keep @`eta=0.01`: `n=1` 0.027% -> `n=4` 0.095%, a
   3.5x spread). Under the mean the spread nearly vanishes: `n=1` 0.027% -> `n=4`
   0.033%, a 1.2x spread. Normalizing the ladder magnitude removes almost all of
   the depth floor penalty -- the deeper targets are not intrinsically much
   noisier; the sum was just applying a `~n`x larger effective SD weight at depth.

2. **But the mean also destroys depth's early-convergence benefit, so depth now
   buys nothing.** The sum's depth tradeoff had a *good* side: deeper ladders
   converged 2x faster early (sum keep @`eta=0.03`, ctx=3k: `n=4` 3.47% vs `n=1`
   6.48%). That speedup was the summed gradient magnitude, not the multi-horizon
   information: under the mean, deeper is now *slightly slower* early (mean keep
   @`eta=0.03`, ctx=3k: `n=4` 9.06% vs `n=1` 6.48%) and marginally worse at the
   tail. So the mean converts depth from a genuine **speed/floor tradeoff** into a
   near-no-op that is slightly harmful at every horizon. On this stationary
   problem the multi-horizon *information*, magnitude held fixed, provides no
   early-descent benefit -- the sum's acceleration was purely a larger step.

3. **`n=1` still wins, the launch verdict is unchanged, and stability is (if
   anything) better.** `n=1` is bit-identical to the sum study, so it remains the
   best SD arm and the keep-vs-detach launch story is exactly as before (keep
   converges faster and is safe on the A-init; detach reaches a slightly lower
   floor where it converges but its `eta=0.01` column is still under-converged at
   300k). The whole mean grid is `ndiv=0` at 100% stability -- including the
   single `n=4, eta=1.0` corner that lost one trajectory under the sum -- because
   the mean's smaller effective SD magnitude at depth never stresses `F`.

**Best configuration found (converged at 300k): `keep_launch=True, n=1,
eta=0.01`, 0.027% excess at 100% stability** -- identical to the sum A-init study
(the winner is unchanged; the mean's contribution is that `n=2..4` are now
near-tied with it rather than 2-3.5x worse).

## Per-arm best (each at its own best step; L=300000)

floor = 0.0546. `%stbl` at the best step in parentheses; sum reference from
[`nstep_sd_ladder_launch_ainit_findings.md`](nstep_sd_ladder_launch_ainit_findings.md).

| arm | mean keep | sum keep | mean detach | sum detach |
|---|---|---|---|---|
| M4 (a-priori control) | 0.089% @0.01 (100%) | 0.089% @0.01 | 0.089% @0.01 (100%) | 0.092% @0.01 |
| SD+anchor n=1 | **0.027% @0.01 (100%)** | **0.027% @0.01** | 0.040% @0.03 (100%) | 0.044% @0.03 |
| SD+anchor n=2 | 0.031% @0.01 (100%) | 0.055% @0.01 | 0.060% @0.03 (100%) | 0.068% @0.01 |
| SD+anchor n=3 | 0.033% @0.01 (100%) | 0.081% @0.01 | 0.059% @0.01 (100%) | 0.072% @0.01 |
| SD+anchor n=4 | 0.033% @0.01 (100%) | 0.095% @0.01 | 0.056% @0.01 (100%) | 0.083% @0.01 |

`n=1` matches the sum study by construction (the tiny M4/`n=1` reference diffs vs
sum are stored-run/code drift, not the reduction). The signal is the deeper arms:
under keep the mean nearly ties them to `n=1` (0.031-0.033% vs the sum's
0.055-0.095%); under detach the mean even lets deeper arms edge *below* the
under-converged `n=1` @`eta=0.01`, but `n=1` at its own best step (`eta=0.03`,
0.040%) still wins.

## Steady excess grids: depths compressed, 100% stable everywhere

Steady excess (%) per `(arm, step)`, tail `%stable` in parentheses. `ndiv=0` for
**every** cell in both variants (the sum's lone `n=4, eta=1.0` casualty is gone).

### keep_launch=True (`sd_depth_ainit_mean_L300k`)

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.089 (100%) | 0.269 (100%) | 0.777 (100%) | 2.234 (100%) | 8.844 (100%) |
| SD+anchor n=1 | **0.027 (100%)** | 0.080 (100%) | 0.263 (100%) | 0.692 (100%) | 2.125 (100%) |
| SD+anchor n=2 | 0.031 (100%) | 0.092 (100%) | 0.298 (100%) | 0.736 (100%) | 2.195 (100%) |
| SD+anchor n=3 | 0.033 (100%) | 0.101 (100%) | 0.314 (100%) | 0.760 (100%) | 2.193 (100%) |
| SD+anchor n=4 | 0.033 (100%) | 0.103 (100%) | 0.326 (100%) | 0.791 (100%) | 2.212 (100%) |

### keep_launch=False / detached (`sd_depth_ainit_mean_detach_L300k`)

| arm | eta=0.01 | eta=0.03 | eta=0.1 | eta=0.3 | eta=1.0 |
|---|---|---|---|---|---|
| M4 constant | 0.089 (100%) | 0.269 (100%) | 0.777 (100%) | 2.234 (100%) | 8.859 (100%) |
| SD+anchor n=1 | 0.251* (100%) | **0.040 (100%)** | 0.144 (100%) | 0.439 (100%) | 1.470 (100%) |
| SD+anchor n=2 | 0.077 (100%) | 0.060 (100%) | 0.208 (100%) | 0.628 (100%) | 1.965 (100%) |
| SD+anchor n=3 | 0.059 (100%) | 0.073 (100%) | 0.253 (100%) | 0.738 (100%) | 2.124 (100%) |
| SD+anchor n=4 | 0.056 (100%) | 0.078 (100%) | 0.267 (100%) | 0.776 (100%) | 2.219 (100%) |

`*` detached `n=1, eta=0.01` is under-converged at 300k (still descending; the
same signature as the sum study, which needed `L=1000000` to resolve it).

Compare the sum keep grid, where `n=4` ran 0.095 / 0.269 / 0.669 / 1.487 / 4.229
across the same steps -- roughly `n`x the `n=1` row. The mean rows sit within
~1.05-1.2x of `n=1` everywhere. Median `|lambda(F_hat)|` reads **0.82 at
`eta <= 0.1`** across all depths and drifts to 0.74-0.80 at `eta=1.0` -- the same
contractive basin (inside the true spectrum `|eig(F)| in [0.778, 0.900]`) as the
sum A-init study, unchanged by the reduction.

![mean keep_launch=True depth sweep](../output/sd_depth_ainit_mean_L300k_depth_sweep.png)

![mean keep_launch=False depth sweep](../output/sd_depth_ainit_mean_detach_L300k_depth_sweep.png)

## Convergence: depth loses its early-speed edge under the mean

`n=1` excess vs context (%) -- identical to the sum A-init study by construction
(the launch story is unchanged: keep @0.01 converged by 200k and lowest at this
horizon; detach @0.01 still descending):

| ctx | keep @0.01 | detach @0.01 | keep @0.03 | detach @0.03 |
|---|---|---|---|---|
| 3,000 | 19.82 | 21.35 | 6.48 | 6.68 |
| 10,000 | 5.80 | 6.13 | 2.13 | 3.51 |
| 30,000 | 2.02 | 3.42 | 0.462 | 1.95 |
| 100,000 | 0.259 | 1.674 | 0.076 | 0.213 |
| 200,000 | 0.028 | 0.573 | 0.056 | 0.031 |
| 300,000 | **0.024** | 0.179 | 0.074 | **0.040** |

![mean keep vs detach convergence, A-init](../output/sd_depth_ainit_mean_launch_convergence.png)

The depth story is where the mean diverges from the sum. Excess vs context (%) at
`eta=0.03`, `keep_launch=True`, per depth:

| ctx | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| 3,000 | 6.48 | 7.70 | 8.74 | 9.06 |
| 10,000 | 2.13 | 2.11 | 2.20 | 2.24 |
| 30,000 | 0.462 | 0.413 | 0.456 | 0.467 |
| 100,000 | 0.076 | 0.083 | 0.078 | 0.077 |
| 300,000 | **0.074** | 0.080 | 0.086 | 0.086 |

Under the **sum**, this same table had the deeper ladders *leading* by ~2x early
(n=4 3.47 vs n=1 6.48 at 3k) before inverting to `n=1`-lowest at the tail -- a
clean speed/floor fan-out. Under the **mean**, the early lead is **gone**: deeper
is marginally *slower* at 3k, the depths are essentially on top of each other from
10k onward, and `n=1` is (barely) lowest at the tail. The multi-horizon gradient's
early acceleration was its summed magnitude; average it away and the extra
horizons add nothing on this stationary problem.

![mean A-init depth convergence, eta=0.03](../output/sd_depth_ainit_mean_depth_convergence_eta0p03.png)

Left = keep, right = detach; color = depth. (Detach @`eta=0.03` shows the deeper
arms reaching the mid-context floor a bit sooner -- 30k: n=4 1.37 vs n=1 1.95 --
because detaching weakens `n=1`'s gradient more than it weakens the averaged deep
ladder; but `n=1` still wins the tail, 0.040 vs 0.065.)

## Stability: the mean is (marginally) more stable than the sum

Diverged trajectories (out of 16) at `eta=1.0`, `L=300000`:

| arm | A-init sum keep | A-init sum detach | A-init mean keep | A-init mean detach |
|---|---|---|---|---|
| SD+anchor n=1 | 0 | 0 | **0** | **0** |
| SD+anchor n=2 | 0 | 0 | **0** | **0** |
| SD+anchor n=3 | 0 | 0 | **0** | **0** |
| SD+anchor n=4 | 1 | 1 | **0** | **0** |

The A-init already made both launch variants essentially fully stable; the mean
removes the last casualty (the `n=4, eta=1.0` corner, 1/16 under the sum in both
variants) by shrinking the effective SD magnitude at depth, so the deepest ladder
no longer stresses `F` even at the most aggressive gain. `rad` stays contractive
(0.74-0.82) everywhere.

## Reading

1. **The sum's depth tradeoff was a magnitude artifact.** Summing the ladder
   applies a `~n`x larger effective SD weight at depth `n`; that single fact
   produced *both* halves of the sum study's depth verdict -- the faster early
   convergence (bigger step) and the higher asymptotic floor (bigger noise ball).
   The mean holds the magnitude fixed and both halves collapse: floors compress to
   near-`n=1`, and the early speedup disappears.

2. **The multi-horizon information alone buys nothing here.** With magnitude
   controlled, adding autonomous horizons `k = 2..n` neither accelerates early
   descent nor lowers the floor on this stationary system -- deeper is uniformly a
   hair worse. So the `F^k` ladder's only real lever was its summed gradient
   magnitude, which the plain single-step term (or simply a larger `alpha` / gain)
   supplies more directly.

3. **`n=1` remains the recommendation, now for a simpler reason.** Under the sum
   `n=1` won the converged floor but conceded early speed to deeper ladders (a
   real tradeoff a tracking task might have wanted). Under the mean `n=1` wins
   essentially everywhere with no tradeoff surrendered, so the case for a
   single-step ladder is unconditional on this problem. The launch verdict (keep
   is fast and safe on the A-init; detach floors slightly lower where converged)
   is unchanged, since it lives entirely at `n=1`.

## Caveats / follow-ups

- **`n=1` is unchanged by the reduction** (sum and mean are bit-identical there,
  verified in the parity test and the smoke run), so this study genuinely only
  re-prices depth `n>=2`; the launch and best-config conclusions are inherited
  from [`nstep_sd_ladder_launch_ainit_findings.md`](nstep_sd_ladder_launch_ainit_findings.md).
- **Detached `eta=0.01` is under-converged at 300k** (mean `n=1` 0.179%, still
  descending) -- the same signature as the sum study; the keep-vs-detach floor
  race at the smallest gain would need `L=1000000` on that column, though it does
  not affect the depth verdict (which is read at converged cells).
- **Non-stationary implication.** The sum study flagged depth's *early-speed*
  benefit as the property that might matter under drift. This study shows that
  benefit is purely a gradient-magnitude effect -- so under non-stationarity a
  deeper *summed* ladder and a larger `alpha`/gain at `n=1` are likely
  interchangeable accelerators, and the mean reduction removes depth as an
  independent knob. If a drift study wants the ladder's early speed, it should use
  the sum (or tune the gain), not the mean.
