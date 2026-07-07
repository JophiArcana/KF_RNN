# Test time training for LQE problems
_Exported on 6/28/2026 at 19:02:57 PDT from Cursor (3.8.11)_

---

**User**

@docs/minimal_convergent_filter_proposal.md @docs/self_distillation_study_manifest.md I am doing a study on test time training for LQE problems. I would like to start with the absolute most minimal method that might give 0 convergence. E.g. while the proposal suggests that stability projection and warm starts are required for convergence gaurantees, it is fine to omit them for simplicity if it will still probably work. I would also like to know what code modifications are necessary to the current code structure to do this.

This current results show that the convergence stops at above 0, meaning the regret fundamentally is not sub-constant @output/sd_m3m4_L10000/sd6_od2_analytical_error_curve.png 

---

**Cursor**

I'll research the proposal docs, the current code structure, and the results image to design the most minimal test-time training method. Let me start by reading the key files in parallel.

Now let me examine the actual code structure to understand exactly what hooks exist and how the scan/loop works.

Let me look at how the analytical error curve is produced and how per-step filter parameters are extracted, plus the optimizer/scan primitives.

Let me confirm the base predictor's parameters and the `eu.sgd_step` / `eu.scan` primitives so the plan is precise.

`sgd_step`/`scan` live in the `ecliseutils` package. Let me locate them to be precise about signatures.

I'll skip killing the process and just search for the package location.

Found the `ecliseutils` source. Let me read `sgd_step` and `scan` to confirm the exact primitives I'll build on.

I now have a complete picture. Let me confirm my understanding of the key fact that shapes the plan: the plotted analytical-error curve comes from `_adapt_and_measure` (a CPU re-implementation of the TTT loop), not from the model's `forward`/scan.

Here's what I found and the resulting plan.

## What causes the non-zero plateau

The current `sd6_od2 L=10000` curve flattens at ~0.2% excess (M4, absolute ~2e-4) because of a **constant SGD gain**. `_optimizer_step` does `theta - self.step_size * grads` with a fixed `step_size=0.03` ([rnn_ttt.py:267-282](src/kf_rnn/model/sequential/rnn_ttt.py)). A constant gain converges only to a noise ball of radius ∝√η (Dieuleveut–Durmus–Bach), which is exactly the proposal's §2.4 diagnosis and the gap to the online-FIR curves (which sit 1–2 orders lower). To get sub-constant regret you need a **vanishing gain**.

## The absolute-minimal method I'll propose

M4 objective (already `beta0=1, beta2=0, alpha=0`) + a vanishing gain:

- **Core (one knob):** decaying step `η_t = η_0·(t+1)^{−a}`, `a≈0.67` (Robbins–Monro; `a∈(½,1)`).
- **Companion (makes it η₀-robust + asymptotically efficient):** Polyak–Ruppert tail averaging — report `θ̄ = mean(θ_t)` over `t ≥ n₀` instead of `θ_t`.
- **Deliberately omitted** (per your call): stability projection, warm start.
- **Kept truncated:** the `window=4` gradient (no RTRL yet). This leaves an `O(|λ_cl|^4)` bias, but on this system the optimal FIR reaches the floor by R≈4, so that residual is small. RTRL is the clearly-labeled next step if a residual floor remains.

Before I write code, two quick decisions I want to confirm rather than assume.

> **How much of the change should I wire up for the first run?**
> - Both the model hooks (decaying step in _optimizer_step) AND the readout (Polyak tail-averaging θ̄ in the study's _adapt_and_measure), plus a runnable --methods minimal preset (Recommended) ✔
> - Just the decaying step (Robbins–Monro), no averaging — the single smallest change; add averaging only if the curve is too noisy
> - Only the model-side hooks; I'll wire the study/preset myself

> **Keep the truncated window=4 gradient for this first minimal attempt, or also switch to the exact RTRL/sensitivity-carry gradient now?**
> - Keep window=4 (truncated) — most minimal; treat RTRL as a separate follow-up tier (Recommended) ✔
> - Include the exact RTRL gradient now (larger change, removes the slow-pole bias)

Both confirmed: full wiring (model hooks + readout + `--methods minimal` preset), and keep `window=4` for this first attempt. Let me write up the plan.

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

I'll implement the plan. Let me start with the model hooks in `rnn_ttt.py`.

Now the Config and `__init__` changes:

Now thread `t` into the base `_optimizer_step` and add a decayed-rate helper:

Now the scan call site in `run_sequence.step`:

Now the `RnnSelfDistillPredictor._optimizer_step` override:

Those are pre-existing import-resolution warnings (linter not using the conda env), not from my edits. Now the readout in `_adapt_and_measure`.

Now add the `minimal_methods()` preset after `m3m4_methods()`:

Now wire the global CLI knobs and forward them into the `Config` in `run_method`:

Now the argparse: `--methods` choices, and the new `--step-decay`/`--polyak-burnin`:

Now select the preset in `main`:

Only pre-existing import warnings. Now mark preset done and run a quick CPU smoke test before submitting the full job.

The constant-step baseline ran fine. The duplicate-kwarg error is the per-method override colliding with the explicit kwargs. Let me merge so `method.cfg` wins.

End-to-end run works (both curves produce, save, and plot). The short L=80 horizon is just a smoke test — the decay+avg method's advantage only appears at long horizons. Let me clean up the throwaway smoke output and submit the full L=10000 job.

No `sbatch` here (this is a compute shell, not the login node). Let me check for a local GPU to run directly.

An L40S GPU is available locally. I'll run the full L=10000 sweep directly on it (instead of `sbatch`), in the background.

Let me do a quick smoke check that the run started cleanly (not failed to launch).

The launcher returned (the `nohup` python process is detached). Let me check the run log to confirm the actual sweep is progressing.

Log is empty — likely Python stdout block-buffering when redirected to a file. Let me verify the process is actually alive and using the GPU.

The full sweep (PID 3102264) is alive and on the GPU; the empty log is just Python block-buffering when redirected. The implementation is validated end-to-end by the smoke test. This L=10000 / 16-trajectory / 2-method run will take a while (the analogous 6-method run used a 4h budget; 2 methods should be well under an hour). Let me monitor for progress in slices rather than block the whole time.

Still running (low GPU util is expected — `_adapt_and_measure` is a per-step Python loop). The launcher shell already returned, so there's no completion notification wired up. Let me start a tracked watcher that blocks until the final plot is produced, so I can compare results when it finishes without polling disruptively.

The run completed. Let me look at the resulting analytical-error curve to read what actually happened to the minimal method during adaptation.

The run finished and the comparison is conclusive. Let me mark the final todo complete and report.

All five to-dos are complete. The plan is implemented, validated, and the headline run is done.

## What I changed

Model hooks — [src/kf_rnn/model/sequential/rnn_ttt.py](src/kf_rnn/model/sequential/rnn_ttt.py):
- Added `step_decay` (exponent `a`) and `polyak_burnin` config fields (defaults `0.0` / `-1.0` preserve current behavior).
- Added `_step_size_at(t) = step_size * (t+1)**(-step_decay)`; threaded the global step `t` into both `_optimizer_step` implementations and the scan call site.

Readout + preset — [scripts/self_distillation_losses.py](scripts/self_distillation_losses.py):
- `_adapt_and_measure` now passes `t` to the update, maintains the Polyak tail-average `theta_bar` (from `polyak_burnin * L`), and measures/returns the reported (averaged) filter.
- Added `minimal_methods()` (constant-step M4 control + `step_decay=0.67, polyak_burnin=0.1` minimal method), `--methods minimal`, and `--step-decay`/`--polyak-burnin` overrides.

I smoke-tested on CPU, then ran the full sweep on the local L40S (no `sbatch` on this node, so I ran it directly).

## Result: the minimal method, as specified, does not reach zero — it goes closed-loop-unstable

![minimal vs baseline](/pa/home/wentinn.liao/KF_RNN/output/sd_minimal_L10000/sd6_od2_analytical_error_curve.png)

| method | excess over floor | rel IR err | % stable at tail |
|---|---|---|---|
| M4 constant (baseline) | 0.3% | 0.030 | 26% |
| M4 decay+avg (minimal) | **inf** | 0.817 | **0%** |

- The **constant baseline reproduces the known ~0.3% plateau** (control works).
- The **decay+avg curve descends early (~step 100) then rises and cuts off ~step 3500** (gaps = every trajectory unstable). The averaged parameter point `F̄(I−K̄H̄)` has spectral radius ≥ 1, so `filter_analytical_error` returns `+inf`, and `relIR=0.817` shows it converged to a poor filter.

Two compounding causes, both tied to pieces we deliberately omitted:
1. No stability projection — averaging `(F,H,K)` independently does not preserve closed-loop contractivity, so the reported filter tips over the unit circle (exactly the §1.4 / §6.3 pathology the proposal flagged as load-bearing).
2. The `a=0.67`, `η₀=0.03` schedule shrinks the gain to ~6e-5 by the end, before the filter locks on — so it never converges, then the unstable readout dominates.

So this is a clean negative result: on `sd6_od2`, the vanishing gain alone is not enough — the stability projection is in fact load-bearing for this analytical-error readout. The knobs are now all in place to test the next levers without further plumbing: a gentler schedule (`--step-decay 0.51`, larger `polyak_burnin`), and then the omitted structural pieces (stability projection, warm start) and the exact RTRL gradient.

Want me to try a gentler decay/averaging schedule, or move straight to adding the stability projection?

---

**User**

Why does standard M4 not need the stability projection?

---

**Cursor**

Short answer: **standard M4 doesn't actually escape the stability problem — it hugs the unit circle just like the minimal method does.** It only survives the *readout* because of two crutches that the decay+avg method removes. Here's the mechanism.

## What "stable" means in this readout

`filter_analytical_error` returns a finite number only if the filter's closed loop `F(I−KH)` is a strict contraction (`ρ < 1`); otherwise `+inf`. The plotted curve is a `nanmedian` over the 16 trajectories — so a trajectory contributes only when its snapshot happens to be on the stable side.

Look at the tail-stability numbers from the run:

| method | median &#124;λ(F̂)&#124; | rel IR err | % stable at tail |
|---|---|---|---|
| M4 constant | 1.000 | **0.030** | **26%** |
| M4 decay+avg | 1.000 | 0.817 | **0%** |

M4 constant is **only 26% stable** — i.e. 74% of its trajectory snapshots are *also* closed-loop-unstable at any given tail step. Both methods sit right on the unit circle (`median|λ(F̂)|=1.0`). M4 is not safely inside the stable region; it's balancing on the edge.

## The two crutches M4 has and the minimal method loses

1. **Proximity to the inherently-stable optimum (`relIR 0.030` vs `0.817`).** The population PEM minimizer is the Kalman filter, whose closed loop `F(I−KH)` is automatically a contraction. So any estimator that *reaches* the optimum needs no projection — stability comes for free at the solution. M4 constant converges right next to the true filter (relIR 0.03), so it lives in the inherently-stable neighborhood. The decay+avg method **never got there**: the aggressive schedule (`a=0.67` drives `η` to ~6e-5 by step 10000) froze the iterate at a far-off, closed-loop-unstable point (relIR 0.82) before it converged. That under-convergence — not the averaging per se — is the primary cause of the `inf`.

2. **Dithering survivorship (constant gain) vs collapse (vanishing gain + averaging).** A constant gain keeps the iterate rattling around in a noise ball straddling `ρ≈1`, so at any snapshot a minority of trajectories (here ~26%) land on the stable side and rescue the median. Two of our changes destroy that:
   - the **vanishing gain** stops the dithering, so there's no longer "some momentarily-stable snapshot" to keep the median finite;
   - **Polyak averaging** collapses the cloud into a *single* point `θ̄`, and stability is not preserved under parameter-space averaging — `ρ(F̄(I−K̄H̄))` is not the average of the per-snapshot `ρ`'s, and can be larger than any of them.

## So why is the projection "non-negotiable" for the minimal method but not M4?

Because the projection's job is to keep you in the stable region *while the estimator is transiting / under-converged / being averaged*. M4 implicitly avoids needing it by (a) staying near the inherently-stable optimum and (b) never collapsing off the dithering cloud. The moment you push for true sub-constant regret — vanish the gain, average to one point, and stop relying on survivorship — you lose both crutches, and any excursion to `ρ≥1` becomes a permanent `inf` rather than a transient blip. That's exactly the §1.4 / §6.3 pathology the proposal flagged.

The practical implication for next steps: a gentler schedule that actually converges to the optimum (`a≈0.51`, larger `polyak_burnin`, or warm start) might land `θ̄` in the stable neighborhood and avoid the `inf` — but the robust fix the proposal prescribes is to add the stability projection so convergence quality can never turn into an unstable readout.

