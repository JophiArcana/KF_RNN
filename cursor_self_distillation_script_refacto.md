# Self distillation script refactor
_Exported on 7/7/2026 at 13:12:52 PDT from Cursor (3.8.11)_

---

**User**

I am doing a test-time training study in @scripts/self_distillation_losses.py. However, the script code has become a bit bulky. Is it possible to to restructure it to better integrate with the src/ code currently in place?

---

**Cursor**

I'll investigate the script and how the `src/` code is structured so I can propose a good restructuring plan.

Let me explore the model code and the related scripts to understand what already exists in `src/` and how the script duplicates logic.

Let me look at the TTT model in `src/` and the helper scripts that the main script reuses.

Let me examine the helper scripts the main script imports from, and the broader infrastructure layout.

Let me check which sibling scripts import from `self_distillation_losses.py`, since restructuring must not break them.

Let me check how the sibling plot/analyze scripts consume the results (e.g. `MethodResult`, `load_results`), since the persistence format matters for restructuring.

Let me check the infrastructure layout and where `analytical_error` lives, to know natural homes for extracted code.

I now have a thorough picture. A few key facts shape the options:

- The script (~1464 lines) hand-rolls a batched, multi-method online TTT replay (`_make_sd_grad_fn`, `_adapt_and_measure_batched_multi`, etc., ~350 lines) that reimplements `RnnSelfDistillPredictor._window_loss` + `RnnInContextPredictor.forward` from `src/`, because the model's `forward` exposes neither per-step snapshots nor per-row (per-method) vectorized knobs.
- It also duplicates genuinely reusable, model-adjacent helpers: `kalman_ir` (also copied in `ttt_impulse.py` / `ttt_impulse_video.py`), `filter_analytical_error` (imported by `ttt_impulse_video.py`), `_spectral_radius`/`_eig_unsafe`, and it imports eigen-phase metrics + `_draw_complex_plane` from `self_distillation_eigenphase.py`.
- Sibling scripts (`plot_sd_*`, `analyze_mech_sweep.py`) only read the `.pt` checkpoints as plain dicts, so the persistence schema must stay stable, but they don't import Python symbols. Only `ttt_impulse_video.py` imports a symbol (`filter_analytical_error`).

Before I write the plan, two scope questions:

> **How aggressive should the restructuring be?**
> - Moderate (Recommended): move reusable, model-adjacent code (kalman_ir, analytical-error/spectral helpers, eigen-phase metrics) into src/kf_rnn, slim the script to a thin driver (presets + FIR glue + plotting). Keep the batched multi-method loop in the script but have it reuse src pieces.
> - Aggressive: also re-architect the perf-critical batched multi-method TTT replay into a first-class model/infra capability in src (e.g. an adapt/replay-with-snapshots API on the TTT predictors), largely replacing the hand-rolled loop.
> - Minimal: only de-duplicate across scripts via a shared scripts/ helper module; no changes under src/.

> **Where should the extracted src code live?**
> - New subpackage src/kf_rnn/analysis/ (ttt replay/snapshots, filter-error metrics, eigen/IR metrics) - keeps study/analysis code grouped and separate from core model.
> - Attach to existing modules (e.g. kalman_ir + filter error near model/sequential, metrics into infrastructure/experiment/metrics.py) - fewer new files.
> - No preference - you decide the best placement.

---

**User**

What is the main blocker preventing the current TTT loop from being implemented as a separate model like rnn_ttt.py?

---

**Cursor**

The short answer: the model in `rnn_ttt.py` throws away exactly the thing the study needs. `RnnInContextPredictor.forward` runs the whole online adaptation inside an `eu.scan` under `eu.multi_vmap` and returns only the *predictions*, never the *per-step fast-weight trajectory* `theta_t`:

```165:180:src/kf_rnn/model/sequential/rnn_ttt.py
            xs = (torch.arange(L), actions_d, observations)
            _, (preds, states) = eu.scan(step, (theta_init, state_0, entering_window), xs)
            return preds, states     # [L x O_D], [L x S_D]

        # Map the per-sequence routine over every leading batch dim so the
        # propagation/update math always sees 2-D weights and a single trajectory.
        mapped = eu.multi_vmap(run_sequence, len(batch_shape), in_dims=0, randomness="different")
        observation_estimations, state_estimations = mapped(state_0, actions_d, observations)
```

Everything the script computes — the closed-form analytical error at each sampled step, the spectral radius of `F_hat`, the impulse response, and the Polyak tail-average — is a function of the adapted `(F, H, K)` *at intermediate steps*, i.e. of `theta_t`. The model gives no hook to observe or snapshot that carry, which is the entire reason `_adapt_and_measure` re-implements the loop as an explicit Python loop that snapshots `theta_bar` at `sampled_set`:

```395:402:scripts/self_distillation_losses.py
        if t in sampled_set:
            step_errs[t] = filter_analytical_error(theta_bar["F"], theta_bar["H"], theta_bar["K"], lsg_td)
            ...
            raw_step_errs[t] = (step_errs[t] if theta_bar is theta
                                else filter_analytical_error(theta["F"], theta["H"], theta["K"], lsg_td))
```

So even a *single* method can't reuse `forward` as-is. That is the main blocker: **the adaptation trajectory `theta_t` is a discarded internal of `forward` (buried in `scan`/`multi_vmap`), and the model exposes only final predictions, not the parameter trajectory or a stepwise hook.**

There are two secondary blockers, both layered on top of that one:

1. **Reporting features that live outside the model.** `polyak_burnin` exists on the `Config` but `forward`/`_optimizer_step` never actually average — the tail-average (`theta_sum`/`theta_bar`) is implemented only in the script's replay. So the "reported filter" semantics the study relies on aren't in the model yet.

2. **Per-row (multi-method) knobs vs. scalar config.** The model bakes each method's identity into *Python control flow* over scalar config: `if self.beta0 != 0.0`, `for k in range(1, min(self.sd_horizon, j+1)+1)`, `if k in self.adapt_keys`. The script's speed win comes from stacking `M` methods into one dispatch-bound loop, which forces those branches to become *data*: per-row tensors (`alpha_v`, `beta0_v`, `sd_horizon_v`, `adapt_mask`, ...) and `torch.where` gating instead of `if`/`range`:

```660:666:scripts/self_distillation_losses.py
        for _ in range(n_steps):
            grads = grad_step(theta, s_start, win_obs, alpha_v, beta0_v, beta2_v, sd_horizon_v)
            new_theta: dict[str, torch.Tensor] = {}
            for k, v in theta.items():
                cand = v - _rowwise(eta, v) * (grads[k] + _rowwise(wd_v, v) * v)
                new_theta[k] = torch.where(_rowwise(adapt_mask[k], v) > 0, cand, v)
            theta = new_theta
```

A single `nn.Module` with scalar attributes can't represent rows that use different weights/adapt-sets/horizons; you'd need either to run `M` separate model instances (semantically fine but ~M× slower — the former approach) or to give the model a data-driven loss/step where those knobs are tensors. The `torch.where` gating in `_make_sd_grad_fn` (guarding `0 * inf = nan` from inactive-but-diverged terms) is a further consequence of that batching that the per-method model never has to deal with.

In other words: making this a proper `src/` model isn't blocked by anything fundamental — it's blocked by the model needing (a) a **trajectory/stepwise API** that yields `theta_t` (so snapshots, Polyak averaging, and analytical-error readouts can hang off it), and secondarily (b) a **data-driven (tensorized) loss/optimizer step** if you want to keep the multi-method batched speedup rather than looping `M` instances.

Want me to fold this into a concrete plan (e.g. add an `adapt_trajectory`/replay-with-snapshots method to the TTT predictors, plus an optional tensorized multi-method variant)?

---

**User**

Is it possible to integrate it as an actual training function instead? I.e. each gradient step is one call to the train func. The "forward pass" of the model is running its current parameters as a static IIR. If we need polyak averaging, then we hold a separate F'/H'/K'/B', the gradient updates happen to those and then F/H/K/B are updated with online averaging

---

**Cursor**

Good question — this maps the TTT loop onto the existing stage/engine framework. Let me check how training stages actually execute (the `SGDStage`, the engine loop, and how per-step results/metrics are recorded) before answering.

Yes — and this is actually the cleaner integration. It dissolves the "main blocker" from my previous answer by *inverting the control flow*: instead of exposing the `theta_t` trajectory from inside `forward`, you make the engine's outer loop *be* the timestep loop. There's already a precedent for exactly this shape — `OnlineLeastSquaresStage` is an iterative, per-"epoch" online stage that consumes the trace incrementally through `self._state`:

```265:284:src/kf_rnn/infrastructure/experiment/stages.py
class OnlineLeastSquaresStage(TrainingStage):
    ...
    def step(self, ctx: TrainingContext) -> StepResult:
        reference_module = ctx.model_pair[0]
        return reference_module.train_least_squares_online(
            ctx.thp, ctx.exclusive, ctx.model_pair, self._state
        )

    def is_done(self, ctx: TrainingContext) -> bool:
        ...
```

## How your proposal maps on

- **`forward` = static IIR.** You'd use the plain `SequentialPredictor._forward` Kalman recurrence (i.e. a normal `RnnPredictor`), *not* `RnnInContextPredictor.forward`. That immediately deletes the reason the script needs `torch.func.functional_call` / `torch.func.grad`: with a static forward and an external loop, the window loss is just a plain autograd loss + `loss.backward()`, exactly like `SGDStage`.
- **Each gradient step = one `stage.step`.** A `SelfDistillTTTStage.step` builds the sliding window `[w0:t+1]` for the current `t` from `self._state`, computes the SD window loss (β0/β2/α ladder), takes one SGD step, increments `t`; `is_done` is `t >= L`.
- **Polyak via a shadow iterate.** Exactly as you describe: the optimizer owns the *raw* iterate `(F',H',K',B')` in `self._state`, and after each step you write the online average into the model's actual `(F,H,K,B)`. Since the engine evaluates metrics and checkpoints off `model_pair[1]` (the model params), storing the *average* there means every readout automatically sees the reported filter. (`torch.optim.swa_utils.AveragedModel` is essentially this.)
- **The per-step analytical-error curve comes for free.** The engine calls `evaluate_metrics()` before *every* step (training.py line 383), and `validation_analytical` already computes the closed-form `analytical_error`. So the script's hand-rolled `_adapt_and_measure` snapshotting is replaced by the engine's normal per-epoch metrics — over the *static* current params, which is precisely what `filter_analytical_error` does today.
- **The other knobs map to standard primitives:** `adapt_keys` → which params go in the optimizer group; `step_decay` → an LR scheduler; `weight_decay` → optimizer weight decay.

So the whole `_make_sd_grad_fn` / `_adapt_and_measure_batched*` apparatus (~350 lines, plus all the `functional_call`/`vmap`/`torch.where`-gating) largely disappears.

## The real frictions

1. **Per-epoch evaluation cost is the biggest practical blocker.** The engine unconditionally runs the *full* `evaluate_metrics()` (overfit, validation, analytical, impulse, ...) before every `stage.step`:

```382:386:src/kf_rnn/infrastructure/experiment/training.py
        while not stage.is_done(ctx):
            r = evaluate_metrics()

            reference_module.train()
            train_result, log = stage.step(ctx)
```

"One timestep per step" then pays a whole-dataset metric evaluation *per timestep*. The current script deliberately snapshots analytical error only every `--analytic-stride` steps and computes nothing else, keeping the loop dispatch-bound. To keep it cheap you'd either (a) let `step` advance several timesteps internally (reintroducing a small inner loop), or (b) make the metric cadence configurable in the engine. As-is, L=10000 would mean 10000 full metric passes.

2. **Per-trajectory vs shared parameters — the deepest mismatch.** The training pipeline optimizes *one shared* parameter set against the whole train dataset; the ensemble `[N×E]` dim is for replication/resampling, and `_sample_dataset_indices` draws subsequences across all systems/traces for *every* model member. The study instead needs *N independent* adapted filters (one per trajectory) so it can report median / quantiles / fraction-stable over trajectories. Making "one filter per trace, trained only on that trace" requires mapping trajectories onto the ensemble dim and feeding each member its own trace — which is not how the dataset layout `[N_model × E × S × B × L]` and the sampler are wired. This is the main thing you'd have to bend.

3. **Multi-method stacking.** In the stage world the natural expression is *M separate runs* (each method = a model config + the SD stage), which is clean and makes the per-row-knob tensorization problem vanish — but it gives up the fused M× batched-loop speedup. Recovering that speedup means re-tensorizing the knobs across the ensemble dim, the same problem as today.

4. **Checkpoint re-pointing convention (minor).** The engine assumes the optimizer's params *are* the model params (training.py lines 325–333). The shadow-average scheme deliberately separates them (optimizer → raw iterate, model → average), so that convention needs a small accommodation.

## Bottom line

It's feasible and architecturally *nicer* than exposing a trajectory API on `forward`: `forward` becomes a static filter, the SD loss becomes plain autograd, Polyak is a shadow iterate, and the error curve is just per-epoch metrics. The two things that decide whether it's worth it are (2) reworking the pipeline to give each trajectory its own parameters and (1) the per-step metric-evaluation cadence. Multi-method batching (3) is the feature you'd most likely trade away for cleanliness.

Want me to turn this into a concrete plan (a new `SelfDistillTTTStage` + a static-forward `RnnSelfDistill*` model config, per-trajectory ensemble wiring, and the shadow-average step), with the script reduced to presets + plotting?

---

**User**

Isn't the point of the ensemble dimension training different models on different realizations of data from the same system, and the n_experiment dimension is sampling different systems from the same distribution of systems?

---

**Cursor**

Good — let me verify the exact semantics rather than rely on my summary. Let me look at how the experiment/ensemble/system/trace dimensions are defined and populated.

Let me confirm how systems and datasets are generated across the experiment/ensemble dims.

You're right, and my earlier "deep mismatch" framing was wrong. Let me correct it with what the data-construction code actually does.

Systems are sampled with a group shape of `(n_experiments, n_systems)`:

```182:185:src/kf_rnn/infrastructure/experiment/internals.py
        def sample_system_parameters_with_sub_hyperparameters(_, sub_HP: ExperimentConfig) -> TensorDict:
            return sub_HP.system.distribution.for_split(ds_type).sample_parameters(
                schema.resolve_splits(sub_HP.system), (HP.experiment.n_experiments, max_n_systems)
            )
```

so each `n_experiments` index is an **independent system draw from the distribution** — exactly your description of N.

And for the training dataset, one system generates `ensemble_size * n_traces` traces which are then split along the ensemble axis:

```261:268:src/kf_rnn/infrastructure/experiment/internals.py
    max_batch_size = (HP.experiment.ensemble_size if ds_type == TRAINING_DATASET_TYPES[0] else 1) * max_n_traces
    ...
        if ds_type == TRAINING_DATASET_TYPES[0]:
            return dataset.unflatten(2, (HP.experiment.ensemble_size, max_n_traces)).permute(0, 2, 1, 3, 4)
```

so the ensemble axis E is **different data realizations (trace draws / noise) from the same system** — exactly your description of E. And `_extract_dataset_from_indices` indexes per `(N, E)` member (`n_experiment_idx`, `ensemble_size_idx`), while the stacked params are `[N × E × ...]` — so each `(N,E)` member is genuinely an independent model trained on its own data slice, not a shared parameter set. I was wrong to call it "replication/resampling, not per-trajectory."

## What that changes about the blockers

The consequence is that per-trajectory TTT maps *naturally* onto existing machinery, rather than fighting it:

- **Trajectories → ensemble dim.** Set `n_experiments = 1` (one system), `ensemble_size = N` (the N trajectories), `n_systems = n_traces = 1`. Each ensemble member is then an independent filter adapting online to its own single trace — precisely the study's setup. The median / quantile / fraction-stable readouts over trajectories become the engine's usual reductions over the model-shape tail (`print_losses` already does `.mean(-1).median(-1)`).
  - One caveat: the study starts all N trajectories from the *same* re-seeded init, whereas `stack_module_arr` gives each ensemble member an independent random init. You'd force identical inits (broadcast one draw), which is a small thing.
- **Methods → a config sweep (or separate runs).** Each method is just a different SD-weight model config, which the sweep engine already runs as separate cells. That makes the per-row-knob tensorization problem disappear entirely (each cell has scalar config again). The only thing you give up is the fused M×-in-one-loop speedup — but the engine parallelizes the ensemble anyway.

So blocker #2 from my last message largely dissolves. The blockers that actually remain are:

1. **Per-step metric-evaluation cadence (still the big one).** The engine calls the full `evaluate_metrics()` before *every* `stage.step` (training.py line 383). If each step is one online timestep, an L=10000 run pays ~10000 full metric passes. You'd batch several timesteps per `step`, or make the metric cadence configurable.
2. **Sequential online-state bookkeeping.** Unlike `SGDStage` (which samples random subsequences from a fresh sampled initial state), the TTT stage must carry the *filtered state* `x_t^+` and the entering-state window across `step` calls and slide the window `[w0:t+1]` — persisted in `self._state`, like `OnlineLeastSquaresStage` carries its `t`. It's a genuinely different (but legitimate) stage, not something the framework hands you for free.
3. **Polyak shadow vs the optimizer/model-param convention** (minor): the engine assumes the optimizer's tensors *are* the model params (training.py lines 325–333); the averaged-report + raw-iterate split needs a small accommodation on the checkpoint path.

Net: your training-function framing is a good fit — the ensemble/experiment axes already give you per-trajectory-and-per-system parallelism, methods become a sweep, and the SD loss becomes plain autograd on a static forward. The remaining work is a new online sequential stage plus deciding the metric cadence.

Want me to write this up as a concrete plan now (new `SelfDistillTTTStage` with persisted online state + Polyak shadow, a static-forward SD model config, ensemble=trajectories / sweep=methods wiring, and the script cut down to config + plotting)?

---

**User**

Yes, you can write the concrete plan.
1. You can integrate the metric cadence into the plan
2. That's fine, have the state store the filtered state x_t^+ and whatever else is needed

---

**Cursor**

I'll gather a few more specifics so the plan is precise: the metrics system (for cadence), the stage test conventions, `Predictor.run`, and how experiments are launched from scripts.

Let me confirm the experiment entrypoint and a representative script that drives it, so the plan's wiring is accurate.

I have everything I need. Here's the concrete plan.

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

I'll start implementing. Let me mark the first todo in progress and read the key source files I need to understand precisely.

The static forward and analytical-error machinery are clear. Let me examine how experiment results are returned/extracted (`ResultGrid`, `get_result_attr`) and the experiment `__init__` exports, so the driver rewrite is accurate.

Let me check the distribution/system API (`ContinuousDistribution`, `sample_parameters`, `system_type`) and how `systems`/`initialization` are passed to the engine.

Let me understand the sweep dimension naming and initialization alignment so a single shared-data sweep works.

The engine plumbing is clear enough. Now let me implement. Starting with the shared analysis helpers module in src.

Now the static-forward SD model in `rnn_ttt.py`.

Now let me read the stages and training engine in full to add the stage and cadence.

Now let me read the training.py engine (SGDStage, `_run_training`, `_build_model_ensemble`, registration, evaluate_metrics).

Let me read the script's core TTT loop to port the exact online-state semantics.

Let me read the metrics module to mirror the analytical-error metric exactly.

Now let me add the guarded metrics to `metrics.py`.

Now the `RuntimeConfig.metric_frequency` in schema and the cadence gating + stage in training.py.

Now the cadence gating in `_run_training`:

Now the `SelfDistillTTTStage` in `training.py`:

Model, stage, cadence, and metric are in place. Let me update todos and inspect the helper scripts to dedupe imports.

Now the eigenphase script. Let me view its imports and helper definitions.

Now remove the local `_draw_complex_plane` definition in the eigenphase script since it's imported.

Let me smoke-test that the new src modules import and the model/stage/metric wire up correctly before the driver rewrite.

There's a project-local `.conda` env. Let me use it.

Core src integration verified. Now the driver rewrite. Let me read the current script's structure (imports, presets, `MethodResult`, `run_methods`, save/load, `main`).

Let me check the `LTISystem` constructor to see if I can cheaply build a group-shaped system for the engine metric.

Let me check for any other references to the functions I'm about to move/delete (tests, other scripts).

Now I'll rewrite the driver. First, the imports:

Now I'll delete the whole block from `# SECTION: Metrics` through `_adapt_and_measure_batched` (lines 253–488), replacing with a short note:

Now let me view the current `finalize_method_result` and `run_methods` region to rewrite `run_methods`.

Now I'll rewrite `run_methods` to build a per-method `Config`, drive the `SelfDistillTTTStage` in src, and record the reported/raw error curves:

Now let me syntax-check and run a tiny end-to-end of the driver.

The `_build_model_ensemble` initialization leaves the stacked params as broadcast (expanded) views, so in-place `copy_` fails. I'll re-clone them to contiguous leaves in `_init_state` (as `SGDStage` does):

The stacked params land on the settings device (cuda) regardless of default; I'll move the ensemble to the observations' device in the driver (like `_fir_pair` does).

Driver works end-to-end. Now the tests. Let me read the existing `test_stage_recipes.py` and the `RnnSelfDistillPredictor` internals to build a parity test.

Now let me look at `RnnSelfDistillPredictor`/`RnnInContextPredictor` internals to write a parity reference test.

Now let me write a numerical parity test comparing the new stage against the original `RnnSelfDistillPredictor` fast-weight loop.

Parity is exact. Now let me run the stage-recipes test and verify a `_run_training`-driven path exercises the cadence + metric.

Let me add an integration test for the cadence + metric via `_run_training`.

All tests pass. Let me verify the driver `--reload` path and run the broader existing test suite for regressions.

Let me run the FIR path once (the part I didn't exercise) and the existing lightweight tests for regressions.

The FIR path and all plotting work. Let me run the two most relevant existing tests to confirm no regressions, then clean up smoke artifacts.

All tests pass. Let me mark the remaining todos complete.

All to-dos are complete. The TTT self-distillation loop is now integrated as a first-class training stage.

## Summary

The refactor inverts control so the engine's stage loop *is* the online timestep loop:

**Model** (`src/kf_rnn/model/sequential/rnn_ttt.py`) — Added `RnnSelfDistillTTTPredictor(RnnPredictor)`: inherited static IIR `forward`, a pure-function batched `window_loss`, a static `kalman_posterior` step, SD/init knobs in its `Config`, per-method init tweaks in `__init__`, and `training_recipe -> ["self_distill_ttt"]`.

**Stage** (`infrastructure/experiment/training.py`) — Added `SelfDistillTTTStage` with persisted online state (filtered state, entering-state window, raw shadow iterate, Polyak accumulators), manual decoupled SGD on the raw iterate, and Polyak averaging folded into the reported `nn.Parameter`s. Registered with a `requires` guard.

**Cadence** — `RuntimeConfig.metric_frequency` gates `evaluate_metrics()` in `_run_training`, with the online step index stored per results row.

**Metrics** — Guarded `sd_analytical` + `sd_frac_stable` wrap analytical error with the divergence/spectral-radius guard, returning per-trajectory values.

**Helpers** — `kalman_ir`, spectral/analytical-error, and eigen-phase/complex-plane helpers consolidated into `src/kf_rnn/analysis.py`; `ttt_impulse.py`, `ttt_impulse_video.py`, and `self_distillation_eigenphase.py` now import from there.

**Driver** — `self_distillation_losses.py` reduced to presets → `Config`, drives the stage, computes final scalars from the reported `model_pair`, keeps FIR + plotting, and writes the legacy `<tag>_save.pt` payload so sibling scripts + `--reload` keep working.

**Tests** — `test_stage_recipes` extended; new `test_self_distill_ttt_parity.py` confirms exact numerical parity vs the original `RnnSelfDistillPredictor` loop (`max|Δ| ≤ 6.66e-16` across M4/M3+post/Polyak/M1) plus a cadence+metric integration check. Verified `--reload` and the FIR path end-to-end.

Let me check how the current rewritten driver is structured compared to the `run_experiments` example.

