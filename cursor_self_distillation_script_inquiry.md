# Self-distillation script inquiry
_Exported on 7/8/2026 at 15:38:32 PDT from Cursor (3.8.11)_

---

**User**

@cursor_self_distillation_script_refacto.md I recently just finished implementation of the plan@/pa/home/wentinn.liao/.cursor/plans/ttt_self-distillation_as_training_stage_e49d9572.plan.md. But does the new @scripts/self_distillation_losses.py not use run_experiments as a driver? An example is @scripts/impulse_response_length.py

---

**Cursor**

I'll read the relevant files to answer this.

Let me look at the plan and how `run_experiments` and `SelfDistillTTTStage` fit together, to answer whether the script should be using `run_experiments`.

You're right — the rewritten script does not go through `run_experiments`. It bypasses the experiment engine and drives the stage machinery by hand, which is a deviation from what the plan's "driver" todo specified ("presets -> Config sweep, **run_experiments driver**").

## What it does instead

In `impulse_response_length.py`, configs go into a `configurations` sweep and `run_experiments(cfg, configurations, ...)` owns everything: dataset generation, the training loop via `_run_training`, metrics, and result persistence.

In the new `self_distillation_losses.py`, `run_methods` reproduces that plumbing manually for each method:

```443:458:scripts/self_distillation_losses.py
    for m in methods:
        cfg = _method_config(m, args, problem_shape)
        # Re-seed so every method draws the SAME base init (kaiming H etc.); the
        # per-method M1 (H=0) / frozen-K tweaks are applied in the model __init__.
        # Broadcasting the single init across the N trajectories reproduces the old
        # identical-init semantics.
        torch.manual_seed(args.seed)
        init_td = eu.parameter_td(cfg.cls(cfg)).detach()
        ref, stacked = _build_model_ensemble(cfg, (1, N), init_td)
```

and `_drive_stage` hand-builds a `TrainingContext` with a `SimpleNamespace` standing in for the engine's `exclusive`, then loops `stage.step(ctx)` itself, even incrementing `stage.t` manually and snapshotting `batched_filter_analytical_error` at the sampled steps.

## Consequences of the deviation

- **The engine features built for this plan go unused by the driver.** Plan sections 3 and 4 added `RuntimeConfig.metric_frequency` (gating `evaluate_metrics()` in `_run_training`) and the guarded `sd_analytical` / `sd_frac_stable` metrics precisely so the error-vs-step curve would "fall out of the engine." Those are implemented and registered in src, but since the script never enters `_run_training`, they're only exercised by the tests — the script re-implements the same sampling loop in `_drive_stage`.
- **It imports engine internals.** `_build_model_ensemble` is a private helper of `training.py`, and faking `exclusive` with `SimpleNamespace` couples the script to the engine's internal attribute layout rather than its public entry point.
- **No experiment persistence/caching from the engine** — the script keeps its own `<tag>_save.pt` adapter (which the plan did call for as legacy compatibility, but it was meant to be assembled *from* engine results, not to replace them).

## Why it may have ended up this way (partly legitimate)

Two things the manual drive gets that the `run_experiments` path doesn't hand you directly:

1. **The raw-iterate diagnostic curve.** `_drive_stage` reads `stage.state.theta_raw` at each sampled step to compute the no-Polyak error curve. Engine metrics only see the *reported* model parameters, so the raw shadow iterate's error would need either a stage-exposed metric or to be dropped.
2. **Shared observations across methods.** All methods must adapt on the *same* trajectories with the same re-seeded init. Through `run_experiments`, each sweep cell would regenerate data through the engine's dataset pipeline; you'd have to pass the pre-generated system/dataset in (which `run_experiments` does support — `impulse_response_length.py` passes `system2` — but wiring one fixed observation set as "N ensemble members, one trace" through that path takes more care).

So the current state is: the model/stage/metric/cadence integration into src happened as planned, but the final inversion — letting `run_experiments` be the driver with methods as a sweep — was not completed, and the script sits at a halfway point where it uses the stage but re-implements the harness around it. If you want, I can finish that last step: turn the method presets into a `("method", {"model": [...]})` sweep, run it through `run_experiments` with `metric_frequency`/`sd_analytical`, and keep the raw-iterate curve via a small stage-level hook (or make it optional).

---

**User**

Yes, you can run it through run_experiments

---

**Cursor**

I'll dig into the experiment engine to understand `run_experiments` precisely before rewiring the driver. Let me read the core files.

Now let me read the training engine and the internals for dataset construction.

Let me read the metrics module and the internals for how systems/datasets are constructed and passed in.

This is a substantial rewrite. Let me gather the remaining context: config schema, static constants, how `initialization`/`systems` LabeledArrays are built elsewhere, and the analysis helpers.

Let me find existing examples of passing `systems`/`initialization` LabeledArrays and how `load_system_and_args` builds them, plus the analysis helpers.

Let me look at how existing scripts build the `systems`/`initialization` arrays and call the engine.

Let me read the `ResultGrid` API and the base `RnnPredictor` to finalize the extraction logic.

Let me check the parity/integration test to see the exact expected result shapes from the engine path.

Let me read the system class to see how to wrap the sampled system for the engine and what `td()`/losses look like.

Let me check the experiment package exports and how `eu.parameter_td`/`eu.array_of` are used, to finalize imports.

Let me check the sweep classification helpers to confirm the valid split falls back cleanly to train.

I now have a complete picture. Let me set up a todo list and start editing.

Now the imports. Let me update the analysis import block and the engine imports.

Now let me rewrite the driver. Let me view the current `_method_config`/`_drive_stage`/`run_methods` region.

Now let me update `main`. Let me re-read the current `main`.

Now the run branch (method selection + call) and the FIR/sampled_steps line:

Let me check for lint errors.

All remaining references are in comments. Now let me smoke-test. Let me find the environment.

Device mismatch. The engine's `stack_module_arr` lands params on the settings `DEVICE` regardless of the default device. Let me check settings.

The `stack_module_arr` uses `ecliseutils.settings.DEVICE`, which `torch.set_default_device` doesn't change. Let me check `eu.configure`.

`eu.configure` sets both the ecliseutils device and torch defaults. Let me use it in `main` instead of `torch.set_default_device`.

I need to import `DTYPE` and `PRECISION`. Let me update the settings import.

The engine-driven run works end-to-end. Now let me verify the `--reload` path.

Let me test the default `ladder` methods (exercises Polyak-averaging probe) and then clean up.

That was a sandbox backend hiccup on the chained command. Let me run them separately with full permissions.

All method sets work through the engine. Let me clean up the smoke-test artifacts.

Done. `scripts/self_distillation_losses.py` now drives the study through the experiment engine instead of hand-rolling the stage loop.

## What changed

The driver no longer touches engine internals (`SelfDistillTTTStage`, `TrainingContext`, `_build_model_ensemble`, or a manual `stage.step` loop). Instead, `run_methods` builds a real sweep and hands off to the engine:

- **Methods → a sweep.** Each preset becomes a `RnnSelfDistillTTTPredictor.Config`, and the list is passed as a single `("method", {"model": [...]})` sweep to `run_training_experiments`, exactly the shape `impulse_response_length.py` uses for `run_experiments`.
- **One shared system.** A single draw is sampled at the engine's group shape `(1, 1)` (`lsg_group`, passed via `systems=`) with the same draw's unbatched view `[0, 0]` (`lsg`) backing the closed-form truth/floor/FIR baselines. The engine generates the shared observation trace once from it, so every method cell sees identical data.
- **Trajectories → the ensemble.** `ensemble_size = N`, `n_experiments = n_systems = n_traces = 1`, `total_sequence_length.train = L`, and a per-method `initialization` `LabeledArray` broadcasts each method's re-seeded (post-tweak) init across all `N` members.
- **The error curve falls out of the engine.** `metric_frequency = analytic_stride` and `metrics.training = {"sd_analytical", "sd_frac_stable"}` produce the per-step curve; `run_methods` reads `output["sd_analytical"]`/`output["step"]` and the reported `learned_kfs` params, and reuses the existing `finalize_method_result` for the spectral/IR/phase readouts.

`--reload`, the legacy `<tag>_save.pt` persistence, the FIR benchmarks, and all plotting are unchanged.

## Two deviations worth flagging

1. **`run_training_experiments`, not `run_experiments`.** The study is purely a training-stage analysis (no test phase), and it needs the returned `info_dict` to recover the shared observation trace for the FIR baselines. `run_experiments` would additionally run the whole testing pipeline (which the study discards) and doesn't surface the train `info_dict` as cleanly. `run_training_experiments` is the training half of `run_experiments`, so this is the honest fit.

2. **The raw-iterate diagnostic curve is dropped.** Engine metrics only see the *reported* (Polyak-averaged) model params, not the stage's raw shadow iterate, so I folded the raw curve onto the reported one (identical when Polyak averaging is off). The plots already degrade gracefully when the raw curve matches. Recovering it would need a stage-level metric hook exposing `theta_raw`.

I also had to route device selection through `eu.configure` rather than `torch.set_default_device`, because the engine's `stack_module_arr` builds modules on `ecliseutils.settings.DEVICE` — a bare default-device change left models on cuda while the metric ran on cpu.

Verified end-to-end on CPU: `m3m4`, `ladder` (incl. the M3+post Polyak probe), and `minimal` (decay+Polyak) all run, produce all six plots, write the save file, and `--reload` re-renders from it.

