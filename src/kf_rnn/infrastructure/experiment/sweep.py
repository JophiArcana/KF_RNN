"""Structural predicates for classifying hyperparameter-sweep parameter names.

Two distinct name spaces show up in the experiment plumbing:

1. Full sweep-parameter names (the keys swept over in `dimensions` / `iterparams`),
   which carry the top-level branch prefix, e.g. "system.distribution.train",
   "dataset.n_traces.test", "model.ir_length". Use the `is_*_param*` predicates.

2. Branch-relative support-hyperparameter names (the keys of
   `schema.config_leaves(HP.system)` / `config_leaves(HP.dataset)`), e.g.
   "distribution.train", "S_D". Use the `support_param_*` predicates.

Classification is positional: per-split values live exclusively in
``schema.Split`` fields, whose leaves are named ``<field>.<split>`` with
``<split>`` one of ``schema.SPLIT_NAMES``. A parameter "targets" a dataset type
exactly when its final path segment is that split name — ``config_leaves`` only
emits a split leaf when it was explicitly authored or swept in.
"""
from kf_rnn.infrastructure.config.schema import SPLIT_NAMES


def _segments(name: str) -> list[str]:
    return name.split(".")


# SECTION: Predicates over full sweep-parameter names (prefixed)

def is_experiment_param(name: str) -> bool:
    return _segments(name)[0] == "experiment"

def is_dataset_param(name: str) -> bool:
    return _segments(name)[0] == "dataset"

def is_dataset_param_of_type(name: str, ds_type: str) -> bool:
    """A dataset.* sweep parameter scoped to the given dataset type, e.g. "dataset.n_traces.train"."""
    segs = _segments(name)
    return len(segs) >= 2 and segs[0] == "dataset" and segs[-1] == ds_type

def is_system_param_of_type(name: str, ds_type: str) -> bool:
    """A system.* sweep parameter scoped to the given dataset type."""
    segs = _segments(name)
    return len(segs) >= 2 and segs[0] == "system" and segs[-1] == ds_type

def is_system_nonauxiliary_param_of_type(name: str, ds_type: str) -> bool:
    """A system.* (but not system.auxiliary.*) sweep parameter scoped to the given dataset type."""
    segs = _segments(name)
    return is_system_param_of_type(name, ds_type) and "auxiliary" not in segs[1:-1]


# SECTION: Predicates over branch-relative support-hyperparameter names

def support_param_targets_type(name: str, ds_type: str) -> bool:
    """A support hyperparameter scoped to the given dataset type, e.g. "distribution.train"."""
    assert ds_type in SPLIT_NAMES, f"Unknown dataset type {ds_type!r}; expected one of {SPLIT_NAMES}."
    segs = _segments(name)
    return len(segs) >= 2 and segs[-1] == ds_type

def support_param_is_nonauxiliary_of_type(name: str, ds_type: str) -> bool:
    """A non-auxiliary support hyperparameter scoped to the given dataset type."""
    segs = _segments(name)
    return support_param_targets_type(name, ds_type) and "auxiliary" not in segs[:-1]
