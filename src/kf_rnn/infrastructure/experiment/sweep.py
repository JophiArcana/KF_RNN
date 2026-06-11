import re

"""
Named predicates for classifying hyperparameter-sweep parameter names.

Two distinct name spaces show up in the experiment plumbing:

1. Full sweep-parameter names (the keys swept over in `dimensions` / `iterparams`),
   which carry the top-level prefix, e.g. "system.distribution.train",
   "dataset.n_traces.test", "experiment.metrics". Use the `is_*_param*` predicates.

2. Prefix-stripped support-hyperparameter names (the keys of
   `utils.nested_vars(HP.system)` / `utils.nested_vars(HP.dataset)`), e.g.
   "distribution.train", "S_D". Use the `support_param_*` predicates.

Centralizing the regexes here keeps the subtle patterns (anchoring, the
`auxiliary` negative lookahead, the dataset-type suffix) in one tested place.
"""


# SECTION: Predicates over full sweep-parameter names (prefixed)

def is_experiment_param(name: str) -> bool:
    return bool(re.match(r"experiment\.", name))

def is_dataset_param(name: str) -> bool:
    return bool(re.match(r"dataset\.", name))

def is_dataset_param_of_type(name: str, ds_type: str) -> bool:
    """A dataset.* sweep parameter scoped to the given dataset type, e.g. "dataset.n_traces.train"."""
    return bool(re.match(rf"dataset(\..*\.|\.){re.escape(ds_type)}$", name))

def is_system_param_of_type(name: str, ds_type: str) -> bool:
    """A system.* sweep parameter scoped to the given dataset type."""
    return bool(re.match(rf"system(\..*\.|\.){re.escape(ds_type)}$", name))

def is_system_nonauxiliary_param_of_type(name: str, ds_type: str) -> bool:
    """A system.* (but not system.auxiliary.*) sweep parameter scoped to the given dataset type."""
    return bool(re.match(rf"system\.((?!auxiliary\.).)*\.{re.escape(ds_type)}$", name))


# SECTION: Predicates over prefix-stripped support-hyperparameter names

def support_param_targets_type(name: str, ds_type: str) -> bool:
    """A support hyperparameter scoped to the given dataset type, e.g. "distribution.train"."""
    return bool(re.match(rf"(?!\.).*\.{re.escape(ds_type)}", name))

def support_param_is_nonauxiliary_of_type(name: str, ds_type: str) -> bool:
    """A non-auxiliary support hyperparameter scoped to the given dataset type."""
    return bool(re.match(rf"((?!auxiliary\.).)*\.{re.escape(ds_type)}$", name))
