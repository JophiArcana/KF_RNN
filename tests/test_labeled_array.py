"""Unit tests for the in-house ``LabeledArray`` / ``LabeledDataset`` layer.

Pure numpy (no torch), so it can run standalone:
    PYTHONPATH=. python tests/test_labeled_array.py
"""
from collections import OrderedDict

import numpy as np

from infrastructure.labeled_array import LabeledArray, LabeledDataset, array_of


def test_construction_and_attrs() -> None:
    la = LabeledArray(np.arange(6).reshape(2, 3), ("a", "b"))
    assert la.shape == (2, 3)
    assert la.ndim == 2
    assert la.dims == ("a", "b")
    assert list(la.ravel()) == list(range(6))

    # ndim / dims mismatch is rejected.
    try:
        LabeledArray(np.zeros((2, 3)), ("a",))
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError on dim count mismatch")

    # A bare (non-ndarray) value is wrapped into a 0-d object array.
    scalar = LabeledArray("hello", ())
    assert scalar.ndim == 0 and scalar.dims == ()
    assert scalar.values[()] == "hello"
    print("OK  construction_and_attrs")


def test_take_drops_and_ignores() -> None:
    la = LabeledArray(np.arange(24).reshape(2, 3, 4), ("a", "b", "c"))

    # Indexing a subset of dims drops exactly those dims.
    taken = la.take({"b": 1})
    assert taken.dims == ("a", "c")
    assert taken.shape == (2, 4)
    assert np.array_equal(taken.values, la.values[:, 1, :])

    # Irrelevant dim names are ignored.
    taken2 = la.take({"b": 1, "z": 99})
    assert taken2.dims == ("a", "c")
    assert np.array_equal(taken2.values, la.values[:, 1, :])

    # Indexing every dim yields a 0-d labeled scalar.
    full = la.take({"a": 1, "b": 2, "c": 3})
    assert full.dims == ()
    assert full.values[()] == la.values[1, 2, 3]
    print("OK  take_drops_and_ignores")


def test_put_stores_object_reference() -> None:
    la = LabeledArray(np.empty((2, 2), dtype=object), ("x", "y"))

    payload = {"k": 5}
    la.put({"x": 0, "y": 1}, payload)
    assert la.values[0, 1] is payload

    # A list must be stored as a single reference, not broadcast across the cell.
    lst = [1, 2, 3]
    la.put({"x": 1, "y": 0}, lst)
    assert la.values[1, 0] is lst

    # put must use every dim of the array.
    try:
        la.put({"x": 0}, payload)
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError when an index is missing")
    print("OK  put_stores_object_reference")


def test_broadcast_transpose_and_expand() -> None:
    # Source dims (b, a); broadcast to union (a, b, c).
    base = np.arange(6).reshape(2, 3)  # indexed [b, a]
    la = LabeledArray(base, ("b", "a"))

    out = la.broadcast(OrderedDict([("a", 3), ("b", 2), ("c", 4)]))
    assert out.dims == ("a", "b", "c")
    assert out.shape == (3, 2, 4)

    # out[a, b, c] should equal base[b, a] for every c (expanded axis).
    for a in range(3):
        for b in range(2):
            for c in range(4):
                assert out.values[a, b, c] == base[b, a]

    # Result is a real copy (independent storage).
    out.values[0, 0, 0] = -999
    assert base[0, 0] == 0
    print("OK  broadcast_transpose_and_expand")


def test_ufunc_dim_preservation() -> None:
    a = LabeledArray(np.ones((2, 3)), ("a", "b"))
    b = LabeledArray(2 * np.ones((2, 3)), ("a", "b"))

    s = a + b
    assert isinstance(s, LabeledArray)
    assert s.dims == ("a", "b")
    assert np.allclose(s.values, 3.0)

    neg = -a
    assert isinstance(neg, LabeledArray) and np.allclose(neg.values, -1.0)

    # A reduction that changes ndim drops the labels (returns a raw scalar/ndarray).
    total = np.add.reduce(a, axis=None)
    assert not isinstance(total, LabeledArray)
    print("OK  ufunc_dim_preservation")


def test_labeled_dataset_take() -> None:
    ds = LabeledDataset({
        "full": LabeledArray(np.arange(24).reshape(2, 3, 4), ("a", "b", "c")),
        "partial": LabeledArray(np.arange(6).reshape(2, 3), ("a", "b")),
    })

    taken = ds.take({"c": 2, "b": 0})
    assert taken["full"].dims == ("a",)
    assert np.array_equal(taken["full"].values, np.arange(24).reshape(2, 3, 4)[:, 0, 2])
    # "partial" has no "c" dim; only "b" is applied.
    assert taken["partial"].dims == ("a",)
    assert np.array_equal(taken["partial"].values, np.arange(6).reshape(2, 3)[:, 0])

    # Membership / mutation API.
    assert "full" in ds and "missing" not in ds
    ds["new"] = array_of(LabeledArray(np.zeros(2), ("a",)))  # exercise __setitem__
    assert "new" in ds
    print("OK  labeled_dataset_take")


if __name__ == "__main__":
    test_construction_and_attrs()
    test_take_drops_and_ignores()
    test_put_stores_object_reference()
    test_broadcast_transpose_and_expand()
    test_ufunc_dim_preservation()
    test_labeled_dataset_take()
    print("ALL_OK")
