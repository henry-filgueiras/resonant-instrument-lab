#!/usr/bin/env python3
"""Unit tests for pure helpers in `sim.derived` that don't need a sim run.

Focused on state-derivation primitives whose contract can be exercised
from hand-constructed inputs. Regime-level numerical validation lives
in `test_regimes.py`; detector thresholds live in `test_detectors.py`.

Run directly:
    python tests/test_derived.py
Or via pytest:
    python -m pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from sim.derived import pulse_times_of  # noqa: E402


# ---------------------------------------------------------------------------
# pulse_times_of — per-node ordered pulse frame indices.

def test_pulse_times_of_normal_multi_node():
    pf = np.zeros((10, 3), dtype=bool)
    pf[[1, 4, 7], 0] = True
    pf[[0, 9], 1] = True
    # node 2: deliberately no pulses

    out = pulse_times_of(pf)

    assert isinstance(out, tuple)
    assert len(out) == 3
    np.testing.assert_array_equal(out[0], np.array([1, 4, 7]))
    np.testing.assert_array_equal(out[1], np.array([0, 9]))
    assert out[2].shape == (0,)
    assert out[2].dtype == np.int64


def test_pulse_times_of_empty_and_no_pulse_cases():
    # no-pulse nodes yield empty int64 arrays, not None, not omitted
    pf = np.zeros((5, 4), dtype=bool)
    pf[2, 1] = True
    out = pulse_times_of(pf)
    assert len(out) == 4
    for i, arr in enumerate(out):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.int64
        assert arr.ndim == 1
        if i == 1:
            np.testing.assert_array_equal(arr, [2])
        else:
            assert arr.shape == (0,)

    # all-False input: every node empty, tuple still length N
    all_false = pulse_times_of(np.zeros((8, 3), dtype=bool))
    assert len(all_false) == 3
    for arr in all_false:
        assert arr.shape == (0,) and arr.dtype == np.int64

    # T == 0: every node present, all empty
    zero_T = pulse_times_of(np.zeros((0, 2), dtype=bool))
    assert len(zero_T) == 2
    for arr in zero_T:
        assert arr.shape == (0,) and arr.dtype == np.int64

    # N == 0: empty tuple
    assert pulse_times_of(np.zeros((5, 0), dtype=bool)) == ()


def test_pulse_times_of_ordering_and_monotonicity():
    # sparse random pulses — every returned sequence must be strictly ascending
    rng = np.random.default_rng(0)
    pf = rng.random((200, 5)) > 0.8
    out = pulse_times_of(pf)
    assert len(out) == 5
    for i, arr in enumerate(out):
        if arr.size >= 2:
            assert np.all(np.diff(arr) > 0), f"node {i} not strictly ascending"
        # recomputable from the column mask — no reordering, no filtering
        np.testing.assert_array_equal(arr, np.flatnonzero(pf[:, i]))

    # all-True: every frame fires on every node
    out_all = pulse_times_of(np.ones((6, 2), dtype=bool))
    for arr in out_all:
        np.testing.assert_array_equal(arr, np.arange(6))
        assert arr.dtype == np.int64


def test_pulse_times_of_shape_and_type_contract():
    # int-castable input (not just strict bool) is accepted
    pf = np.zeros((4, 2), dtype=np.uint8)
    pf[1, 0] = 1
    pf[3, 1] = 1
    out = pulse_times_of(pf)
    assert len(out) == 2
    for arr in out:
        assert arr.ndim == 1
        assert arr.dtype == np.int64
    np.testing.assert_array_equal(out[0], [1])
    np.testing.assert_array_equal(out[1], [3])

    # 1D input rejected
    try:
        pulse_times_of(np.zeros(5, dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on 1D pulse_fired")

    # 3D input rejected
    try:
        pulse_times_of(np.zeros((3, 4, 2), dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on 3D pulse_fired")


if __name__ == "__main__":
    test_pulse_times_of_normal_multi_node()
    test_pulse_times_of_empty_and_no_pulse_cases()
    test_pulse_times_of_ordering_and_monotonicity()
    test_pulse_times_of_shape_and_type_contract()
    print("ok: pulse_times_of")
