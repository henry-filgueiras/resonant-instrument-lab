"""Pure helper functions over exported simulator state (DESIGN_V0.md §9.8).

The physical-summary primitives that the simulator, the regime smoke
tests, and (eventually) the detector layer all share. Every function in
this module is a pure function of state-like arrays:

- no thresholds,
- no label output,
- no YAML loading,
- no mutation.

Detector semantics (confidence, windows, compound declarations) live
elsewhere — this module only answers *what is the state*, not *what
does the state mean*.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


def kuramoto_order(theta):
    """Global Kuramoto order parameter `r = |mean(exp(i·θ))|`.

    Parameters
    ----------
    theta : np.ndarray
        Phases in radians, consistent with `np.exp(1j * theta)`. Accepted
        shapes: `(T, N)` (time series of N phases) or `(N,)` (one frame).

    Returns
    -------
    float or np.ndarray
        For 2D input, shape `(T,)` with dtype matching the phase dtype's
        real counterpart (float32 in → float32 out). For 1D input, a
        Python `float`.
    """
    theta = np.asarray(theta)
    if theta.ndim not in (1, 2):
        raise ValueError(f"theta must be 1D or 2D, got {theta.ndim}D (shape {theta.shape})")
    r = np.abs(np.mean(np.exp(1j * theta), axis=-1))
    return float(r) if theta.ndim == 1 else r


def local_kuramoto_order(theta, members):
    """`kuramoto_order` restricted to a subset of nodes.

    Thin wrapper over `kuramoto_order(theta[..., members])`. Kept as a
    named helper so detector code reads as "local coherence of cluster X"
    rather than "fancy-index then Kuramoto".

    Parameters
    ----------
    theta : np.ndarray
        Shape `(T, N)` or `(N,)`.
    members : array-like of int
        1D collection of node indices. Order is irrelevant; the result
        does not depend on it. Must be non-empty.

    Returns
    -------
    float or np.ndarray
        Same shape semantics as `kuramoto_order`.
    """
    members = np.asarray(members, dtype=int)
    if members.ndim != 1:
        raise ValueError(f"members must be 1D, got shape {members.shape}")
    if members.size == 0:
        raise ValueError("members must contain at least one index")
    return kuramoto_order(np.asarray(theta)[..., members])


def mean_phase_velocity(phase_vel, start_idx=None, end_idx=None):
    """Per-node mean phase velocity over a half-open time window.

    Parameters
    ----------
    phase_vel : np.ndarray
        Shape `(T, N)` — typically `state.npz['phase_vel']`.
    start_idx, end_idx : int or None
        Standard Python slice bounds on the time axis. `None` means the
        respective end of the array. Negative indices count from the end
        (e.g. `start_idx=-200` takes the last 200 frames).

    Returns
    -------
    np.ndarray
        Shape `(N,)`. The per-node time-mean over
        `phase_vel[start_idx:end_idx]`.

    Raises
    ------
    ValueError
        If the resolved window is empty.
    """
    phase_vel = np.asarray(phase_vel)
    if phase_vel.ndim != 2:
        raise ValueError(f"phase_vel must be 2D (T, N), got shape {phase_vel.shape}")
    window = phase_vel[start_idx:end_idx]
    if window.shape[0] == 0:
        raise ValueError(
            f"empty time window: start_idx={start_idx}, end_idx={end_idx}, T={phase_vel.shape[0]}"
        )
    return window.mean(axis=0)


class VelocitySplit(NamedTuple):
    """Result of `split_by_largest_velocity_gap`.

    Index arrays are sorted by node index (not by velocity) so they read
    naturally as membership sets.
    """

    low_indices: np.ndarray     # (k,) — nodes below the gap
    high_indices: np.ndarray    # (N-k,) — nodes above the gap
    gap: float                  # magnitude of the largest adjacent gap in sorted velocities
    bimodal_ratio: float        # gap / next-largest-gap; inf if only one gap exists; 0.0 if all gaps zero


def split_by_largest_velocity_gap(mean_vel):
    """Partition nodes into two groups at the largest adjacent gap in sorted mean velocities.

    Bare-minimum bimodality test, **not** a general clustering algorithm.
    Use only when the physics is expected to produce one dominant gap
    (e.g. the TWO_CLUSTER regime); do not use as a substitute for the
    eventual multi-regime `cluster_assignments` helper.

    Parameters
    ----------
    mean_vel : np.ndarray
        Shape `(N,)`, `N ≥ 2`. Typically the output of `mean_phase_velocity`.

    Returns
    -------
    VelocitySplit
        `low_indices` and `high_indices` are sorted ascending by node
        index. `gap` is the absolute size of the selected split. `bimodal_ratio`
        is `gap / next_largest_gap` (or `+inf` when only one gap exists).

    Raises
    ------
    ValueError
        If `mean_vel` is not 1D or has fewer than 2 elements.

    Determinism note
    ----------------
    `np.argsort` is called with `kind="stable"` and `np.argmax` returns
    the first (smallest-index) maximum on ties, so the output is a pure
    function of the input values and ordering.
    """
    mean_vel = np.asarray(mean_vel)
    if mean_vel.ndim != 1:
        raise ValueError(f"mean_vel must be 1D, got shape {mean_vel.shape}")
    if mean_vel.shape[0] < 2:
        raise ValueError(f"need at least 2 nodes to split, got {mean_vel.shape[0]}")

    order = np.argsort(mean_vel, kind="stable")
    v_sorted = mean_vel[order]
    gaps = np.diff(v_sorted)
    split_at = int(np.argmax(gaps))
    max_gap = float(gaps[split_at])

    other = np.delete(gaps, split_at)
    if other.size == 0:
        ratio = float("inf") if max_gap > 0.0 else 0.0
    else:
        max_other = float(other.max())
        if max_other > 0.0:
            ratio = max_gap / max_other
        elif max_gap > 0.0:
            ratio = float("inf")
        else:
            ratio = 0.0

    low = np.sort(order[: split_at + 1])
    high = np.sort(order[split_at + 1 :])
    return VelocitySplit(low_indices=low, high_indices=high, gap=max_gap, bimodal_ratio=ratio)


def sustained_windows(condition, min_duration_frames, hysteresis_frames=0):
    """Find windows where a boolean time series stays True for a sustained span.

    The core primitive for sustained-condition detectors: given a 1D
    boolean series over time (e.g. `r(t) < 0.3`, or "cluster A is
    locked"), return the frame-index windows in which the condition
    holds long enough to be considered sustained, with brief
    interruptions absorbed by hysteresis.

    Parameters
    ----------
    condition : array-like, shape `(T,)`
        Time-indexed boolean (or bool-castable) series.
    min_duration_frames : int, >= 1
        Minimum span of a returned window, measured as `end - start` on
        the returned half-open interval (i.e. the full span including
        any absorbed hysteresis gaps, not the count of True frames).
    hysteresis_frames : int, >= 0, default 0
        A stretch of False between two True runs is *absorbed* into the
        window if its length is `<= hysteresis_frames`. `0` disables
        absorption — only strictly contiguous True runs are considered.

    Returns
    -------
    list[tuple[int, int]]
        Sorted, non-overlapping half-open intervals `[start, end)` on
        the frame axis, each satisfying:
          - `0 <= start < end <= T`;
          - `condition[start]` and `condition[end - 1]` are both True
            (trailing False hysteresis is never included);
          - any False stretch strictly between two True sub-runs inside
            the window has length `<= hysteresis_frames`;
          - `end - start >= min_duration_frames`.
        `condition[start:end]` slices exactly the window's frames.

    Edge cases
    ----------
    - `T == 0` → `[]`.
    - all-False → `[]`.
    - all-True → `[(0, T)]` if `T >= min_duration_frames`, else `[]`.
    - True runs (after hysteresis merging) shorter than
      `min_duration_frames` are dropped silently.
    - `hysteresis_frames = 0` means adjacent True frames separated by
      even a single False frame stay in separate runs.

    Purity
    ------
    No config, no wall-clock time, no thresholds — callers convert
    seconds to frames themselves via `control_rate_hz`. The primitive
    is unit-agnostic: the same function works over control-rate bools,
    pulse bools, or any other 1D boolean time series.
    """
    arr = np.asarray(condition)
    if arr.ndim != 1:
        raise ValueError(f"condition must be 1D, got shape {arr.shape}")
    if min_duration_frames < 1:
        raise ValueError(f"min_duration_frames must be >= 1, got {min_duration_frames}")
    if hysteresis_frames < 0:
        raise ValueError(f"hysteresis_frames must be >= 0, got {hysteresis_frames}")

    arr = arr.astype(bool, copy=False)
    T = arr.shape[0]
    if T == 0:
        return []

    padded = np.concatenate(([False], arr, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    if starts.size == 0:
        return []

    merged_starts = [int(starts[0])]
    merged_ends = [int(ends[0])]
    for s, e in zip(starts[1:], ends[1:]):
        gap = int(s) - merged_ends[-1]
        if gap <= hysteresis_frames:
            merged_ends[-1] = int(e)
        else:
            merged_starts.append(int(s))
            merged_ends.append(int(e))

    return [
        (s, e)
        for s, e in zip(merged_starts, merged_ends)
        if e - s >= min_duration_frames
    ]
