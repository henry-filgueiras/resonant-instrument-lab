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


class ClusterAssignments(NamedTuple):
    """Result of `cluster_assignments`.

    Clusters are ordered ascending by mean-velocity centroid — cluster 0
    is the slowest-drifting group, cluster `k-1` the fastest. `members[i]`
    is sorted by node index (not by velocity) so each entry reads as a
    plain membership set.
    """

    labels: np.ndarray                  # (N,) int — cluster id per node, in {0, …, k-1}
    members: tuple                      # k-tuple of np.ndarray[int], sorted by node index
    separability_ratio: float           # min(selected_gaps) / max(unselected_gaps); see docstring


def cluster_assignments(mean_vel, n_clusters):
    """Partition `N` nodes into `n_clusters` groups by the `n_clusters - 1` largest
    adjacent gaps in the sorted `mean_vel` array.

    This is the minimum-useful clustering primitive for the garden: the
    physics of coupled oscillators makes coherent groups share a common
    mean phase velocity (each cluster's collective rotation rate), so a
    1D gap-based split on per-node mean velocity is the honest and
    deterministic thing to do at `N = 8`. It generalizes the earlier k=2
    special case (`split_by_largest_velocity_gap`, now removed) to
    arbitrary `n_clusters` without pulling in sklearn-grade machinery.

    Why a derived-state primitive, not an ontology commitment:
    this helper does **not** decide *how many* clusters exist — the
    caller does. It does not threshold, label, or emit detector
    confidence. It only answers "given that you want `k` groups over a
    1D velocity array, here is the partition with the largest inter-group
    gaps and a scalar measure of how clean the split is".

    Parameters
    ----------
    mean_vel : array-like, shape `(N,)`
        Per-node mean phase velocity. Typically the output of
        `mean_phase_velocity(phase_vel, start_idx, end_idx)`.
    n_clusters : int, `1 <= n_clusters <= N`
        Desired number of groups. `n_clusters == 1` is a valid
        degenerate call — all nodes in one cluster.

    Returns
    -------
    ClusterAssignments
        - `labels`: `(N,)` int array of cluster ids (`0 <= labels[i] < n_clusters`).
        - `members`: `n_clusters`-tuple of int arrays, each sorted by
          node index. Clusters are ordered ascending by mean-velocity
          centroid (`members[0]` is the slowest group).
        - `separability_ratio`: scalar diagnostic on how clean the split
          is — `min(selected_gaps) / max(unselected_gaps)`. Large values
          mean the chosen inter-cluster boundaries are much wider than
          any remaining intra-cluster gap. Edge values:
            * `n_clusters == 1`: `+inf` (no inter-cluster boundaries to weaken).
            * `n_clusters == N`: `+inf` if any selected gap is positive, else `0.0`.
            * all velocities identical: `0.0` (all gaps are zero).

    Raises
    ------
    ValueError
        If `mean_vel` is not 1D, is empty, or `n_clusters` is outside
        `[1, N]`.

    Determinism & tie-breaking
    --------------------------
    - Sorting nodes by velocity uses `np.argsort(kind="stable")`, so
      nodes with identical velocities keep their node-index order.
    - Gap selection sorts gaps in descending order via
      `np.argsort(-gaps, kind="stable")`, which breaks ties between
      equal-magnitude gaps in favor of the one with the smaller
      position in the sorted-velocity axis (i.e. the earlier/slower
      boundary wins). This is deterministic for any fixed input; near-
      identical gaps are sensitive to input precision, as with any gap
      heuristic.

    Scope
    -----
    1D velocity signal only. Phase-based sub-clustering inside a
    velocity-homogeneous group (e.g. anti-phase clusters locked at the
    same rotation rate) is deferred — not needed by any current or
    near-term detector.
    """
    mean_vel = np.asarray(mean_vel)
    if mean_vel.ndim != 1:
        raise ValueError(f"mean_vel must be 1D, got shape {mean_vel.shape}")
    N = mean_vel.shape[0]
    if N < 1:
        raise ValueError("mean_vel must contain at least one node")
    if not isinstance(n_clusters, (int, np.integer)) or isinstance(n_clusters, bool):
        raise ValueError(f"n_clusters must be an int, got {type(n_clusters).__name__}")
    if n_clusters < 1 or n_clusters > N:
        raise ValueError(f"n_clusters must be in [1, N={N}], got {n_clusters}")

    order = np.argsort(mean_vel, kind="stable")
    v_sorted = mean_vel[order]
    gaps = np.diff(v_sorted)  # shape (N-1,); empty if N == 1

    n_selected = n_clusters - 1
    if n_selected == 0:
        selected_positions = np.empty(0, dtype=np.intp)
    else:
        ranking = np.argsort(-gaps, kind="stable")
        selected_positions = np.sort(ranking[:n_selected])

    # Split sorted order into n_clusters contiguous slices at the selected positions.
    boundaries = np.concatenate(([0], selected_positions + 1, [N]))
    members = tuple(
        np.sort(order[boundaries[i]:boundaries[i + 1]])
        for i in range(n_clusters)
    )

    labels = np.empty(N, dtype=np.int64)
    for i, m in enumerate(members):
        labels[m] = i

    if n_selected == 0:
        sep = float("inf")
    else:
        selected_gaps = gaps[selected_positions]
        min_sel = float(selected_gaps.min())
        mask = np.ones(gaps.shape[0], dtype=bool)
        mask[selected_positions] = False
        unselected = gaps[mask]
        if unselected.size == 0:
            sep = float("inf") if min_sel > 0.0 else 0.0
        else:
            max_unsel = float(unselected.max())
            if max_unsel > 0.0:
                sep = min_sel / max_unsel
            elif min_sel > 0.0:
                sep = float("inf")
            else:
                sep = 0.0

    return ClusterAssignments(labels=labels, members=members, separability_ratio=sep)


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


def pulse_times_of(pulse_fired):
    """Per-node ordered pulse frame indices extracted from `pulse_fired`.

    Rhythm-side analogue of `kuramoto_order`: the minimum pure-function
    projection of the `(T, N)` pulse mask into the form that
    rhythm-facing detectors (`phase_beating`, `flam`, `polyrhythmic`)
    all want — one ascending sequence of pulse frame indices per node.
    No pairing, no interval stats, no thresholds; those belong to the
    detector layer.

    Parameters
    ----------
    pulse_fired : array-like, shape `(T, N)`
        Per-frame pulse indicator as exported by the simulator. True at
        `(t, i)` means node `i`'s phase-zero crossing fired on frame
        `t` (see `DESIGN_V0.md §9.3`). Accepts any bool-castable dtype.

    Returns
    -------
    tuple[np.ndarray, ...]
        Length-N tuple. Entry `i` is a 1D `int64` array of frame indices
        at which node `i` fired, in strictly ascending order. Nodes with
        no pulses yield an empty `int64` array of shape `(0,)` — never
        `None`, never omitted. `len(result) == N` always.

    Raises
    ------
    ValueError
        If `pulse_fired` is not 2D.

    Edge cases
    ----------
    - `T == 0` → every entry is an empty `int64` array; length is still `N`.
    - `N == 0` → empty tuple.
    - All-False input → every entry is an empty `int64` array.
    - All-True input → every entry is `np.arange(T, dtype=np.int64)`.

    Units
    -----
    Frame indices, **not** sample indices and **not** seconds. Callers
    that want seconds convert via `frames / control_rate_hz` themselves
    — matching the unit policy of `sustained_windows` and keeping this
    helper config-free. Pulse-to-audio-sample mapping uses `audio_rate_hz`
    and is a rendering concern, not a detector concern.

    Scope
    -----
    Lives in `sim/derived.py` because it is a pure projection of
    exported state with no detector semantics — detectors will build
    pair-wise offset series, inter-pulse intervals, and ratio checks on
    top of this. Returning raw frame indices (rather than pre-computing
    intervals or a 2D ragged array) keeps those downstream choices
    open without commitment.
    """
    arr = np.asarray(pulse_fired)
    if arr.ndim != 2:
        raise ValueError(f"pulse_fired must be 2D (T, N), got shape {arr.shape}")
    bool_arr = arr.astype(bool, copy=False)
    _, N = bool_arr.shape
    return tuple(
        np.flatnonzero(bool_arr[:, i]).astype(np.int64, copy=False)
        for i in range(N)
    )
