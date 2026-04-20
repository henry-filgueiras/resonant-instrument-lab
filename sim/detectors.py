"""First semantic-detector layer.

Thin, deterministic detectors built on top of `sim.derived` primitives.
Two detectors in this pass — both sustained-condition detectors derived
from the global Kuramoto order parameter:

- `detect_phase_locked` — sustained high global coherence.
- `detect_drifting`     — sustained low global coherence.

Threshold policy
----------------
Thresholds and sustained-window settings are hardcoded as module-level
constants. `DESIGN_V0.md §10.7` calls for versioned YAML threshold
files; we defer that bureaucracy until enough detectors exist to
justify it. For now the constants *are* the tuning surface, and every
detector names its thresholds inline in its docstring.

Confidence doctrine
-------------------
Per `DESIGN_V0.md §3` and `ONTOLOGY.md`: confidence is *evidence
margin*, not epistemic uncertainty. A fired detector returns a scalar
that measures how far inside the defining region the signal sat,
averaged across the firing windows. Low-but-positive margins mean the
detector fired but the evidence was thin — this is a feature, not a
bug. Hysteresis dips are *not* excluded from the mean, so they pull
the margin toward zero (again, a feature — they are part of what made
the window qualify).

Non-goals for this pass
-----------------------
No detector registry, no YAML loading, no label serialization schema,
no `labels.json` emit path, no `sim.detect` orchestration entry point.
Callers invoke detectors directly from their module.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from sim.derived import kuramoto_order, sustained_windows


class DetectionResult(NamedTuple):
    """Uniform result shape for sustained-condition detectors.

    Fields
    ------
    fired : bool
        True iff at least one sustained window satisfying the detector's
        defining condition was found. Equivalent to `bool(windows)`.
    windows : list[tuple[int, int]]
        Sorted, non-overlapping half-open frame-index intervals
        `[start, end)` (same convention as `sustained_windows`). Empty
        when `fired is False`.
    confidence : float
        Evidence margin — detector-specific. Each detector's docstring
        names the exact formula and the scale. `0.0` when `fired is
        False`. Not a probability.
    longest_window_frames : int
        Span of the longest firing window, in frames. `0` when
        `fired is False`. Redundant with `windows` but convenient.
    """

    fired: bool
    windows: list
    confidence: float
    longest_window_frames: int


# --- phase_locked -----------------------------------------------------------
# Sustained r(t) >= R_LOCK for at least PHASE_LOCKED_MIN_S seconds, with
# brief dips up to PHASE_LOCKED_HYST_S absorbed by hysteresis.
R_LOCK = 0.9
PHASE_LOCKED_MIN_S = 1.0
PHASE_LOCKED_HYST_S = 0.1


def detect_phase_locked(theta, control_rate_hz):
    """Fire on sustained global phase lock.

    Defining condition
    ------------------
    Global Kuramoto `r(t) >= R_LOCK = 0.9` for a contiguous
    `sustained_windows` span of at least `PHASE_LOCKED_MIN_S = 1.0` s,
    with False dips of up to `PHASE_LOCKED_HYST_S = 0.1` s absorbed.

    Parameters
    ----------
    theta : np.ndarray, shape `(T, N)`
        Phase trajectories from `state.npz`.
    control_rate_hz : int
        Frames per second of the `theta` array. Used only to convert the
        seconds-valued thresholds above into frame counts.

    Returns
    -------
    DetectionResult
        `confidence = mean(r[frames_in_firing_windows] - R_LOCK)`, in
        `[-R_LOCK, 1 - R_LOCK]` in principle, effectively `[0, 0.1]` for
        a clean lock. Hysteresis frames where `r < R_LOCK` pull the mean
        down — a thin-margin lock returns a small confidence.
    """
    r = kuramoto_order(theta)
    condition = r >= R_LOCK
    windows = sustained_windows(
        condition,
        min_duration_frames=int(PHASE_LOCKED_MIN_S * control_rate_hz),
        hysteresis_frames=int(PHASE_LOCKED_HYST_S * control_rate_hz),
    )
    if not windows:
        return DetectionResult(fired=False, windows=[], confidence=0.0, longest_window_frames=0)
    frames = np.concatenate([r[s:e] for s, e in windows])
    confidence = float((frames - R_LOCK).mean())
    longest = max(e - s for s, e in windows)
    return DetectionResult(
        fired=True, windows=windows, confidence=confidence, longest_window_frames=longest
    )


# --- drifting ---------------------------------------------------------------
# Sustained r(t) < R_DRIFT for at least DRIFTING_MIN_S seconds, with
# brief rises up to DRIFTING_HYST_S absorbed.
R_DRIFT = 0.5
DRIFTING_MIN_S = 2.0
DRIFTING_HYST_S = 0.25


def detect_drifting(theta, control_rate_hz):
    """Fire on sustained low global coherence.

    Defining condition
    ------------------
    Global Kuramoto `r(t) < R_DRIFT = 0.5` for a contiguous
    `sustained_windows` span of at least `DRIFTING_MIN_S = 2.0` s, with
    upward spikes of up to `DRIFTING_HYST_S = 0.25` s absorbed.

    Window settings are the same ones the regime fixture test already
    validated — locked and two_cluster state produce zero qualifying
    windows, so these parameters discriminate the regime cleanly.

    Parameters
    ----------
    theta : np.ndarray, shape `(T, N)`
        Phase trajectories from `state.npz`.
    control_rate_hz : int
        Frames per second of the `theta` array.

    Returns
    -------
    DetectionResult
        `confidence = mean(R_DRIFT - r[frames_in_firing_windows])`, in
        `[-(1 - R_DRIFT), R_DRIFT]` in principle; for a clean drift the
        value sits around `0.15 – 0.25`. Hysteresis frames with
        `r >= R_DRIFT` pull the mean down, as intended.
    """
    r = kuramoto_order(theta)
    condition = r < R_DRIFT
    windows = sustained_windows(
        condition,
        min_duration_frames=int(DRIFTING_MIN_S * control_rate_hz),
        hysteresis_frames=int(DRIFTING_HYST_S * control_rate_hz),
    )
    if not windows:
        return DetectionResult(fired=False, windows=[], confidence=0.0, longest_window_frames=0)
    frames = np.concatenate([r[s:e] for s, e in windows])
    confidence = float((R_DRIFT - frames).mean())
    longest = max(e - s for s, e in windows)
    return DetectionResult(
        fired=True, windows=windows, confidence=confidence, longest_window_frames=longest
    )
