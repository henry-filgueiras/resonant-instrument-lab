"""First semantic-detector layer.

Thin, deterministic detectors built on top of `sim.derived` primitives.
Two global-coherence detectors, three pairwise rhythm detectors, one
cluster-structure detector, and one counterfactual detector:

- `detect_phase_locked`     — sustained high global coherence.
- `detect_drifting`         — sustained low global coherence.
- `detect_phase_beating`    — near-frequency node pairs whose unwrapped
  phase difference drifts linearly across the analysis window.
- `detect_flam`             — frequency-locked node pairs whose pulse
  times land repeatedly with a small but nonzero offset and stand
  alone (no other node pulses between them).
- `detect_polyrhythmic`     — node pairs whose observed pulse rates sit
  in a small-integer ratio (2:3 or 3:4) within tolerance, sustained
  across sub-windows of the run.
- `detect_dominant_cluster` — a velocity-gap partition that yields a
  coherent, nontrivially-sized sub-group cleanly separated from the
  rest of the ensemble.
- `detect_unstable_bridge`  — (counterfactual) nodes whose ablation
  collapses a previously coherent baseline cluster's local coherence.
  The first detector that routes evidence through `sim.ablate`.

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

from sim.derived import (
    cluster_assignments,
    kuramoto_order,
    local_kuramoto_order,
    mean_phase_velocity,
    pulse_times_of,
    sustained_windows,
)


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


# --- phase_beating ---------------------------------------------------------
# Pairwise rhythm detector. Asks: are there node pairs close in mean phase
# velocity (but not locked together) whose relative phase — sampled at the
# pulse times of one member — drifts linearly through at least one full
# beat cycle across the analysis window?
#
# Pulse-domain honest. This is NOT acoustic continuous-tone beating —
# there is no amplitude envelope modulation in v0. It is "the offset
# between two pulse trains slowly walks around the circle".
#
# Analysis window
# ---------------
# One window per run: frames `[warmup, T)`. Warmup skip is large enough
# to outlast the locking transients in `regime_locked`; in drifting and
# the beating fixture the dynamics are stationary, so the exact boundary
# is not critical. The detector fires once globally — its single output
# window spans the full analysis region, not a per-pair span.
#
# Velocity gate — PHASE_BEATING_DW_{MIN,MAX}
# ------------------------------------------
# `|Δω̄|` must sit inside this band, measured as |mean(phase_vel[ws:, i])
# - mean(phase_vel[ws:, j])| in rad/s.
# - Lower bound excludes locked pairs (mean-velocity difference at or
#   near zero). It is conservative — genuine same-cluster members in
#   `regime_locked` and the intra-cluster pairs in `regime_two_cluster`
#   both sit far below this.
# - Upper bound excludes the inter-cluster pairs in `regime_two_cluster`
#   (~5.6 rad/s) and the candidate pairs in `regime_drifting` (min
#   ~1.59 rad/s). It reflects the "close in mean phase velocity" half
#   of the detector's intent: ~1.5 rad/s corresponds to a beat period
#   of ~4.2 s, a genuinely slow pulse-arrival drift.
#
# Pulse and coverage gates
# ------------------------
# A pair qualifies only if the reference node fired at least
# `PHASE_BEATING_MIN_PULSES` times inside the window (enough samples to
# fit a line through), the partner fired at least once (nontrivial pulse
# activity), and the unwrapped partner-phase sequence at those pulse
# instants spans at least `PHASE_BEATING_MIN_COVERAGE` radians — i.e.
# one full 2π beat cycle actually completed inside the window.
#
# Structure gate
# --------------
# The unwrapped sequence must be well-fit by a line: `R² >=
# PHASE_BEATING_MIN_R2`. Two clocks running at steady different rates
# produce a perfectly linear drift; noise, abrupt slips, or mode
# changes pull R² down. R² is the single structural claim this detector
# makes. We deliberately avoid fancier spectral machinery.
#
# Confidence
# ----------
# Mean of (R² - PHASE_BEATING_MIN_R2) across firing pairs. Scale is
# `[0, 1 - PHASE_BEATING_MIN_R2]` = `[0, 0.1]` in practice. Thin-margin
# pairs pull the mean toward zero, matching the `detect_phase_locked`
# convention.
PHASE_BEATING_WARMUP_S = 1.0
PHASE_BEATING_DW_MIN = 0.25        # rad/s
PHASE_BEATING_DW_MAX = 1.5         # rad/s
PHASE_BEATING_MIN_PULSES = 4
PHASE_BEATING_MIN_COVERAGE = 2.0 * np.pi
PHASE_BEATING_MIN_R2 = 0.9


def _phase_beating_pair_evidence(theta, phase_vel, pulse_fired, control_rate_hz):
    """Per-pair evidence list for `detect_phase_beating`.

    Iterates unordered pairs `i < j`, applies the velocity/pulse/
    coverage/structure gates in that order, and returns a list of dicts
    for every qualifying pair:

        {"i": i, "j": j, "dw": Δω̄_signed, "slope": rad/s,
         "r2": float, "coverage_rad": float, "n_pulses_ref": int}

    Shared by the detector itself and by tests that want to inspect
    which pairs contributed without re-running the simulator.
    """
    theta = np.asarray(theta)
    phase_vel = np.asarray(phase_vel)
    pulse_fired = np.asarray(pulse_fired)
    if theta.ndim != 2 or phase_vel.shape != theta.shape or pulse_fired.shape != theta.shape:
        raise ValueError(
            "theta, phase_vel, pulse_fired must share shape (T, N); "
            f"got {theta.shape}, {phase_vel.shape}, {pulse_fired.shape}"
        )
    T, N = theta.shape
    ws = int(PHASE_BEATING_WARMUP_S * control_rate_hz)
    we = T
    if we - ws < PHASE_BEATING_MIN_PULSES:
        return []
    mean_v = phase_vel[ws:we].mean(axis=0)
    pulses = pulse_times_of(pulse_fired)
    evidence = []
    for i in range(N):
        pi = pulses[i]
        pi = pi[(pi >= ws) & (pi < we)]
        if pi.size < PHASE_BEATING_MIN_PULSES:
            continue
        times_i = pi.astype(np.float64) / float(control_rate_hz)
        for j in range(i + 1, N):
            dw = float(mean_v[j] - mean_v[i])
            if not (PHASE_BEATING_DW_MIN <= abs(dw) <= PHASE_BEATING_DW_MAX):
                continue
            pj = pulses[j]
            pj = pj[(pj >= ws) & (pj < we)]
            if pj.size < 1:
                continue
            seq = np.unwrap(theta[pi, j].astype(np.float64))
            coverage = float(abs(seq[-1] - seq[0]))
            if coverage < PHASE_BEATING_MIN_COVERAGE:
                continue
            A = np.vstack([times_i, np.ones_like(times_i)]).T
            (slope, icpt), *_ = np.linalg.lstsq(A, seq, rcond=None)
            fit = A @ np.array([slope, icpt])
            ss_res = float(np.sum((seq - fit) ** 2))
            ss_tot = float(np.sum((seq - seq.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
            if r2 < PHASE_BEATING_MIN_R2:
                continue
            evidence.append({
                "i": i, "j": j, "dw": dw,
                "slope": float(slope), "r2": float(r2),
                "coverage_rad": coverage,
                "n_pulses_ref": int(pi.size),
            })
    return evidence


def detect_phase_beating(theta, phase_vel, pulse_fired, control_rate_hz):
    """Fire on near-frequency node pairs with slow structured pulse-offset drift.

    Defining condition
    ------------------
    Over the analysis window `[warmup, T)` with warmup =
    `PHASE_BEATING_WARMUP_S = 1.0` s, at least one unordered pair
    `(i, j)` satisfies all of:

    - `PHASE_BEATING_DW_MIN <= |Δω̄| <= PHASE_BEATING_DW_MAX`
      (0.25 rad/s .. 1.5 rad/s — "close but not locked");
    - `i` fired at least `PHASE_BEATING_MIN_PULSES = 4` times in the
      window; `j` fired at least once;
    - the unwrapped sequence `θ_j` sampled at `i`'s pulse times covers
      `>= PHASE_BEATING_MIN_COVERAGE = 2π` radians (one full beat);
    - a linear fit through that sequence has `R² >=
      PHASE_BEATING_MIN_R2 = 0.9`.

    Parameters
    ----------
    theta : np.ndarray, shape `(T, N)`
        Wrapped phases from `state.npz`. Unwrapping is applied internally
        to the per-pair sampled series.
    phase_vel : np.ndarray, shape `(T, N)`
        Per-frame phase velocity (rad/s) from `state.npz`.
    pulse_fired : np.ndarray, shape `(T, N)`, bool
        Per-frame pulse indicator from `state.npz`.
    control_rate_hz : int
        Frames per second of the input arrays.

    Returns
    -------
    DetectionResult
        `fired` is True iff at least one pair qualified. `windows` is a
        single analysis-window entry `[(warmup_frames, T)]` when fired
        — this detector does not carve per-pair sub-windows; the
        evidence is a property of the whole window. `confidence` is the
        mean of `(R² - PHASE_BEATING_MIN_R2)` across firing pairs.
        `longest_window_frames` is the length of the analysis window.

    Notes
    -----
    For tests and debugging that need to inspect *which* pairs
    contributed, call `_phase_beating_pair_evidence` directly. The
    detector-level return deliberately collapses that list into the
    standard sustained-detector result shape; no pair metadata leaks
    into `DetectionResult`.
    """
    T = np.asarray(theta).shape[0]
    ws = int(PHASE_BEATING_WARMUP_S * control_rate_hz)
    evidence = _phase_beating_pair_evidence(theta, phase_vel, pulse_fired, control_rate_hz)
    if not evidence:
        return DetectionResult(fired=False, windows=[], confidence=0.0, longest_window_frames=0)
    mean_margin = float(np.mean([e["r2"] - PHASE_BEATING_MIN_R2 for e in evidence]))
    return DetectionResult(
        fired=True,
        windows=[(ws, T)],
        confidence=mean_margin,
        longest_window_frames=T - ws,
    )


# --- flam -------------------------------------------------------------------
# Pairwise pulse-timing detector. Asks: are there node pairs that repeatedly
# land close together in time with a small but nonzero offset, producing a
# flam-like near-unison rather than exact synchrony?
#
# Pulse-domain honest. No acoustic claim about wave interference. The
# only evidence this detector uses is pulse frame indices — a flam here
# is "two pulse streams land 15–50 ms apart, repeatedly, with no other
# node's pulse between them".
#
# Velocity gate — FLAM_DW_MAX
# ---------------------------
# `|Δω̄|` must be `<= FLAM_DW_MAX`. Flam requires a pair whose mean
# phase velocities are essentially equal — a near-lock — so offsets can
# be consistent across time. Pairs with meaningful velocity difference
# drift through all offsets and belong to `phase_beating`, not flam.
# Set tight (0.25 rad/s) to exclude every candidate pair in the
# `phase_beating` fixture (|Δω̄| ≈ 1.22 rad/s there) and every
# inter-cluster pair in `two_cluster` (~5.6 rad/s).
#
# Offset band — FLAM_MIN_OFFSET_S, FLAM_MAX_OFFSET_S
# --------------------------------------------------
# `|t_j − t_i|` must sit in [15 ms, 50 ms]. The lower bound keeps
# exact-lock coincidences out — the `regime_locked` ensemble has
# adjacent-pair offsets of 2–10 ms, well below 15 ms. The upper bound
# reflects the musical intuition of a flam: a sliver of audible offset,
# not a cross-rhythm.
#
# Isolation — the crowd-exclusion rule
# ------------------------------------
# A pair event qualifies as flam iff NO third node's pulse sits
# within `FLAM_ISOLATION_MARGIN_S` of either pair pulse. Without this
# rule, cluster-edge pairs masquerade as flams: in `regime_locked` the
# extreme pair reaches ~20 ms offset with intermediate nodes firing
# *between* them (caught by "strictly between"); in
# `regime_two_cluster` the leading pair within a cluster fires 15 ms
# apart with the rest of its cluster firing 5–10 ms *just outside*
# the pair window (NOT caught by "strictly between" alone — needs the
# margin). The honest flam signature is a pair that stands apart in
# time from every other pulse-producing node by at least the chosen
# margin on both sides.
#
# Evidence gates — FLAM_MIN_EVENTS, FLAM_MIN_SPAN_S
# -------------------------------------------------
# A pair must accumulate at least `FLAM_MIN_EVENTS` qualifying events
# whose first-to-last span is `>= FLAM_MIN_SPAN_S` seconds. One-off
# near-coincidences don't count; flam is a sustained rhythmic
# relationship.
#
# Confidence
# ----------
# Mean over qualifying pairs of `(n_events − FLAM_MIN_EVENTS) /
# FLAM_MIN_EVENTS` — "how many times over the minimum did we see".
# Zero when a pair just made the bar, unbounded upward (but in
# practice `<= 5` for an 8–15 s run at 2 Hz pair rate). `0.0` when the
# detector does not fire, matching the other detectors' convention.
FLAM_WARMUP_S = 1.0
FLAM_DW_MAX = 0.25          # rad/s
FLAM_MIN_OFFSET_S = 0.015   # 15 ms
FLAM_MAX_OFFSET_S = 0.050   # 50 ms
FLAM_ISOLATION_MARGIN_S = 0.015  # 15 ms — one FLAM_MIN_OFFSET worth
FLAM_MIN_EVENTS = 6
FLAM_MIN_SPAN_S = 2.0


def _flam_pair_evidence(phase_vel, pulse_fired, control_rate_hz):
    """Per-pair evidence list for `detect_flam`.

    Iterates unordered pairs `i < j`, applies the velocity / offset /
    isolation / count / span gates, and returns a list of dicts for
    every qualifying pair:

        {"i": i, "j": j, "dw": Δω̄_signed, "n_events": int,
         "median_abs_offset_s": float, "offset_std_s": float,
         "span_s": float}

    Shared by the detector itself and by tests that want to inspect
    which pairs contributed without re-running the simulator.
    """
    phase_vel = np.asarray(phase_vel)
    pulse_fired = np.asarray(pulse_fired)
    if pulse_fired.ndim != 2 or phase_vel.shape != pulse_fired.shape:
        raise ValueError(
            "phase_vel and pulse_fired must share shape (T, N); "
            f"got {phase_vel.shape}, {pulse_fired.shape}"
        )
    T, N = pulse_fired.shape
    ws = int(FLAM_WARMUP_S * control_rate_hz)
    we = T
    if we <= ws:
        return []
    mean_v = phase_vel[ws:we].mean(axis=0)
    pulses = pulse_times_of(pulse_fired)
    # Per-node pulse times in seconds, restricted to analysis window.
    times = []
    nodes = []
    for i in range(N):
        p = pulses[i]
        p = p[(p >= ws) & (p < we)]
        t = p.astype(np.float64) / float(control_rate_hz)
        times.append(t)
        nodes.append(np.full(t.shape[0], i, dtype=np.int64))
    # Global sorted (time, node) index for fast "pulse strictly between"
    # isolation queries via np.searchsorted.
    all_times = np.concatenate(times) if N else np.empty(0, dtype=np.float64)
    all_nodes = np.concatenate(nodes) if N else np.empty(0, dtype=np.int64)
    if all_times.size:
        order = np.argsort(all_times, kind="stable")
        all_times = all_times[order]
        all_nodes = all_nodes[order]

    evidence = []
    for i in range(N):
        ti = times[i]
        if ti.size == 0:
            continue
        for j in range(i + 1, N):
            dw = float(mean_v[j] - mean_v[i])
            if abs(dw) > FLAM_DW_MAX:
                continue
            tj = times[j]
            if tj.size == 0:
                continue
            # For each pulse in i, find nearest pulse in j.
            ins = np.searchsorted(tj, ti)
            signed_offsets = []
            event_times = []
            for k, t_i in enumerate(ti):
                ii = int(ins[k])
                candidates = []
                if ii > 0:
                    candidates.append(tj[ii - 1])
                if ii < tj.size:
                    candidates.append(tj[ii])
                if not candidates:
                    continue
                t_j = min(candidates, key=lambda x: abs(x - t_i))
                off = t_j - t_i
                aoff = abs(off)
                if not (FLAM_MIN_OFFSET_S <= aoff <= FLAM_MAX_OFFSET_S):
                    continue
                lo = min(t_i, t_j) - FLAM_ISOLATION_MARGIN_S
                hi = max(t_i, t_j) + FLAM_ISOLATION_MARGIN_S
                # Isolation: no other node pulse within the padded window
                # [lo, hi]. The margin excludes cluster-edge pairs whose
                # neighbors fire just outside the pair window.
                lo_idx = int(np.searchsorted(all_times, lo, side="left"))
                hi_idx = int(np.searchsorted(all_times, hi, side="right"))
                intruder = False
                for m in range(lo_idx, hi_idx):
                    k_node = int(all_nodes[m])
                    if k_node != i and k_node != j:
                        intruder = True
                        break
                if intruder:
                    continue
                signed_offsets.append(off)
                event_times.append(t_i)
            if len(signed_offsets) < FLAM_MIN_EVENTS:
                continue
            span = float(event_times[-1] - event_times[0])
            if span < FLAM_MIN_SPAN_S:
                continue
            arr = np.asarray(signed_offsets, dtype=np.float64)
            evidence.append({
                "i": i, "j": j, "dw": dw,
                "n_events": int(arr.size),
                "median_abs_offset_s": float(np.median(np.abs(arr))),
                "offset_std_s": float(arr.std()),
                "span_s": span,
            })
    return evidence


def detect_flam(phase_vel, pulse_fired, control_rate_hz):
    """Fire on pair-wise, isolated, repeated small-offset pulse coincidences.

    Defining condition
    ------------------
    Over the analysis window `[warmup, T)` with warmup =
    `FLAM_WARMUP_S = 1.0` s, at least one unordered pair `(i, j)`
    satisfies all of:

    - `|Δω̄| <= FLAM_DW_MAX = 0.25 rad/s` (near-locked);
    - at least `FLAM_MIN_EVENTS = 6` qualifying "flam events", where
      a flam event is a pulse in `i` paired with its nearest pulse in
      `j` such that (a) `FLAM_MIN_OFFSET_S <= |t_j − t_i| <=
      FLAM_MAX_OFFSET_S` (15–50 ms band) and (b) no third node `k`
      fired a pulse strictly between the two pair pulses (isolation);
    - first-to-last flam-event span `>= FLAM_MIN_SPAN_S = 2.0` s
      (sustained, not a one-off cluster of near-coincidences).

    Parameters
    ----------
    phase_vel : np.ndarray, shape `(T, N)`
        Per-frame phase velocity (rad/s) from `state.npz`. Used only
        for the velocity gate.
    pulse_fired : np.ndarray, shape `(T, N)`, bool
        Per-frame pulse indicator from `state.npz`.
    control_rate_hz : int
        Frames per second of the input arrays.

    Returns
    -------
    DetectionResult
        `fired` is True iff at least one pair qualified. `windows` is
        a single analysis-window entry `[(warmup_frames, T)]` — flam
        is a whole-window pair-level property, not a per-event
        sub-window. `confidence` is the mean of `(n_events −
        FLAM_MIN_EVENTS) / FLAM_MIN_EVENTS` across firing pairs.
        `longest_window_frames` is the length of the analysis window.

    Notes
    -----
    `theta` is deliberately not required — the detector is a pure
    function of `phase_vel` (for the velocity gate) and `pulse_fired`
    (for all timing logic). For debugging / test inspection of
    *which* pairs contributed, call `_flam_pair_evidence` directly;
    the detector-level return collapses pair metadata into the
    standard `DetectionResult` shape.
    """
    T = int(np.asarray(pulse_fired).shape[0])
    ws = int(FLAM_WARMUP_S * control_rate_hz)
    evidence = _flam_pair_evidence(phase_vel, pulse_fired, control_rate_hz)
    if not evidence:
        return DetectionResult(fired=False, windows=[], confidence=0.0, longest_window_frames=0)
    margins = [(e["n_events"] - FLAM_MIN_EVENTS) / FLAM_MIN_EVENTS for e in evidence]
    mean_margin = float(np.mean(margins))
    return DetectionResult(
        fired=True,
        windows=[(ws, T)],
        confidence=mean_margin,
        longest_window_frames=T - ws,
    )


# --- polyrhythmic ----------------------------------------------------------
# Pairwise pulse-rate detector. Asks: are there node pairs whose observed
# pulse rates sit in a small-integer ratio (2:3 or 3:4) over a sustained
# interval?
#
# Pulse-domain honest. No musicality, groove, or meter claim — the only
# signal this detector reads is the per-node ordered pulse-frame list
# exported by the simulator. "Polyrhythmic" here means exactly
# "rate_i / rate_j is within a few percent of p/q for some small p:q,
# and holds across sub-windows of the run".
#
# Ratio family — POLY_RATIO_SET
# -----------------------------
# `((2, 3), (3, 4))`. Deliberately narrow:
# - 1:2 is excluded because "half-time" is a weak claim — every
#   frequency-doubled pair passes, including the ones whose alignment
#   is an intrinsic-frequency coincidence with zero coupling and no
#   rhythmic cohesion.
# - 4:5 is excluded because at 2 % tolerance, intrinsic-frequency
#   coincidences in the existing negative fixtures (e.g.
#   `regime_drifting`'s 2.10/2.60 Hz pair at ratio 0.808 ≈ 4:5 within
#   1 %) would force false positives. Adding 4:5 would require
#   retuning `regime_drifting`, which is out of scope for this pass.
# The narrow set still covers the two most rhythmically distinct
# small-integer polyrhythms and is enough to exercise the detector.
#
# Analysis window
# ---------------
# `[warmup, T)` with warmup = `POLY_WARMUP_S = 1.0` s, matching the
# other pair detectors. One window per run; the detector fires once
# globally and its single `windows` entry spans the full analysis
# region (per-pair sub-windows are an internal sustained-gate
# mechanism, not exposed on `DetectionResult`).
#
# Rate measurement
# ----------------
# For each node, `rate = (n_pulses - 1) / (last_pulse_time -
# first_pulse_time)`. This is the mean-inter-pulse-interval estimator,
# free of the ±1-pulse window-boundary quantization that would make
# `n / window_duration` unstable at short windows. It matches the
# pulse-domain honesty of the detector: only pulse timestamps are
# used — no phase, no phase velocity.
#
# Near-unity exclusion — POLY_NEAR_UNITY_MAX
# ------------------------------------------
# Pairs with `min_rate / max_rate > 0.85` are skipped before the ratio
# search. Those are phase_beating / flam / phase_locked territory —
# the pair shares a near-common rate, not a small-integer ratio.
# `0.85` sits above the `(3, 4) = 0.75` target with headroom but well
# below any plausible "essentially same rate" pair (locked pairs land
# within 0.1 % of 1.0).
#
# Whole-window ratio gate — POLY_RATIO_TOL
# ----------------------------------------
# Pick the best `(p, q)` in `POLY_RATIO_SET` minimizing
# `|ratio - p/q|`; require relative deviation
# `|ratio - p/q| / (p/q) <= POLY_RATIO_TOL = 0.02` (2 %). Threshold
# chosen from a survey of worst-case negatives across the existing
# fixtures:
#
#   fixture                  nearest small-ratio pair    rel dev
#   regime_drifting          (3.05, 4.45) → 2:3          2.7 %
#   regime_drifting          (2.10, 3.05) → 2:3          3.2 %
#   regime_two_cluster       inter-cluster   → 2:3       5.0 %
#   regime_phase_beating     (4.5, 7.0)     → 2:3        3.6 %
#   regime_phase_beating     (5.5, 7.0)     → 3:4        4.8 %
#   regime_flam              (4, 5) → 4:5 (not in set) no match
#
# 2 % cleanly rejects all of them while the intended positive fixture
# (`regime_polyrhythmic`, intrinsic 2.0 Hz / 3.0 Hz) sits far inside
# the band (deviation < 0.5 %).
#
# Sub-window sustained gate
# -------------------------
# Split the analysis window into `POLY_SUB_WINDOWS = 3` equal thirds.
# Each sub-window must carry at least `POLY_MIN_PULSES_PER_SUB = 3`
# pulses on each pair member, and its per-window rate ratio must
# match the SAME `(p, q)` within `POLY_SUB_WINDOW_RATIO_TOL = 0.04`
# (4 %). The looser sub-window tolerance acknowledges that shorter
# windows carry fewer pulses and thus noisier IPI-based rate
# estimates; 4 % still rejects the drifting near-misses, which stay
# close to their whole-window deviation across thirds.
#
# Pulse-count floor — POLY_MIN_PULSES_PER_NODE
# --------------------------------------------
# A pair is skipped unless each member fired at least 8 pulses in the
# analysis window. At 2 Hz over an 8 s window that's comfortably met;
# pairs with too few pulses can't support a stable rate estimate,
# let alone a sustained-across-thirds claim.
#
# Confidence
# ----------
# Mean over firing pairs of `POLY_RATIO_TOL - ratio_dev` — "how far
# inside the 2 % tolerance band the pair sat". Scale `[0, 0.02]` in
# practice; the positive fixture's pair (near-zero deviation) sits
# near the top of the band. `0.0` when the detector does not fire,
# matching the other detectors' convention.
POLY_WARMUP_S = 1.0
POLY_RATIO_SET = ((2, 3), (3, 4))
POLY_RATIO_TOL = 0.02
POLY_NEAR_UNITY_MAX = 0.85
POLY_SUB_WINDOWS = 3
POLY_SUB_WINDOW_RATIO_TOL = 0.04
POLY_MIN_PULSES_PER_NODE = 8
POLY_MIN_PULSES_PER_SUB = 3


def _poly_rate_from_pulse_times(times_s):
    """Mean inter-pulse-interval rate (Hz) from a 1D ascending array of
    pulse timestamps in seconds. Returns `None` if the array is too
    short or degenerate (zero-span)."""
    n = int(times_s.shape[0])
    if n < 2:
        return None
    span = float(times_s[-1] - times_s[0])
    if span <= 0.0:
        return None
    return (n - 1) / span


def _polyrhythmic_pair_evidence(pulse_fired, control_rate_hz):
    """Per-pair evidence list for `detect_polyrhythmic`.

    Iterates unordered pairs `i < j`, applies the pulse-count /
    near-unity / whole-window-ratio / sustained-across-thirds gates
    in that order, and returns a list of dicts for every qualifying
    pair:

        {"i": i_slow, "j": j_fast, "p": int, "q": int,
         "rate_slow": float, "rate_fast": float, "ratio": float,
         "ratio_dev": float, "sub_window_ratios": list[float],
         "sub_window_devs": list[float],
         "n_pulses_slow": int, "n_pulses_fast": int}

    `i_slow` / `j_fast` are node indices reordered so that
    `rate_slow < rate_fast`; `p < q` matches (`p:q = slow:fast`).

    Shared by the detector itself and by tests that want to inspect
    which pairs contributed without re-running the simulator.
    """
    arr = np.asarray(pulse_fired)
    if arr.ndim != 2:
        raise ValueError(
            f"pulse_fired must be 2D (T, N), got shape {arr.shape}"
        )
    T, N = arr.shape
    ws = int(POLY_WARMUP_S * control_rate_hz)
    we = T
    if we <= ws:
        return []
    sub_edges_f = [
        ws + int(round(k * (we - ws) / POLY_SUB_WINDOWS))
        for k in range(POLY_SUB_WINDOWS + 1)
    ]
    sub_edges_s = [f / float(control_rate_hz) for f in sub_edges_f]
    pulses = pulse_times_of(arr)
    times_s = []
    for i in range(N):
        p = pulses[i]
        p = p[(p >= ws) & (p < we)]
        times_s.append(p.astype(np.float64) / float(control_rate_hz))

    evidence = []
    for i in range(N):
        ti = times_s[i]
        if ti.size < POLY_MIN_PULSES_PER_NODE:
            continue
        ri = _poly_rate_from_pulse_times(ti)
        if ri is None:
            continue
        for j in range(i + 1, N):
            tj = times_s[j]
            if tj.size < POLY_MIN_PULSES_PER_NODE:
                continue
            rj = _poly_rate_from_pulse_times(tj)
            if rj is None:
                continue
            r_lo = min(ri, rj)
            r_hi = max(ri, rj)
            if r_hi <= 0.0:
                continue
            ratio = r_lo / r_hi
            if ratio > POLY_NEAR_UNITY_MAX:
                continue
            best_pq = None
            best_dev = float("inf")
            for (p, q) in POLY_RATIO_SET:
                target = p / q
                dev = abs(ratio - target) / target
                if dev < best_dev:
                    best_dev = dev
                    best_pq = (p, q)
            if best_dev > POLY_RATIO_TOL:
                continue
            target_pq = best_pq[0] / best_pq[1]
            sub_ratios = []
            sub_devs = []
            sub_ok = True
            for k in range(POLY_SUB_WINDOWS):
                s_start_s = sub_edges_s[k]
                s_end_s = sub_edges_s[k + 1]
                sub_ti = ti[(ti >= s_start_s) & (ti < s_end_s)]
                sub_tj = tj[(tj >= s_start_s) & (tj < s_end_s)]
                if (sub_ti.size < POLY_MIN_PULSES_PER_SUB
                        or sub_tj.size < POLY_MIN_PULSES_PER_SUB):
                    sub_ok = False
                    break
                sub_ri = _poly_rate_from_pulse_times(sub_ti)
                sub_rj = _poly_rate_from_pulse_times(sub_tj)
                if sub_ri is None or sub_rj is None:
                    sub_ok = False
                    break
                sub_hi = max(sub_ri, sub_rj)
                if sub_hi <= 0.0:
                    sub_ok = False
                    break
                sub_lo = min(sub_ri, sub_rj)
                sub_ratio = sub_lo / sub_hi
                sub_dev = abs(sub_ratio - target_pq) / target_pq
                if sub_dev > POLY_SUB_WINDOW_RATIO_TOL:
                    sub_ok = False
                    break
                sub_ratios.append(float(sub_ratio))
                sub_devs.append(float(sub_dev))
            if not sub_ok:
                continue
            if ri < rj:
                i_slow, j_fast = i, j
                rate_slow, rate_fast = ri, rj
                n_slow, n_fast = int(ti.size), int(tj.size)
            else:
                i_slow, j_fast = j, i
                rate_slow, rate_fast = rj, ri
                n_slow, n_fast = int(tj.size), int(ti.size)
            evidence.append({
                "i": i_slow, "j": j_fast,
                "p": int(best_pq[0]), "q": int(best_pq[1]),
                "rate_slow": float(rate_slow),
                "rate_fast": float(rate_fast),
                "ratio": float(ratio),
                "ratio_dev": float(best_dev),
                "sub_window_ratios": sub_ratios,
                "sub_window_devs": sub_devs,
                "n_pulses_slow": n_slow,
                "n_pulses_fast": n_fast,
            })
    return evidence


def detect_polyrhythmic(pulse_fired, control_rate_hz):
    """Fire on pair-wise small-integer pulse-rate ratios sustained across the run.

    Defining condition
    ------------------
    Over the analysis window `[warmup, T)` with warmup =
    `POLY_WARMUP_S = 1.0` s, at least one unordered pair `(i, j)`
    satisfies all of:

    - each member fired at least `POLY_MIN_PULSES_PER_NODE = 8`
      pulses in the window;
    - `min(rate_i, rate_j) / max(rate_i, rate_j) <=
      POLY_NEAR_UNITY_MAX = 0.85` (the pair does NOT share a
      near-common rate — that's phase_beating / flam / locked
      territory);
    - there exists `(p, q)` in `POLY_RATIO_SET = ((2, 3), (3, 4))`
      whose relative deviation from the observed rate ratio is
      `<= POLY_RATIO_TOL = 0.02` (2 %);
    - in each of `POLY_SUB_WINDOWS = 3` equal thirds of the analysis
      window, both nodes fired at least `POLY_MIN_PULSES_PER_SUB = 3`
      pulses, and the sub-window rate ratio matches the SAME `(p, q)`
      within `POLY_SUB_WINDOW_RATIO_TOL = 0.04` (4 %).

    Parameters
    ----------
    pulse_fired : np.ndarray, shape `(T, N)`, bool
        Per-frame pulse indicator from `state.npz`.
    control_rate_hz : int
        Frames per second of the input array.

    Returns
    -------
    DetectionResult
        `fired` is True iff at least one pair qualified. `windows` is
        a single analysis-window entry `[(warmup_frames, T)]` — the
        detector does not carve per-pair sub-windows into its public
        result; the evidence is a property of the whole analysis
        window, and the sub-window split is an internal sustained
        gate, not a caller-visible partition. `confidence` is the
        mean of `(POLY_RATIO_TOL - ratio_dev)` across firing pairs
        — "how far inside the 2 % tolerance band the pair sat", in
        `[0, POLY_RATIO_TOL]`. `longest_window_frames` is the length
        of the analysis window.

    Notes
    -----
    Depends only on `pulse_fired` — no `theta`, no `phase_vel`.
    This is the pulse-domain honesty: a small-integer rate-ratio
    claim needs nothing beyond pulse timestamps to state. For tests
    and debugging that need to inspect *which* pairs contributed,
    call `_polyrhythmic_pair_evidence` directly; the detector-level
    return deliberately collapses the pair list into the standard
    `DetectionResult` shape.
    """
    T = int(np.asarray(pulse_fired).shape[0])
    ws = int(POLY_WARMUP_S * control_rate_hz)
    evidence = _polyrhythmic_pair_evidence(pulse_fired, control_rate_hz)
    if not evidence:
        return DetectionResult(fired=False, windows=[], confidence=0.0, longest_window_frames=0)
    mean_margin = float(np.mean([POLY_RATIO_TOL - e["ratio_dev"] for e in evidence]))
    return DetectionResult(
        fired=True,
        windows=[(ws, T)],
        confidence=mean_margin,
        longest_window_frames=T - ws,
    )


# --- dominant_cluster ------------------------------------------------------
# Cluster-structure detector. Asks: over the tail analysis window, does
# the per-node mean-phase-velocity signal partition cleanly into two
# groups, each nontrivially sized, with at least one group internally
# phase-coherent?
#
# Divergence from the original ontology sketch
# --------------------------------------------
# `ONTOLOGY.md` sketches the label with a size-dominance rule
# (`|largest| >= size_frac · N`, `r_dom = 0.8`) and documents that
# balanced 4/4 splits at `N = 8` never fire — i.e. the original sketch
# cannot name the current `regime_two_cluster` fixture, which is
# exactly a 4/4 split. We deliberately evolve the definition: the
# detector fires when at least one coherent sub-group stands cleanly
# apart from the rest, whether or not any single sub-group occupies a
# majority of the ensemble. The name is kept ("a group is running its
# own show") but the criterion is coherence + separation, not size
# majority. The pivot is recorded in `DIRECTORS_NOTES.md`; if later
# detectors (`unstable_bridge`, `brittle_lock`) need the strict
# size-majority interpretation, that's a cluster-membership helper on
# top of this detector, not a change to its firing rule.
#
# Window & partition
# ------------------
# Analysis is the tail `DOMINANT_CLUSTER_TAIL_S = 3.0` s — the same
# tail window the summary builder already uses for the 2-way
# velocity-separability stat. `cluster_assignments(v_tail,
# n_clusters=2)` partitions the N nodes into two groups by the largest
# adjacent gap in sorted mean velocity, and returns a separability
# ratio = `selected_gap / max(unselected_gap)`. A large ratio means
# the chosen inter-cluster boundary is much wider than any remaining
# intra-cluster gap — what "clean two-way split" means here.
#
# Separability gate — DOMINANT_CLUSTER_MIN_SEP
# --------------------------------------------
# `sep >= 5.0`. Survey across the five fixtures on the 3 s tail:
#   - `regime_two_cluster`: sep ≈ 706       (fires, by design)
#   - `regime_flam`        (N=4): sep ≈ 1.90
#   - `regime_phase_beating`:     sep ≈ 1.79
#   - `regime_drifting`:          sep ≈ 1.42
#   - `regime_locked`:            sep ≈ 0.00 (all velocities equal)
# Five is well above every negative (next-highest is 1.9) and far below
# the positive (706). The ratio is scale-free; 5 is not threshold-gamed.
#
# Cluster-size gate — DOMINANT_CLUSTER_MIN_SIZE
# ---------------------------------------------
# `min(|C_0|, |C_1|) >= 2`. Excludes the degenerate split where one
# node sits far from seven others (e.g. a single outlier in
# `regime_drifting`, where the sorted-velocity extrema produce a 1+7
# split). "At least two nontrivial clusters exist" is the task's
# explicit ask; a singleton cluster is not a group.
#
# N precondition — DOMINANT_CLUSTER_MIN_N
# ---------------------------------------
# Need at least 4 nodes before a "two nontrivial clusters" claim makes
# sense (2+2 minimum). Below `N = 4` the detector stays silent.
#
# Coherence gate — DOMINANT_CLUSTER_MIN_LOCAL_R
# ---------------------------------------------
# At least one cluster must have `local_r >= 0.9` across the tail
# window. This is the same 0.9 threshold `detect_phase_locked` uses on
# global `r` — a sub-group is called coherent when it would lock by
# the same standard the whole ensemble is.
#
# Confidence
# ----------
# Mean of `(local_r − DOMINANT_CLUSTER_MIN_LOCAL_R)` across qualifying
# clusters — same "how far above the coherence floor" shape as
# `detect_phase_locked`. Observed on `regime_two_cluster`: both
# clusters at `local_r ≈ 0.993` → margin ≈ 0.093. Scale is
# `[0, 1 − 0.9]` = `[0, 0.1]` in practice.
DOMINANT_CLUSTER_TAIL_S = 3.0
DOMINANT_CLUSTER_MIN_N = 4
DOMINANT_CLUSTER_MIN_SIZE = 2
DOMINANT_CLUSTER_MIN_SEP = 5.0
DOMINANT_CLUSTER_MIN_LOCAL_R = 0.9


def _dominant_cluster_evidence(theta, phase_vel, control_rate_hz):
    """Per-cluster evidence list for `detect_dominant_cluster`.

    Applies the whole-partition preconditions (`N >= MIN_N`, non-empty
    tail window, `sep >= MIN_SEP`, every cluster size `>= MIN_SIZE`)
    first; returns `[]` if any fails. Otherwise returns one dict per
    cluster whose tail-window local `r` clears
    `DOMINANT_CLUSTER_MIN_LOCAL_R`:

        {"cluster_idx": int, "size": int, "local_r": float,
         "mean_vel": float, "separability_ratio": float,
         "members": np.ndarray[int]}

    Shared by the detector itself and by tests that want to inspect
    which clusters contributed without re-running the simulator.
    Clusters are ordered ascending by mean-velocity centroid (the
    `cluster_assignments` convention) — `cluster_idx = 0` is the
    slower group.
    """
    theta = np.asarray(theta)
    phase_vel = np.asarray(phase_vel)
    if theta.ndim != 2 or phase_vel.shape != theta.shape:
        raise ValueError(
            "theta and phase_vel must share shape (T, N); "
            f"got {theta.shape}, {phase_vel.shape}"
        )
    T, N = theta.shape
    if N < DOMINANT_CLUSTER_MIN_N:
        return []
    tail_frames = min(int(DOMINANT_CLUSTER_TAIL_S * control_rate_hz), T)
    if tail_frames < 1:
        return []
    v_tail = mean_phase_velocity(phase_vel, start_idx=-tail_frames)
    ca = cluster_assignments(v_tail, n_clusters=2)
    sep = ca.separability_ratio
    # `sep` can be +inf for degenerate inputs; numpy's inf > 5 is True, so
    # the comparison does the right thing. NaN cannot occur here — the
    # primitive returns 0.0 in the all-equal edge case.
    if not (sep >= DOMINANT_CLUSTER_MIN_SEP):
        return []
    for m in ca.members:
        if m.size < DOMINANT_CLUSTER_MIN_SIZE:
            return []
    theta_tail = theta[-tail_frames:]
    evidence = []
    for idx, m in enumerate(ca.members):
        r_local = float(local_kuramoto_order(theta_tail, m).mean())
        if r_local < DOMINANT_CLUSTER_MIN_LOCAL_R:
            continue
        evidence.append({
            "cluster_idx": idx,
            "size": int(m.size),
            "local_r": r_local,
            "mean_vel": float(v_tail[m].mean()),
            "separability_ratio": float(sep) if np.isfinite(sep) else float("inf"),
            "members": np.asarray(m),
        })
    return evidence


def detect_dominant_cluster(theta, phase_vel, control_rate_hz):
    """Fire on a coherent, well-separated velocity sub-group in the tail window.

    Defining condition
    ------------------
    Over the tail analysis window `[T − tail_frames, T)` with
    `tail_frames = DOMINANT_CLUSTER_TAIL_S · control_rate_hz`:

    - `N >= DOMINANT_CLUSTER_MIN_N = 4` (need enough nodes for two
      nontrivial groups);
    - `cluster_assignments(v_tail, n_clusters=2).separability_ratio >=
      DOMINANT_CLUSTER_MIN_SEP = 5.0` (clean 2-way split);
    - every cluster has size `>= DOMINANT_CLUSTER_MIN_SIZE = 2`;
    - at least one cluster has tail-window `local_r >=
      DOMINANT_CLUSTER_MIN_LOCAL_R = 0.9`.

    Parameters
    ----------
    theta : np.ndarray, shape `(T, N)`
        Phase trajectories from `state.npz`.
    phase_vel : np.ndarray, shape `(T, N)`
        Per-frame phase velocity (rad/s) from `state.npz`.
    control_rate_hz : int
        Frames per second of the input arrays.

    Returns
    -------
    DetectionResult
        `fired` is True iff at least one cluster cleared the coherence
        floor. `windows` is a single tail-window entry
        `[(T − tail_frames, T)]` — the detector does not carve
        per-cluster sub-windows; the evidence is a property of the
        whole tail. `confidence` is the mean of `(local_r −
        DOMINANT_CLUSTER_MIN_LOCAL_R)` across qualifying clusters.
        `longest_window_frames` is the tail-window length.

    Notes
    -----
    For tests and debugging that need to inspect *which* clusters
    contributed, call `_dominant_cluster_evidence` directly. The
    detector-level return deliberately collapses that list into the
    standard `DetectionResult` shape; per-cluster membership does not
    leak into the public result here (the counterfactual detectors
    that need a cluster handle — `unstable_bridge` — will call the
    evidence helper directly).
    """
    T = int(np.asarray(theta).shape[0])
    tail_frames = min(int(DOMINANT_CLUSTER_TAIL_S * control_rate_hz), T)
    evidence = _dominant_cluster_evidence(theta, phase_vel, control_rate_hz)
    if not evidence:
        return DetectionResult(fired=False, windows=[], confidence=0.0, longest_window_frames=0)
    mean_margin = float(
        np.mean([e["local_r"] - DOMINANT_CLUSTER_MIN_LOCAL_R for e in evidence])
    )
    ws = T - tail_frames
    return DetectionResult(
        fired=True,
        windows=[(ws, T)],
        confidence=mean_margin,
        longest_window_frames=tail_frames,
    )


# --- unstable_bridge -------------------------------------------------------
# Counterfactual detector. Asks: is there a node whose ablation collapses
# a previously coherent baseline cluster's local coherence?
#
# First detector that consumes `sim.ablate`. Per `DESIGN_V0.md §10.1`,
# counterfactual detectors must route intervention evidence through
# `sim.ablate` only — no ad-hoc back door. Applicability is strict:
# the baseline run must already exhibit a coherent cluster structure
# (`dominant_cluster` must fire), otherwise there is no structure to
# destabilize and the detector is silent by construction.
#
# Detection rule (ONTOLOGY.md sketch, v0)
# ---------------------------------------
# For each baseline cluster (as returned by `cluster_assignments` over
# the tail-window mean velocity) whose tail-window `local_r` clears
# `DOMINANT_CLUSTER_MIN_LOCAL_R = 0.9` — "there is coherence to lose":
#   For each node k in that cluster with at least 2 surviving members:
#     Run `sim.ablate.ablate_node(cfg, node=k)` into a scratch dir.
#     Compute tail-window `local_r` of the surviving cluster members
#     (baseline cluster minus {k}) on the ablated state.
#     Flag k as a bridge if `r_post < DOMINANT_CLUSTER_MIN_LOCAL_R`.
#
# The gate is the SAME 0.9 threshold `detect_phase_locked` and
# `detect_dominant_cluster` both use: a cluster is "coherent" when it
# would lock by the ensemble's own standard. The bridge claim is that
# ablating one node specifically makes the cluster stop qualifying —
# crossing the same line, in the other direction.
#
# Why re-use `cluster_assignments` instead of the `_dominant_cluster_
# evidence` cluster list
# -----------------------------------------------------------------
# The raw partition sees every node; the evidence helper filters to
# coherent clusters only. We want both clusters visible so that a
# later, subtler detector (e.g. "bridge between two partially-coherent
# groups") can inherit this scaffolding without API change. The
# applicability gate still comes from `_dominant_cluster_evidence`
# (we require it to be non-empty), so silent-baseline fixtures skip
# the whole ablation loop.
#
# Budget
# ------
# O(N) sim reruns per detector call, fully deterministic. At v0
# fixture sizes (N ≤ 8) this is cheap (~0.5 s per ablation → ~4 s
# worst case). No candidate-set pruning — the brief prefers simplicity
# over cleverness at current sizes.
#
# Confidence
# ----------
# Mean over bridge nodes of `(r_pre − r_post)` — "how much coherence
# was lost, per bridge, averaged over bridges". Scale is `[0, 1]` in
# principle; on the `regime_unstable_bridge` fixture a single bridge
# fires with a ~0.33 margin. `0.0` when the detector does not fire,
# matching the other detectors' convention.
#
# Per-node bridge identity
# ------------------------
# Lives behind `_unstable_bridge_evidence`, per the project's
# established pattern. `DetectionResult` stays the uniform shape.
UNSTABLE_BRIDGE_TAIL_S = DOMINANT_CLUSTER_TAIL_S  # reuse — same 3 s tail
UNSTABLE_BRIDGE_MIN_CLUSTER_SIZE = 3              # need ≥ 2 survivors after removing k


def _unstable_bridge_evidence(cfg, theta, phase_vel, control_rate_hz, *, work_dir=None):
    """Per-bridge evidence list for `detect_unstable_bridge`.

    Applies the baseline applicability gate (`dominant_cluster` must
    fire) first; returns `[]` if it fails. Otherwise iterates every
    baseline cluster with tail-window `local_r >=
    DOMINANT_CLUSTER_MIN_LOCAL_R` and size `>=
    UNSTABLE_BRIDGE_MIN_CLUSTER_SIZE`, running one ablation per member
    and recording the nodes whose removal drops the surviving-member
    `local_r` below the same gate:

        {"node": int, "cluster_idx": int,
         "cluster_members": np.ndarray[int],
         "baseline_local_r": float, "ablated_local_r": float,
         "drop": float}

    Parameters
    ----------
    cfg : dict
        Validated v0 config. Required — this detector runs counterfactual
        simulations via `sim.ablate.ablate_node` and therefore needs the
        scene + run spec, not just observable state.
    theta, phase_vel : np.ndarray, shape `(T, N)`
        Baseline state arrays from simulating `cfg`. The caller is
        expected to have already run the baseline; the detector does
        not re-simulate the baseline to save one sim.
    control_rate_hz : int
        Frames per second of the state arrays.
    work_dir : pathlib.Path | str | None, optional
        Scratch directory to write ablation artifact bundles into. If
        `None`, a fresh `TemporaryDirectory` is created and cleaned up
        at function exit. Pass an explicit path to keep ablated runs
        on disk for inspection.

    Shared by the detector itself and by tests that want to inspect
    which nodes contributed without re-running the simulator.
    """
    import tempfile
    from pathlib import Path

    # Late imports to keep `sim.detectors` loadable in environments
    # that don't have the full sim stack (e.g. partial unit tests).
    from sim.ablate import ablate_node

    theta = np.asarray(theta)
    phase_vel = np.asarray(phase_vel)
    if theta.ndim != 2 or phase_vel.shape != theta.shape:
        raise ValueError(
            "theta and phase_vel must share shape (T, N); "
            f"got {theta.shape}, {phase_vel.shape}"
        )

    baseline_ev = _dominant_cluster_evidence(theta, phase_vel, control_rate_hz)
    if not baseline_ev:
        return []

    T, _ = theta.shape
    tail_frames = min(int(UNSTABLE_BRIDGE_TAIL_S * control_rate_hz), T)
    if tail_frames < 1:
        return []

    v_tail = mean_phase_velocity(phase_vel, start_idx=-tail_frames)
    ca = cluster_assignments(v_tail, n_clusters=2)
    theta_tail = theta[-tail_frames:]

    def _run_loop(work_path):
        out = []
        for cidx, members in enumerate(ca.members):
            if members.size < UNSTABLE_BRIDGE_MIN_CLUSTER_SIZE:
                continue
            r_pre = float(local_kuramoto_order(theta_tail, members).mean())
            if r_pre < DOMINANT_CLUSTER_MIN_LOCAL_R:
                continue
            for k in members:
                k = int(k)
                survivors = np.array(
                    [int(m) for m in members if int(m) != k],
                    dtype=np.int64,
                )
                sub = work_path / f"ablate_c{cidx}_n{k}"
                ablate_node(cfg, sub, node=k)
                with np.load(sub / "state.npz") as st:
                    a_theta = st["theta"]
                r_post = float(
                    local_kuramoto_order(a_theta[-tail_frames:], survivors).mean()
                )
                if r_post < DOMINANT_CLUSTER_MIN_LOCAL_R:
                    out.append({
                        "node": k,
                        "cluster_idx": int(cidx),
                        "cluster_members": np.asarray(members),
                        "baseline_local_r": r_pre,
                        "ablated_local_r": r_post,
                        "drop": r_pre - r_post,
                    })
        return out

    if work_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            return _run_loop(Path(tmp))
    return _run_loop(Path(work_dir))


def detect_unstable_bridge(cfg, theta, phase_vel, control_rate_hz, *, work_dir=None):
    """Fire on nodes whose ablation collapses a baseline cluster's coherence.

    Defining condition
    ------------------
    Over the tail analysis window `[T − tail_frames, T)` with
    `tail_frames = UNSTABLE_BRIDGE_TAIL_S · control_rate_hz` (3 s):

    - `detect_dominant_cluster` must fire on the baseline run —
      applicability gate. Runs without any coherent cluster structure
      cannot produce a bridge.
    - For at least one baseline cluster (from `cluster_assignments(
      v_tail, n_clusters=2)`) whose size is `>=
      UNSTABLE_BRIDGE_MIN_CLUSTER_SIZE = 3` and tail-window `local_r
      >= DOMINANT_CLUSTER_MIN_LOCAL_R = 0.9`: there exists a member
      node `k` such that, on a counterfactual run produced by
      `sim.ablate.ablate_node(cfg, node=k)`, the tail-window `local_r`
      of the surviving members (baseline cluster minus {k}) falls
      *below* `DOMINANT_CLUSTER_MIN_LOCAL_R`.

    Parameters
    ----------
    cfg : dict
        Validated v0 config — the baseline scene. Required because
        ablations re-simulate from this config via `sim.ablate`.
    theta : np.ndarray, shape `(T, N)`
        Baseline phase trajectories.
    phase_vel : np.ndarray, shape `(T, N)`
        Baseline per-frame phase velocities.
    control_rate_hz : int
        Frames per second of the state arrays.
    work_dir : pathlib.Path | str | None, optional
        Scratch directory for ablation artifact bundles. `None` uses
        a short-lived `TemporaryDirectory`; pass a real path to keep
        the bundles for inspection.

    Returns
    -------
    DetectionResult
        `fired` is True iff at least one bridge node was found.
        `windows` is a single tail-window entry `[(T − tail_frames, T)]`
        — the analysis boundary, same as `detect_dominant_cluster`.
        `confidence` is the mean of `(baseline_local_r −
        ablated_local_r)` across bridge nodes — "how much coherence
        the ablation cost, averaged over bridges". Scale `[0, 1]` in
        principle. `longest_window_frames` is the tail-window length.

    Notes
    -----
    The public result deliberately collapses per-bridge detail into
    the standard `DetectionResult` shape — consistent with the other
    pair/cluster detectors. To inspect *which* nodes were flagged (and
    with what baseline/ablated coherence), call
    `_unstable_bridge_evidence` directly. The evidence helper is the
    canonical place per-node bridge identity lives.
    """
    T = int(np.asarray(theta).shape[0])
    tail_frames = min(int(UNSTABLE_BRIDGE_TAIL_S * control_rate_hz), T)
    evidence = _unstable_bridge_evidence(
        cfg, theta, phase_vel, control_rate_hz, work_dir=work_dir
    )
    if not evidence:
        return DetectionResult(
            fired=False, windows=[], confidence=0.0, longest_window_frames=0
        )
    mean_drop = float(np.mean([e["drop"] for e in evidence]))
    ws = T - tail_frames
    return DetectionResult(
        fired=True,
        windows=[(ws, T)],
        confidence=mean_drop,
        longest_window_frames=tail_frames,
    )
