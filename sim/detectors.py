"""First semantic-detector layer.

Thin, deterministic detectors built on top of `sim.derived` primitives.
Two global-coherence detectors plus two pairwise rhythm detectors:

- `detect_phase_locked`  — sustained high global coherence.
- `detect_drifting`      — sustained low global coherence.
- `detect_phase_beating` — near-frequency node pairs whose unwrapped
  phase difference drifts linearly across the analysis window.
- `detect_flam`          — frequency-locked node pairs whose pulse
  times land repeatedly with a small but nonzero offset and stand
  alone (no other node pulses between them).

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

from sim.derived import kuramoto_order, pulse_times_of, sustained_windows


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
