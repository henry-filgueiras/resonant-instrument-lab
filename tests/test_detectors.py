#!/usr/bin/env python3
"""Detector-level tests.

Exercise the semantic detectors in `sim.detectors` against the three
regime fixtures (`locked`, `drifting`, `two_cluster`). Each detector
must fire on its namesake regime and stay silent on the other two.

These are not simulator smoke tests (those live in `test_regimes.py`) —
the simulator is trusted here; what we're checking is that the detector
thresholds and sustained-window settings correctly discriminate the
regimes that the fixtures already validate as physically distinct.

Run directly:
    python tests/test_detectors.py
Or via pytest:
    python -m pytest tests/
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from sim import load, simulate  # noqa: E402
from sim.detectors import (  # noqa: E402
    DRIFTING_MIN_S,
    FLAM_DW_MAX,
    FLAM_MAX_OFFSET_S,
    FLAM_MIN_EVENTS,
    FLAM_MIN_OFFSET_S,
    FLAM_WARMUP_S,
    PHASE_BEATING_DW_MAX,
    PHASE_BEATING_DW_MIN,
    PHASE_BEATING_MIN_R2,
    PHASE_BEATING_WARMUP_S,
    PHASE_LOCKED_MIN_S,
    _flam_pair_evidence,
    _phase_beating_pair_evidence,
    detect_drifting,
    detect_flam,
    detect_phase_beating,
    detect_phase_locked,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _run(config_path):
    cfg = load(config_path)
    with tempfile.TemporaryDirectory() as tmp:
        simulate(cfg, tmp, config_path=str(config_path))
        with np.load(Path(tmp) / "state.npz") as state:
            artifacts = {k: state[k] for k in state.files}
    return cfg, artifacts


def _assert_silent(result, detector_name, regime_name):
    assert not result.fired, (
        f"{detector_name} should not fire on {regime_name}, but did: "
        f"{len(result.windows)} window(s), longest = {result.longest_window_frames} frames"
    )
    assert result.windows == [], f"{detector_name} on {regime_name}: expected no windows"
    assert result.confidence == 0.0, (
        f"{detector_name} on {regime_name}: expected confidence 0.0 for silent detector, "
        f"got {result.confidence}"
    )
    assert result.longest_window_frames == 0


# ---------------------------------------------------------------------------
# phase_locked — fires only on the locked regime.

def test_detect_phase_locked_across_regimes():
    # (a) fires cleanly on locked
    cfg, art = _run(CONFIGS_DIR / "regime_locked.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_phase_locked(art["theta"], rate)
    assert result.fired, "phase_locked should fire on regime_locked"
    assert result.longest_window_frames >= int(PHASE_LOCKED_MIN_S * rate), (
        f"expected a >= {PHASE_LOCKED_MIN_S}s lock window, got "
        f"{result.longest_window_frames / rate:.2f}s"
    )
    assert result.confidence > 0.05, (
        f"expected solid margin above R_LOCK (observed r ≈ 0.995 → margin ≈ 0.09), "
        f"got confidence {result.confidence:.4f}"
    )

    # (b) silent on drifting — r never reaches R_LOCK
    cfg_d, art_d = _run(CONFIGS_DIR / "regime_drifting.yaml")
    _assert_silent(
        detect_phase_locked(art_d["theta"], cfg_d["run"]["control_rate_hz"]),
        "phase_locked", "regime_drifting",
    )

    # (c) silent on two_cluster — r spikes above R_LOCK but never sustains the window
    cfg_t, art_t = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    _assert_silent(
        detect_phase_locked(art_t["theta"], cfg_t["run"]["control_rate_hz"]),
        "phase_locked", "regime_two_cluster",
    )

    print(
        f"ok: phase_locked — locked fired (conf={result.confidence:.4f}, "
        f"longest={result.longest_window_frames/rate:.2f}s); drifting & two_cluster silent"
    )


# ---------------------------------------------------------------------------
# drifting — fires only on the drifting regime.

def test_detect_drifting_across_regimes():
    # (a) fires cleanly on drifting
    cfg, art = _run(CONFIGS_DIR / "regime_drifting.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_drifting(art["theta"], rate)
    assert result.fired, "drifting should fire on regime_drifting"
    assert result.longest_window_frames >= int(DRIFTING_MIN_S * rate), (
        f"expected a >= {DRIFTING_MIN_S}s low-coherence window, got "
        f"{result.longest_window_frames / rate:.2f}s"
    )
    assert result.confidence > 0.10, (
        f"expected mean r well below R_DRIFT (observed r ≈ 0.32 → margin ≈ 0.18), "
        f"got confidence {result.confidence:.4f}"
    )

    # (b) silent on locked — r is pinned near 1.0
    cfg_l, art_l = _run(CONFIGS_DIR / "regime_locked.yaml")
    _assert_silent(
        detect_drifting(art_l["theta"], cfg_l["run"]["control_rate_hz"]),
        "drifting", "regime_locked",
    )

    # (c) silent on two_cluster — low-r valleys stay below the 2.0s window minimum
    cfg_t, art_t = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    _assert_silent(
        detect_drifting(art_t["theta"], cfg_t["run"]["control_rate_hz"]),
        "drifting", "regime_two_cluster",
    )

    print(
        f"ok: drifting — drifting fired (conf={result.confidence:.4f}, "
        f"longest={result.longest_window_frames/rate:.2f}s); locked & two_cluster silent"
    )


# ---------------------------------------------------------------------------
# phase_beating — fires only on the dedicated beating fixture.
#
# The three original regime fixtures are all negatives for this detector:
#   - locked: every pair sits at |Δω̄| ≈ 0, below the lower velocity gate.
#   - drifting: the smallest candidate pair has |Δω̄| ≈ 1.59 rad/s, above
#     the 1.5 rad/s "close in mean phase velocity" upper gate — the
#     drifting regime has genuinely structured pairwise drift, but the
#     pairs are no longer near-frequency by this detector's definition.
#   - two_cluster: intra-cluster pairs are locked (|Δω̄| ≈ 0), inter-
#     cluster pairs sit at ~5.6 rad/s (above the upper gate).
# `regime_phase_beating` is the single positive fixture. Its pair (0,1)
# sits at |Δω̄| ≈ 1.22 rad/s with beat period ~5.1 s; the 12 s run covers
# ~2 full cycles of the 2π coverage requirement.

def test_detect_phase_beating_across_regimes():
    # (a) fires on the dedicated positive fixture
    cfg, art = _run(CONFIGS_DIR / "regime_phase_beating.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_phase_beating(
        art["theta"], art["phase_vel"], art["pulse_fired"], rate
    )
    assert result.fired, "phase_beating should fire on regime_phase_beating"
    # Analysis window is the run minus the warmup skip.
    expected_window_frames = art["theta"].shape[0] - int(PHASE_BEATING_WARMUP_S * rate)
    assert result.longest_window_frames == expected_window_frames
    assert result.windows == [(int(PHASE_BEATING_WARMUP_S * rate), art["theta"].shape[0])]
    # R² margin above the 0.9 floor; the fixture's pair sits near R² ≈ 0.998.
    assert result.confidence > 0.05, (
        f"expected R² margin > 0.05 above floor {PHASE_BEATING_MIN_R2}, "
        f"got confidence {result.confidence:.4f}"
    )

    # Confirm exactly the (0, 1) pair contributed — the distractor nodes
    # sit too far away in frequency to enter the candidate band.
    evidence = _phase_beating_pair_evidence(
        art["theta"], art["phase_vel"], art["pulse_fired"], rate
    )
    assert len(evidence) == 1, f"expected one qualifying pair, got {len(evidence)}: {evidence}"
    e = evidence[0]
    assert (e["i"], e["j"]) == (0, 1)
    assert PHASE_BEATING_DW_MIN <= abs(e["dw"]) <= PHASE_BEATING_DW_MAX
    assert e["r2"] >= PHASE_BEATING_MIN_R2

    # (b) silent on locked — every pair has Δω̄ ≈ 0, below lower gate
    cfg_l, art_l = _run(CONFIGS_DIR / "regime_locked.yaml")
    _assert_silent(
        detect_phase_beating(
            art_l["theta"], art_l["phase_vel"], art_l["pulse_fired"],
            cfg_l["run"]["control_rate_hz"],
        ),
        "phase_beating", "regime_locked",
    )

    # (c) silent on drifting — smallest candidate |Δω̄| ≈ 1.59 rad/s is
    # above the upper gate; the drifting regime has structured pairwise
    # drift but its pairs are not "close in mean phase velocity" by this
    # detector's threshold policy.
    cfg_d, art_d = _run(CONFIGS_DIR / "regime_drifting.yaml")
    _assert_silent(
        detect_phase_beating(
            art_d["theta"], art_d["phase_vel"], art_d["pulse_fired"],
            cfg_d["run"]["control_rate_hz"],
        ),
        "phase_beating", "regime_drifting",
    )

    # (d) silent on two_cluster — intra-cluster pairs locked (|Δω̄| ≈ 0),
    # inter-cluster pairs at ~5.6 rad/s (above upper gate).
    cfg_t, art_t = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    _assert_silent(
        detect_phase_beating(
            art_t["theta"], art_t["phase_vel"], art_t["pulse_fired"],
            cfg_t["run"]["control_rate_hz"],
        ),
        "phase_beating", "regime_two_cluster",
    )

    print(
        f"ok: phase_beating — phase_beating fixture fired "
        f"(conf={result.confidence:.4f}, |Δω̄|={abs(e['dw']):.3f} rad/s, "
        f"beat~{(2 * np.pi) / abs(e['dw']):.2f} s, R²={e['r2']:.4f}); "
        f"locked, drifting, two_cluster silent"
    )


# ---------------------------------------------------------------------------
# flam — fires only on the dedicated flam fixture.
#
# The four earlier regime fixtures are all negatives for this detector:
#   - regime_locked: the extreme pair in an 8-node lock produces ~20 ms
#     offsets (inside the flam band) but intermediate nodes fire between
#     them, caught by the isolation margin.
#   - regime_drifting: every candidate pair has |Δω̄| > 1.5 rad/s,
#     above the 0.25 rad/s velocity gate.
#   - regime_two_cluster: the leading intra-cluster pair fires 15 ms
#     apart with the right velocity match, but the rest of the cluster
#     fires 5–10 ms just outside the pair window, caught by the
#     isolation margin.
#   - regime_phase_beating: the beating pair sits at |Δω̄| ≈ 1.22 rad/s,
#     above the velocity gate; it drifts through all offsets rather
#     than holding one.
# `regime_flam` is the single positive fixture: two nodes at 2.00 /
# 2.20 Hz locked together with a ~33 ms phase offset, two distractors
# at 4 / 5 Hz placed far away.

def test_detect_flam_across_regimes():
    # (a) fires cleanly on the dedicated positive fixture
    cfg, art = _run(CONFIGS_DIR / "regime_flam.yaml")
    rate = cfg["run"]["control_rate_hz"]
    result = detect_flam(art["phase_vel"], art["pulse_fired"], rate)
    assert result.fired, "flam should fire on regime_flam"
    expected_window_frames = art["pulse_fired"].shape[0] - int(FLAM_WARMUP_S * rate)
    assert result.longest_window_frames == expected_window_frames
    assert result.windows == [
        (int(FLAM_WARMUP_S * rate), art["pulse_fired"].shape[0])
    ]
    # 11 qualifying events vs 6 minimum → margin ≈ (11-6)/6 ≈ 0.83.
    assert result.confidence > 0.3, (
        f"expected a comfortable margin over {FLAM_MIN_EVENTS}-event floor, "
        f"got confidence {result.confidence:.4f}"
    )

    # Exactly the (0, 1) pair contributes; offset is in band and consistent.
    evidence = _flam_pair_evidence(art["phase_vel"], art["pulse_fired"], rate)
    assert len(evidence) == 1, f"expected one qualifying pair, got {len(evidence)}: {evidence}"
    e = evidence[0]
    assert (e["i"], e["j"]) == (0, 1)
    assert abs(e["dw"]) <= FLAM_DW_MAX
    assert FLAM_MIN_OFFSET_S <= e["median_abs_offset_s"] <= FLAM_MAX_OFFSET_S
    assert e["n_events"] >= FLAM_MIN_EVENTS

    # (b) silent on locked — extreme pair has crowd members inside margin
    cfg_l, art_l = _run(CONFIGS_DIR / "regime_locked.yaml")
    _assert_silent(
        detect_flam(art_l["phase_vel"], art_l["pulse_fired"],
                    cfg_l["run"]["control_rate_hz"]),
        "flam", "regime_locked",
    )

    # (c) silent on drifting — all candidate pairs above the velocity gate
    cfg_d, art_d = _run(CONFIGS_DIR / "regime_drifting.yaml")
    _assert_silent(
        detect_flam(art_d["phase_vel"], art_d["pulse_fired"],
                    cfg_d["run"]["control_rate_hz"]),
        "flam", "regime_drifting",
    )

    # (d) silent on two_cluster — intra-cluster leading pair has neighbors
    # firing just outside the pair window, inside the isolation margin
    cfg_t, art_t = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    _assert_silent(
        detect_flam(art_t["phase_vel"], art_t["pulse_fired"],
                    cfg_t["run"]["control_rate_hz"]),
        "flam", "regime_two_cluster",
    )

    # (e) silent on phase_beating — beating pair |Δω̄| > velocity gate
    cfg_b, art_b = _run(CONFIGS_DIR / "regime_phase_beating.yaml")
    _assert_silent(
        detect_flam(art_b["phase_vel"], art_b["pulse_fired"],
                    cfg_b["run"]["control_rate_hz"]),
        "flam", "regime_phase_beating",
    )

    print(
        f"ok: flam — flam fixture fired "
        f"(conf={result.confidence:.4f}, |Δω̄|={abs(e['dw']):.3f} rad/s, "
        f"median offset={e['median_abs_offset_s']*1000:.1f} ms, "
        f"n_events={e['n_events']}); locked, drifting, two_cluster, "
        f"phase_beating silent"
    )


if __name__ == "__main__":
    test_detect_phase_locked_across_regimes()
    test_detect_drifting_across_regimes()
    test_detect_phase_beating_across_regimes()
    test_detect_flam_across_regimes()
