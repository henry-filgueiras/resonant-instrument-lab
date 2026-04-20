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
    PHASE_LOCKED_MIN_S,
    detect_drifting,
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


if __name__ == "__main__":
    test_detect_phase_locked_across_regimes()
    test_detect_drifting_across_regimes()
