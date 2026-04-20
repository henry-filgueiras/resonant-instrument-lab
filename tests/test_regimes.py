#!/usr/bin/env python3
"""Smoke tests for regime reference configs.

Each test runs a reference config through `sim.simulate` into a tempdir,
reads the exported state, and asserts a simple numerical signature of
the regime label using the shared helpers in `sim.derived`. These are
simulator sanity checks, not detectors — they consume the same pure
state-derivation primitives that the detector layer will use later.

Run directly:
    python tests/test_regimes.py
Or via pytest:
    python -m pytest tests/
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from sim import load, simulate  # noqa: E402
from sim.derived import (  # noqa: E402
    cluster_assignments,
    kuramoto_order,
    local_kuramoto_order,
    mean_phase_velocity,
    sustained_windows,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _run(config_path):
    cfg = load(config_path)
    with tempfile.TemporaryDirectory() as tmp:
        simulate(cfg, tmp, config_path=str(config_path))
        with np.load(Path(tmp) / "state.npz") as state:
            artifacts = {k: state[k] for k in state.files}
    return cfg, artifacts


# ---------------------------------------------------------------------------
# LOCKED — sustained global coherence above the phase_locked threshold.

def test_locked_regime_reaches_phase_lock():
    cfg, art = _run(CONFIGS_DIR / "regime_locked.yaml")
    rate = cfg["run"]["control_rate_hz"]
    r = kuramoto_order(art["theta"])
    n_last = int(1.0 * rate)
    r_tail = r[-n_last:]
    r_mean = float(r_tail.mean())
    r_min = float(r_tail.min())
    assert r_mean > 0.9, f"expected sustained lock: last-1s mean r > 0.9, got {r_mean:.4f}"
    assert r_min > 0.85, f"expected stable lock: last-1s min r > 0.85, got {r_min:.4f}"
    print(f"ok: locked — last-1s mean r = {r_mean:.4f}, min r = {r_min:.4f}")


# ---------------------------------------------------------------------------
# TWO_CLUSTER — intermediate global coherence + two distinct velocity groups.

def test_two_cluster_regime_shows_bimodal_structure():
    cfg, art = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    rate = cfg["run"]["control_rate_hz"]
    tail_s = 3.0
    n_tail = int(tail_s * rate)
    theta_tail = art["theta"][-n_tail:]

    # (1) global r sits in a middle band — neither drift nor global lock.
    r = kuramoto_order(theta_tail)
    r_mean = float(r.mean())
    assert 0.35 < r_mean < 0.85, (
        f"expected middle-band coherence (0.35 < tail mean r < 0.85), got {r_mean:.4f}"
    )

    # (2) per-node mean phase velocity splits bimodally into two coherent groups.
    v_per_node = mean_phase_velocity(art["phase_vel"], start_idx=-n_tail)
    clusters = cluster_assignments(v_per_node, n_clusters=2)
    assert clusters.separability_ratio > 3.0, (
        f"expected clean bimodal split (selected gap >> remaining gaps), "
        f"got separability_ratio = {clusters.separability_ratio:.1f}"
    )

    # (3) the split is roughly balanced (4+4 expected, 3+5 tolerated).
    slow, fast = clusters.members
    slow_count, fast_count = len(slow), len(fast)
    assert abs(slow_count - fast_count) <= 2, (
        f"expected roughly balanced split, got {slow_count}+{fast_count}"
    )

    # (4) each sub-cluster is internally coherent in the tail.
    r_slow = float(local_kuramoto_order(theta_tail, slow).mean())
    r_fast = float(local_kuramoto_order(theta_tail, fast).mean())
    assert r_slow > 0.8 and r_fast > 0.8, (
        f"expected intra-cluster lock (r > 0.8 each), got r_slow={r_slow:.3f}, r_fast={r_fast:.3f}"
    )

    print(
        f"ok: two_cluster — tail mean r = {r_mean:.4f}, split = {slow_count}+{fast_count}, "
        f"intra r = ({r_slow:.3f}, {r_fast:.3f}), separability = {clusters.separability_ratio:.1f}"
    )


# ---------------------------------------------------------------------------
# DRIFTING — sustained low global coherence, separated from locked and two_cluster.

def test_drifting_regime_sustains_low_coherence():
    cfg, art = _run(CONFIGS_DIR / "regime_drifting.yaml")
    rate = cfg["run"]["control_rate_hz"]
    r = kuramoto_order(art["theta"])

    # (1) Threshold r(t) into a "low coherence" indicator and look for a
    # sustained window. Hysteresis absorbs the brief sub-second upward
    # spikes that happen when drifting phases pass near each other;
    # min_duration rejects the half-second valleys seen in two_cluster.
    thr = 0.5
    low = r < thr
    windows = sustained_windows(
        low,
        min_duration_frames=int(2.0 * rate),
        hysteresis_frames=int(0.25 * rate),
    )
    assert windows, (
        f"expected a sustained low-coherence window (r < {thr}); "
        f"got none (mean r = {r.mean():.3f}, max r = {r.max():.3f})"
    )
    longest = max(e - s for s, e in windows)
    assert longest >= int(3.0 * rate), (
        f"expected a ≥ 3 s sustained low-coherence window, got {longest / rate:.2f} s"
    )

    # (2) Pin tail mean well below two_cluster's middle band (observed ≈ 0.63)
    # and far below locked's near-unity coherence. This is the regime
    # separator — the same `sustained_windows` call above returns 0 frames
    # for locked (r stays > 0.95) and < 1.0 s for two_cluster (short valleys).
    tail_s = 2.0
    n_tail = int(tail_s * rate)
    r_tail_mean = float(r[-n_tail:].mean())
    assert r_tail_mean < 0.40, (
        f"expected tail mean r < 0.40 to separate drifting from two_cluster's "
        f"middle band, got {r_tail_mean:.3f}"
    )

    frac_low = float(low.mean())
    print(
        f"ok: drifting — tail mean r = {r_tail_mean:.4f}, frac(r<{thr}) = {frac_low:.3f}, "
        f"{len(windows)} sustained window(s), longest = {longest} f ({longest/rate:.2f} s)"
    )


if __name__ == "__main__":
    test_locked_regime_reaches_phase_lock()
    test_two_cluster_regime_shows_bimodal_structure()
    test_drifting_regime_sustains_low_coherence()
