#!/usr/bin/env python3
"""Smoke tests for regime reference configs.

Each test runs a reference config through sim.simulate into a tempdir,
reads the exported state, and asserts a simple numerical signature of
the regime label. These are simulator sanity checks, not detectors.

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

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _run(config_path):
    cfg = load(config_path)
    with tempfile.TemporaryDirectory() as tmp:
        simulate(cfg, tmp, config_path=str(config_path))
        with np.load(Path(tmp) / "state.npz") as state:
            artifacts = {k: state[k] for k in state.files}
    return cfg, artifacts


def _kuramoto(theta_T_N):
    """Per-frame global order parameter r(t), shape (T,)."""
    return np.abs(np.mean(np.exp(1j * theta_T_N), axis=-1))


# ---------------------------------------------------------------------------
# LOCKED — sustained global coherence above the phase_locked threshold.

def test_locked_regime_reaches_phase_lock():
    cfg, art = _run(CONFIGS_DIR / "regime_locked.yaml")
    rate = cfg["run"]["control_rate_hz"]
    r = _kuramoto(art["theta"])
    n_last = int(1.0 * rate)
    r_mean = float(r[-n_last:].mean())
    r_min = float(r[-n_last:].min())
    assert r_mean > 0.9, f"expected sustained lock: last-1s mean r > 0.9, got {r_mean:.4f}"
    assert r_min > 0.85, f"expected stable lock: last-1s min r > 0.85, got {r_min:.4f}"
    print(f"ok: locked — last-1s mean r = {r_mean:.4f}, min r = {r_min:.4f}")


# ---------------------------------------------------------------------------
# TWO_CLUSTER — intermediate global coherence + two velocity groups.

def test_two_cluster_regime_shows_bimodal_structure():
    cfg, art = _run(CONFIGS_DIR / "regime_two_cluster.yaml")
    rate = cfg["run"]["control_rate_hz"]
    N = cfg["scene"]["N"]

    # Tail window chosen at ~2.6× the expected drift period (1.13 s).
    tail_s = 3.0
    n_tail = int(tail_s * rate)
    theta_tail = art["theta"][-n_tail:]
    pv_tail = art["phase_vel"][-n_tail:]

    # (1) Global r sits in a middle band — neither drift nor global lock.
    r = _kuramoto(theta_tail)
    r_mean = float(r.mean())
    assert 0.35 < r_mean < 0.85, (
        f"expected middle-band coherence (0.35 < tail mean r < 0.85), got {r_mean:.4f}"
    )

    # (2) Per-node mean phase velocity splits bimodally: the biggest gap
    # between sorted mean velocities dominates the other gaps.
    v_per_node = pv_tail.mean(axis=0)
    order = np.argsort(v_per_node)
    v_sorted = v_per_node[order]
    gaps = np.diff(v_sorted)
    split_idx = int(np.argmax(gaps))
    max_gap = float(gaps[split_idx])
    other_gaps = np.delete(gaps, split_idx)
    max_other_gap = float(other_gaps.max()) if other_gaps.size else 0.0
    ratio = max_gap / max(max_other_gap, 1e-9)
    assert ratio > 3.0, (
        f"expected bimodal velocity structure (max gap >> others), "
        f"got max_gap={max_gap:.3f}, next={max_other_gap:.3f}, ratio={ratio:.1f}"
    )

    # (3) The split is roughly balanced (4+4 expected; 3+5 tolerated).
    low_count = split_idx + 1
    high_count = N - low_count
    assert abs(low_count - high_count) <= 2, (
        f"expected roughly balanced split, got {low_count}+{high_count}"
    )

    # (4) Each sub-cluster is internally coherent in the tail.
    lo_members = list(order[:low_count])
    hi_members = list(order[low_count:])
    r_lo = float(np.abs(np.mean(np.exp(1j * theta_tail[:, lo_members]), axis=-1)).mean())
    r_hi = float(np.abs(np.mean(np.exp(1j * theta_tail[:, hi_members]), axis=-1)).mean())
    assert r_lo > 0.8 and r_hi > 0.8, (
        f"expected intra-cluster lock (r > 0.8 each), got r_lo={r_lo:.3f}, r_hi={r_hi:.3f}"
    )

    print(
        f"ok: two_cluster — tail mean r = {r_mean:.4f}, split = {low_count}+{high_count}, "
        f"intra r = ({r_lo:.3f}, {r_hi:.3f}), bimodal ratio = {ratio:.1f}"
    )


if __name__ == "__main__":
    test_locked_regime_reaches_phase_lock()
    test_two_cluster_regime_shows_bimodal_structure()
