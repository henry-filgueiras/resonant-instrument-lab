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
    kuramoto_order,
    local_kuramoto_order,
    mean_phase_velocity,
    split_by_largest_velocity_gap,
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

    # (2) per-node mean phase velocity splits bimodally.
    v_per_node = mean_phase_velocity(art["phase_vel"], start_idx=-n_tail)
    split = split_by_largest_velocity_gap(v_per_node)
    assert split.bimodal_ratio > 3.0, (
        f"expected bimodal velocity structure (gap >> next), "
        f"got gap={split.gap:.3f}, ratio={split.bimodal_ratio:.1f}"
    )

    # (3) the split is roughly balanced (4+4 expected, 3+5 tolerated).
    low_count = len(split.low_indices)
    high_count = len(split.high_indices)
    assert abs(low_count - high_count) <= 2, (
        f"expected roughly balanced split, got {low_count}+{high_count}"
    )

    # (4) each sub-cluster is internally coherent in the tail.
    r_lo = float(local_kuramoto_order(theta_tail, split.low_indices).mean())
    r_hi = float(local_kuramoto_order(theta_tail, split.high_indices).mean())
    assert r_lo > 0.8 and r_hi > 0.8, (
        f"expected intra-cluster lock (r > 0.8 each), got r_lo={r_lo:.3f}, r_hi={r_hi:.3f}"
    )

    print(
        f"ok: two_cluster — tail mean r = {r_mean:.4f}, split = {low_count}+{high_count}, "
        f"intra r = ({r_lo:.3f}, {r_hi:.3f}), bimodal ratio = {split.bimodal_ratio:.1f}"
    )


if __name__ == "__main__":
    test_locked_regime_reaches_phase_lock()
    test_two_cluster_regime_shows_bimodal_structure()
