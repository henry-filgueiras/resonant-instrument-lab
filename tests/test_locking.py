#!/usr/bin/env python3
"""Smoke test: the LOCKED regime config reaches sustained phase lock.

Runs configs/regime_locked.yaml through sim.simulate, then asserts that
the last-second mean Kuramoto order parameter r sits comfortably above
the `phase_locked` threshold of 0.9 used throughout DESIGN_V0 / ONTOLOGY.

Invoke as:
    python tests/test_locking.py          # exit 0 on pass
or as a pytest discovery:
    python -m pytest tests/
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from sim import load, simulate  # noqa: E402

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "regime_locked.yaml"
R_THRESHOLD = 0.9  # matches the `phase_locked` detector's r_lock (ONTOLOGY.md)
LAST_WINDOW_S = 1.0


def kuramoto_order(theta_T_N):
    """Per-frame r(t), shape (T,)."""
    return np.abs(np.mean(np.exp(1j * theta_T_N), axis=-1))


def test_locked_regime_reaches_phase_lock():
    cfg = load(CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        simulate(cfg, tmp, config_path=str(CONFIG))
        with np.load(Path(tmp) / "state.npz") as state:
            theta = state["theta"]

    r = kuramoto_order(theta)
    n_last = int(LAST_WINDOW_S * cfg["run"]["control_rate_hz"])
    r_final_mean = float(r[-n_last:].mean())
    r_final_min = float(r[-n_last:].min())

    assert r_final_mean > R_THRESHOLD, (
        f"expected sustained lock: last-{LAST_WINDOW_S}s mean r > {R_THRESHOLD}, "
        f"got {r_final_mean:.4f}"
    )
    assert r_final_min > R_THRESHOLD - 0.05, (
        f"expected stable lock: last-{LAST_WINDOW_S}s min r > {R_THRESHOLD - 0.05}, "
        f"got {r_final_min:.4f}"
    )
    print(
        f"ok: last-{LAST_WINDOW_S}s mean r = {r_final_mean:.4f}, "
        f"min r = {r_final_min:.4f} (threshold {R_THRESHOLD})"
    )


if __name__ == "__main__":
    test_locked_regime_reaches_phase_lock()
