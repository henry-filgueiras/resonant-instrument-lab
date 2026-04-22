#!/usr/bin/env python3
"""Smoke tests for the counterfactual perturbation paths (sim.ablate).

Proves each intervention is real and artifact-preserving:

- `ablate_node` (decouple + silence): running `regime_two_cluster` with
  one node decoupled changes the *other* nodes' phase trajectories,
  wipes the ablated node's pulses, and emits a valid `ablation.json`
  manifest — while leaving `(T, N)` artifact shapes intact.
- `nudge_node` (detune in place): running `regime_brittle_lock` with
  one node's natural frequency shifted changes phase trajectories
  globally (coupling is preserved so the perturbation propagates),
  preserves pulse firing on the nudged node, and emits a valid
  `nudge.json` manifest.

Both perturbations are deterministic: byte-identical `state.npz` across
repeated runs with the same `(config + seed + perturbation)` tuple.

Run directly:
    python tests/test_ablate.py
Or via pytest:
    python -m pytest tests/test_ablate.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from sim import ablate_node, load, nudge_node, simulate  # noqa: E402

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_state(run_dir):
    with np.load(Path(run_dir) / "state.npz") as st:
        return {k: st[k] for k in st.files}


def test_ablate_node_is_not_a_noop():
    """Ablation must change non-ablated node trajectories, not just silence one row."""
    cfg = load(CONFIGS_DIR / "regime_two_cluster.yaml")
    N = int(cfg["scene"]["N"])
    target = 0  # group-A node; removing it should perturb group A

    with tempfile.TemporaryDirectory() as tmp:
        baseline_dir = Path(tmp) / "baseline"
        ablated_dir = Path(tmp) / "ablated"

        simulate(cfg, baseline_dir, config_path=str(CONFIGS_DIR / "regime_two_cluster.yaml"))
        ablate_node(
            cfg,
            ablated_dir,
            node=target,
            config_path=str(CONFIGS_DIR / "regime_two_cluster.yaml"),
        )

        base = _load_state(baseline_dir)
        abl = _load_state(ablated_dir)

        # --- shape invariants: ablation preserves the (T, N) artifact contract.
        assert base["theta"].shape == abl["theta"].shape
        assert base["pulse_fired"].shape == abl["pulse_fired"].shape
        assert abl["theta"].shape[1] == N

        # --- the ablated node is silenced; the baseline fires real pulses there.
        n_baseline_pulses = int(base["pulse_fired"][:, target].sum())
        n_ablated_pulses = int(abl["pulse_fired"][:, target].sum())
        assert n_baseline_pulses > 0, (
            f"baseline sanity check: node {target} should pulse, got {n_baseline_pulses}"
        )
        assert n_ablated_pulses == 0, (
            f"ablated node {target} must have zero pulses, got {n_ablated_pulses}"
        )

        # --- load-bearing assertion: the *other* nodes' trajectories change.
        # If ablation were "post-hoc silencing" only, other nodes' theta would be
        # byte-identical to baseline. Decoupling means they evolve differently.
        other_mask = np.ones(N, dtype=bool)
        other_mask[target] = False
        base_theta_others = base["theta"][:, other_mask]
        abl_theta_others = abl["theta"][:, other_mask]
        max_abs_diff = float(np.max(np.abs(abl_theta_others - base_theta_others)))
        assert max_abs_diff > 1e-3, (
            f"expected non-trivial counterfactual effect on other nodes, "
            f"max|Δθ| = {max_abs_diff:.3e} (would indicate ablation was a no-op)"
        )

        # --- manifest file exists and carries the documented schema.
        manifest_path = ablated_dir / "ablation.json"
        assert manifest_path.is_file(), "ablation.json should exist in ablated run dir"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == 1
        assert manifest["semantics"] == "decouple_and_silence"
        assert manifest["ablated_nodes"] == [target]
        assert manifest["baseline_config_path"].endswith("regime_two_cluster.yaml")

        # --- baseline runs do NOT emit ablation.json (presence = counterfactual).
        assert not (baseline_dir / "ablation.json").exists(), (
            "baseline run must not emit ablation.json"
        )

        print(
            f"ok: ablate_node — baseline pulses@n{target}={n_baseline_pulses}, "
            f"ablated pulses@n{target}={n_ablated_pulses}, "
            f"max|Δθ| on other nodes = {max_abs_diff:.3f} rad"
        )


def test_ablate_node_determinism():
    """Byte-identical state.npz across two runs of the same ablation."""
    cfg = load(CONFIGS_DIR / "regime_two_cluster.yaml")

    with tempfile.TemporaryDirectory() as tmp:
        a_dir = Path(tmp) / "a"
        b_dir = Path(tmp) / "b"
        ablate_node(cfg, a_dir, node=3)
        ablate_node(cfg, b_dir, node=3)
        a_bytes = (a_dir / "state.npz").read_bytes()
        b_bytes = (b_dir / "state.npz").read_bytes()
        assert a_bytes == b_bytes, "ablated state.npz must be byte-identical across runs"
    print("ok: ablate_node determinism — state.npz byte-identical across repeats")


def test_nudge_node_is_not_a_noop():
    """A whole-run frequency nudge must propagate through coupling and
    shift the nudged node's phase progression, while preserving
    artifact shapes, pulse firing on the nudged node, and emitting a
    valid nudge.json manifest."""
    cfg = load(CONFIGS_DIR / "regime_brittle_lock.yaml")
    N = int(cfg["scene"]["N"])
    target = 4          # outer-ω node — the nudge will break the lock
    delta_hz = 0.25     # same magnitude detect_brittle_lock uses

    with tempfile.TemporaryDirectory() as tmp:
        baseline_dir = Path(tmp) / "baseline"
        nudged_dir = Path(tmp) / "nudged"

        simulate(cfg, baseline_dir, config_path=str(CONFIGS_DIR / "regime_brittle_lock.yaml"))
        nudge_node(
            cfg,
            nudged_dir,
            node=target,
            delta_hz=delta_hz,
            config_path=str(CONFIGS_DIR / "regime_brittle_lock.yaml"),
        )

        base = _load_state(baseline_dir)
        ndg = _load_state(nudged_dir)

        # --- shape invariants: nudge preserves the (T, N) artifact contract.
        assert base["theta"].shape == ndg["theta"].shape
        assert base["pulse_fired"].shape == ndg["pulse_fired"].shape
        assert ndg["theta"].shape[1] == N

        # --- the nudged node KEEPS firing pulses (unlike ablation).
        # Coupling is preserved; the node still crosses phase zero.
        n_nudged_pulses = int(ndg["pulse_fired"][:, target].sum())
        assert n_nudged_pulses > 0, (
            f"nudged node {target} should still pulse (coupling preserved), "
            f"got {n_nudged_pulses} pulses"
        )

        # --- nudged node's theta evolves differently — it's running at a
        # higher natural frequency and coupling isn't strong enough to
        # fully pull it back on this marginal-lock fixture.
        self_diff = float(np.max(np.abs(ndg["theta"][:, target] - base["theta"][:, target])))
        assert self_diff > 1e-2, (
            f"expected nudged node theta to diverge from baseline; "
            f"max|Δθ_self| = {self_diff:.3e}"
        )

        # --- OTHER nodes must also feel the perturbation (coupling carries it).
        # If nudge were only a local effect, other nodes would match baseline.
        other_mask = np.ones(N, dtype=bool)
        other_mask[target] = False
        max_abs_other = float(np.max(np.abs(ndg["theta"][:, other_mask] - base["theta"][:, other_mask])))
        assert max_abs_other > 1e-3, (
            f"expected coupling to propagate the nudge to other nodes; "
            f"max|Δθ_other| = {max_abs_other:.3e} (would indicate nudge stayed local)"
        )

        # --- manifest file exists and carries the documented schema.
        manifest_path = nudged_dir / "nudge.json"
        assert manifest_path.is_file(), "nudge.json should exist in nudged run dir"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == 1
        assert manifest["semantics"] == "detune_in_place"
        assert manifest["node"] == target
        assert manifest["delta_hz"] == delta_hz
        assert manifest["baseline_config_path"].endswith("regime_brittle_lock.yaml")

        # --- baseline runs do NOT emit nudge.json (presence = counterfactual).
        assert not (baseline_dir / "nudge.json").exists(), (
            "baseline run must not emit nudge.json"
        )
        # --- nudged runs do not accidentally emit ablation.json either.
        assert not (nudged_dir / "ablation.json").exists(), (
            "nudged run must not emit ablation.json (different perturbation kind)"
        )

        print(
            f"ok: nudge_node — node {target} +{delta_hz:.2f} Hz, pulses preserved "
            f"({n_nudged_pulses}), max|Δθ_self|={self_diff:.3f} rad, "
            f"max|Δθ_other|={max_abs_other:.3f} rad"
        )


def test_nudge_node_determinism():
    """Byte-identical state.npz across two runs of the same nudge."""
    cfg = load(CONFIGS_DIR / "regime_brittle_lock.yaml")

    with tempfile.TemporaryDirectory() as tmp:
        a_dir = Path(tmp) / "a"
        b_dir = Path(tmp) / "b"
        nudge_node(cfg, a_dir, node=2, delta_hz=0.25)
        nudge_node(cfg, b_dir, node=2, delta_hz=0.25)
        a_bytes = (a_dir / "state.npz").read_bytes()
        b_bytes = (b_dir / "state.npz").read_bytes()
        assert a_bytes == b_bytes, "nudged state.npz must be byte-identical across runs"
    print("ok: nudge_node determinism — state.npz byte-identical across repeats")


if __name__ == "__main__":
    test_ablate_node_is_not_a_noop()
    test_ablate_node_determinism()
    test_nudge_node_is_not_a_noop()
    test_nudge_node_determinism()
