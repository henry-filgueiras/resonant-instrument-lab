#!/usr/bin/env python3
"""Smoke test for the node-ablation counterfactual path (sim.ablate).

Proves the intervention is real: running `regime_two_cluster` with one
node decoupled+silenced changes the *other* nodes' phase trajectories
and wipes the ablated node's pulses, while leaving the artifact shape
contracts intact and emitting a valid `ablation.json` manifest.

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

from sim import ablate_node, load, simulate  # noqa: E402

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


if __name__ == "__main__":
    test_ablate_node_is_not_a_noop()
    test_ablate_node_determinism()
