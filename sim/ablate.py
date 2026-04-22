"""Counterfactual (intervention-grounded) run entry points.

`sim.ablate` is the project's single, narrow counterfactual surface
(DESIGN_V0.md §9.9, §10.1). Each entry point wraps `sim.garden.simulate`
with one specific, deterministic perturbation and writes a small JSON
manifest sibling so the artifact bundle self-documents which
intervention produced it. Baseline runs emit none of those sibling
files; their presence is itself the marker "this is a counterfactual
artifact bundle".

Semantics of `ablate_node(k)` — "decouple and silence":

- Node `k` is removed from the coupling graph: every `K[k, j]` and
  `K[j, k]` is zeroed, so node `k` neither drives nor is driven by
  any other node for the whole run.
- `pulse_fired[:, k]` is forced to all-False — the node emits no
  audible rhythmic events.
- Node `k`'s `theta` continues to integrate under its own natural
  frequency (and noise if `eta > 0`). This keeps `state.npz` at a
  constant `(T, N)` shape so detectors written against baseline
  artifacts also consume ablated artifacts without shape branching.
- The resulting trajectories of the *other* nodes are the honest
  counterfactual — "what would have happened without node k" — and
  that is the load-bearing signal counterfactual detectors will key on.

Semantics of `nudge_node(k, delta_hz)` — "detune in place":

- Node `k`'s natural frequency is shifted by `delta_hz` Hz for the
  whole run (applied at t=0, held constant thereafter). Every other
  parameter — topology, coupling matrix, noise seed, `pulse_fired`
  derivation — is untouched.
- The node remains fully coupled: the network can still capture it
  back into synchrony (a robust lock) or fail to (a brittle lock).
  This is the load-bearing distinction — the intervention probes
  lock *robustness*, not causal removal.
- `(T, N)` artifact shapes are preserved; `pulse_fired[:, k]` is the
  node's own zero-crossings under the shifted dynamics.

Each entry point writes exactly one manifest sibling file into the
output directory:

- `ablate_node`  → `ablation.json`  (schema version 1, legacy name retained)
- `nudge_node`   → `nudge.json`     (schema version 1)

Determinism: `(config + seed + perturbation)` → byte-identical
artifacts. The RNG stream still derives from `seed` alone.

Arbitrary interventions (mid-run ablation, edge ablation, topology
edits, combined ablation+nudge) remain deferred. Each new
counterfactual detector earns one narrow sibling seam here; we do not
grow a generic intervention registry.
"""
from __future__ import annotations

import json
from pathlib import Path

from sim.garden import simulate

ABLATION_SCHEMA_VERSION = 1
ABLATION_SEMANTICS = "decouple_and_silence"

NUDGE_SCHEMA_VERSION = 1
NUDGE_SEMANTICS = "detune_in_place"


def ablate_node(cfg, out_dir, node, config_path=None):
    """Run the simulator with node `node` ablated for the whole duration.

    Parameters
    ----------
    cfg : dict
        A validated v0 config (`sim.config.load`'s return value).
    out_dir : str | Path
        Destination directory for the ablated run artifacts. Created if
        missing. Will contain `state.npz`, `events.jsonl`,
        `topology.json`, `audio.wav`, an optional frozen `config.yaml`
        copy if `config_path` is given, and an `ablation.json` manifest.
    node : int
        Node index to ablate, in `[0, N)`.
    config_path : str | Path, optional
        If given, the baseline config file is copied verbatim into the
        ablated run's directory as `config.yaml`, matching the `sim.run`
        surface. The ablation itself is recorded in `ablation.json`, not
        in `config.yaml`, so the baseline config remains byte-identical
        to its source — the intervention lives next to the frozen config,
        not inside it.
    """
    N = int(cfg["scene"]["N"])
    node = int(node)
    if not 0 <= node < N:
        raise ValueError(f"node {node} out of range [0, {N})")

    out = Path(out_dir)
    simulate(cfg, out, config_path=config_path, ablated_nodes={node})
    _write_ablation_json(out, [node], config_path)


def nudge_node(cfg, out_dir, node, delta_hz, config_path=None):
    """Run the simulator with node `node`'s natural frequency shifted by `delta_hz`.

    The shift is applied at t=0 and held constant for the whole run;
    coupling, noise, and every other config parameter are preserved.
    This is the counterfactual probe for lock-robustness detectors
    (first consumer: `detect_brittle_lock`).

    Parameters
    ----------
    cfg : dict
        A validated v0 config (`sim.config.load`'s return value).
    out_dir : str | Path
        Destination directory for the nudged run artifacts. Created if
        missing. Will contain `state.npz`, `events.jsonl`,
        `topology.json`, `audio.wav`, an optional frozen `config.yaml`
        copy if `config_path` is given, and a `nudge.json` manifest.
    node : int
        Node index to nudge, in `[0, N)`.
    delta_hz : float
        Frequency offset (Hz) added to `cfg.scene.nodes[node].omega_0_hz`
        for the whole run. May be positive or negative. `0.0` is a valid
        (no-op) value — useful as a control — but is not a common caller
        pattern.
    config_path : str | Path, optional
        If given, the baseline config file is copied verbatim into the
        nudged run's directory as `config.yaml`. The nudge itself is
        recorded in `nudge.json`, not in `config.yaml`, so the baseline
        config remains byte-identical to its source.
    """
    N = int(cfg["scene"]["N"])
    node = int(node)
    if not 0 <= node < N:
        raise ValueError(f"node {node} out of range [0, {N})")
    delta = float(delta_hz)

    out = Path(out_dir)
    simulate(cfg, out, config_path=config_path, omega_offsets={node: delta})
    _write_nudge_json(out, node, delta, config_path)


def _write_ablation_json(out_dir, ablated_nodes, baseline_config_path):
    manifest = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "semantics": ABLATION_SEMANTICS,
        "ablated_nodes": sorted(int(k) for k in ablated_nodes),
        "baseline_config_path": (
            str(baseline_config_path) if baseline_config_path is not None else None
        ),
    }
    path = Path(out_dir) / "ablation.json"
    with path.open("w") as f:
        json.dump(manifest, f, sort_keys=True, indent=2)
        f.write("\n")
    return path


def _write_nudge_json(out_dir, node, delta_hz, baseline_config_path):
    manifest = {
        "schema_version": NUDGE_SCHEMA_VERSION,
        "semantics": NUDGE_SEMANTICS,
        "node": int(node),
        "delta_hz": float(delta_hz),
        "baseline_config_path": (
            str(baseline_config_path) if baseline_config_path is not None else None
        ),
    }
    path = Path(out_dir) / "nudge.json"
    with path.open("w") as f:
        json.dump(manifest, f, sort_keys=True, indent=2)
        f.write("\n")
    return path
