"""Counterfactual (intervention-grounded) run entry point.

`sim.ablate` is the single, narrow counterfactual surface for v0
(DESIGN_V0.md §9.9, §10.1). The only intervention exposed here is
**node ablation**. Arbitrary interventions (mid-run ablation, edge
ablation, topology edits) are deferred.

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

The ablated run writes the same artifact set as a baseline run plus
one additional small sibling file, `ablation.json`, documenting which
nodes were ablated and under what semantics. Baseline runs do not get
this file, so its presence is itself the marker "this is a
counterfactual artifact bundle".

Determinism: `(config + seed + ablated_nodes)` → byte-identical
artifacts. The RNG stream still derives from `seed` alone.
"""
from __future__ import annotations

import json
from pathlib import Path

from sim.garden import simulate

ABLATION_SCHEMA_VERSION = 1
ABLATION_SEMANTICS = "decouple_and_silence"


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
