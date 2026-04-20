#!/usr/bin/env python3
"""Validate a Resonant Instrument Lab run config against DESIGN_V0.md §9.2.

Usage:
    python scripts/validate_config.py [path/to/config.yaml]

If no path is given, defaults to configs/regime_drifting.yaml.
Exits 0 on success; nonzero on failure with a precise diagnostic.
"""
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("error: PyYAML required (pip install pyyaml)")

SCHEMA_VERSION = 1
VOICE_PALETTE = 8
OMEGA_LO, OMEGA_HI = 0.5, 8.0

TOP_REQ  = {"version", "scene", "run"}
TOP_OK   = {"version", "meta", "scene", "run", "events"}
META_REQ = {"name"}
META_OK  = {"name", "notes"}
SCENE    = {"N", "nodes", "coupling", "noise"}
NODE     = {"pos", "omega_0_hz", "gamma", "voice"}
COUPLING = {"K0", "sigma"}
NOISE    = {"eta"}
RUN      = {"duration_s", "control_rate_hz", "audio_rate_hz", "seed"}
EVENT_SHAPES = {
    "setK":        {"t", "type", "K0"},
    "nudge":       {"t", "type", "node", "delta_hz"},
    "impulse":     {"t", "type", "node"},
    "move":        {"t", "type", "node", "delta_pos"},
    "ablate_node": {"t", "type", "node"},
}


class E(Exception): ...


def keys(path, obj, req, ok):
    if not isinstance(obj, dict):
        raise E(f"{path}: expected mapping, got {type(obj).__name__}")
    missing = req - obj.keys()
    if missing: raise E(f"{path}: missing required keys {sorted(missing)}")
    extra = obj.keys() - ok
    if extra:   raise E(f"{path}: unknown keys {sorted(extra)} (allowed: {sorted(ok)})")


def num(path, v, lo=None, hi=None, integer=False):
    if integer:
        if isinstance(v, bool) or not isinstance(v, int):
            raise E(f"{path}: expected int, got {type(v).__name__} ({v!r})")
    else:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise E(f"{path}: expected number, got {type(v).__name__} ({v!r})")
    if lo is not None and v < lo: raise E(f"{path}: {v} < min {lo}")
    if hi is not None and v > hi: raise E(f"{path}: {v} > max {hi}")
    return v


def pair(path, v, lo=None, hi=None):
    if not isinstance(v, list) or len(v) != 2:
        raise E(f"{path}: expected [x, y] list of length 2, got {v!r}")
    num(f"{path}[0]", v[0], lo, hi)
    num(f"{path}[1]", v[1], lo, hi)


def validate(doc):
    keys("config", doc, TOP_REQ, TOP_OK)
    num("config.version", doc["version"], SCHEMA_VERSION, SCHEMA_VERSION, integer=True)
    if "meta" in doc:
        keys("config.meta", doc["meta"], META_REQ, META_OK)
        for k in ("name", "notes"):
            if k in doc["meta"] and not isinstance(doc["meta"][k], str):
                raise E(f"config.meta.{k}: expected string, got {type(doc['meta'][k]).__name__}")
    s = doc["scene"]
    keys("config.scene", s, SCENE, SCENE)
    N = num("config.scene.N", s["N"], lo=2, integer=True)
    if not isinstance(s["nodes"], list):
        raise E(f"config.scene.nodes: expected list, got {type(s['nodes']).__name__}")
    if len(s["nodes"]) != N:
        raise E(f"config.scene.nodes: length {len(s['nodes'])} != scene.N={N}")
    for i, n in enumerate(s["nodes"]):
        p = f"config.scene.nodes[{i}]"
        keys(p, n, NODE, NODE)
        pair(f"{p}.pos", n["pos"], 0.0, 1.0)
        num(f"{p}.omega_0_hz", n["omega_0_hz"], OMEGA_LO, OMEGA_HI)
        num(f"{p}.gamma", n["gamma"], 0.0, 1.0)
        num(f"{p}.voice", n["voice"], 0, VOICE_PALETTE - 1, integer=True)
    keys("config.scene.coupling", s["coupling"], COUPLING, COUPLING)
    num("config.scene.coupling.K0", s["coupling"]["K0"], lo=0.0)
    num("config.scene.coupling.sigma", s["coupling"]["sigma"], lo=1e-6)
    keys("config.scene.noise", s["noise"], NOISE, NOISE)
    num("config.scene.noise.eta", s["noise"]["eta"], lo=0.0)
    r = doc["run"]
    keys("config.run", r, RUN, RUN)
    duration = num("config.run.duration_s", r["duration_s"], lo=1e-6)
    num("config.run.control_rate_hz", r["control_rate_hz"], lo=1, integer=True)
    num("config.run.audio_rate_hz", r["audio_rate_hz"], lo=1, integer=True)
    num("config.run.seed", r["seed"], lo=0, integer=True)
    events = doc.get("events", [])
    if not isinstance(events, list):
        raise E(f"config.events: expected list, got {type(events).__name__}")
    last_t = -float("inf")
    for i, ev in enumerate(events):
        p = f"config.events[{i}]"
        if not isinstance(ev, dict) or "type" not in ev:
            raise E(f"{p}: not a mapping with a 'type' field (got {ev!r})")
        et = ev["type"]
        if et not in EVENT_SHAPES:
            raise E(f"{p}.type: unknown {et!r}; allowed {sorted(EVENT_SHAPES)}")
        keys(p, ev, EVENT_SHAPES[et], EVENT_SHAPES[et])
        t = num(f"{p}.t", ev["t"], lo=0.0, hi=duration)
        if t < last_t:
            raise E(f"{p}.t: events must be sorted ascending; {t} < previous {last_t}")
        last_t = t
        if "node" in ev:
            num(f"{p}.node", ev["node"], lo=0, hi=N - 1, integer=True)
        if et == "setK":
            num(f"{p}.K0", ev["K0"], lo=0.0)
        elif et == "nudge":
            num(f"{p}.delta_hz", ev["delta_hz"])
        elif et == "move":
            pair(f"{p}.delta_pos", ev["delta_pos"])


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/regime_drifting.yaml")
    if not path.is_file():
        sys.exit(f"error: config not found: {path}")
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        sys.exit(f"error: YAML parse failure in {path}: {e}")
    try:
        validate(doc)
    except E as e:
        sys.exit(f"error: {e}")
    print(f"ok: {path} validates against schema v{SCHEMA_VERSION}")


if __name__ == "__main__":
    main()
