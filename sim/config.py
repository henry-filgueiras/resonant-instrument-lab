"""Config loading + validation for v0 simulator configs (DESIGN_V0.md §9.2)."""
from __future__ import annotations

from pathlib import Path

import yaml

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
# `move` is deferred — see DESIGN_V0.md §9.5. Until pos_t is exported
# in state.npz, accepting `move` events would silently lie to detectors
# about mid-run geometry. Rejecting it at the schema layer is the honest
# choice for v0. Reintroduce when the export lands.
EVENT_SHAPES = {
    "setK":        {"t", "type", "K0"},
    "nudge":       {"t", "type", "node", "delta_hz"},
    "impulse":     {"t", "type", "node"},
    "ablate_node": {"t", "type", "node"},
}


class ConfigError(Exception):
    """Raised when a config violates the v1 schema."""


def _keys(path, obj, req, ok):
    if not isinstance(obj, dict):
        raise ConfigError(f"{path}: expected mapping, got {type(obj).__name__}")
    missing = req - obj.keys()
    if missing: raise ConfigError(f"{path}: missing required keys {sorted(missing)}")
    extra = obj.keys() - ok
    if extra:   raise ConfigError(f"{path}: unknown keys {sorted(extra)} (allowed: {sorted(ok)})")


def _num(path, v, lo=None, hi=None, integer=False):
    if integer:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ConfigError(f"{path}: expected int, got {type(v).__name__} ({v!r})")
    else:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{path}: expected number, got {type(v).__name__} ({v!r})")
    if lo is not None and v < lo: raise ConfigError(f"{path}: {v} < min {lo}")
    if hi is not None and v > hi: raise ConfigError(f"{path}: {v} > max {hi}")
    return v


def _pair(path, v, lo=None, hi=None):
    if not isinstance(v, list) or len(v) != 2:
        raise ConfigError(f"{path}: expected [x, y] list of length 2, got {v!r}")
    _num(f"{path}[0]", v[0], lo, hi)
    _num(f"{path}[1]", v[1], lo, hi)


def validate(doc):
    """Validate a loaded config dict. Raises ConfigError on any violation."""
    _keys("config", doc, TOP_REQ, TOP_OK)
    _num("config.version", doc["version"], SCHEMA_VERSION, SCHEMA_VERSION, integer=True)
    if "meta" in doc:
        _keys("config.meta", doc["meta"], META_REQ, META_OK)
        for k in ("name", "notes"):
            if k in doc["meta"] and not isinstance(doc["meta"][k], str):
                raise ConfigError(f"config.meta.{k}: expected string, got {type(doc['meta'][k]).__name__}")
    s = doc["scene"]
    _keys("config.scene", s, SCENE, SCENE)
    N = _num("config.scene.N", s["N"], lo=2, integer=True)
    if not isinstance(s["nodes"], list):
        raise ConfigError(f"config.scene.nodes: expected list, got {type(s['nodes']).__name__}")
    if len(s["nodes"]) != N:
        raise ConfigError(f"config.scene.nodes: length {len(s['nodes'])} != scene.N={N}")
    for i, n in enumerate(s["nodes"]):
        p = f"config.scene.nodes[{i}]"
        _keys(p, n, NODE, NODE)
        _pair(f"{p}.pos", n["pos"], 0.0, 1.0)
        _num(f"{p}.omega_0_hz", n["omega_0_hz"], OMEGA_LO, OMEGA_HI)
        _num(f"{p}.gamma", n["gamma"], 0.0, 1.0)
        _num(f"{p}.voice", n["voice"], 0, VOICE_PALETTE - 1, integer=True)
    _keys("config.scene.coupling", s["coupling"], COUPLING, COUPLING)
    _num("config.scene.coupling.K0", s["coupling"]["K0"], lo=0.0)
    _num("config.scene.coupling.sigma", s["coupling"]["sigma"], lo=1e-6)
    _keys("config.scene.noise", s["noise"], NOISE, NOISE)
    _num("config.scene.noise.eta", s["noise"]["eta"], lo=0.0)
    r = doc["run"]
    _keys("config.run", r, RUN, RUN)
    duration = _num("config.run.duration_s", r["duration_s"], lo=1e-6)
    _num("config.run.control_rate_hz", r["control_rate_hz"], lo=1, integer=True)
    _num("config.run.audio_rate_hz", r["audio_rate_hz"], lo=1, integer=True)
    _num("config.run.seed", r["seed"], lo=0, integer=True)
    events = doc.get("events", [])
    if not isinstance(events, list):
        raise ConfigError(f"config.events: expected list, got {type(events).__name__}")
    last_t = -float("inf")
    for i, ev in enumerate(events):
        p = f"config.events[{i}]"
        if not isinstance(ev, dict) or "type" not in ev:
            raise ConfigError(f"{p}: not a mapping with a 'type' field (got {ev!r})")
        et = ev["type"]
        if et not in EVENT_SHAPES:
            raise ConfigError(f"{p}.type: unknown {et!r}; allowed {sorted(EVENT_SHAPES)}")
        _keys(p, ev, EVENT_SHAPES[et], EVENT_SHAPES[et])
        t = _num(f"{p}.t", ev["t"], lo=0.0, hi=duration)
        if t < last_t:
            raise ConfigError(f"{p}.t: events must be sorted ascending; {t} < previous {last_t}")
        last_t = t
        if "node" in ev:
            _num(f"{p}.node", ev["node"], lo=0, hi=N - 1, integer=True)
        if et == "setK":
            _num(f"{p}.K0", ev["K0"], lo=0.0)
        elif et == "nudge":
            _num(f"{p}.delta_hz", ev["delta_hz"])


def load(path):
    """Load and validate a config file. Returns the parsed dict.

    Raises ConfigError on validation failure, FileNotFoundError if missing,
    and propagates yaml.YAMLError on malformed input.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    doc = yaml.safe_load(p.read_text())
    validate(doc)
    return doc
