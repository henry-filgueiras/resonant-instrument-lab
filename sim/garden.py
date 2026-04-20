"""v0 Coupled Oscillator Garden — contract-honoring stub.

Intentionally NOT a physics simulator yet. This module honours the shape
contracts of DESIGN_V0.md §9 (export schemas + determinism) so the
detector layer, dataset pipeline, and real dynamics can land in clean
seams without rewriting the artifact plumbing.

Stubbed behaviour:
- no pairwise coupling; each node advances at its intrinsic rate
- amplitude is constant 1.0
- move events have no dynamical effect (position-time-series is a
  deferred contract gap — see DIRECTORS_NOTES.md)
- ablate_node events have no effect
- audio is silent at the contract-specified duration, rate, channel count
"""
from __future__ import annotations

import io
import json
import shutil
import wave
import zipfile
from pathlib import Path

import numpy as np

# Fixed zip mtime so state.npz bytes do not depend on wall-clock.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def simulate(cfg, out_dir, config_path=None):
    """Run the stub simulation for `cfg` and write artifacts to `out_dir`.

    `cfg` must already be validated (pass it through sim.config.load).
    `config_path`, if given, is copied verbatim to out_dir/config.yaml.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    N = int(cfg["scene"]["N"])
    duration_s = float(cfg["run"]["duration_s"])
    control_rate_hz = int(cfg["run"]["control_rate_hz"])
    audio_rate_hz = int(cfg["run"]["audio_rate_hz"])
    seed = int(cfg["run"]["seed"])

    T = int(round(duration_s * control_rate_hz)) + 1
    dt = 1.0 / control_rate_hz

    nodes = cfg["scene"]["nodes"]
    omega = np.array([n["omega_0_hz"] for n in nodes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-np.pi, np.pi, size=N)

    events = cfg.get("events", [])
    ev_by_idx = {}
    for ev in events:
        idx = min(int(round(float(ev["t"]) * control_rate_hz)), T - 1)
        ev_by_idx.setdefault(idx, []).append(ev)

    theta = np.zeros((T, N), dtype=np.float64)
    K0_t = np.zeros(T, dtype=np.float64)
    current_K0 = float(cfg["scene"]["coupling"]["K0"])
    current_omega = omega.copy()

    theta[0] = theta0
    for ev in ev_by_idx.get(0, []):
        et = ev["type"]
        if et == "setK":      current_K0 = float(ev["K0"])
        elif et == "nudge":   current_omega[ev["node"]] += float(ev["delta_hz"])
        elif et == "impulse": theta[0, ev["node"]] = 0.0
    K0_t[0] = current_K0

    two_pi = 2.0 * np.pi
    for k in range(1, T):
        theta[k] = theta[k-1] + two_pi * current_omega * dt
        for ev in ev_by_idx.get(k, []):
            et = ev["type"]
            if et == "setK":      current_K0 = float(ev["K0"])
            elif et == "nudge":   current_omega[ev["node"]] += float(ev["delta_hz"])
            elif et == "impulse": theta[k, ev["node"]] = 0.0
        K0_t[k] = current_K0

    theta_w = (theta + np.pi) % two_pi - np.pi

    phase_vel = np.zeros_like(theta_w)
    dtheta = (theta_w[1:] - theta_w[:-1] + np.pi) % two_pi - np.pi
    phase_vel[1:] = dtheta / dt
    phase_vel[0] = phase_vel[1]

    A = np.ones((T, N), dtype=np.float64)

    pulse_fired = np.zeros((T, N), dtype=bool)
    pulse_fired[1:] = (theta_w[:-1] < 0.0) & (theta_w[1:] >= 0.0)

    t = np.arange(T, dtype=np.float64) * dt

    _write_state_npz(
        out / "state.npz",
        t=t.astype(np.float32),
        theta=theta_w.astype(np.float32),
        phase_vel=phase_vel.astype(np.float32),
        A=A.astype(np.float32),
        pulse_fired=pulse_fired,
        K0_t=K0_t.astype(np.float32),
    )
    _write_events_jsonl(out / "events.jsonl", events)
    _write_topology_json(out / "topology.json", cfg, theta0)
    _write_silent_wav(out / "audio.wav", duration_s, audio_rate_hz)
    if config_path is not None:
        shutil.copyfile(config_path, out / "config.yaml")


def _write_state_npz(path, **arrays):
    """Write a .npz with sorted, timestamp-zeroed zip entries for byte-identity."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, arrays[name], allow_pickle=False)
            zi = zipfile.ZipInfo(filename=f"{name}.npy", date_time=_ZIP_DATE)
            zi.compress_type = zipfile.ZIP_STORED
            zf.writestr(zi, buf.getvalue())


def _write_events_jsonl(path, events):
    with path.open("w") as f:
        for i, ev in enumerate(events):
            record = dict(ev)
            record["id"] = f"evt_{i:03d}"
            record["source"] = "scheduled"
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def _write_topology_json(path, cfg, initial_theta):
    topo = {
        "version": 1,
        "N": cfg["scene"]["N"],
        "nodes": [
            {
                "index": i,
                "pos": list(n["pos"]),
                "omega_0_hz": n["omega_0_hz"],
                "gamma": n["gamma"],
                "voice": n["voice"],
                "initial_theta": float(initial_theta[i]),
                "initial_A": 1.0,
            }
            for i, n in enumerate(cfg["scene"]["nodes"])
        ],
        "coupling": dict(cfg["scene"]["coupling"]),
        "noise": dict(cfg["scene"]["noise"]),
    }
    with path.open("w") as f:
        json.dump(topo, f, sort_keys=True, indent=2)
        f.write("\n")


def _write_silent_wav(path, duration_s, audio_rate_hz):
    n_samples = int(round(duration_s * audio_rate_hz))
    payload = np.zeros(n_samples * 2, dtype=np.int16).tobytes()  # stereo
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(audio_rate_hz)
        w.writeframes(payload)
