"""v0 Coupled Oscillator Garden — real distance-weighted Kuramoto dynamics.

Integrates DESIGN_V0.md §2.2:

    dθᵢ/dt = 2π·fᵢ + Σⱼ Kᵢⱼ · sin(θⱼ − θᵢ) + ξᵢ(t)
    Kᵢⱼ    = K₀ · exp(−dᵢⱼ / σ)

Units: `fᵢ = omega_0_hz` in Hz (promoted to rad/s by 2π); `K₀` is in
rad/s; `σ` is in unit-square distance; `ξᵢ(t)` is Gaussian white noise
with intensity `η` (rad/√s), discretized as `η · √dt · 𝒩(0,1)` per step.
Explicit Euler step with `dt = 1 / control_rate_hz`.

Still stubbed:
- amplitude is constant 1.0 (γ is accepted in config but unused)
- audio is silent-but-format-correct
- `ablate_node` events in the config schema have no runtime effect yet
  (mid-run ablation is deferred; the static-from-t=0 counterfactual path
  is driven by `simulate(..., ablated_nodes=...)`, wrapped by sim.ablate)
- `move` events are rejected at the schema layer (sim/config.py)
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


def simulate(cfg, out_dir, config_path=None, ablated_nodes=None, omega_offsets=None):
    """Run the simulator for `cfg` and write artifacts to `out_dir`.

    `cfg` must already be validated (pass it through sim.config.load).
    `config_path`, if given, is copied verbatim to out_dir/config.yaml.

    `ablated_nodes` is an optional iterable of node indices to ablate for
    the whole run. Ablation semantics: the node is decoupled from every
    pair in the coupling matrix (it neither drives nor is driven by any
    other node) and its pulses are silenced in `pulse_fired`. The node's
    `theta` continues to integrate under its own natural frequency and
    noise so per-frame arrays retain shape `(T, N)` — but because the
    node is causally detached, other nodes' trajectories are the
    counterfactual trajectories the intervention is meant to expose.
    The public counterfactual entry point is `sim.ablate.ablate_node`;
    callers generally do not pass `ablated_nodes` directly.

    `omega_offsets` is an optional `{node_idx: delta_hz}` mapping applied
    as a **whole-run** perturbation to each listed node's natural
    frequency. Semantics: for node k with offset Δf, the integrator uses
    `omega[k] += Δf` from t=0 for the entire run. Coupling is preserved
    — the nudged node remains fully connected and can still be captured
    or pulled out of phase; that is what makes the perturbation an
    honest probe of lock robustness rather than a causal removal. The
    public counterfactual entry point is `sim.ablate.nudge_node`;
    callers generally do not pass `omega_offsets` directly.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    N = int(cfg["scene"]["N"])
    ablated = frozenset(int(k) for k in (ablated_nodes or ()))
    for k in ablated:
        if not 0 <= k < N:
            raise ValueError(f"ablated node {k} out of range [0, {N})")
    nudges = {int(k): float(v) for k, v in (omega_offsets or {}).items()}
    for k in nudges:
        if not 0 <= k < N:
            raise ValueError(f"nudged node {k} out of range [0, {N})")
    duration_s = float(cfg["run"]["duration_s"])
    control_rate_hz = int(cfg["run"]["control_rate_hz"])
    audio_rate_hz = int(cfg["run"]["audio_rate_hz"])
    seed = int(cfg["run"]["seed"])

    T = int(round(duration_s * control_rate_hz)) + 1
    dt = 1.0 / control_rate_hz
    sqrt_dt = np.sqrt(dt)
    two_pi = 2.0 * np.pi

    nodes = cfg["scene"]["nodes"]
    positions = np.array([n["pos"] for n in nodes], dtype=np.float64)
    omega = np.array([n["omega_0_hz"] for n in nodes], dtype=np.float64)
    for k, delta_hz in nudges.items():
        omega[k] += delta_hz

    sigma = float(cfg["scene"]["coupling"]["sigma"])
    eta = float(cfg["scene"]["noise"]["eta"])
    current_K0 = float(cfg["scene"]["coupling"]["K0"])

    # Distance-kernel shape (static in v0 now that `move` is rejected).
    D = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    K_shape = np.exp(-D / sigma)
    np.fill_diagonal(K_shape, 0.0)
    # Node ablation: zero rows and columns so the node is mutually decoupled.
    # Applied to K_shape (not just K_mat) so later `setK` events preserve ablation.
    for k in ablated:
        K_shape[k, :] = 0.0
        K_shape[:, k] = 0.0
    K_mat = current_K0 * K_shape

    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-np.pi, np.pi, size=N)

    events = cfg.get("events", [])
    ev_by_idx = {}
    for ev in events:
        idx = min(int(round(float(ev["t"]) * control_rate_hz)), T - 1)
        ev_by_idx.setdefault(idx, []).append(ev)

    theta = np.zeros((T, N), dtype=np.float64)
    K0_t = np.zeros(T, dtype=np.float64)
    current_omega = omega.copy()

    theta[0] = theta0
    for ev in ev_by_idx.get(0, []):
        et = ev["type"]
        if et == "setK":
            current_K0 = float(ev["K0"])
            K_mat = current_K0 * K_shape
        elif et == "nudge":
            current_omega[ev["node"]] += float(ev["delta_hz"])
        elif et == "impulse":
            theta[0, ev["node"]] = 0.0
    K0_t[0] = current_K0

    for k in range(1, T):
        theta_prev = theta[k-1]
        # diff[i, j] = theta_j - theta_i
        diff = theta_prev[None, :] - theta_prev[:, None]
        coupling = (K_mat * np.sin(diff)).sum(axis=1)
        rate = two_pi * current_omega + coupling
        if eta > 0.0:
            theta[k] = theta_prev + rate * dt + eta * sqrt_dt * rng.standard_normal(N)
        else:
            theta[k] = theta_prev + rate * dt
        for ev in ev_by_idx.get(k, []):
            et = ev["type"]
            if et == "setK":
                current_K0 = float(ev["K0"])
                K_mat = current_K0 * K_shape
            elif et == "nudge":
                current_omega[ev["node"]] += float(ev["delta_hz"])
            elif et == "impulse":
                theta[k, ev["node"]] = 0.0
        K0_t[k] = current_K0

    theta_w = (theta + np.pi) % two_pi - np.pi

    phase_vel = np.zeros_like(theta_w)
    dtheta = (theta_w[1:] - theta_w[:-1] + np.pi) % two_pi - np.pi
    phase_vel[1:] = dtheta / dt
    phase_vel[0] = phase_vel[1]

    A = np.ones((T, N), dtype=np.float64)

    pulse_fired = np.zeros((T, N), dtype=bool)
    pulse_fired[1:] = (theta_w[:-1] < 0.0) & (theta_w[1:] >= 0.0)
    for k in ablated:
        pulse_fired[:, k] = False

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
    _write_pulse_wav(
        out / "audio.wav",
        pulse_fired=pulse_fired,
        omega_hz=omega,
        control_rate_hz=control_rate_hz,
        audio_rate_hz=audio_rate_hz,
        duration_s=duration_s,
    )
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


# Pulse-synth parameters. Click length 60 ms with exp decay tau 12 ms —
# short enough that even dense 4 Hz rasters don't smear into a drone,
# long enough that the click has an audible pitched tail. Per-node pitch
# is a linear function of omega_0_hz so nudges (the canonical +0.25 Hz
# whole-run perturbation) are themselves audible as a small pitch shift.
_CLICK_DURATION_S = 0.060
_CLICK_DECAY_TAU_S = 0.012
_CLICK_PITCH_BASE_HZ = 180.0
_CLICK_PITCH_PER_HZ = 80.0
_CLICK_AMPLITUDE = 0.30


def _write_pulse_wav(path, *, pulse_fired, omega_hz, control_rate_hz,
                      audio_rate_hz, duration_s):
    """Write a mono 16-bit PCM WAV whose sound is driven by `pulse_fired`.

    Each True entry in `pulse_fired[:, k]` splats a short pitched click
    into the audio buffer at the sample corresponding to that control
    frame. Per-node pitch is `_CLICK_PITCH_BASE_HZ + _CLICK_PITCH_PER_HZ
    * omega_hz[k]`, so nodes with different natural frequencies are
    audibly distinguishable and a `nudge_node(+0.25 Hz)` intervention
    produces a small but perceivable pitch shift on the nudged node.

    Deterministic (no RNG) so `(config + seed + intervention)` still
    maps to byte-identical artifacts — the existing determinism
    contract on atlas builds depends on this.
    """
    n_samples = int(round(duration_s * audio_rate_hz))
    if n_samples <= 0:
        _write_empty_wav(path, audio_rate_hz)
        return

    # Precompute one click-shape envelope; each node scales it by its
    # own pitched sinusoid. Reusing the envelope keeps this O(N·clicks)
    # with a small constant, fine for ~6 s runs at N≤8.
    click_len = min(int(round(_CLICK_DURATION_S * audio_rate_hz)), n_samples)
    t_click = np.arange(click_len, dtype=np.float64) / float(audio_rate_hz)
    envelope = np.exp(-t_click / _CLICK_DECAY_TAU_S).astype(np.float64)

    out = np.zeros(n_samples, dtype=np.float64)
    pulse_fired = np.asarray(pulse_fired)
    T = pulse_fired.shape[0]
    N = pulse_fired.shape[1]
    sample_stride = float(audio_rate_hz) / float(control_rate_hz)

    for k in range(N):
        frames = np.flatnonzero(pulse_fired[:, k])
        if frames.size == 0:
            continue
        pitch_hz = _CLICK_PITCH_BASE_HZ + _CLICK_PITCH_PER_HZ * float(omega_hz[k])
        tone = np.sin(2.0 * np.pi * pitch_hz * t_click) * envelope * _CLICK_AMPLITUDE
        for frame in frames:
            s = int(round(frame * sample_stride))
            if s >= n_samples:
                continue
            e = min(s + click_len, n_samples)
            out[s:e] += tone[: e - s]

    # Soft clip into [-1, 1] so overlapping pulses don't wrap on overflow.
    np.tanh(out, out=out)
    pcm = np.round(out * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio_rate_hz)
        w.writeframes(pcm.tobytes())


def _write_empty_wav(path, audio_rate_hz):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio_rate_hz)
        w.writeframes(b"")
