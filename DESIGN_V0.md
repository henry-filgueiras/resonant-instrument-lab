# Resonant Instrument Lab — v0 Design Note

_A scoped, skeptical plan for a single-researcher prototype._

---

## 1. Framing

### Why this is a better target than generic music AI

Generic music-generation and music-captioning projects fail as single-researcher prototypes for three compounding reasons:

- **No ground truth.** "Does this sound musical?" is unfalsifiable. Human rating is expensive, subjective, and biased.
- **Dataset friction.** Audio corpora are heavy, licensing-messy, and stylistically narrow. A small team cannot realistically iterate on them.
- **The model has no handle.** A caption like *"upbeat electronic track"* neither constrains the model's representations nor lets us verify that the model understood anything about the signal.

The Resonant Instrument Lab swaps the target: instead of arbitrary music, we study a **bounded synthetic world** whose internal state is fully observable and whose semantic labels are *computable from that state*. Every claim the model makes — *"these nodes are phase-locked"*, *"the groove will collapse if node 3 is detuned"* — is either confirmable or disconfirmable by re-running the simulator.

This converts the research loop from *taste* into **grounded perception + hypothesis testing**. That is the only regime in which one researcher plus a small local model can do work that is both cheap and checkable.

### What we are actually building

```
simulator state  →  audio + views  →  small LM  →  description / intervention
                                                           ↓
                                                      simulator
                                                  (verifies the claim)
```

Success = the loop closes. The model's words predict sim behavior; the model's suggested interventions move the sim in the direction it claimed they would.

### How this assists instrument-making in practice

Once the loop closes on the toy world, the same pattern ports to real instruments (physical models, modular synth patches, generative systems):

- **Observe:** model describes current behavior in musician-native terms.
- **Compare:** model contrasts takes / patches.
- **Diagnose:** model flags brittleness, stuck regions, dead edges of parameter space.
- **Steer:** model proposes a single, minimal intervention that achieves a requested shift.

The point is not that a tiny LM will out-compose anyone. The point is that a tiny LM grounded in signal + state can be a *technical collaborator* during patch design, voicing, and debugging.

---

## 2. Toy World Spec — Coupled Oscillator Garden (v0)

### 2.1 Node state

`N = 8` nodes in the unit square. Each node carries:

| field   | type       | notes                                      |
|---------|------------|--------------------------------------------|
| `pos`   | `(x, y)`   | 2D position, unit square                   |
| `ω₀`    | Hz         | intrinsic pulse rate, 0.5–8 Hz             |
| `θ`     | rad        | current phase                              |
| `A`     | 0–1        | output amplitude                           |
| `γ`     | 0–1        | damping / leakiness                        |
| `voice` | int        | fixed pitch + timbre (v0: 8-voice palette) |

### 2.2 Dynamics

Kuramoto-style phase coupling with a distance kernel:

```
dθᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ · sin(θⱼ − θᵢ) + ξᵢ(t)
Kᵢⱼ    = K₀ · exp(−dᵢⱼ / σ)
```

Amplitude is a driven relaxation: `dAᵢ/dt = −γᵢ·Aᵢ + pulse_input`, with pulses firing on phase-zero crossings.

Knobs that matter:

- `K₀` — global coupling scalar. The most important knob; sweeping it crosses regime boundaries.
- `σ` — coupling range; controls whether geometry matters.
- `η` — optional white noise amplitude on `ξᵢ(t)`.

### 2.3 Audio rendering

- Each phase-zero crossing fires a percussive grain for that node's voice (fixed pitch + short envelope + stereo pan from `pos.x`).
- **Audio layer is pulse-driven, not audio-rate.** Emergent *rhythm* is the musical content. This is cheap to simulate and makes coupling-induced timing differences audible as grooves, flams, and polyrhythms.
- Audio-rate tone coupling (sub-second beating / interference in continuous tones) is explicitly deferred to v0.1.

### 2.4 Events / gestures (v0, scheduled not interactive)

- `nudge(node, Δω)` — retune a node
- `impulse(node)` — force a phase reset
- `move(node, Δpos)` — reshape coupling geometry
- `setK(K₀)` — global coupling sweep

### 2.5 Exports per run

- `audio.wav` — stereo, 22050 Hz, 4–8 s
- `state.npz` — `θ(t), A(t)` per node at 200 Hz control-rate
- `events.jsonl` — scheduled + emitted events
- `topology.json` — positions, voice assignments, static params
- `labels.json` — ground-truth labels from §3 detectors, with time windows and confidence

---

## 3. Semantic Ontology

Compact (13 labels), multi-label, testable. Each label has a **deterministic detector** computed from state / events. Detectors are part of the simulator package, not the model — they are ground truth.

Full per-label detail (intuition, detector sketch, inputs, knobs, confidence source, failure modes, overlaps) lives in [`ONTOLOGY.md`](./ONTOLOGY.md). The list:

`phase_locked`, `drifting`, `polyrhythmic`, `phase_beating`, `dominant_cluster`, `brittle_lock`, `unstable_bridge`, `groove`, `groove_collapse`, `beat_bloom`, `tension_break`, `flam`, `regime_change`.

A clip is labelled with the (possibly multiple) labels whose detectors fired, plus time windows and per-detector confidence scores.

### 3.1 Detector confidence — doctrine

Detector confidence is **evidence margin**, not epistemic uncertainty. It measures *how strongly and stably* the defining condition was met — e.g. how far `r` exceeded the lock threshold, for how long, across how many nodes. Detectors are deterministic; graded confidence scores simply let downstream tasks distinguish a marginal lock from a rock-solid one, or a fleeting bloom from a sustained one.

### 3.2 Note on "beating" in a pulse-first world

`phase_beating` and `beat_bloom` do **not** refer to audio-rate acoustic beating between continuous tones. v0 is pulse-driven; what these detectors measure is the slow drift of pulse-arrival times between near-frequency streams — the rhythmic cousin of acoustic beating. True Hz-scale continuous-tone beating is deferred to v0.1 (see §8).

---

## 4. Data Generation Plan

A **regime-template sampler** ensures each label appears enough for training. Uniform random sampling over parameters under-covers interesting regimes.

Regime templates:

- `LOCKED` — high `K₀`, low `η`
- `DRIFTING` — low `K₀`
- `TWO_CLUSTER` — bimodal `ω₀`, medium `K₀`
- `BRITTLE` — high `K₀` near bifurcation + low-amp noise
- `SWEEP` — `K₀` ramps across a bifurcation mid-clip
- `PERTURBED` — stable regime + scheduled `impulse` / `nudge`
- `COLLAPSE` — groove regime + scheduled parameter step

Per run:

1. Simulate with fixed seed.
2. Export audio, state, events, topology.
3. Run detectors → `labels.json`.
4. Template-bank task generator emits `{task_type, prompt, target}` tuples seeded by the labels (see §5).

**v0 target scale:** ~5 000 clips × ~5 tasks = ~25 k examples. Fits comfortably on a laptop (<10 GB).

---

## 5. Task Suite

All tasks return short text (one sentence to a small JSON blob). Each task type has a template bank; targets are generated from detector output, not by an LLM, so they are verifiable.

### T1. Caption
**In:** audio (+ optional state).
**Out:** ≤25-word description of present state.
**Example →** *"Mostly phase-locked with a dominant four-node cluster; one outlier node drifts."*

### T2. Semantic QA
**In:** clip + question.
**Out:** `{yes | no | unclear}` + one-line justification.
**Ex. Q:** *"Is this a brittle lock?"* **A:** *"Yes — order parameter is near 1 but a small nudge to node 3 would break it."*

### T3. Compare
**In:** clip A, clip B.
**Out:** single sentence naming the salient difference using the ontology.
**Example →** *"A is drifting; B is in a groove led by nodes 1 and 4."*

### T4. Anomaly / instability
**In:** clip.
**Out:** `{label, window_s}` or `null`.
**Example →** `{"label": "tension_break", "window_s": [3.1, 3.6]}`

### T5. Explain change
**In:** clip spanning a regime change + events log.
**Out:** ≤2 sentences connecting event → outcome.
**Example →** *"Raising K₀ past ~0.8 collapsed the two sub-clusters into one; the prior polyrhythm disappeared."*

### T6. Suggest intervention (the headline task)
**In:** clip + goal (e.g. *"more tension"*, *"stabilize"*, *"create contrast"*).
**Out:** a single structured action:
```json
{"type": "nudge", "node": 3, "delta_hz": 0.15, "why": "detunes the dominant cluster's anchor; should introduce phase-beating between the two streams"}
```
Evaluated by actually applying the intervention — see §7 L3.

---

## 6. Model Plan

All local. Single consumer GPU or Apple-silicon laptop.

### Encoders (trained from scratch or lightly fine-tuned)
- **Spectrogram encoder** — small CNN (~1–2 M params) over 64-bin mel, OR a frozen small pretrained CLAP-style encoder with a trained linear adapter. One-day bake-off picks the winner; don't pre-commit.
- **State encoder** — 2–3-layer transformer over per-node `(θ, A)` downsampled to 50 Hz. ~500 k params.
- **Topology encoder** — concat node features into a short token sequence with positional embedding from `pos`. **No GNN in v0** — unjustified complexity at this scale.

Encoders emit ~32–64 soft tokens, prefixed to the LM context.

### Language model (frozen base + LoRA)
- Base: **Qwen2.5-0.5B-Instruct** or **Llama-3.2-1B-Instruct**. Both run locally on CPU or consumer GPU.
- **Frozen:** base weights.
- **Trained:** LoRA adapters (attention + MLP), encoder weights, input projection.

### Training
- Instruction SFT across T1–T6 with mixed sampling. No RL, no preference tuning in v0.
- **Auxiliary losses only if justified.** Contrastive clip↔caption is on the table *if* SFT plateaus; otherwise don't.

### Budget
- ≤24 h training on a single consumer GPU (4090 or MLX on M-series).
- Inference: <2 s per response on CPU acceptable.

---

## 7. Evaluation Plan

Three tiers. Each must pass before moving on.

### L1. Label accuracy
Held-out sim runs with known labels.
- T2 / T4: per-label F1, macro-F1 aggregate.
- T1 / T3: label-coverage F1 via extracting ontology terms from generated text.

### L2. Robustness
- **Audio:** add pink noise, clip, resample — accuracy curves.
- **State:** jitter `θ`, drop nodes from view — graceful degradation?
- **Near-bifurcation configs:** labels are only useful if they hold up near regime edges.

### L3. Intervention grounding — **the real test**
For each T6 output:
1. Model proposes an intervention with a predicted effect.
2. **Apply it in the simulator.** Measure actual effect via detectors.
3. Score: did the predicted-effect label become true / strengthen, relative to baselines?

Baselines:
- Random intervention of same action type.
- Hand-coded heuristic policy (e.g. *"to stabilize, raise K₀; to destabilize, add noise"*).

The model must beat both, convincingly, on average. This is what distinguishes the project from caption-style benchmarks; a model that memorizes surface patterns fails here.

### Generalization probes
- Train on `N=8`, evaluate on `N ∈ {4, 12}`.
- Train on voice palette A, evaluate on palette B (timbre shift).
- Hold out one regime template and test zero-shot on it.

---

## 8. Out of scope for v0

- Real-time interactive playing / MIDI I/O.
- Audio-rate tone coupling (Hz-scale beating in continuous tones). Deferred to v0.1.
- Graph neural networks; learned topology.
- Preference tuning / RLHF.
- Any third-party music dataset.
- Multimodal benchmarks beyond the internal suite.
- Human listening studies. Acknowledged as eventually needed, but not blocking v0.

---

## 9. Simulator contract

Implementation-facing. The layer the detector code, data pipeline, and (eventually) model training will call into. No code yet — but the shape is frozen here.

### 9.1 Vocabulary

- **Scene** — the static description of a garden: node count, per-node static params, coupling params, noise amplitude. A scene alone does not produce a run.
- **Run** — a scene + `duration_s` + `seed` + scheduled `events[]`, executed once. Deterministic under `(config, seed)`.
- **Frame / step** — one control-rate tick (default 200 Hz).
- **Ablation** — a scheduled or injected perturbation applied to a run's config, producing a *new* run. Used by counterfactual detectors.

### 9.2 Run config (canonical YAML schema)

```yaml
version: 1
meta:                         # optional; human-readable identity for the config
  name: regime_drifting
  notes: >                    # optional string
    Free-form description, kept out of state/topology exports.
scene:
  N: 8
  nodes:                      # length must equal N
    - {pos: [0.20, 0.30], omega_0_hz: 2.5, gamma: 0.10, voice: 0}
    - {pos: [0.45, 0.55], omega_0_hz: 3.1, gamma: 0.10, voice: 1}
    # ...
  coupling:
    K0: 0.80                  # initial value; events may mutate
    sigma: 0.40               # distance-kernel width
  noise:
    eta: 0.02                 # 0 disables
run:
  duration_s: 8.0
  control_rate_hz: 200
  audio_rate_hz: 22050
  seed: 42
events:                       # optional, sorted by t at load; default []
  - {t: 3.0, type: setK,    K0: 1.10}
  - {t: 5.5, type: nudge,   node: 3, delta_hz: 0.15}
  - {t: 6.2, type: impulse, node: 5}
  - {t: 7.0, type: move,    node: 2, delta_pos: [0.05, -0.02]}
```

All fields required unless marked optional. Unknown keys are an error, not a warning. `meta` (if present) is human-facing identity only — it is **not** copied into exported `topology.json` or any other artifact the detector layer reads, which preserves the §10.2 boundary against privileged latent truth.

Concrete ranges (enforced by `scripts/validate_config.py`, see §9.9):

- `version` must equal `1`.
- `scene.N` integer ≥ 2; `len(scene.nodes) == scene.N`.
- `node.pos[i]` ∈ `[0.0, 1.0]`; `omega_0_hz` ∈ `[0.5, 8.0]`; `gamma` ∈ `[0.0, 1.0]`; `voice` integer ∈ `[0, 7]` (v0 palette).
- `coupling.K0 ≥ 0`; `coupling.sigma > 0`; `noise.eta ≥ 0`.
- `run.duration_s > 0`; `control_rate_hz, audio_rate_hz, seed` integer ≥ 0 (rates ≥ 1).
- `events[i].t ∈ [0, duration_s]`; `events` sorted ascending by `t`; `node` indices ∈ `[0, N-1]`.

### 9.3 Per-step exports (control rate)

Written to `state.npz`. Arrays of shape `(T, N)` unless noted. `T = duration_s * control_rate_hz + 1`.

| field          | dtype     | shape  | notes                                         |
|----------------|-----------|--------|-----------------------------------------------|
| `t`            | float32   | (T,)   | seconds from run start                        |
| `theta`        | float32   | (T, N) | phase, radians, wrapped `[−π, π]`             |
| `phase_vel`    | float32   | (T, N) | rad/s, derived from `theta` with finite diff  |
| `A`            | float32   | (T, N) | amplitude, `[0, 1]`                           |
| `pulse_fired`  | bool      | (T, N) | true on phase-zero crossings this frame       |
| `K0_t`         | float32   | (T,)   | global coupling over time (mutated by events) |

Detectors may read any of these. The simulator must **not** export additional "helper" latent fields (e.g. a boolean `is_locked`) — see §10.

### 9.4 Audio rendering

- Stereo PCM at `audio_rate_hz = 22050`, `int16` WAV.
- Each `pulse_fired[t, i]` schedules a voice-`voice_i` grain at the corresponding sample index.
- Grain = short envelope × fixed-pitch tone for that voice. Voice palette is static data, shared across runs (v0: 8 voices).
- Stereo pan is a monotonic function of `pos_i.x`.
- **No tone-level coupling** (audio-rate beating between continuous tones is v0.1, see §8).

### 9.5 Events / gestures

Both scheduled (from config) and simulator-emitted parameter changes are logged to `events.jsonl`. One event per line:

```json
{"t": 3.0, "type": "setK", "K0": 1.10, "id": "evt_000", "source": "scheduled"}
{"t": 5.5, "type": "nudge", "node": 3, "delta_hz": 0.15, "id": "evt_001", "source": "scheduled"}
```

Pulse firings are **not** events — they live in `state.npz`. Events are reserved for things that change the config of the simulation mid-run.

Event types (v0, closed set):
- `setK` — `{K0: float}`
- `nudge` — `{node: int, delta_hz: float}` — modifies `omega_0_hz[node]` permanently from `t`
- `impulse` — `{node: int}` — sets `theta[node] = 0` at `t`
- `move` — `{node: int, delta_pos: [dx, dy]}` — modifies position (affects coupling kernel from `t`)
- `ablate_node` — `{node: int}` — sets `A[node] = 0` and zeros coupling to/from node for remainder of run (counterfactual only; not used in primary-generation configs)

### 9.6 Output artifact layout

```
runs/<run_id>/
  config.yaml       # frozen copy of the config actually used
  topology.json     # scene + initial state (positions, voices, static params)
  state.npz         # per-step arrays (§9.3)
  events.jsonl      # §9.5
  audio.wav         # stereo 22050 Hz int16
  labels.json       # detector outputs — written by detector layer, NOT by sim
```

`labels.json` is out of the simulator's scope. The simulator's artifact set is everything else.

### 9.7 Determinism

- `(config + seed)` → byte-identical `state.npz`, `events.jsonl`, `topology.json`, `audio.wav`. WAV header timestamps must be zeroed.
- No wall-clock dependencies in dynamics, audio, or RNG seeding.
- RNG stream is derived from `seed` alone; no `/dev/urandom` or time-based entropy.
- Pinning the numeric library stack (NumPy / SciPy versions) is acceptable for v0 — document in `requirements.txt` when code lands.

### 9.8 Derived-state helpers (provided, not stored)

Detectors may call, but the outputs are not written to `state.npz`:

- `kuramoto_order(theta) → r(t)` of shape `(T,)`
- `local_order(theta, members) → r(t)` for a subset of nodes
- `cluster_assignments(theta, phase_vel, t) → labels(t, N)` with documented DBSCAN params
- `pulse_times_of(pulse_fired, t, i) → array of float seconds`
- `cluster_signature(labels, phase_vel, window) → (count, size_hist, freq_hist)`

Helpers live in `sim.derived` (or equivalent). They are pure functions of exported state. Detectors must not import private simulator internals.

### 9.9 CLI / API surface (v0)

```
python scripts/validate_config.py [PATH]                 # schema check only (implemented)
python -m sim.run     --config PATH --out DIR [--seed N]
python -m sim.ablate  --run DIR    --perturbation JSON --out DIR [--seed N]
python -m sim.detect  --run DIR    [--detectors LIST]      # writes labels.json
```

`validate_config.py` is the first (and currently only) implemented entry point. Everything under `sim.*` is still contract-only.

Programmatic:

```python
from sim import Garden

g   = Garden.from_config("configs/regime_locked.yaml", seed=42)
run = g.simulate()                          # returns RunArtifacts
run.save("runs/demo")

alt_cfg = g.config.with_ablation({"type": "ablate_node", "node": 3})
alt     = Garden.from_config(alt_cfg, seed=42).simulate()
```

`sim.ablate` is the single counterfactual entry point — all ablation-style detectors go through it. The simulator must not expose a different back door for counterfactuals.

---

## 10. Detector contract

Implementation-facing rules for the detector layer. Goal: detectors grounded in sim outputs, with a sharp boundary against cheating by reading privileged ground truth.

### 10.1 Fair-game inputs

A detector **may** read:

- Anything in `state.npz` (§9.3).
- `topology.json` (positions, voices, per-node static params, initial coupling params).
- `events.jsonl` (scheduled + emitted parameter-change events).
- Derived-state helpers (§9.8).
- Outputs of other detectors — **only** when the detector is declared *compound* (see §10.5) and the dependency is documented in `ONTOLOGY.md`.
- In-sim ablation reruns via `sim.ablate` (§9.9) — only for detectors declared *counterfactual*, with a bounded rollout budget.

### 10.2 Privileged latent truth — forbidden

A detector **must not** read:

- The regime-template name used by the generator (e.g. `LOCKED`, `BRITTLE`). Regime templates are data-generation categories; labels must be re-derived from observable state.
- The random seed, RNG state, or any internal simulator state not exported.
- Any "convenience" latent field asserting the answer (e.g. a hypothetical `is_locked` flag). The simulator contract forbids exporting such fields; the detector contract independently forbids reading them.
- The outputs of upstream data-generation logic (e.g. the intended bifurcation time of a `SWEEP` template).

If a field exists that looks like latent truth, it is a contract bug and gets removed.

### 10.3 Eval-only signals — not detection inputs

These signals exist and are useful, but only for evaluation reporting and for driving data generation:

- Regime-template name (stratified metric reporting).
- Human-assigned audit tags from the sniff test.
- Intended bifurcation times in `SWEEP` configs (used to score `tension_break` / `regime_change` timing, not to detect them).

The detector function's input signature must not accept any of these. A code-level lint rule is the cleanest enforcement.

### 10.4 Output schema

Each detector returns (and `labels.json` is a JSON array of):

```json
{
  "label": "phase_locked",
  "fired": true,
  "windows": [
    {"t_start": 1.24, "t_end": 4.31, "confidence": 0.87}
  ],
  "per_node": null,
  "notes": "optional freeform string for audit trails",
  "detector_version": "phase_locked@0.1",
  "thresholds_ref": "detectors/phase_locked.yaml#v0.1"
}
```

- `fired: false` detectors may be omitted from `labels.json` or included with empty `windows`; a dataset convention picks one. v0 default: **include all, with empty `windows` on non-firing.** Makes downstream joins trivial.
- `per_node` is `null` unless the label is per-node (e.g. `unstable_bridge`, `dominant_cluster`).
- `windows` must be non-overlapping and sorted by `t_start`.
- Times in seconds, aligned to `state.npz` frames.

### 10.5 Compound and counterfactual declarations

Each detector declares exactly one of these types in its YAML config:

- `direct` — reads only §10.1 primary inputs + helpers.
- `compound` — additionally reads other detectors' outputs. Must list which ones.
- `counterfactual` — additionally calls `sim.ablate`. Must declare a rollout budget.

A detector cannot be both compound and counterfactual in v0. If that need arises, lift the dependency out or restructure.

### 10.6 Time-window semantics

- All windows are in seconds relative to run start, snapped to the nearest control-rate frame.
- Hysteresis and minimum-duration parameters merge adjacent satisfying intervals before emitting windows.
- Detectors may use future samples in v0 (offline / whole-run analysis is the default; streaming / online detection is not a v0 requirement). Document any detector that genuinely requires full-run access in its YAML (e.g. `regime_change` needs before/after context).

### 10.7 Confidence semantics

Confidence is **evidence margin**, `[0, 1]`, not probability-of-correctness.

Each detector's confidence function must:
- Be a documented deterministic function of observable quantities (not a learned score).
- Degrade smoothly as the condition approaches its threshold from above.
- Scale with sustain (longer windows above threshold → higher confidence, up to a cap).
- Be expressible in one or two lines in the detector YAML / pseudocode.

Typical form: `sigmoid(k * (observed − threshold)) * min(1, duration / min_duration)`.

### 10.8 Threshold representation

- Every detector has a YAML config at `detectors/<label>.yaml` with:
  ```yaml
  detector: phase_locked
  version: 0.1
  thresholds:
    r_lock: 0.9
    min_duration_s: 1.0
    hysteresis_s: 0.2
  confidence:
    k: 10.0
    duration_cap_s: 2.0
  ```
- No magic numbers in detector code. Changing a threshold → version bump. The `thresholds_ref` field in detector output pins the exact config used.
- Threshold tuning during the sniff test is expected; that is exactly what the version field is for.

### 10.9 Scope: runs, not streams

- v0 detectors operate on whole exported runs.
- Detector implementations **should** be causal where possible (especially the cheap ones like `phase_locked`), to make future streaming ports less painful. But this is a preference, not a contract.
- `regime_change` and the counterfactual detectors are not causal. That is declared in their YAML.

### 10.10 Counterfactual pattern (`brittle_lock`, `unstable_bridge`)

The only API for counterfactual evidence is `sim.ablate` (§9.9). Pattern:

```python
def detect_brittle_lock(run, cfg):
    if not phase_locked(run, cfg_pl).fired:                    # compound-on-direct is fine here
        return not_fired("phase_locked did not fire")
    for delta in cfg.perturbation_ladder:                      # ascending
        perturbation = select_perturbation(run, delta, cfg)    # deterministic choice
        alt = sim.ablate(run.config_path, perturbation, seed=run.seed)
        if not phase_locked(alt, cfg_pl).fired:
            return fired(conf=1.0 - delta / cfg.max_delta, notes=f"broken at {delta}")
    return not_fired("did not break within ladder")
```

Rules:

- The counterfactual detector **never** inspects the simulator's internal state during the ablation rerun. It only reads the resulting `RunArtifacts`, like any other detector.
- The rollout budget is a YAML field and hard-enforced. A detector that tries to run `N+1` rollouts fails closed.
- Ablation reruns inherit the parent run's seed unless the detector explicitly overrides (and documents why).

### 10.11 Determinism

Detectors are deterministic functions of `(artifacts, config)`. No RNG. If an underlying helper is non-deterministic (e.g. DBSCAN tie-breaking), it must be seeded from a constant or the parent run's seed — documented in the helper, not the detector.

### 10.12 What the detector layer can assume exists

Summary (i.e. the contract with §9):

- Full `state.npz`, `topology.json`, `events.jsonl` for the run.
- Derived-state helpers in `sim.derived`.
- `sim.ablate(config_or_path, perturbation, seed) → RunArtifacts` for counterfactual detectors.
- Absence of any "answer" field inside `state.npz`. If one appears, it is a sim contract violation.

---

## 11. First coding milestone

**Ship the simulator, detectors, and dataset dump. No model yet.**

Rationale: nothing can be trained or evaluated without ground-truth data, and the nastiest surprises live in the dynamics and in detector definitions. Earn those surprises first.

### Acceptance criteria

1. `python -m sim.run --config configs/regime_drifting.yaml --seed 42 --out runs/demo` produces `audio.wav`, `state.npz`, `events.jsonl`, `topology.json`, `labels.json`.
2. **Determinism:** same seed + config → byte-identical `state.npz`.
3. Detectors implemented for at least `phase_locked`, `drifting`, `polyrhythmic`, `phase_beating`, `dominant_cluster`, `groove`, `groove_collapse`. Remaining ontology entries can land in a follow-up.
4. `scripts/generate_dataset.py` produces **200 runs** across ≥4 regime templates, total <500 MB, in <30 min on a laptop.
5. **Sniff test:** pick 10 random clips blind, listen, read labels. ≥8 should feel right. Record the disagreements in `DIRECTORS_NOTES.md` for a follow-up detector pass.
6. No ML code yet. Those directories should not exist.

### Suggested layout

```
sim/
  garden.py        # state + dynamics
  render.py        # audio synthesis
  detectors.py     # label functions
  run.py           # CLI entry
configs/
  regime_*.yaml
scripts/
  generate_dataset.py
runs/              # gitignored, outputs
```
