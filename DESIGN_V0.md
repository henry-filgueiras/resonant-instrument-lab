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

Compact (13 labels), multi-label, testable. Each label has a **deterministic detector** computed from state / events.

| label                          | condition (sketch)                                                                                                                                         | instrument-native gloss       |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| `phase_locked`                 | Kuramoto order parameter `r > 0.9` sustained ≥ 1 s                                                                                                         | "it locked in"                |
| `drifting`                     | `r < 0.3` sustained ≥ 1 s                                                                                                                                  | "loose, wandering"            |
| `polyrhythmic`                 | ≥ 2 stable sub-clusters with distinct mean freq                                                                                                            | "two rhythms at once"         |
| `phase_beating` (was `beating`) | two pulse streams with small Δf whose pulse-arrival times drift with a slow periodic offset — the pulse-domain analogue of acoustic beating, not wave-level interference | "wobble"                      |
| `dominant_cluster`             | one cluster holds >50% of nodes at highest local `r`                                                                                                       | "the big group is running it" |
| `brittle_lock`                 | `r > 0.9` **and** smallest perturbation-to-unlock < threshold                                                                                              | "locked but fragile"          |
| `unstable_bridge`              | a single node whose coupling contribution is load-bearing for cluster coherence — ablating it in-sim drops the cluster's `r` below the lock threshold      | "held by a thread"            |
| `groove`                       | `r` stable in `[0.6, 0.85]` with small bounded oscillation                                                                                                 | "it's breathing"              |
| `groove_collapse`              | `groove` → `drifting` within a short window                                                                                                                | "lost the pocket"             |
| `beat_bloom` (was `interference_bloom`) | transient rise-and-fall in the `phase_beating` index for a node pair — streams approach near-coherence, briefly interlock, then drift apart          | "flutter then settle"         |
| `tension_break`                | `K₀` rising followed by sudden cluster reorganization                                                                                                      | "it snapped"                  |
| `flam`                         | pulse pair with 10–40 ms phase offset                                                                                                                      | "flammed unison"              |
| `regime_change`                | detected cluster-count change sustained past a hysteresis window                                                                                           | "different piece now"         |

A clip is labelled with the (possibly multiple) labels whose detectors fired, plus time windows and per-detector confidence scores. Detectors are part of the simulator package, not the model — they are ground truth.

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

## 9. First coding milestone

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
