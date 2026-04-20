# Director's Notes

Living design doc. Two sections: **Current Canon** (architecture, invariants, present-state truth; edit in place) and **Resolved Dragons and Pivots** (append-only devlog). When a fact stops being true in Canon, move it **verbatim** into the archive instead of editing or deleting.

---

## Current Canon

### Project thesis
Teach a small local language model to ground musician-native semantics (phase locking, groove, brittleness, interference, etc.) in a bounded synthetic world where those semantics are computable from state. Evaluate on **intervention grounding** — does the model's suggested action actually move the sim the way it claimed?

### Toy world — Coupled Oscillator Garden (v0)
- `N = 8` nodes in the unit square.
- Kuramoto-style phase coupling with distance kernel `Kᵢⱼ = K₀ · exp(−dᵢⱼ/σ)`.
- Optional white-noise perturbation `ξᵢ(t)` with amplitude `η`.
- Per-node fields: `pos, ω₀, θ, A, γ, voice`.
- Audio is **pulse-driven**: each phase-zero crossing fires a percussive grain with a fixed-pitch voice; stereo pan from `pos.x`. Audio-rate tone coupling is deferred to v0.1.
- Scheduled (not real-time) events: `nudge`, `impulse`, `move`, `setK`.

### Run exports
`audio.wav` (22050 Hz stereo), `state.npz` (`θ`, `A` at 200 Hz), `events.jsonl`, `topology.json`, `labels.json`.

### Ontology (13 labels, multi-label, detector-grounded)
`phase_locked`, `drifting`, `polyrhythmic`, `phase_beating`, `dominant_cluster`, `brittle_lock`, `unstable_bridge`, `groove`, `groove_collapse`, `beat_bloom`, `tension_break`, `flam`, `regime_change`. Each has a deterministic detector; labels are ground truth, not model outputs. Detector confidence is **evidence margin** (how strongly / stably the defining condition was met), not epistemic uncertainty.

### Data pipeline
Regime-template sampler (`LOCKED`, `DRIFTING`, `TWO_CLUSTER`, `BRITTLE`, `SWEEP`, `PERTURBED`, `COLLAPSE`). v0 target: ~5 000 clips × ~5 tasks = ~25 k examples.

### Task suite
T1 caption, T2 semantic QA, T3 compare, T4 anomaly, T5 explain change, T6 suggest intervention. T6 is the headline evaluable task.

### Model stack (planned)
- Trained encoders: small CNN (or adapter over frozen small CLAP) on mel spectrograms; small transformer on per-node state; topology as concat tokens (no GNN in v0).
- Frozen base LM (Qwen2.5-0.5B or Llama-3.2-1B) + LoRA.
- SFT only for v0; auxiliary losses only if SFT plateaus.

### Evaluation tiers
L1 label accuracy → L2 noise / perturbation robustness → L3 intervention grounding against random + heuristic baselines. Generalization probes across `N`, voice palette, held-out regime.

### Out of scope for v0
Real-time interaction, audio-rate tone coupling, GNNs, RLHF, third-party music datasets, human listening studies.

### Implementation contracts (frozen shape, no code yet)
- Simulator contract — `DESIGN_V0.md §9`. Defines scene vs run, canonical YAML, per-step export schema (`state.npz`), audio rendering, event schema, output artifact layout, determinism rules, derived-state helpers, and the `sim.run` / `sim.ablate` / `sim.detect` surface.
- Detector contract — `DESIGN_V0.md §10`. Fair-game vs privileged vs eval-only input boundary; output schema; confidence = evidence margin; thresholds in versioned YAML; counterfactual detectors only via `sim.ablate`.
- Per-label reference — `ONTOLOGY.md` at repo root.

### First coding milestone
Simulator + detectors + dataset dump only. **No model code yet.** Acceptance criteria are recorded in `DESIGN_V0.md §11` (renumbered from §9 when the contracts were inserted).

### Working conventions
- Design narrative: `DESIGN_V0.md`.
- Per-label implementation sheet: `ONTOLOGY.md`.
- Project canon and pivots: this file.
- Per-exchange: update this file, commit design + notes together, **do not push**.

---

## Resolved Dragons and Pivots

### 2026-04-20 — Claude Opus 4.7
Initial design pass for v0. Key pivots made while drafting:

- **Pulse-driven audio over audio-rate tone coupling.** The obvious starting assumption was continuous tones with Hz-scale beating. Chose pulses instead because (a) it's cheaper to simulate, (b) emergent *rhythm* is the strongest musical phenomenon at the timescale of short clips, and (c) beating can be approximated at the perceptual level by tight near-unison pulses (the `flam` label). Audio-rate tone coupling reconsidered in v0.1.
- **No GNN in v0.** Tempting but unjustified at `N = 8` with known geometry. Concat tokens with positional embeds are sufficient. Revisit only if topology reasoning visibly under-performs.
- **First milestone is data, not model.** Initial draft had a minimal end-to-end model as the first milestone. Reversed: detector definitions are the hardest part of the project and must be sniff-tested against human ears before any model training is meaningful.
- **Intervention grounding as headline eval.** Chosen over label-F1 because label accuracy can be gamed by surface features; running the model's suggestion through the sim and measuring its effect is the only test that distinguishes grounded understanding from pattern matching.
- **Frozen LM + LoRA over full fine-tune.** Single-researcher scale; full fine-tune of even a 0.5B model is overkill for the data budget and loses generality.

### 2026-04-20 — Claude Opus 4.7 (doc pass 2)
Terminology tightening for the pulse-first v0 world. No scope changes.

- **`beating` → `phase_beating`.** The original name silently inherits acoustic-wave semantics (Hz-scale amplitude modulation between continuous tones). v0 is pulse-driven; what the detector measures is slow periodic drift between pulse-arrival times of near-frequency streams. Renamed to name the actual quantity. Old name kept in parentheses in the ontology table so the drift is explicit.
- **`interference_bloom` → `beat_bloom`.** Same audio-rate overclaim as above; "interference" strongly suggests wave-level superposition. The phenomenon is a transient excursion of the `phase_beating` index — two streams briefly approach coherence and fall back out. `beat_bloom` ties it to its steady-state sibling and drops the misleading connotation.
- **`unstable_bridge` — name kept, detector redefined.** Old wording assumed a discrete edge set ("edge whose removal splits the graph"); the garden is distance-weighted, fully connected with no discrete edges. Redefined as a **load-bearing node** whose in-sim ablation drops the cluster's order parameter below the lock threshold. Name kept because the musical "bridge about to collapse" connotation is worth preserving.
- **Detector confidence doctrine recorded.** Confidence in `labels.json` is evidence margin (how strongly / stably a condition was met), not epistemic uncertainty. Detectors are deterministic; graded scores exist so downstream tasks can distinguish marginal from solid.
- **README promoted to a proper front door.** Public-facing summary surfacing the project thesis, the v0 shape, and — most prominently — intervention grounding as the headline evaluation principle.

### 2026-04-20 — Claude Opus 4.7 (doc pass 3 — implementation contracts)
Paperwork-shaped implementation bridge. Still no code.

- **`ONTOLOGY.md` extracted.** The per-label detail previously embedded in `DESIGN_V0.md §3` now lives in a dedicated file, reshaped into a consistent per-label schema (kind / intuition / detector sketch / inputs / knobs / confidence source / failure modes / overlaps). `DESIGN_V0.md §3` shrinks to a pointer plus the existing doctrine on confidence and pulse-domain framing. Rationale: the table was narrative-shaped, but detector code needs a reference-shaped artifact.
- **Simulator contract — `DESIGN_V0.md §9` (new).** Freezes the shape before any code: scene-vs-run vocabulary, canonical YAML, per-step export fields, audio-rendering rules, event schema, artifact layout, determinism, derived-state helpers, CLI / programmatic surface. `sim.ablate` is designated as the only counterfactual back door.
- **Detector contract — `DESIGN_V0.md §10` (new).** The key boundary: fair-game inputs (state / topology / events / helpers / `sim.ablate`) vs privileged latent truth (regime-template name, seed, any "answer" fields) vs eval-only signals (regime name for stratified reporting, intended bifurcation times). Output schema, windowed semantics, confidence = evidence margin, thresholds in versioned YAML, compound-vs-counterfactual declarations. `ablate_node` added to the event-type closed set so `unstable_bridge` has a concrete counterfactual primitive.
- **First coding milestone renumbered §9 → §11.** Acceptance criteria unchanged.
- **Working convention recorded: simulator must never export "answer" latent fields** (e.g. a hypothetical `is_locked`). Dual-enforced — sim contract forbids writing them, detector contract forbids reading them. This is the single most load-bearing boundary for keeping the detection problem honest.
- **Known-weak-spot list published in `ONTOLOGY.md`.** `groove`, `flam`, `polyrhythmic`, coarse-`N=8` `dominant_cluster` threshold, and ablation cluster-matching for `unstable_bridge`. Flagged but not reopened — these are sniff-test-driven iterations, not design work.
