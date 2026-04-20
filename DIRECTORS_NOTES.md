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

### Implementation contracts (frozen shape, no engine code yet)
- Simulator contract — `DESIGN_V0.md §9`. Defines scene vs run, canonical YAML, per-step export schema (`state.npz`), audio rendering, event schema, output artifact layout, determinism rules, derived-state helpers, and the `sim.run` / `sim.ablate` / `sim.detect` surface.
- Detector contract — `DESIGN_V0.md §10`. Fair-game vs privileged vs eval-only input boundary; output schema; confidence = evidence margin; thresholds in versioned YAML; counterfactual detectors only via `sim.ablate`.
- Per-label reference — `ONTOLOGY.md` at repo root.

### First implemented artifacts
- `sim/config.py` — config loader + validator; shared by the validator CLI and the run pipeline. Executable event set: `setK`, `nudge`, `impulse`, `ablate_node`. `move` is schema-rejected pending `pos_t` export.
- `sim/garden.py` — real distance-weighted Kuramoto integrator per §2.2 (explicit Euler, SDE noise). Amplitude and audio remain stubbed to constant 1.0 and silent-but-format-correct WAV.
- `sim/derived.py` — pure helpers over exported state per §9.8: `kuramoto_order`, `local_kuramoto_order`, `mean_phase_velocity`, `split_by_largest_velocity_gap` (returning a `VelocitySplit` NamedTuple), `sustained_windows` (find half-open frame-index windows where a 1D bool condition holds for a minimum span, with gap-tolerance hysteresis). No thresholds, no label output — state-derivation primitives only, shared by tests today and the detector layer tomorrow.
- `sim/__init__.py` — re-exports `load`, `validate`, `simulate`, `ConfigError`, `SCHEMA_VERSION`. Helpers stay in `sim.derived` (explicit import) to keep the package surface narrow.
- `scripts/validate_config.py` — thin CLI wrapping `sim.config.load`.
- `scripts/run_sim.py` — run one config end-to-end; writes the full §9.6 artifact set to `--out`.
- `tests/test_regimes.py` — automated regime smoke tests (one function per regime fixture — `locked`, `two_cluster`, `drifting`). Runs each config through `sim.simulate` into a tempdir and asserts a simple numerical signature of the regime label. Stdlib-only; runnable via `python tests/test_regimes.py` or `pytest`.
- `configs/regime_drifting.yaml` — DRIFTING reference (low K0, drifted behaviour; test asserts the 8 s run contains a ≥ 3 s sustained low-coherence window (`r < 0.5`, 0.25 s hysteresis) and tail mean `r < 0.40`; observed tail mean `r ≈ 0.29`, single merged sustained window spanning the full 8 s).
- `configs/regime_locked.yaml` — LOCKED reference; test asserts last-1s `r > 0.9`, observed `r = 0.9953`.
- `configs/regime_two_cluster.yaml` — TWO_CLUSTER reference; test asserts middle-band global `r` + bimodal velocity split + intra-cluster coherence. Observed tail mean `r = 0.628`, clean 4+4 split with bimodal ratio ~700.
- `requirements.txt` — `numpy>=2.0`, `pyyaml>=6.0`.
- `.gitignore` — excludes `runs/` outputs and Python bytecode.

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

### 2026-04-20 — Claude Opus 4.7 (doc pass 4 — first implemented artifact)
Turned the §9.2 simulator contract from prose into a machine-checkable pair: a reference config and a schema validator. Still no simulator dynamics, no detectors, no ML code.

- **Added `configs/regime_drifting.yaml`.** Eight nodes on a rough ring in the unit square, spread of intrinsic frequencies wide enough that low `K0=0.12` plus `eta=0.03` will not lock. Four events (`setK`, `nudge`, `impulse`, `move`) deliberately included to exercise every primary event type in the schema without rescuing the regime. `ablate_node` intentionally absent — it is counterfactual-only per `DESIGN_V0.md §9.5`.
- **Added `scripts/validate_config.py`.** Stdlib + PyYAML; ~100 lines; closed-set key validation, numeric range and type checks with bool rejection, event-type switch on the §9.5 closed set, ascending-time invariant on events, per-node index bounds. Validated the reference config clean; eight synthetic bad-config cases each produced a path-and-value diagnostic pointing to the exact offender.
- **Schema resolution: `meta` block added as an optional top-level field** with required `name: str` and optional `notes: str`. Rationale: gave the reference config a place to name its regime and explain itself without filename-as-identity brittleness. Kept out of `topology.json` and other detector-facing exports so it does not leak into §10.2 privileged-truth territory. Updated §9.2 to reflect.
- **Schema range pinning.** §9.2 previously gave units but not range bounds for most fields; `validate_config.py` forced the question. Resolved conservatively: `omega_0_hz ∈ [0.5, 8.0]` (from §2.1), `voice ∈ [0, 7]` (v0's 8-voice palette), `pos` and `gamma` ∈ `[0, 1]`, `K0 ≥ 0`, `sigma > 0` (strict — it divides in the coupling kernel), `eta ≥ 0`. Ranges are now enumerated inline in §9.2 so future detector / sim code cannot drift from them silently.
- **`sim.run` CLI surface now has a running companion.** §9.9 gained a line pinning `scripts/validate_config.py` as the first implemented entry point; everything under `sim.*` is still contract-only.
- **PyYAML adopted as the first (and so far only) external dependency.** No `requirements.txt` yet — defer until there are two or more deps. Validator fails fast with a one-line install hint if PyYAML is missing.

### 2026-04-20 — Claude Opus 4.7 (doc pass 5 — first executable scaffold)
Thinnest possible executable pipeline. Still no real dynamics; still no detectors, ablation, or ML code.

- **Package layout: `sim/` created.** `sim.config` (moved from `scripts/validate_config.py`), `sim.garden` (stub simulator), `sim.__init__` (re-exports). `scripts/validate_config.py` is now a thin CLI over `sim.config.load` — single source of truth for schema validation. Both CLIs prepend the repo root to `sys.path` so they run from a fresh clone with no packaging.
- **Stub evolution rule: per-node intrinsic-rate phase advance, no coupling.** Chosen over alternatives (zero-motion placeholder; random walk; trivial Kuramoto) because (a) it's genuinely deterministic, (b) it lets `pulse_fired` actually fire at sensible rates so downstream detector scaffolding sees non-trivial data, (c) it makes `nudge` events visible via a phase-velocity change, and (d) the seam where real pairwise coupling goes is a single line inside the main integration loop. Amplitude is constant 1.0, `move` and `ablate_node` events are recorded but have no dynamical effect — clearly marked in the module docstring as stubs.
- **Artifacts contract honored on disk.** `state.npz`, `events.jsonl`, `topology.json`, `audio.wav`, `config.yaml` all written per §9.6. `state.npz` contains all six §9.3 arrays with correct dtypes and `(T, N)` or `(T,)` shapes. Events include generated `id` (`evt_NNN`) and `source: "scheduled"` fields per §9.5. Audio is silent stereo int16 WAV of the contract-specified duration and rate — contract-correct but explicitly no content.
- **Byte-level determinism verified.** Two back-to-back runs of the same config produced `cmp -s`-identical `state.npz`, `events.jsonl`, `topology.json`, `audio.wav`, and `config.yaml`. The one non-obvious choice: `.npz` is written as a manually-constructed zip with `ZIP_STORED` and a fixed `(1980,1,1)` date for every entry, because `numpy.savez`'s default path uses wall-clock zip timestamps that break byte-identity. JSON outputs use `sort_keys=True` to pin dict ordering. `config.yaml` is copied verbatim rather than re-emitted.
- **Deferred contract gap flagged: position time series is not exported.** `move` events change geometry but §9.3 does not include a `pos_t` field, so a detector that needs mid-run positions has no signal to read. The scaffold logs move events faithfully to `events.jsonl` and leaves this for a follow-up contract pass (either add `pos_t: (T, N, 2)` to `state.npz`, or declare positions static-after-t0 and forbid `move` events). **Not resolved in this pass — `move` events are recorded but have no effect on exported state.**
- **`requirements.txt` added** (deps threshold crossed: PyYAML + NumPy). `numpy>=2.0` chosen for Python 3.14 support; `pyyaml>=6.0` carried forward.
- **Smoke test done and torn down.** Validator + two runs + byte-diff + nudge-effect sanity check in a throwaway venv; test fixtures deleted before commit. The scaffold is the artifact, not the test runs.

### 2026-04-20 — Claude Opus 4.7 (doc pass 6 — real coupling + first smoke test)
Replaced the stub phase integration with real distance-weighted Kuramoto coupling. Added the LOCKED reference config and a minimal automated smoke test. Resolved the `move`-event truth gap by removing `move` from the executable schema.

- **Real coupling live.** `sim/garden.py` now integrates §2.2 exactly: `dθ/dt = 2π·f + Σ Kᵢⱼ sin(θⱼ−θᵢ) + ξ`, with `Kᵢⱼ = K₀ · exp(−dᵢⱼ/σ)` (zero diagonal) precomputed once from static positions and rescaled by `K₀` on every `setK` event. Explicit Euler at `dt = 1 / control_rate_hz`. Noise is Ito-discretized as `η · √dt · 𝒩(0,1)`; `η = 0` skips the RNG draw so noise-off runs consume identical RNG state whatever the config. Determinism preserved — two back-to-back runs of `regime_locked.yaml` produced byte-identical `state.npz`, `events.jsonl`, `topology.json`, `audio.wav`, and `config.yaml`.
- **Units pinned in §2.2.** Writing real dynamics forced the question. `fᵢ = omega_0_hz` in Hz (promoted to 2π·f in the integrator); `K₀` in rad/s; `σ` in unit-square distance; `η` in rad/√s. The contract had been silent on most of these; the clarification is now load-bearing and documented inline.
- **`move`-event truth gap — chose Option B (remove).** Reasoning: making `move` real is not cheap — it requires a new `pos_t: (T, N, 2)` field in `state.npz`, a `K_mat` rebuild on every move event, and an audio-pan-tracking decision. That is a full contract extension. The honest minimum was dropping `move` from `EVENT_SHAPES` in `sim/config.py`, removing the move event from `configs/regime_drifting.yaml`, and marking §9.5 "Deferred" for `move` with a pointer back to `pos_t`. The validator now emits `unknown event type 'move'; allowed [ablate_node, impulse, nudge, setK]` on any config that still references it. Reintroduce when `pos_t` lands.
- **`ablate_node` asymmetry acknowledged.** Kept in the schema even though `sim/garden.py` ignores it at runtime. Reasoning: `ablate_node` is the canonical counterfactual perturbation primitive for the future `sim.ablate` path (§10.10). Removing it would require re-adding later; leaving it inert is forward-compatible. `move` was different — its inertness was a contract hole, not a future feature.
- **LOCKED reference config — `configs/regime_locked.yaml`.** Eight nodes clustered near (0.5, 0.5), ω∈[2.00, 2.40] Hz (Δω in angular terms ≈ 2.5 rad/s), `K₀ = 1.50 rad/s` through a σ = 0.30 kernel. At the typical 0.10–0.15 pair spacing each node sees ~6–7 rad/s of total coupling — well above the Kuramoto bifurcation. Measured `r = 0.9953` over the last 1.0 s of the run. Zero events, zero noise; ω is static.
- **First automated test — `tests/test_locking.py`.** Runs `regime_locked.yaml` through `sim.simulate` into a tempdir, loads `state.npz`, computes `r(t) = |mean(exp(iθ))|`, and asserts both `mean(r_{last 1s}) > 0.9` and `min(r_{last 1s}) > 0.85`. Stdlib-only (no pytest dependency); runnable as `python tests/test_locking.py` or discoverable via `pytest`. Observed lock is so tight that the margin above threshold is > 0.09 — robust to small numerical variation.
- **Cross-checked that DRIFTING still drifts.** Sanity-running `configs/regime_drifting.yaml` through the real integrator: last-1s mean `r = 0.267`, below the `drifting` detector's `r_drift = 0.3` — parameters still honour their regime label after the coupling change.
- **Not done in this pass.** Amplitude dynamics (γ still unused); audio grain synthesis (still silent); `ablate_node` runtime handling; any further regime configs. Deliberately scoped out per the user's guardrails.

### 2026-04-20 — Claude Opus 4.7 (doc pass 7 — intermediate regime fixture)
Added the first intermediate-regime fixture and the numerical sanity test that confirms the simulator produces more than just "drift" and "full lock".

- **Tuning logic for `configs/regime_two_cluster.yaml`.** Two groups of four nodes clustered at ≈(0.25, 0.46) and ≈(0.75, 0.46); inter-centroid distance 0.5. Chose `σ = 0.15` so the distance kernel itself does most of the work: `exp(-0.10/0.15) ≈ 0.51` intra vs `exp(-0.50/0.15) ≈ 0.036` inter — a ~14× ratio. `K₀ = 2.0 rad/s` puts per-node intra-cluster coupling at ~3 rad/s (safely above the 1.26 rad/s intra-Δω, so each group locks) and per-node cross-cluster coupling at ~0.29 rad/s (far below the 5.53 rad/s inter-mean-ω gap, so groups stay separate). Bimodal ω: group A in `[2.00, 2.20]` Hz, group B in `[2.90, 3.10]` Hz, giving a drift period `≈ 1.13 s`.
- **Assertion shape in `tests/test_regimes.py::test_two_cluster_regime_shows_bimodal_structure`.** Four checks on a 3-second tail window (≈ 2.6 drift periods, enough for a stable time-average of the oscillating global `r`):
  1. Global Kuramoto `r` tail mean in `(0.35, 0.85)` — rejects both drift and global lock.
  2. Per-node mean phase velocity splits: the largest adjacent gap in the sorted velocity array dominates the others by at least 3× (measured ratio: **706.4**).
  3. The split is roughly balanced (4+4 tolerated, 3+5 still accepted). Observed: exact 4+4.
  4. Each sub-cluster's own local Kuramoto `r` exceeds 0.8. Observed: 0.993 / 0.993.
- **Predicted vs measured tail `r`.** For two internally-locked clusters drifting linearly apart, `r_global(t) = r_intra · |cos((φ_A − φ_B)/2)|`, so the time-average is `E[r_global] ≈ (2/π) · r_intra ≈ 0.637 · 0.993 = 0.632`. Measured: **0.6276**. Within 1%.
- **Test file consolidated: `tests/test_locking.py` → `tests/test_regimes.py`.** Both regime tests now live in one file with a shared `_run(config_path)` helper and a shared `_kuramoto(theta)` utility. The consolidation is forward-looking: future regime fixtures (`BRITTLE`, `SWEEP`, `PERTURBED`, `COLLAPSE`) will land here as additional `test_*` functions rather than spawning one file per fixture.
- **Sanity check on DRIFTING.** Not retested in this pass beyond what doc pass 6 already measured (tail mean `r = 0.267`). DRIFTING remains covered by the implicit test-via-validator-and-run; not worth adding a dedicated numerical test for "this config fails to lock" until we have a detector framework to test against.
- **What is still not done.** No detector framework. No clustering / DBSCAN machinery — the test uses `np.argsort` + `np.diff` on eight per-node mean velocities, which is robust specifically because the physics produces a dominant single gap. When the detector layer lands, it will need a more general cluster-assignment helper (`sim.derived.cluster_assignments`, per §9.8); this test deliberately does not prototype that.

### 2026-04-20 — Claude Opus 4.7 (doc pass 9 — sustained-window primitive + DRIFTING sanity test)
Closed the simulator sanity triangle (`locked`, `two_cluster`, now `drifting`) and landed the first reusable time-window primitive that sustained-condition detectors (`phase_locked`, `drifting`, `groove`, …) will compose over. No detector framework, no YAML thresholds, no label schema — just a state-level primitive and a regime fixture test.

- **`sustained_windows(condition, min_duration_frames, hysteresis_frames=0)` in `sim/derived.py`.** Takes a 1D bool-castable time series and returns sorted, non-overlapping half-open `[start, end)` frame-index intervals. The interval span (including any absorbed hysteresis gaps) is what gets compared to `min_duration_frames`; the window's endpoints always land on True frames (trailing False hysteresis is never included). `hysteresis_frames=0` disables merging and returns strict contiguous True runs only. Pure function — frame counts only, no config / wall-clock / thresholds. Callers convert seconds to frames themselves via `control_rate_hz`. Edge cases verified in-session: empty input → `[]`, all-False → `[]`, all-True → `[(0, T)]` if it meets `min_duration_frames` else `[]`, 2D input rejected, `min_duration_frames < 1` rejected, `hysteresis_frames < 0` rejected, tiny sub-threshold runs dropped silently, mid-run False gaps `> hysteresis_frames` split into separate windows.
- **Interval convention chosen: half-open `[start, end)`.** Rationale: matches Python slicing (`condition[start:end]` yields exactly the window's frames) and NumPy indexing, and `end - start` gives span directly without off-by-one. The alternative (closed `[start, end]`) would have the nicer "last True index is `end`" property but forces every caller to write `+ 1` for length and for slicing — worse for the three or four upcoming sustained-condition detectors.
- **Hysteresis semantics chosen: gap-absorption at merge time.** Alternatives considered: (a) count only True frames against `min_duration_frames` (rejected — makes "sustained for 3 s" mean "at least 3 s of True out of some larger span", which is a statistical test, not a sustained-condition test); (b) require the condition to hold at both endpoints *and* for all but `hysteresis_frames` interior frames (rejected — harder to implement deterministically, and equivalent in most cases to the span-based rule once `min_duration_frames` is comparable to hysteresis). Current rule: a False stretch between two True runs of length `<= hysteresis_frames` is folded into the window; anything longer splits. Simple, deterministic, and the only interior-False-frame count that matters is the one used for merging.
- **DRIFTING sanity test — `tests/test_regimes.py::test_drifting_regime_sustains_low_coherence`.** Two checks on the 8 s run of `configs/regime_drifting.yaml`: (1) `sustained_windows(r < 0.5, min=2.0 s, hyst=0.25 s)` returns at least one window, with longest `>= 3.0 s`; (2) tail-2 s mean `r < 0.40`, pinning drifting well below two_cluster's middle band (observed `0.63`) and locked's near-unity (observed `0.995`). Observed: one merged sustained window spanning the full 1599 frames (8.00 s); tail mean `r = 0.2902`; fraction of frames with `r < 0.5` = `0.859`.
- **Regime separation is structural, not numeric.** The same `sustained_windows(r < 0.5, min=2.0 s, hyst=0.25 s)` call returns `[]` on both locked (r stays > 0.95, so `low` is never True) and two_cluster (whose longest merged sub-second valley is 86 frames = 0.43 s, well below the 2.0 s floor). Noted in the test's inline comment — the primitive itself is the discriminator, we do not need separate threshold knobs per regime.
- **Existing `configs/regime_drifting.yaml` worked unchanged.** No parameter tuning needed. Tail mean `r ≈ 0.29` on the same seed/config that the earlier doc-pass-6 cross-check measured at `r ≈ 0.27`; the small difference is the different tail length (2 s vs 1 s) used for averaging, not a simulator change — `state.npz` output is still byte-identical run-to-run.
- **Not done in this pass.** No detector registry. No label schema. No thresholds YAML. No notebook work. No new configs. Just one primitive and one test.

### 2026-04-20 — Claude Opus 4.7 (doc pass 8 — derived-state helpers)
Extracted the physical-summary primitives out of the tests into the first cut of `sim/derived.py`. Pure functions only; no thresholds, no label output, no YAML loading. Refactored `tests/test_regimes.py` to consume them.

- **Four helpers landed in `sim/derived.py`.**
  - `kuramoto_order(theta)` — `(T, N) → (T,)` or `(N,) → float`. Shape semantics explicitly documented; accepts any dtype for which `np.exp(1j * theta)` is defined (dtype of output matches the phase dtype's real counterpart).
  - `local_kuramoto_order(theta, members)` — thin wrapper that fancy-indexes `theta[..., members]` then calls `kuramoto_order`. Kept as a named helper because "local coherence of cluster X" is the phrase future detectors will want.
  - `mean_phase_velocity(phase_vel, start_idx=None, end_idx=None)` — per-node time-mean over a Python-style slice of `(T, N)`. Negative indices work. Empty windows raise rather than silently returning NaN.
  - `split_by_largest_velocity_gap(mean_vel) → VelocitySplit(low_indices, high_indices, gap, bimodal_ratio)`. Explicitly scoped as a bare-minimum bimodality test; docstring warns against using it as a general cluster-assignments substitute. Index arrays sorted by node index (not by velocity) for readability; `bimodal_ratio = gap / next-largest-gap`, with `+inf` when only one gap exists and `0.0` when all gaps are zero.
- **`VelocitySplit` as a `typing.NamedTuple`.** Chosen over `dataclass` for the smaller API surface and the native tuple-destructure ergonomics. Immutable.
- **Helpers deliberately not re-exported from `sim/__init__.py`.** Callers import from `sim.derived` explicitly. Matches §9.8's "helpers live in `sim.derived`" wording and keeps the package surface from ballooning as more primitives land (`cluster_assignments`, `pulse_times_of`, `cluster_signature`, etc.).
- **Test refactor is implementation-only.** Same assertions, same tolerances, same numbers. Measured post-refactor: locked `r = 0.9953`; two_cluster tail mean `r = 0.6276`, split 4+4, intra `(0.993, 0.993)`, bimodal ratio 706.4. Byte-for-byte identical to doc pass 7's measurements — the behavior didn't change because the math didn't change; only the code path did.
- **Edge-case sanity verified in-session.** 1D Kuramoto returns Python `float`; 2D returns `(T,)`. `local_kuramoto_order` preserves the 2D caller's time axis. `mean_phase_velocity` with `start_idx=-n` works on `(T, N)` and raises on empty windows. `split_by_largest_velocity_gap` handles `N=2` (`bimodal_ratio = inf`) and all-equal velocities (`ratio = 0.0`) without crashing.
